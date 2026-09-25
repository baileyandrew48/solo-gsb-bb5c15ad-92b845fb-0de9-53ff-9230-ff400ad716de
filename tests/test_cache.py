"""SQLite 持久幂等缓存测试。

覆盖：读写与 LRU 访问顺序、跨“重启”（重新打开同一数据库文件）恢复、
成功与错误记录格式、写入/访问/淘汰事务原子性（无法识别的存储版本
应在启动时明确报错）、等价写法（H2O 与 OH2）共享签名、旧格式签名
记录的 GET 兼容。
"""

import json
import sqlite3

import pytest

from app.cache import (
    SCHEMA_VERSION,
    CacheSchemaVersionError,
    ReactionCache,
    build_signature,
    reaction_id_for,
    species_key,
)
from app.parser import parse_formula
from app.balancer import Species


def _species(text: str) -> Species:
    formula = parse_formula(text)
    return Species(raw=text, normalized=formula.normalized, formula=formula)


def _make_record(cache: ReactionCache, tag: str, *, error=None):
    """以唯一元素标签造一条反应记录，避免与其它测试的签名碰撞。"""
    reactants = [_species(f"{tag}2")]
    products = [_species(tag)]
    sig = build_signature(reactants, products)
    return cache.put(
        sig,
        reactants,
        products,
        None if error is not None else {"k": 2},
        [tag],
        error,
    ), sig


# ------------------------------------------------------------ 基本读写
def test_put_get_roundtrip():
    c = ReactionCache(":memory:")
    record, sig = _make_record(c, "He")
    assert record.reaction_id == reaction_id_for(sig)
    fetched = c.get(record.reaction_id)
    assert fetched is not None
    assert fetched.reaction_id == record.reaction_id
    assert fetched.signature == sig
    assert fetched.coefficients == {"k": 2}
    assert fetched.elements == ["He"]
    assert fetched.error is None
    assert fetched.canonical_reactants == record.canonical_reactants
    assert c.get("rxn_missing") is None
    assert c.get_by_signature(sig) is not None
    assert len(c) == 1


def test_error_record_roundtrip():
    c = ReactionCache(":memory:")
    err = {
        "code": "FORMULA_PARSE_ERROR",
        "message": "无法解析",
        "details": {"input": "Xx"},
    }
    record, _ = _make_record(c, "Ar", error=err)
    fetched = c.get(record.reaction_id)
    assert fetched.error == err
    assert fetched.coefficients == {}
    assert fetched.elements == ["Ar"]


def test_upsert_does_not_duplicate():
    c = ReactionCache(":memory:")
    record, sig = _make_record(c, "Ne")
    c.put(sig, [_species("Ne2")], [_species("Ne")], {"k": 3}, ["Ne"], None)
    assert len(c) == 1
    fetched = c.get(record.reaction_id)
    assert fetched.coefficients == {"k": 3}


# ------------------------------------------------------------ LRU 淘汰
def test_lru_evicts_least_recently_accessed(tmp_path):
    db = tmp_path / "c.db"
    c = ReactionCache(db, maxsize=3)
    r1, _ = _make_record(c, "Li")
    r2, _ = _make_record(c, "Be")
    r3, _ = _make_record(c, "B")
    # 访问 r1：顺序变为 r2(最旧), r3, r1
    assert c.get(r1.reaction_id) is not None
    r4, _ = _make_record(c, "C")  # 应淘汰最久未访问的 r2
    assert c.get(r2.reaction_id) is None
    for rid in (r1.reaction_id, r3.reaction_id, r4.reaction_id):
        assert c.get(rid) is not None
    assert len(c) == 3


def test_get_updates_last_accessed_order(tmp_path):
    db = tmp_path / "c.db"
    c = ReactionCache(db, maxsize=2)
    r1, _ = _make_record(c, "N")
    r2, _ = _make_record(c, "O")
    c.get(r1.reaction_id)  # r1 比 r2 新
    _make_record(c, "F")   # 淘汰 r2
    assert c.get(r1.reaction_id) is not None
    assert c.get(r2.reaction_id) is None


# ------------------------------------------------------------ 跨重启持久化
def test_records_survive_reopen(tmp_path):
    db = tmp_path / "persist.db"
    c1 = ReactionCache(db)
    ok, sig = _make_record(c1, "Na")
    err = {"code": "IMPOSSIBLE_BALANCE", "message": "无解", "details": {}}
    bad, bad_sig = _make_record(c1, "Mg", error=err)
    c1.close()

    c2 = ReactionCache(db)  # 模拟容器重启
    fetched = c2.get(ok.reaction_id)
    assert fetched is not None
    assert fetched.coefficients == {"k": 2}
    assert fetched.canonical_reactants == ok.canonical_reactants
    bad_fetched = c2.get(bad.reaction_id)
    assert bad_fetched is not None
    assert bad_fetched.error == err
    assert c2.get_by_signature(sig) is not None
    assert c2.get_by_signature(bad_sig) is not None
    assert len(c2) == 2


def test_lru_order_survives_reopen(tmp_path):
    db = tmp_path / "lru.db"
    c1 = ReactionCache(db, maxsize=2)
    r1, _ = _make_record(c1, "Al")
    r2, _ = _make_record(c1, "Si")
    c1.get(r1.reaction_id)  # r1 最新
    c1.close()

    c2 = ReactionCache(db, maxsize=2)
    _make_record(c2, "P")  # 重启后仍应淘汰最旧的 r2
    assert c2.get(r2.reaction_id) is None
    assert c2.get(r1.reaction_id) is not None


# ------------------------------------------------------------ 存储版本校验
def test_unknown_schema_version_raises(tmp_path):
    db = tmp_path / "future.db"
    c = ReactionCache(db)
    c.close()
    # 模拟未来版本写入的库
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE cache_meta SET schema_version = ? WHERE key='schema_version'",
        (SCHEMA_VERSION + 999,),
    )
    conn.commit()
    conn.close()

    with pytest.raises(CacheSchemaVersionError) as exc:
        ReactionCache(db)
    assert "无法识别的缓存存储版本" in str(exc.value)


def test_missing_version_meta_raises(tmp_path):
    db = tmp_path / "meta.db"
    c = ReactionCache(db)
    c.close()
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM cache_meta")
    conn.commit()
    conn.close()

    with pytest.raises(CacheSchemaVersionError):
        ReactionCache(db)


# ------------------------------------------------------------ 并发
def test_concurrent_writes_remain_consistent(tmp_path):
    import threading

    db = tmp_path / "mt.db"
    c = ReactionCache(db, maxsize=10_000)

    # 用可解析且随 i 变化的下标构造 100 个互不相同的签名：
    # 反应物 H_{2..101} / O2，生成物 H2O / CO2——H 下标取 i+2（≥2），
    # 全部合法且每个 i 一个新签名。
    def record_at(i: int):
        reactants = [_species(f"H{i + 2}"), _species("O2")]
        products = [_species("H2O"), _species("CO2")]
        sig = build_signature(reactants, products)
        c.put(sig, reactants, products, {"k": 1}, ["C", "H", "O"], None)

    def worker(start: int):
        for i in range(start, start + 25):
            record_at(i)

    threads = [threading.Thread(target=worker, args=(base,))
               for base in range(0, 100, 25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 100 个唯一签名 -> 100 条记录（并发写入也不应丢失或损坏）
    assert len(c) == 100

    # 重新打开做完整性校验
    c.close()
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


# ------------------------------------------------------------ 端到端：重启恢复
def test_service_balance_record_recovers_after_reopen(tmp_path, monkeypatch):
    """通过 service 层写入的成功/错误记录，重建缓存实例后仍可取回。"""
    from app import service
    from fastapi.testclient import TestClient
    from app.main import app

    db = tmp_path / "svc.db"
    persistent = ReactionCache(db)
    monkeypatch.setattr(service, "cache", persistent)

    client = TestClient(app)
    resp = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    )
    assert resp.status_code == 200
    rid = resp.json()["reaction_id"]

    err_resp = client.post(
        "/api/v1/balance", json={"reactants": ["Zz"], "products": ["H2O"]}
    )
    assert err_resp.status_code == 400
    err_rid = err_resp.json()["reaction_id"]
    persistent.close()

    # 模拟容器重启：用同一数据库文件重新打开并挂回 service
    reopened = ReactionCache(db)
    monkeypatch.setattr(service, "cache", reopened)

    got = client.get(f"/api/v1/reactions/{rid}")
    assert got.status_code == 200
    body = got.json()
    assert body["reaction_id"] == rid
    assert body["cached"] is True
    assert body["coefficients"]["vector"] == [2, 1, 2]

    got_err = client.get(f"/api/v1/reactions/{err_rid}")
    assert got_err.status_code == 400
    assert got_err.json()["error"]["code"] == "FORMULA_PARSE_ERROR"
    assert got_err.json()["reaction_id"] == err_rid
    reopened.close()


# ------------------------------------------------------------ 签名：等价写法
def test_signature_ignores_element_order():
    """H2O 与 OH2 组成、电荷相同：签名不得混入规范化文本。"""
    reactants = [_species("H2"), _species("O2")]
    assert build_signature(reactants, [_species("H2O")]) == build_signature(
        reactants, [_species("OH2")]
    )


def test_signature_preserves_sides():
    """同一组物种整体换侧（反应物 <-> 生成物）必须得到不同签名。"""
    left = [_species("H2"), _species("O2")]
    right = [_species("H2O")]
    assert build_signature(left, right) != build_signature(right, left)


def test_signature_includes_charge_and_multiplicity():
    """电荷不同则签名不同；同一物种在同一侧的重数也计入签名。"""
    assert build_signature([_species("SO4^2-")], []) != build_signature(
        [_species("SO4")], []
    )
    assert build_signature([_species("H2"), _species("H2")], []) != build_signature(
        [_species("H2")], []
    )


def test_equivalent_writings_share_record():
    """缓存层：H2O 写入的记录可被 OH2 的签名命中。"""
    c = ReactionCache(":memory:")
    reactants = [_species("H2"), _species("O2")]
    sig = build_signature(reactants, [_species("H2O")])
    record = c.put(sig, reactants, [_species("H2O")], {"k": 2}, ["H", "O"], None)

    sig_equiv = build_signature(reactants, [_species("OH2")])
    assert sig_equiv == sig
    hit = c.get_by_signature(sig_equiv)
    assert hit is not None
    assert hit.reaction_id == record.reaction_id
    assert len(c) == 1


# ------------------------------------------------------------ 旧格式签名兼容
def _legacy_signature(reactants: list[Species], products: list[Species]) -> str:
    """修复前的签名格式：组成键之外还混入规范化文本（用于模拟旧记录）。"""
    payload = {
        "r": sorted((species_key(sp), sp.normalized) for sp in reactants),
        "p": sorted((species_key(sp), sp.normalized) for sp in products),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_legacy_signature_record_remains_gettable(tmp_path):
    """旧代码（签名含规范化文本）写入的记录：旧 reaction_id 重启后仍可 GET。"""
    db = tmp_path / "legacy.db"
    reactants = [_species("H2"), _species("O2")]
    products = [_species("H2O")]

    c1 = ReactionCache(db)
    legacy_sig = _legacy_signature(reactants, products)
    legacy_record = c1.put(legacy_sig, reactants, products, {"k": 2}, ["H", "O"], None)
    legacy_id = legacy_record.reaction_id
    c1.close()

    c2 = ReactionCache(db)  # 模拟容器重启
    # 旧 reaction_id 仍可按 ID 取回（GET 不经过签名算法）
    fetched = c2.get(legacy_id)
    assert fetched is not None
    assert fetched.reaction_id == legacy_id
    assert fetched.signature == legacy_sig
    assert fetched.coefficients == {"k": 2}

    # 新签名与旧签名不同：同一反应的新写法写入新记录，两条记录共存
    new_sig = build_signature(reactants, products)
    assert new_sig != legacy_sig
    assert c2.get_by_signature(new_sig) is None
    new_record = c2.put(new_sig, reactants, products, {"k": 2}, ["H", "O"], None)
    assert new_record.reaction_id != legacy_id
    assert c2.get(legacy_id) is not None  # 旧记录未被破坏
    assert len(c2) == 2
    c2.close()


def test_legacy_record_gettable_via_api_after_reopen(tmp_path, monkeypatch):
    """旧格式签名记录（含真实配平系数）经 GET 接口在重启后仍完整返回。"""
    from app import service
    from app.balancer import solve_reaction
    from app.service import _coef_map
    from fastapi.testclient import TestClient
    from app.main import app

    db = tmp_path / "legacy_api.db"
    # 模拟旧版本 service 写入：旧格式签名 + 规范系数映射
    solved = solve_reaction(["Na", "H2O"], ["NaOH", "H2"])
    legacy_sig = _legacy_signature(list(solved.reactants), list(solved.products))
    writer = ReactionCache(db)
    legacy_record = writer.put(
        legacy_sig,
        list(solved.reactants),
        list(solved.products),
        _coef_map(solved),
        list(solved.elements),
        None,
    )
    legacy_id = legacy_record.reaction_id
    writer.close()

    # 模拟容器重启：新代码打开同一数据库
    reopened = ReactionCache(db)
    monkeypatch.setattr(service, "cache", reopened)
    client = TestClient(app)

    got = client.get(f"/api/v1/reactions/{legacy_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["reaction_id"] == legacy_id
    assert body["cached"] is True
    assert body["coefficients"]["vector"] == [2, 2, 2, 1]
    assert body["normalized"]["equation"] == "2H2O + 2Na -> 2NaOH + H2"

    # 同一反应以新签名再次 POST：生成新记录，旧记录不受影响
    posted = client.post(
        "/api/v1/balance",
        json={"reactants": ["Na", "H2O"], "products": ["NaOH", "H2"]},
    )
    assert posted.status_code == 200
    assert posted.json()["reaction_id"] != legacy_id
    assert client.get(f"/api/v1/reactions/{legacy_id}").status_code == 200
    reopened.close()


# ------------------------------------------------------------ 端到端：等价写法 + 重启
def test_equivalent_writing_hits_cache_after_reopen(tmp_path, monkeypatch):
    """新签名记录重启后仍命中；等价写法（OH2）与原始写法共享同一记录。"""
    from app import service
    from fastapi.testclient import TestClient
    from app.main import app

    db = tmp_path / "equiv.db"
    persistent = ReactionCache(db)
    monkeypatch.setattr(service, "cache", persistent)

    client = TestClient(app)
    r1 = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    )
    assert r1.status_code == 200
    rid = r1.json()["reaction_id"]
    persistent.close()

    # 模拟容器重启：同一数据库文件重新打开
    reopened = ReactionCache(db)
    monkeypatch.setattr(service, "cache", reopened)

    # 等价写法（产物写作 OH2）：同一 reaction_id，命中缓存
    r2 = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["OH2"]}
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["reaction_id"] == rid
    assert body["cached"] is True
    assert body["coefficients"]["vector"] == [2, 1, 2]
    # 命中后仍按本次输入展示化学式与系数
    assert body["normalized"]["products"] == ["OH2"]
    assert body["normalized"]["equation"] == "2H2 + O2 -> 2OH2"
    assert body["coefficients"]["products"][0]["input"] == "OH2"
    assert body["coefficients"]["products"][0]["normalized"] == "OH2"
    assert body["coefficients"]["products"][0]["coefficient"] == 2

    # GET 仍可取回同一记录
    got = client.get(f"/api/v1/reactions/{rid}")
    assert got.status_code == 200
    assert got.json()["reaction_id"] == rid
    reopened.close()

