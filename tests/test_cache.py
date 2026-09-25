"""SQLite 持久幂等缓存测试。

覆盖：读写与 LRU 访问顺序、跨“重启”（重新打开同一数据库文件）恢复、
成功与错误记录格式、写入/访问/淘汰事务原子性（无法识别的存储版本
应在启动时明确报错）。
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


# ------------------------------------------------------------ 签名与写法无关
def test_signature_ignores_normalized_spelling_and_order():
    """组成键 + 电荷相同即同签名：元素排列（H2O/OH2）、括号、书写顺序均无关。"""
    r = [_species("H2"), _species("O2")]
    sig = build_signature(r, [_species("H2O")])
    # 元素排列顺序不同但组成相同
    assert sig == build_signature(r, [_species("OH2")])
    # 括号风格 / 空白等价写法
    assert sig == build_signature(
        [_species("H2"), _species("O2")], [_species("(H2O)")]
    )
    # 物种书写顺序不同
    assert sig == build_signature([_species("O2"), _species("H2")], [_species("H2O")])
    # 规范化文本没有混入签名
    assert "H2O" not in sig and "OH2" not in sig


def test_signature_preserves_side_charge_and_multiplicity():
    h2o = _species("H2O")
    h2 = _species("H2")
    o2 = _species("O2")
    # 调换反应物 / 生成物侧别 -> 不同签名
    assert build_signature([h2, o2], [h2o]) != build_signature([h2o], [h2, o2])
    # 同一规范键出现次数（重数）不同 -> 不同签名
    assert build_signature([], [h2o]) != build_signature([], [h2o, h2o])
    # 电荷不同（组成相同）-> 不同签名
    assert build_signature([], [_species("Fe")]) != build_signature(
        [], [_species("Fe3+")]
    )


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


def _legacy_signature(reactants, products) -> str:
    """签名算法升级前（混入规范化文本）的旧签名构造，用于造旧记录。"""
    def key(sp):
        comp = ",".join(f"{el}:{n}" for el, n in sp.formula.key[0])
        return f"[{comp}]{sp.formula.charge:+d}"

    return json.dumps(
        {
            "r": sorted((key(sp), sp.normalized) for sp in reactants),
            "p": sorted((key(sp), sp.normalized) for sp in products),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_legacy_reaction_id_still_gettable_and_new_signature_reopens(tmp_path, monkeypatch):
    """旧 SQLite 记录的旧 reaction_id 继续支持 GET；新签名重启后仍命中。"""
    from app import service
    from fastapi.testclient import TestClient
    from app.main import app

    db = tmp_path / "legacy.db"

    # 1) 用“旧签名”预先写入一条 H2 + O2 -> H2O 的记录（模拟升级前的库）
    reactants = [_species("H2"), _species("O2")]
    products = [_species("H2O")]
    legacy_sig = _legacy_signature(reactants, products)
    legacy_rid = reaction_id_for(legacy_sig)
    seeded = ReactionCache(db)
    seeded.put(
        legacy_sig,
        reactants,
        products,
        {"[H:2]+0": 2, "[O:2]+0": 1, "[H:2,O:1]+0": 2},
        ["H", "O"],
        None,
    )
    seeded.close()

    reopened = ReactionCache(db)  # 模拟升级后重启
    monkeypatch.setattr(service, "cache", reopened)
    client = TestClient(app)

    # 2) 旧 reaction_id 的 GET 查询仍然可用
    got = client.get(f"/api/v1/reactions/{legacy_rid}")
    assert got.status_code == 200
    assert got.json()["reaction_id"] == legacy_rid
    assert got.json()["coefficients"]["vector"] == [2, 1, 2]

    # 3) 新签名第一次请求产生新 reaction_id（未命中旧记录）
    first = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    )
    assert first.status_code == 200
    new_rid = first.json()["reaction_id"]
    assert first.json()["cached"] is False
    assert new_rid != legacy_rid

    # 4) 元素排列等价写法 OH2 命中同一条新记录，回显沿用本次输入
    equiv = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["OH2"]}
    )
    assert equiv.status_code == 200
    body = equiv.json()
    assert body["reaction_id"] == new_rid
    assert body["cached"] is True
    assert body["normalized"]["products"] == ["OH2"]
    assert body["normalized"]["equation"] == "2H2 + O2 -> 2OH2"
    prod = body["coefficients"]["products"][0]
    assert prod["input"] == "OH2"
    assert prod["normalized"] == "OH2"
    assert prod["coefficient"] == 2
    # 配平计算结果复用缓存
    assert body["coefficients"]["vector"] == [2, 1, 2]
    reopened.close()

    # 5) 再次“重启”：新签名记录仍在，OH2 仍命中；旧 ID 仍可 GET
    second = ReactionCache(db)
    monkeypatch.setattr(service, "cache", second)
    client2 = TestClient(app)
    after = client2.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["OH2"]}
    )
    assert after.status_code == 200
    assert after.json()["reaction_id"] == new_rid
    assert after.json()["cached"] is True
    legacy_get = client2.get(f"/api/v1/reactions/{legacy_rid}")
    assert legacy_get.status_code == 200
    assert legacy_get.json()["reaction_id"] == legacy_rid
    assert len(second) == 2
    second.close()


def test_analyze_equivalent_writing_hits_after_reopen(tmp_path, monkeypatch):
    """/analyze 与 /balance 共享签名；重启后等价写法命中且按本次输入展示。"""
    from app import service
    from fastapi.testclient import TestClient
    from app.main import app

    db = tmp_path / "analyze.db"
    c1 = ReactionCache(db)
    monkeypatch.setattr(service, "cache", c1)
    client = TestClient(app)

    first = client.post(
        "/api/v1/analyze", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    )
    assert first.status_code == 200
    rid = first.json()["reaction_id"]
    assert first.json()["cached"] is False
    c1.close()

    c2 = ReactionCache(db)  # 重启
    monkeypatch.setattr(service, "cache", c2)
    second = client.post(
        "/api/v1/analyze", json={"reactants": ["H2", "O2"], "products": ["OH2"]}
    )
    assert second.status_code == 200
    body = second.json()
    assert body["reaction_id"] == rid
    assert body["cached"] is True
    # 本次输入写法回显，缓存中的配平结果复用
    assert body["normalized"]["products"] == ["OH2"]
    assert body["normalized"]["equation"] == "2H2 + O2 -> 2OH2"
    balanced_product = body["balanced"]["species"]["products"][0]
    assert balanced_product["normalized"] == "OH2"
    assert balanced_product["coefficient"] == 2
    assert body["balanced_ok"] is True
    c2.close()

