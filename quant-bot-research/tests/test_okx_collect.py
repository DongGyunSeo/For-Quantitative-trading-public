"""수집기 — 네트워크 없이 로직만 검증한다.

10시간짜리 작업이라 페이지네이션·재개·무결성이 틀리면 안 된다.
가짜 OKX 응답기를 만들어 `Client.get` 만 갈아 끼운다.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "okx_collect", Path(__file__).resolve().parents[1] / "tools" / "okx_collect.py")
OC = importlib.util.module_from_spec(_SPEC)
sys.modules["okx_collect"] = OC
_SPEC.loader.exec_module(OC)

BAR_MS = OC.BAR_MS
#: 패치되기 전의 진짜 클래스. monkeypatch 후에 OC.Client 를 보면 함수다.
REAL_CLIENT = OC.Client


def t0(y=2024, m=1, d=1):
    return OC.ms(datetime(y, m, d))


class FakeOKX:
    """최신순 응답 · after = '그보다 이전' · limit 상한 100 · 마지막 봉 미마감."""

    def __init__(self, first_ts, last_ts, *, unconfirmed_tip=True, max_limit=100):
        self.first, self.last = first_ts, last_ts
        self.tip = unconfirmed_tip
        self.max_limit = max_limit
        self.calls = 0

    def get(self, path, **p):
        self.calls += 1
        if path != "/api/v5/market/history-candles":
            raise AssertionError(path)
        lim = min(int(p.get("limit", 100)), self.max_limit)
        hi = int(p["after"]) - BAR_MS if "after" in p else self.last
        hi = min(hi, self.last)
        out = []
        t = hi
        while t >= self.first and len(out) < lim:
            conf = "0" if (self.tip and t == self.last) else "1"
            px = 100.0 + (t // BAR_MS) % 7
            out.append([str(t), f"{px}", f"{px+1}", f"{px-1}", f"{px+0.5}",
                        "10", "20", "30", conf])
            t -= BAR_MS
        return out


def make(tmp_path, first, last, **kw):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    c = REAL_CLIENT.__new__(REAL_CLIENT)
    fake = FakeOKX(first, last, **kw)
    c.get = fake.get
    c.calls = 0
    return c, fake


# ---------------------------------------------------------------- 페이지네이션


def test_walks_backwards_to_target(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 3)
    c, fake = make(tmp_path, first, last)
    st = OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)
    assert st["done"] and st["reason"] == "목표 도달"
    # [first, last] 는 2*288+1 = 577봉, 미마감 tip 1봉을 빼면 576
    assert st["rows"] == 576
    ts = sorted(int(r[0]) for r in csv.reader(
        OC.part_path("X").open()) if r and r[0] != "ts")
    assert min(ts) == first
    assert max(ts) == last - BAR_MS          # 미마감 tip 은 빠진다
    assert len(set(ts)) == len(ts)
    assert all(b - a == BAR_MS for a, b in zip(ts, ts[1:]))


def test_stops_at_retention_limit(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    c, _ = make(tmp_path, first, last)
    st = OC.fetch_symbol(c, "X-USDT-SWAP", "X", t0(2020, 1, 1))   # 훨씬 과거를 요구
    assert st["done"] and st["reason"] == "보존 한계 도달"
    ts = [int(r[0]) for r in csv.reader(OC.part_path("X").open()) if r and r[0] != "ts"]
    assert min(ts) == first


def test_unconfirmed_bar_is_dropped(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 1) + 10 * BAR_MS
    c, _ = make(tmp_path, first, last, unconfirmed_tip=True)
    OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)
    ts = [int(r[0]) for r in csv.reader(OC.part_path("X").open()) if r and r[0] != "ts"]
    assert last not in ts and (last - BAR_MS) in ts


def test_no_infinite_loop_when_server_repeats(tmp_path):
    """서버가 같은 페이지를 계속 주면 멈춰야 한다."""
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    c, fake = make(tmp_path, first, last)
    stuck = [[str(last), "1", "2", "0.5", "1", "1", "1", "1", "1"]]
    c.get = lambda path, **p: stuck
    st = OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)
    assert st["reason"] == "진전 없음(중복 페이지)" and not st["done"]


def test_lookup_error_is_recorded(tmp_path):
    c, _ = make(tmp_path, t0(), t0() + BAR_MS)

    def boom(path, **p):
        raise LookupError("51001: instrument not found")
    c.get = boom
    st = OC.fetch_symbol(c, "BAD-USDT-SWAP", "BAD", t0())
    assert not st["done"] and "51001" in st["reason"]


# ---------------------------------------------------------------- 재개


def test_resume_continues_from_oldest(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 5)
    mid = t0(2024, 1, 3)
    c, _ = make(tmp_path, first, last)
    st1 = OC.fetch_symbol(c, "X-USDT-SWAP", "X", mid)
    assert st1["done"]
    n1 = st1["rows"]
    st2 = OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)       # 목표를 더 과거로
    assert st2["done"] and st2["rows"] > n1
    ts = sorted(int(r[0]) for r in csv.reader(
        OC.part_path("X").open()) if r and r[0] != "ts")
    assert min(ts) == first and len(set(ts)) == len(ts)
    assert all(b - a == BAR_MS for a, b in zip(ts, ts[1:]))


def test_resume_is_noop_when_already_done(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    c, _ = make(tmp_path, first, last)
    OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)
    before = c.calls
    st = OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)
    assert st["done"] and st["reason"] == "이미 목표 도달"


def test_part_oldest_reads_back(tmp_path):
    OC.RAW = tmp_path / "raw"
    OC.RAW.mkdir(parents=True)
    p = OC.part_path("Y")
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for t in (300, 100, 200):
            w.writerow([t, 1, 2, 0.5, 1, 1])
    assert OC.part_oldest(p) == (100, 3)
    assert OC.part_oldest(tmp_path / "raw" / "none.part.csv") == (None, 0)


# ---------------------------------------------------------------- 마무리


def test_finalize_sorts_dedupes_and_gzips(tmp_path):
    OC.RAW = tmp_path / "raw"
    OC.RAW.mkdir(parents=True)
    p = OC.part_path("Z")
    rows = [(t0() + i * BAR_MS, 1.0, 2.0, 0.5, 1.5, 9.0) for i in (3, 1, 2, 1)]
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        w.writerows(rows)
    r = OC.finalize("Z")
    assert r["ok"] and r["rows"] == 3          # 중복 1건 제거
    assert not p.exists()
    with gzip.open(OC.final_path("Z"), "rt") as f:
        got = list(csv.reader(f))
    assert got[0] == list(OC.COLS)
    ts = [x[0] for x in got[1:]]
    assert ts == sorted(ts)


def test_finalize_reports_gaps(tmp_path):
    OC.RAW = tmp_path / "raw"
    OC.RAW.mkdir(parents=True)
    p = OC.part_path("G")
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for i in (0, 1, 5, 6):                 # 2,3,4 가 빔 -> 공백 3
            w.writerow([t0() + i * BAR_MS, 1, 2, 0.5, 1, 1])
    r = OC.finalize("G")
    assert r["gaps"] == 3 and r["max_gap"] == 3


def test_finalize_flags_bad_ohlc(tmp_path):
    OC.RAW = tmp_path / "raw"
    OC.RAW.mkdir(parents=True)
    p = OC.part_path("B")
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        w.writerow([t0(), 1, 0.2, 0.5, 1, 1])      # high < low
        w.writerow([t0() + BAR_MS, 1, 2, 0.5, 1, 1])
    assert OC.finalize("B")["bad_ohlc"] == 1


def test_finalized_file_loads_with_project_loader(tmp_path):
    """수집 결과가 기존 로더로 바로 읽혀야 한다."""
    from qbot.data.loader import load_ohlcv
    OC.RAW = tmp_path / "raw"
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    c, _ = make(tmp_path, first, last)
    OC.fetch_symbol(c, "X-USDT-SWAP", "X", first)
    OC.finalize("X")
    df, q = load_ohlcv(OC.final_path("X"))
    assert q.gaps == 0 and q.dupes == 0 and q.bad_ohlc == 0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.dtype) == "datetime64[us, UTC]"


# ---------------------------------------------------------------- 한도


def test_bucket_respects_rate():
    import time
    b = OC.Bucket(rps=50.0, burst=1.0)
    t = time.monotonic()
    for _ in range(11):
        b.take()
    assert time.monotonic() - t >= 0.18        # 10회분 ≈ 0.2s


def test_bucket_slows_down_and_recovers():
    b = OC.Bucket(rps=10.0, burst=5.0)
    b.slow_down()
    assert b.rps == 5.0 and b.throttled == 1
    for _ in range(200):
        b.take()
    assert b.rps > 9.0                         # 천천히 회복


def test_bucket_has_a_floor():
    b = OC.Bucket(rps=10.0)
    for _ in range(20):
        b.slow_down()
    assert b.rps >= 2.0                        # 기본의 20% 아래로는 안 내려간다


# ---------------------------------------------------------------- 시각


def test_iso_roundtrip():
    t = OC.ms(datetime(2026, 7, 5, 12, 35))
    assert OC.iso(t) == "2026-07-05 12:35:00+00:00"


def test_parse_day_is_utc():
    assert OC.parse_day("2021-04-02").tzinfo is timezone.utc


# ---------------------------------------------------------------- CLI 통합


class Args(dict):
    __getattr__ = dict.get


def test_cli_collect_status_verify_end_to_end(tmp_path, monkeypatch, capsys):
    """10시간짜리 작업이 1분 만에 배선 때문에 죽지 않도록 전 경로를 태운다."""
    from qbot.data.loader import load_ohlcv
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    first, last = t0(2024, 1, 1), t0(2024, 1, 4)
    OC.save_json(OC.UNIVERSE, dict(
        created="", bar="5m", target_start="2024-01-01", min_list_date="2022-07-01",
        n_candidates=2, picked=[
            dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP", listTime=first,
                 list_iso=OC.iso(first), ctVal=0.01, last=1.0, notional24h=9e9),
            dict(rank=2, sym="BBB", instId="BBB-USDT-SWAP", listTime=first,
                 list_iso=OC.iso(first), ctVal=0.01, last=1.0, notional24h=8e9)]))

    def fake_client(rps, **kw):
        c = REAL_CLIENT.__new__(REAL_CLIENT)
        c.get = FakeOKX(first, last).get
        c.b = OC.Bucket(rps)
        c.calls = 0
        c.errors = 0
        return c
    monkeypatch.setattr(OC, "Client", fake_client)

    OC.cmd_collect(Args(rps=9.0, start=None, only=None, max_hours=None, limit=None, force=False))
    man = OC.load_json(OC.MANIFEST, {})
    assert man["AAA"]["done"] and man["BBB"]["done"]
    assert man["AAA"]["rows"] == 3 * 288          # 3일치, 미마감 tip 제외

    OC.cmd_status(Args())
    assert "수집 완료 2" in capsys.readouterr().out

    OC.cmd_verify(Args(only=None, keep_part=False))
    for s in ("AAA", "BBB"):
        df, q = load_ohlcv(OC.final_path(s))
        assert len(df) == 3 * 288 and q.gaps == 0 and q.bad_ohlc == 0

    OC.cmd_status(Args())
    assert "압축 완료 2" in capsys.readouterr().out


def test_cli_collect_skips_completed_on_rerun(tmp_path, monkeypatch):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    OC.save_json(OC.UNIVERSE, dict(target_start="2024-01-01", picked=[
        dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP")]))
    seen = []

    def fake_client(rps, **kw):
        c = REAL_CLIENT.__new__(REAL_CLIENT)
        f = FakeOKX(first, last)
        seen.append(f)
        c.get = f.get
        c.b = OC.Bucket(rps)
        c.calls = 0
        c.errors = 0
        return c
    monkeypatch.setattr(OC, "Client", fake_client)
    args = Args(rps=9.0, start=None, only=None, max_hours=None, limit=None, force=False)
    OC.cmd_collect(args)
    n1 = seen[-1].calls
    OC.cmd_collect(args)
    # 두 번째 실행은 할 일이 없으므로 limit 실측조차 하지 않는다
    assert seen[-1].calls == 0 and n1 > 0


# ---------------------------------------------------------------- 유니버스 안전장치


def _uni_client(insts, ticks):
    def get(path, **p):
        if path == "/api/v5/public/instruments":
            return insts
        if path == "/api/v5/market/tickers":
            return ticks
        raise AssertionError(path)
    c = REAL_CLIENT.__new__(REAL_CLIENT)
    c.get = get
    c.b = OC.Bucket(9.0)
    c.calls = 0
    c.errors = 0
    return c


def _inst(sym, list_dt, ct="1"):
    return dict(instId=f"{sym}-USDT-SWAP", settleCcy="USDT", ctType="linear",
                state="live", ctVal=ct, listTime=str(OC.ms(list_dt)))


def _tick(sym, vol):
    return dict(instId=f"{sym}-USDT-SWAP", last="100", vol24h=str(vol))


def test_universe_does_not_pad_with_short_history(tmp_path, monkeypatch, capsys):
    """OKX 토큰화 주식(2025~2026 상장)이 조용히 섞이면 안 된다."""
    OC.UNIVERSE = tmp_path / "universe.json"
    insts = [_inst("BTC", datetime(2020, 1, 1)), _inst("ETH", datetime(2020, 1, 1)),
             _inst("NVDA", datetime(2026, 2, 1)), _inst("AAPL", datetime(2026, 3, 1))]
    ticks = [_tick("NVDA", 9e9), _tick("AAPL", 8e9), _tick("BTC", 5e9), _tick("ETH", 4e9)]
    monkeypatch.setattr(OC, "Client", lambda rps, **k: _uni_client(insts, ticks))
    OC.cmd_universe(Args(rps=9.0, top=4, start="2021-04-02",
                         min_list_date="2022-07-01", allow_short_history=False))
    got = OC.load_json(OC.UNIVERSE, {})["picked"]
    assert [r["sym"] for r in got] == ["BTC", "ETH"]
    out = capsys.readouterr().out
    assert "--allow-short-history" in out and "NVDA" in out      # 다음 순위 안내


def test_universe_pads_only_when_asked(tmp_path, monkeypatch):
    OC.UNIVERSE = tmp_path / "universe.json"
    insts = [_inst("BTC", datetime(2020, 1, 1)), _inst("NVDA", datetime(2026, 2, 1))]
    ticks = [_tick("NVDA", 9e9), _tick("BTC", 5e9)]
    monkeypatch.setattr(OC, "Client", lambda rps, **k: _uni_client(insts, ticks))
    OC.cmd_universe(Args(rps=9.0, top=2, start="2021-04-02",
                         min_list_date="2022-07-01", allow_short_history=True))
    assert [r["sym"] for r in OC.load_json(OC.UNIVERSE, {})["picked"]] == ["BTC", "NVDA"]


def test_universe_ranks_by_notional_not_contracts(tmp_path, monkeypatch):
    """ctVal 이 다르면 계약 수로 줄 세우면 안 된다."""
    OC.UNIVERSE = tmp_path / "universe.json"
    insts = [_inst("BIG", datetime(2020, 1, 1), ct="1000"),
             _inst("SMALL", datetime(2020, 1, 1), ct="0.01")]
    ticks = [_tick("SMALL", 1e6), _tick("BIG", 1e5)]   # 계약 수는 SMALL 이 많다
    monkeypatch.setattr(OC, "Client", lambda rps, **k: _uni_client(insts, ticks))
    OC.cmd_universe(Args(rps=9.0, top=2, start="2021-04-02",
                         min_list_date="2022-07-01", allow_short_history=False))
    picked = OC.load_json(OC.UNIVERSE, {})["picked"]
    assert picked[0]["sym"] == "BIG"                  # 명목가는 BIG 이 1000배


def test_universe_skips_non_usdt_and_inverse(tmp_path, monkeypatch):
    OC.UNIVERSE = tmp_path / "universe.json"
    insts = [_inst("BTC", datetime(2020, 1, 1)),
             dict(instId="BTC-USD-SWAP", settleCcy="BTC", ctType="inverse",
                  state="live", ctVal="100", listTime=str(OC.ms(datetime(2019, 1, 1)))),
             dict(instId="OLD-USDT-SWAP", settleCcy="USDT", ctType="linear",
                  state="suspend", ctVal="1", listTime=str(OC.ms(datetime(2020, 1, 1))))]
    ticks = [_tick("BTC", 1e6), dict(instId="BTC-USD-SWAP", last="100", vol24h="9e9"),
             _tick("OLD", 9e9)]
    monkeypatch.setattr(OC, "Client", lambda rps, **k: _uni_client(insts, ticks))
    OC.cmd_universe(Args(rps=9.0, top=5, start="2021-04-02",
                         min_list_date="2022-07-01", allow_short_history=False))
    assert [r["sym"] for r in OC.load_json(OC.UNIVERSE, {})["picked"]] == ["BTC"]


# ---------------------------------------------------------------- limit 실측


def test_discover_limit_reads_actual_cap(tmp_path):
    """OKX 문서는 history-candles 를 '최대 100' 이라 적어 뒀지만 실측은 300 이다."""
    c, _ = make(tmp_path, t0(2020, 1, 1), t0(2026, 1, 1), max_limit=300)
    assert OC.discover_limit(c) == 300
    c2, _ = make(tmp_path, t0(2020, 1, 1), t0(2026, 1, 1), max_limit=100)
    assert OC.discover_limit(c2) == 100


def test_discover_limit_has_a_floor(tmp_path):
    c, _ = make(tmp_path, t0(2024, 1, 1), t0(2024, 1, 1) + 5 * BAR_MS, max_limit=300)
    assert OC.discover_limit(c) >= 100        # 데이터가 짧아도 100 밑으로 내려가지 않는다


def test_bigger_limit_gives_identical_data(tmp_path):
    """봉 300개씩 받으나 100개씩 받으나 결과가 같아야 한다."""
    first, last = t0(2024, 1, 1), t0(2024, 1, 6)
    out = {}
    for lim in (100, 300):
        d = tmp_path / f"L{lim}"
        c, _ = make(d, first, last, max_limit=300)
        st = OC.fetch_symbol(c, "X-USDT-SWAP", "X", first, limit=lim)
        assert st["done"]
        rows = [tuple(r) for r in csv.reader(OC.part_path("X").open())
                if r and r[0] != "ts"]
        out[lim] = sorted(rows, key=lambda r: int(r[0]))
    assert out[100] == out[300]
    ts = [int(r[0]) for r in out[300]]
    assert all(b - a == BAR_MS for a, b in zip(ts, ts[1:]))


def test_bigger_limit_uses_fewer_calls(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 6)
    calls = {}
    for lim in (100, 300):
        c, fake = make(tmp_path / f"C{lim}", first, last, max_limit=300)
        OC.fetch_symbol(c, "X-USDT-SWAP", "X", first, limit=lim)
        calls[lim] = fake.calls
    assert calls[300] * 2 < calls[100]        # 대략 1/3


def test_collect_uses_discovered_limit(tmp_path, monkeypatch, capsys):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    first, last = t0(2024, 1, 1), t0(2024, 1, 3)
    OC.save_json(OC.UNIVERSE, dict(target_start="2024-01-01", picked=[
        dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP")]))

    def fake_client(rps, **kw):
        c = REAL_CLIENT.__new__(REAL_CLIENT)
        c.get = FakeOKX(first, last, max_limit=300).get
        c.b = OC.Bucket(rps)
        c.calls = 0
        c.errors = 0
        return c
    monkeypatch.setattr(OC, "Client", fake_client)
    OC.cmd_collect(Args(rps=9.0, start=None, only=None, max_hours=None, limit=None, force=False))
    assert "요청당 300봉 (실측)" in capsys.readouterr().out
    assert OC.load_json(OC.MANIFEST, {})["AAA"]["rows"] == 2 * 288


def test_collect_limit_override(tmp_path, monkeypatch, capsys):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    OC.save_json(OC.UNIVERSE, dict(target_start="2024-01-01", picked=[
        dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP")]))
    monkeypatch.setattr(OC, "Client", lambda rps, **k: _fc(first, last, rps))
    OC.cmd_collect(Args(rps=9.0, start=None, only=None, max_hours=None, limit=100, force=False))
    assert "요청당 100봉 (지정)" in capsys.readouterr().out


def _fc(first, last, rps):
    c = REAL_CLIENT.__new__(REAL_CLIENT)
    c.get = FakeOKX(first, last, max_limit=300).get
    c.b = OC.Bucket(rps)
    c.calls = 0
    c.errors = 0
    return c


# ---------------------------------------------------------------- 동시 실행 방지


def test_lock_blocks_second_instance(tmp_path):
    """같은 IP 에서 두 개를 돌리면 한도를 나눠 쓰는 게 아니라 넘는다."""
    a = OC.RunLock(tmp_path / "collect.lock", rps=9.0)
    a.acquire()
    b = OC.RunLock(tmp_path / "collect.lock", rps=9.0)
    with pytest.raises(SystemExit) as e:
        b.acquire()
    assert "넘는다" in str(e.value) and "9.0 req/s" in str(e.value)
    a.release()


def test_stale_lock_is_taken_over(tmp_path, capsys):
    p = tmp_path / "collect.lock"
    OC.save_json(p, dict(pid=123, beat=0.0, started="x", rps=9.0))    # 아주 옛날
    lk = OC.RunLock(p, rps=9.0)
    lk.acquire()
    assert lk.held and "비정상 종료" in capsys.readouterr().out
    assert OC.load_json(p, {})["pid"] != 123
    lk.release()


def test_force_overrides_fresh_lock(tmp_path):
    a = OC.RunLock(tmp_path / "collect.lock", rps=9.0)
    a.acquire()
    b = OC.RunLock(tmp_path / "collect.lock", force=True, rps=9.0)
    b.acquire()
    assert b.held
    b.release()


def test_release_removes_lock(tmp_path):
    lk = OC.RunLock(tmp_path / "collect.lock", rps=9.0)
    lk.acquire()
    assert lk.path.exists()
    lk.release()
    assert not lk.path.exists()
    OC.RunLock(tmp_path / "collect.lock").acquire()      # 다시 잡힌다


def test_collect_releases_lock_after_run(tmp_path, monkeypatch):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    first, last = t0(2024, 1, 1), t0(2024, 1, 2)
    OC.save_json(OC.UNIVERSE, dict(target_start="2024-01-01", picked=[
        dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP")]))
    monkeypatch.setattr(OC, "Client", lambda rps, **k: _fc(first, last, rps))
    OC.cmd_collect(Args(rps=9.0, start=None, only=None, max_hours=None,
                        limit=None, force=False))
    assert not (tmp_path / "collect.lock").exists()


def test_collect_releases_lock_even_on_crash(tmp_path, monkeypatch):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    OC.save_json(OC.UNIVERSE, dict(target_start="2024-01-01", picked=[
        dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP")]))

    def boom(rps, **k):
        raise RuntimeError("네트워크 폭발")
    monkeypatch.setattr(OC, "Client", boom)
    with pytest.raises(RuntimeError):
        OC.cmd_collect(Args(rps=9.0, start=None, only=None, max_hours=None,
                            limit=None, force=False))
    assert not (tmp_path / "collect.lock").exists()


def test_collect_refuses_while_another_runs(tmp_path, monkeypatch):
    OC.ROOT = tmp_path
    OC.RAW = tmp_path / "raw"
    OC.UNIVERSE = tmp_path / "universe.json"
    OC.MANIFEST = tmp_path / "manifest.json"
    OC.save_json(OC.UNIVERSE, dict(target_start="2024-01-01", picked=[
        dict(rank=1, sym="AAA", instId="AAA-USDT-SWAP")]))
    other = OC.RunLock(tmp_path / "collect.lock", rps=9.0)
    other.acquire()                                   # 다른 창에서 돌고 있다
    called = []
    monkeypatch.setattr(OC, "Client", lambda rps, **k: called.append(1))
    with pytest.raises(SystemExit):
        OC.cmd_collect(Args(rps=9.0, start=None, only="AAA", max_hours=None,
                            limit=None, force=False))
    assert called == []                               # 요청 한 번도 안 보냈다
    other.release()


def test_heartbeat_updates_during_collection(tmp_path):
    first, last = t0(2024, 1, 1), t0(2024, 1, 8)
    c, _ = make(tmp_path, first, last, max_limit=300)
    lk = OC.RunLock(tmp_path / "collect.lock", rps=9.0)
    lk.acquire()
    b0 = OC.load_json(lk.path, {})["beat"]
    import time as _t
    _t.sleep(0.02)
    OC.fetch_symbol(c, "X-USDT-SWAP", "X", first, limit=100, flush_every=5, lock=lk)
    got = OC.load_json(lk.path, {})
    assert got["beat"] > b0 and got["sym"] == "X"
    lk.release()
