"""주말 갭 × 4H 본장정렬 × 15m CISD — DST · 주말 제외 · CISD 손계산 · 인과성 · 거울 대칭."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.research import legacy as L
from qbot.research import wgap as W


def _mirror(df: pd.DataFrame, K: float | None = None) -> pd.DataFrame:
    K = K if K is not None else float(df["high"].max() + df["low"].min())
    out = df.copy()
    out["open"], out["close"] = K - df["open"], K - df["close"]
    out["high"], out["low"] = K - df["low"], K - df["high"]
    return out


@pytest.fixture(scope="module")
def wk():
    return synth_5m(30_000, seed=5, vol=0.004)


# ------------------------------------------------------------------ 본장 마스크 · 주말 이벤트


@pytest.mark.parametrize("ts,expect", [
    ("2024-07-12 19:55", True), ("2024-07-12 20:00", False),      # 여름 금 15:55 / 16:00 EDT
    ("2024-07-13 12:00", False), ("2024-07-14 23:00", False),
    ("2024-07-15 13:25", False), ("2024-07-15 13:30", True),      # 여름 월 09:25 / 09:30 EDT
    ("2024-01-12 20:55", True), ("2024-01-12 21:00", False),      # 겨울 금 15:55 / 16:00 EST
    ("2024-01-15 14:25", False), ("2024-01-15 14:30", True),      # 겨울 월 09:25 / 09:30 EST
    ("2024-01-10 03:00", True),
])
def test_weekday_mask_dst(ts, expect):
    idx = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")])
    assert bool(W.weekday_mask(idx)[0]) is expect


def _gap_week(start="2024-07-12", jump_at="2024-07-15 13:30", F=100.0, O=103.0, days=8):
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC").as_unit("us")
    c = np.where(idx >= pd.Timestamp(jump_at, tz="UTC"), O, F)
    df = pd.DataFrame(dict(open=c, high=c * 1.0002, low=c * 0.9998, close=c, volume=1.0), index=idx)
    return df


def test_weekend_events_picks_et_bars_and_week_end():
    df = _gap_week()
    ev = W.weekend_events(df)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert df.index[e.fi] == pd.Timestamp("2024-07-12 19:55", tz="UTC")
    assert df.index[e.mi] == pd.Timestamp("2024-07-15 13:30", tz="UTC")
    assert df.index[e.wend] == pd.Timestamp("2024-07-19 20:00", tz="UTC")     # 그 주 금 16:00 EDT
    assert e.hend == min(len(df), e.mi + W.HOLD) and e.d == -1 and e.gap_bps == pytest.approx(300.0)
    w = W.weekend_events(_gap_week("2024-01-12", "2024-01-15 14:30"))
    assert len(w) == 1
    assert pd.Timestamp(w.monday.iloc[0]) == pd.Timestamp("2024-01-15 14:30", tz="UTC")


def test_base_matches_legacy_when_expiry_is_zero(wk):
    ev = W.weekend_events(wk)
    a = W.base_trades(wk, ev, mtm=False)
    b = L.weekend_gap_trades(wk)
    assert len(a) == len(b) > 5
    assert np.array_equal(a.bar.to_numpy(), b.bar.to_numpy())
    assert np.allclose(a.gross, b.gross) and np.array_equal(a.kind.to_numpy(), b.kind.to_numpy())


# ------------------------------------------------------------------ 본장 4H


def _warp(df, mask, k):
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out.loc[mask, col] = df.loc[mask, col] * k
    return out


def test_weekday_htf_ignores_weekend_bars(wk):
    a = W.htf_trend(wk, weekday_only=True)
    weekend = ~W.weekday_mask(wk.index)
    warped = _warp(wk, weekend, 1.5)
    assert np.array_equal(a, W.htf_trend(warped, weekday_only=True))
    assert not np.array_equal(W.htf_trend(wk, weekday_only=False),
                              W.htf_trend(warped, weekday_only=False))     # 24/7 은 바뀌어야 정상


@pytest.mark.parametrize("weekday_only", [True, False])
def test_htf_trend_is_causal(wk, weekday_only):
    k = 17_000
    a = W.htf_trend(wk, weekday_only=weekday_only)
    fut = np.zeros(len(wk), bool)
    fut[k:] = True
    b = W.htf_trend(_warp(wk, fut, 1.3), weekday_only=weekday_only)
    assert np.array_equal(a[:k], b[:k])
    assert (a != 0).mean() > 0.8


# ------------------------------------------------------------------ CISD 손계산


def _arr(o, c, h=None, l=None):
    o, c = np.asarray(o, float), np.asarray(c, float)
    h = np.maximum(o, c) + 0.2 if h is None else np.asarray(h, float)
    l = np.minimum(o, c) - 0.2 if l is None else np.asarray(l, float)
    return o, h, l, c


def test_cisd_uses_first_open_of_the_series():
    o, h, l, c = _arr([100, 101, 102, 103, 101.5], [101, 102, 103, 101.5, 99.5])
    h[2] = 103.5
    t, lv, ex = W.cisd_scan(o, h, l, c, 0, 5, -1)
    assert (t, lv, ex) == (4, 100.0, 103.5)          # 마지막 양봉 시가(102)였다면 3 에서 났다


def test_cisd_on_the_sweep_bar():
    o, h, l, c = _arr([100, 101, 102], [101, 102, 99], h=[101.2, 102.2, 103.0])
    assert W.cisd_scan(o, h, l, c, 0, 3, -1) == (2, 100.0, 103.0)


def test_cisd_run_is_broken_by_opposite_candle():
    o = [100, 101, 100.5, 101.5, 102.5, 103.5, 102]
    c = [101, 100.5, 101.5, 102.5, 103.5, 102, 100.4]
    o, h, l, c = _arr(o, c)
    t, lv, _ = W.cisd_scan(o, h, l, c, 0, 7, -1)
    assert (t, lv) == (6, 100.5)


def test_cisd_doji_breaks_the_run():
    o, h, l, c = _arr([100, 101, 101, 102], [101, 101, 102, 100.8])
    t, lv, _ = W.cisd_scan(o, h, l, c, 0, 4, -1)
    assert (t, lv) == (3, 101.0)


def test_cisd_walk_back_cap():
    o, h, l, c = _arr([100, 101, 102, 103, 104, 104.5], [101, 102, 103, 104, 105, 100.5])
    assert W.cisd_scan(o, h, l, c, 0, 6, -1)[0] == -1     # 묶음 첫 시가 100 — 종가 100.5 는 못 넘는다
    t, lv, _ = W.cisd_scan(o, h, l, c, 0, 6, -1, cap=2)
    assert (t, lv) == (5, 102.0)                     # 거슬러 오르기가 m−2 에서 멈춘다


def test_cisd_none_and_start_before_series():
    o, h, l, c = _arr([100, 101, 102], [101, 102, 103])
    t, lv, ex = W.cisd_scan(o, h, l, c, 0, 3, -1)
    assert t == -1 and np.isnan(lv) and np.isnan(ex)
    o, h, l, c = _arr([100, 101, 102, 101.8], [101, 102, 101.8, 99.9])
    t, lv, ex = W.cisd_scan(o, h, l, c, 2, 4, -1)    # 시작 이전(주말)의 양봉 묶음을 쓴다
    assert (t, lv) == (3, 100.0) and ex == pytest.approx(102.2)


def test_cisd_mirror_symmetry():
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 0.3, 400))
    o = np.concatenate(([c[0]], c[:-1])) + rng.normal(0, 0.05, 400)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, 400))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, 400))
    K = 250.0
    for s in range(0, 380, 20):
        a = W.cisd_scan(o, h, l, c, s, 400, +1)
        b = W.cisd_scan(K - o, K - l, K - h, K - c, s, 400, -1)
        assert a[0] == b[0]
        if a[0] >= 0:
            assert a[1] == pytest.approx(K - b[1]) and a[2] == pytest.approx(K - b[2])


# ------------------------------------------------------------------ 거래 — 인과성 · 거울


def _all(df, **kw):
    ev = W.weekend_events(df, **kw)
    M, pe = W.ltf_bars(df)
    return ev, W.base_trades(df, ev), W.cisd_entry_trades(df, ev, M, pe), W.postfill(df, ev, M, pe)


def test_cisd_entry_is_causal(wk):
    ev, _, c, _ = _all(wk, min_gap_bps=0)
    assert len(c) >= 3
    p = int(c.bar.iloc[1])
    fut = np.zeros(len(wk), bool)
    fut[p + 1:] = True
    _, _, c2, _ = _all(_warp(wk, fut, 0.8), min_gap_bps=0)
    keep = c.bar <= p
    a, b = c[keep].reset_index(drop=True), c2[c2.bar <= p].reset_index(drop=True)
    assert len(a) == len(b) >= 2
    for k in ("bar", "side", "entry", "risk_bps", "tp_r"):
        assert np.allclose(a[k].to_numpy(float), b[k].to_numpy(float))


def test_trades_mirror_symmetry(wk):
    ev, b0, c, (pf, d1, d2, dc) = _all(wk, min_gap_bps=0)
    mev, mb0, mc, (mpf, md1, md2, mdc) = _all(_mirror(wk), min_gap_bps=0)
    assert np.array_equal(ev.d.to_numpy(), -mev.d.to_numpy())
    for x, y in ((b0, mb0), (c, mc), (d1, md1), (d2, md2), (dc, mdc)):
        assert len(x) == len(y)
        assert np.array_equal(x.bar.to_numpy(), y.bar.to_numpy())
        assert np.array_equal(x.side.to_numpy(), -y.side.to_numpy())
        assert np.allclose(x.gross, y.gross) and np.array_equal(x.kind.to_numpy(), y.kind.to_numpy())
    assert len(c) > 0 and len(d1) > 0 and len(dc) > 0
    assert np.allclose(pf.d0_g.dropna(), mpf.d0_g.dropna())


def test_postfill_d1_is_the_other_side_of_d2(wk):
    _, _, _, (pf, d1, d2, _) = _all(wk, min_gap_bps=0)
    assert len(d1) == len(d2) == int(pf.filled.sum()) > 3
    assert np.allclose(d1.gross, -d2.gross)
    assert np.allclose(d1.risk_bps, d2.risk_bps)
    assert (d1.fee <= d2.fee + 1e-12).all()          # 반전은 지정가 진입(2) · 연속은 역지정가(6)


def _fill_week(fill_low, after):
    """금 종가 100 → 월 09:30 EDT 개장 103 (G 3). 월 15:00 UTC 봉이 fill_low 까지, 그 뒤 after."""
    df = _gap_week()
    t0 = pd.Timestamp("2024-07-15 15:00", tz="UTC")
    i = int(df.index.searchsorted(t0))
    df.iloc[i, df.columns.get_loc("low")] = fill_low
    df.iloc[i, df.columns.get_loc("close")] = 100.5
    for col in ("open", "high", "low", "close"):
        df.iloc[i + 1:, df.columns.get_loc(col)] = after
    return df, i


def test_postfill_fill_bar_extension_counts_as_continuation():
    df, i = _fill_week(96.5, 104.0)                  # 메운 봉에서 한 갭(3) 넘게 더 내려감 → 연속
    ev, _, _, (pf, d1, d2, _) = _all(df)
    assert pf.phi.iloc[0] == i and pf.d0_g.iloc[0] == 1.0
    assert d2.kind.iloc[0] == 1 and d1.kind.iloc[0] == -1 and d1.gross.iloc[0] == -1.0


def test_postfill_reversal_to_monday_open():
    df, i = _fill_week(99.0, 104.0)                  # 살짝 메우고 월 개장가(103) 위로 → 반전
    _, _, _, (pf, d1, d2, _) = _all(df)
    assert pf.d0_g.iloc[0] == -1.0 and pf.d0_kind.iloc[0] == -1
    assert d1.gross.iloc[0] == pytest.approx(1.0) and d1.kind.iloc[0] == 1
    assert pf.fill_dow.iloc[0] == 0 and pf.p1.iloc[0] == pytest.approx(-(104.0 - 100.0) / 3.0)


def test_postfill_ignores_fill_after_friday_close():
    df = _gap_week(days=9)
    t0 = pd.Timestamp("2024-07-19 20:30", tz="UTC")   # 금 16:30 EDT — 주 끝 뒤
    i = int(df.index.searchsorted(t0))
    df.iloc[i, df.columns.get_loc("low")] = 99.0
    _, _, _, (pf, d1, _, _) = _all(df)
    assert not pf.filled.iloc[0] and len(d1) == 0


@pytest.mark.parametrize("start,mon,expect", [
    ("2024-07-12", "2024-07-15 13:30", "2024-07-12 16:00"),     # 여름: 금 16~20 UTC 봉이 마지막(완전)
    ("2024-01-12", "2024-01-15 14:30", "2024-01-12 20:00"),     # 겨울: 금 20~21 UTC 부분 봉도 남긴다
])
def test_weekday_htf_at_monday_open_is_friday_last_bin(start, mon, expect):
    from qbot.data.resample import build_ref_map, resample_htf
    df = _gap_week(start, mon)
    H = resample_htf(df[W.weekday_mask(df.index)], "4h", min_ltf_bars=1)
    ref = build_ref_map(df.index, H)
    mi = int(df.index.searchsorted(pd.Timestamp(mon, tz="UTC")))
    assert H.index[ref[mi]] == pd.Timestamp(expect, tz="UTC")
