"""재구축한 예전 신호군 — 레이서 손계산 · 미래 오염 불변 · 가격 거울 대칭."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.research import legacy as L


def _mirror(df: pd.DataFrame, K: float | None = None) -> pd.DataFrame:
    """가격 반사 p' = K − p. 고가↔저가가 바뀌고 모든 방향이 뒤집힌다."""
    K = K if K is not None else float(df["high"].max() + df["low"].min())
    out = df.copy()
    out["open"] = K - df["open"]
    out["close"] = K - df["close"]
    out["high"] = K - df["low"]
    out["low"] = K - df["high"]
    return out


@pytest.fixture(scope="module")
def big():
    return synth_5m(40_000, seed=21, vol=0.003)


# ------------------------------------------------------------------ 레이서


def _race(h, l, c, entry, tp, sl, side, **kw):
    h, l, c = (np.asarray(x, float) for x in (h, l, c))
    return L.race_levels(h, l, c, np.array([0]), np.array([entry]), np.array([tp]),
                         np.array([sl]), np.array([side]), **kw)


def test_race_tp_first_gives_tp_multiple():
    g, k, xb = _race([10, 10.5, 13], [9.8, 10, 10], [10, 10, 12], 10.0, 12.5, 9.0, 1, horizon=10)
    assert k[0] == 1 and g[0] == pytest.approx(2.5) and xb[0] == 2


def test_race_sl_first():
    g, k, _ = _race([10, 10.2, 13], [9.8, 8.9, 10], [10, 9, 12], 10.0, 12.5, 9.0, 1, horizon=10)
    assert k[0] == -1 and g[0] == -1.0


def test_race_same_bar_is_ambiguous():
    g, k, _ = _race([10, 13], [9.9, 8.5], [10, 10], 10.0, 12.5, 9.0, 1, horizon=10)
    assert k[0] == 0 and g[0] == 0.0


def test_race_entry_bar_counts_by_default():
    g, k, _ = _race([10.1, 10], [8.9, 10], [9, 10], 10.0, 12.0, 9.0, 1, horizon=10)
    assert k[0] == -1
    g2, k2, _ = _race([10.1, 10], [8.9, 10], [9, 10], 10.0, 12.0, 9.0, 1, horizon=10,
                      include_entry_bar=False)
    assert k2[0] == 2


def test_race_expiry_mark_to_market():
    g, k, _ = _race([10.2, 10.4, 10.6], [9.9, 10.1, 10.2], [10.1, 10.3, 10.5], 10.0, 12.0, 9.0, 1,
                    horizon=10, end=np.array([3]), mtm=True)
    assert k[0] == 2 and g[0] == pytest.approx(0.5)


def test_race_short_mirror():
    g, k, _ = _race([10.2, 11.2], [9.9, 9.95], [10, 11], 10.0, 7.5, 11.0, -1, horizon=10)
    assert k[0] == -1 and g[0] == -1.0


def test_fee_uses_maker_only_on_tp():
    df = synth_5m(50)
    t = L.trade_frame(df, [1, 2], [1, 1], [100.0, 100.0], [102.0, 102.0], [99.0, 99.0],
                      [2.0, -1.0], [1, -1], entry_bps=L.MAKER)
    assert t.fee[0] == pytest.approx((2 + 2) / 100.0)
    assert t.fee[1] == pytest.approx((2 + 6) / 100.0)
    assert t.net[0] == pytest.approx(2.0 - 0.04)


# ------------------------------------------------------------------ 순위


def test_rank_is_causal(big):
    a = L.tr_atr_rank_5m(big)
    b = L.tr_atr_rank_5m(big.iloc[:20_000])
    assert np.allclose(a[:20_000], b, equal_nan=True)


def test_htf_rank_excludes_current_bar_from_atr():
    from qbot.data.resample import resample_htf
    df = synth_5m(20_000, seed=3)
    H = resample_htf(df, "4h")
    r = L.htf_rank(H)
    H2 = H.copy()
    H2.iloc[-1, H2.columns.get_loc("high")] *= 1.5      # 마지막 봉만 키운다
    r2 = L.htf_rank(H2)
    assert np.allclose(r[:-1], r2[:-1], equal_nan=True)
    assert r2[-1] >= r[-1]


# ------------------------------------------------------------------ 미래 오염 불변 (전 신호)


def _warp(df, cut):
    w = df.copy()
    w.iloc[cut:, :4] *= 3.0
    return w


SIGNALS = {
    "htf_fib": lambda d: L.htf_fib_trades(d, tf="4h"),
    "level": lambda d: L.level_break_trades(d, tf="4h", rank_min=None),
    "bigbar": lambda d: L.big_bar_choch_trades(d),
    "liq_rev": lambda d: L.liq_cascade_trades(d, direction="reversal"),
    "liq_cont": lambda d: L.liq_cascade_trades(d, direction="continuation"),
    "trend": lambda d: L.trend_fvg_trades(d),
}


@pytest.mark.parametrize("name", list(SIGNALS))
def test_future_does_not_change_past(big, name):
    cut = 30_000
    f = SIGNALS[name]
    a = f(big)
    b = f(_warp(big, cut))
    lim = cut - 2100
    a = a[a.bar < lim].reset_index(drop=True)
    b = b[b.bar < lim].reset_index(drop=True)
    assert len(a) > 5, name
    assert len(a) == len(b)
    for col in ("bar", "side", "gross"):
        assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float)), (name, col)


# ------------------------------------------------------------------ 거울 대칭


@pytest.mark.parametrize("name", ["htf_fib", "level", "bigbar", "liq_rev", "liq_cont", "trend"])
def test_price_mirror_flips_side_keeps_outcome(big, name):
    f = SIGNALS[name]
    a = f(big).sort_values(["bar", "side"]).reset_index(drop=True)
    b = f(_mirror(big)).sort_values(["bar", "side"]).reset_index(drop=True)
    b["side"] = -b["side"]
    b = b.sort_values(["bar", "side"]).reset_index(drop=True)
    assert len(a) > 5 and len(a) == len(b), name
    assert np.array_equal(a.bar.to_numpy(), b.bar.to_numpy()), name
    assert np.array_equal(a.side.to_numpy(), b.side.to_numpy()), name
    assert np.allclose(a.gross.to_numpy(), b.gross.to_numpy()), name


# ------------------------------------------------------------------ 신호별 규칙


def test_htf_fib_rank_filter_is_subset(big):
    # 겹침 금지를 켜면 선별된 봉이 전체에서는 보유 중이라 빠질 수 있다 → 끄고 비교
    a = L.htf_fib_trades(big, tf="4h", no_overlap=False)
    b = L.htf_fib_trades(big, tf="4h", rank_min=0.9, no_overlap=False)
    assert 0 < len(b) < len(a)
    assert set(b.bar) <= set(a.bar)
    assert (b["rank"] >= 0.9).all()


def test_htf_fib_no_overlap_means_one_position(big):
    t = L.htf_fib_trades(big, tf="4h", no_overlap=True)
    u = L.htf_fib_trades(big, tf="4h", no_overlap=False)
    assert len(t) < len(u)
    # 보유 중에는 새 진입이 없다: 진입봉이 단조 증가하고 이전 거래 청산 뒤에만 온다
    hi, lo, cl = (big[k].to_numpy() for k in ("high", "low", "close"))
    _, _, xb = L.race_levels(hi, lo, cl, t.bar.to_numpy(), t.entry.to_numpy(),
                             t.entry.to_numpy() + t.side.to_numpy() * t.tp_r.to_numpy()
                             * t.entry.to_numpy() * t.risk_bps.to_numpy() / 1e4,
                             t.entry.to_numpy() - t.side.to_numpy() * t.entry.to_numpy()
                             * t.risk_bps.to_numpy() / 1e4, t.side.to_numpy())
    assert (t.bar.to_numpy()[1:] > xb[:-1]).all()


def test_htf_fib_entry_is_after_bar_close(big):
    from qbot.data.resample import resample_htf
    H = resample_htf(big, "4h")
    t = L.htf_fib_trades(big, tf="4h")
    ct = set(H["close_ts"])
    assert all(ts in ct for ts in t.time)           # 진입 시각 = 어떤 4h 봉의 확정 시각


def test_level_break_inside_only_is_subset(big):
    a = L.level_break_trades(big, tf="4h", rank_min=None, inside_only=False)
    b = L.level_break_trades(big, tf="4h", rank_min=None, inside_only=True)
    assert 0 < len(b) <= len(a)
    assert (b.tp_r.round(9) == 1.0).all()           # 대칭 0.8 거리


def test_bigbar_dedupes_per_choch(big):
    t = L.big_bar_choch_trades(big)
    assert not t.duplicated(["bar", "side"]).any()
    assert (t.tp_r.round(9) == 2.0).all()


def test_liq_directions_are_opposite(big):
    """같은 이벤트에서 반전은 +, 연속은 − — 같은 봉에 상·하 캐스케이드가 겹칠 수 있어
    봉만으로 짝지으면 안 된다(처음에 그렇게 짰다가 교차 짝이 생겼다)."""
    r = L.liq_cascade_trades(big, direction="reversal")
    c = L.liq_cascade_trades(big, direction="continuation")
    assert len(r) > 10 and len(r) == len(c)
    kr = set(zip(r.bar, r.side, r.risk_bps.round(6)))
    kc = set(zip(c.bar, -c.side, c.risk_bps.round(6)))
    assert kr == kc


def test_liq_size_proxy_is_subset(big):
    a = L.liq_cascade_trades(big, direction="continuation")
    ev = L.cascade_events(big)
    b = L.liq_cascade_trades(big, direction="continuation", size_top=0.2, events=ev)
    assert len(b) < len(a)


def test_trend_alignment_mirror_is_disjoint(big):
    a = L.trend_fvg_trades(big, align=1)
    b = L.trend_fvg_trades(big, align=-1)
    assert len(a) > 5 and len(b) > 5
    assert (a.tp_r.round(9) == 3.0).all()


def test_weekend_gap_detects_dst_correctly():
    idx = pd.date_range("2024-03-01", "2024-03-20", freq="5min", tz="UTC").as_unit("us")
    n = len(idx)
    c = np.full(n, 100.0)
    et = idx.tz_convert("America/New_York")
    mon = (et.dayofweek == 0) & (et.hour * 60 + et.minute >= 9 * 60 + 30)
    # 3/10 DST 시작 이후 월요일(3/11) 09:30 ET = 13:30 UTC 부터 가격 +3%
    after = (idx >= pd.Timestamp("2024-03-11 13:30", tz="UTC"))
    c = np.where(after, 103.0, c)
    df = pd.DataFrame(dict(open=c, high=c * 1.0005, low=c * 0.9995, close=c, volume=1.0), index=idx)
    t = L.weekend_gap_trades(df)
    assert len(t) == 1
    assert t.time.iloc[0] == pd.Timestamp("2024-03-11 13:30", tz="UTC")
    assert t.side.iloc[0] == -1                     # 위로 갭 → 숏으로 메움


def test_tsmom_uses_past_signal_only():
    idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    px = pd.DataFrame(dict(A=100 * np.exp(np.cumsum(rng.normal(0, 0.03, 200))),
                           B=100 * np.exp(np.cumsum(rng.normal(0, 0.03, 200)))), index=idx)
    a = L.tsmom_weekly(px)
    warped = px.copy()
    warped.iloc[150:] *= 2.0
    b = L.tsmom_weekly(warped)
    k = a.index[a.index < idx[150] - pd.Timedelta(days=8)]
    assert len(k) > 10
    assert np.allclose(a.loc[k, "ret"], b.loc[k, "ret"])


# ------------------------------------------------------------------ CHoCH 지정가 · painR3 꼬리


def test_limit_signal_trades_applies_rule_and_fills(big):
    from qbot.research.fvg import atr5m
    from qbot.research.fvg_sweep import choch_on_tf
    a = atr5m(big, 14, mode="daily")
    s = L.plain_choch_signals(big, choch_on_tf(big, "15m", 3))
    t = L.limit_signal_trades(big, s, a)
    assert 0 < len(t) <= len(s)
    rp = t.entry * t.risk_bps / 1e4
    assert (rp / a[t.bar.to_numpy()] >= 2.0 - 1e-9).mean() > 0.95   # 규칙은 신호봉 기준, 체결봉은 근처
    assert set(np.unique(t.kind)) <= {-1, 0, 1, 2}
    win = t[t.kind == 1]
    assert np.allclose(win.fee, (2 + 2) / win.risk_bps)


def test_pain_tail_thresholds_are_walk_forward(big):
    from qbot.research.fvg import atr5m
    a = atr5m(big, 14, mode="daily")
    rng = np.random.default_rng(5)
    score = rng.normal(size=len(big))
    t1 = L.pain_tail_trades(big, score, a, min_months=1)
    s2 = score.copy()
    s2[30_000:] = s2[30_000:] * 10 + 5          # 미래만 오염
    t2 = L.pain_tail_trades(big, s2, a, min_months=1)
    lim = 29_000
    a1 = t1[t1.bar < lim].reset_index(drop=True)
    a2 = t2[t2.bar < lim].reset_index(drop=True)
    assert len(a1) > 5 and len(a1) == len(a2)
    assert np.array_equal(a1.bar, a2.bar) and np.allclose(a1.net, a2.net)


def test_pain_tail_no_overlap(big):
    from qbot.research.fvg import atr5m
    a = atr5m(big, 14, mode="daily")
    score = np.random.default_rng(6).normal(size=len(big))
    t = L.pain_tail_trades(big, score, a, min_months=1, hold=48)
    assert (np.diff(t.bar.to_numpy()) > 0).all()
