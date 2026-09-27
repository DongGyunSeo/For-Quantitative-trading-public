"""관통 기하 · 스윕+CHoCH 진입 — 정의와 룩어헤드."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import daily_atr, synth_5m
from qbot.research.fvg import FVGConfig
from qbot.research.fvg_sweep import choch_on_tf, excursion, sweep_choch


@pytest.fixture(scope="module")
def ex(df5):
    return excursion(df5, FVGConfig(htf="1h", atr_mode="daily"), horizon=576)


def test_excursion_shapes(ex):
    n = len(ex.touch_bar)
    assert n > 50
    for a in (ex.side, ex.near, ex.far, ex.gap, ex.depth, ex.recovered):
        assert len(a) == n
    assert set(np.unique(ex.side)) <= {-1, 1}
    assert (ex.gap > 0).all()


def test_depth_sign_convention(ex):
    """−1 = 갭에 들어오지도 않음, 0 = 먼 끝 정확히 도달, +1 = 갭만큼 더."""
    d = ex.depth[np.isfinite(ex.depth)]
    assert (d >= -1.0001).all()
    assert (d > 0).any() and (d < 0).any()


def test_recovery_requires_a_close_back(ex):
    """회복했다면 회복까지 걸린 봉 수가 기록돼 있어야 한다."""
    assert (ex.bars_to_recover[ex.recovered] >= 0).all()
    assert (ex.bars_to_recover[~ex.recovered] == -1).all()


def test_run_after_only_when_recovered(ex):
    assert np.isnan(ex.run_after[~ex.recovered]).all()
    assert np.isfinite(ex.run_after[ex.recovered]).any()


def test_longer_horizon_weakly_increases_depth_and_recovery(df5):
    """지평을 늘리면 뚫는 비율도 회복하는 비율도 줄지 않는다."""
    a = excursion(df5, FVGConfig(htf="1h", atr_mode="daily"), horizon=144)
    b = excursion(df5, FVGConfig(htf="1h", atr_mode="daily"), horizon=2016)
    assert len(a.touch_bar) == len(b.touch_bar)
    assert np.nanmean(b.depth > 0) >= np.nanmean(a.depth > 0) - 1e-9
    assert b.recovered.mean() >= a.recovered.mean() - 1e-9


def test_excursion_no_lookahead(df5):
    cut = len(df5) // 2
    warped = df5.copy()
    warped.iloc[cut:] *= 3.0
    a = excursion(df5, FVGConfig(htf="1h", atr_mode="daily"), horizon=288).frame()
    b = excursion(warped, FVGConfig(htf="1h", atr_mode="daily"), horizon=288).frame()
    lim = cut - 400
    a = a[a.touch_bar < lim].reset_index(drop=True)
    b = b[b.touch_bar < lim].reset_index(drop=True)
    assert len(a) == len(b)
    assert np.allclose(a.depth.to_numpy(), b.depth.to_numpy(), equal_nan=True)


# ---------------------------------------------------------------- CHoCH 매핑


def test_choch_on_tf_is_sparse_and_ternary(df5):
    for tf in ("5m", "15m", "1h"):
        c = choch_on_tf(df5, tf, 3)
        assert len(c) == len(df5)
        assert set(np.unique(c)) <= {-1, 0, 1}
        assert 0 < (c != 0).mean() < 0.2


def test_higher_tf_gives_fewer_choch(df5):
    a = (choch_on_tf(df5, "5m", 3) != 0).sum()
    b = (choch_on_tf(df5, "1h", 3) != 0).sum()
    assert a > b


def test_choch_on_tf_no_lookahead(df5):
    cut = len(df5) // 2
    full = choch_on_tf(df5, "15m", 3)
    part = choch_on_tf(df5.iloc[:cut], "15m", 3)
    assert np.array_equal(full[:cut - 12], part[:cut - 12])


def test_choch_cache_reuse(df5):
    cache = {}
    a = choch_on_tf(df5, "15m", 3, cache)
    b = choch_on_tf(df5, "15m", 3, cache)
    assert a is b


# ---------------------------------------------------------------- 스윕 진입


def test_sweep_choch_basic(df5, ex):
    ch = choch_on_tf(df5, "15m", 3)
    t = sweep_choch(df5, ex, ch, min_depth=0.0, max_wait=576)
    assert len(t) > 10
    assert (t.risk > 0).all()
    # 손절은 반드시 진입가 반대편
    lo = t[t.side > 0]
    assert (lo.stop < lo.entry).all()
    sh = t[t.side < 0]
    assert (sh.stop > sh.entry).all()


def test_sweep_entry_comes_after_break(df5, ex):
    ch = choch_on_tf(df5, "15m", 3)
    t = sweep_choch(df5, ex, ch, min_depth=0.0, max_wait=576)
    assert (t.break_bar >= t.touch_bar).all()
    assert (t.bar >= t.break_bar).all()


def test_min_depth_filters(df5, ex):
    ch = choch_on_tf(df5, "15m", 3)
    a = sweep_choch(df5, ex, ch, min_depth=0.0, max_wait=576)
    b = sweep_choch(df5, ex, ch, min_depth=1.0, max_wait=576)
    assert len(b) <= len(a)
    if len(b):
        assert (b.depth >= 1.0).all()


def test_buffer_widens_the_stop(df5, ex, atr5):
    ch = choch_on_tf(df5, "15m", 3)
    a = sweep_choch(df5, ex, ch, max_wait=576, buffer_atr=0.0, atr=atr5)
    b = sweep_choch(df5, ex, ch, max_wait=576, buffer_atr=0.5, atr=atr5)
    m = a.merge(b, on=["bar", "side", "touch_bar"], suffixes=("_a", "_b"))
    assert len(m) > 5
    assert (m.risk_b >= m.risk_a - 1e-12).all()


def test_stop_is_the_swept_extreme(df5, ex):
    """손절은 터치~진입 사이의 실제 극단이어야 한다."""
    ch = choch_on_tf(df5, "15m", 3)
    t = sweep_choch(df5, ex, ch, max_wait=576)
    lo = df5["low"].to_numpy(); hi = df5["high"].to_numpy()
    for _, r in t.head(80).iterrows():
        s, e = int(r.touch_bar), int(r.bar)
        want = lo[s:e + 1].min() if r.side > 0 else hi[s:e + 1].max()
        assert abs(r.stop - want) < 1e-9


def test_entries_are_deduplicated(df5, ex):
    """한 봉에 여러 FVG 가 같은 신호를 내면 한 번만 센다 — 안 그러면 t 가 과장된다."""
    ch = choch_on_tf(df5, "15m", 3)
    t = sweep_choch(df5, ex, ch, max_wait=576)
    assert not t.duplicated(subset=["bar", "side"]).any()


def test_short_wait_gives_fewer_entries(df5, ex):
    ch = choch_on_tf(df5, "15m", 3)
    a = sweep_choch(df5, ex, ch, max_wait=24)
    b = sweep_choch(df5, ex, ch, max_wait=576)
    assert len(a) <= len(b)
