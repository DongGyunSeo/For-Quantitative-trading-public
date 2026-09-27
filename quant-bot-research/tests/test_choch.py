"""스윙 탐지 — 확정 지연이 핵심."""
from __future__ import annotations

import numpy as np

from conftest import synth_5m
from qbot.research.choch import confirmed_levels, swing_points


def test_simple_peak():
    h = np.array([1.0, 2, 3, 9, 3, 2, 1])
    l = h.copy()
    is_h, is_l = swing_points(h, l, 3)
    assert is_h[3] and is_h.sum() == 1


def test_simple_trough():
    l = np.array([9.0, 8, 7, 1, 7, 8, 9])
    h = l.copy()
    _, is_l = swing_points(h, l, 3)
    assert is_l[3] and is_l.sum() == 1


def test_edges_never_marked():
    rng = np.random.default_rng(2)
    h = rng.normal(size=200) + 10
    l = h - 1
    is_h, is_l = swing_points(h, l, 5)
    assert not is_h[:5].any() and not is_h[-5:].any()
    assert not is_l[:5].any() and not is_l[-5:].any()


def test_too_short_returns_empty():
    h = np.arange(4.0)
    is_h, is_l = swing_points(h, h, 3)
    assert not is_h.any() and not is_l.any()


def test_ties_are_not_swings():
    h = np.array([1.0, 5, 5, 5, 1, 1, 1])
    is_h, _ = swing_points(h, h, 2)
    assert not is_h.any()


def test_confirmed_levels_lag_by_lb():
    """스윙 바로 그 시점에는 아직 알 수 없다."""
    h = np.array([1.0, 2, 3, 9, 3, 2, 1, 1, 1, 1])
    l = h.copy()
    hv, _ = confirmed_levels(h, l, 3)
    assert not np.isfinite(hv[3]) and not np.isfinite(hv[5])
    assert hv[6] == 9.0


def test_confirmed_levels_no_lookahead():
    df = synth_5m(3000, seed=4)
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    full, _ = confirmed_levels(h, l, 3)
    cut, _ = confirmed_levels(h[:2000], l[:2000], 3)
    assert np.allclose(full[:2000], cut, equal_nan=True)


def test_structure_first_break_is_bos():
    from qbot.research.choch import structure_events
    h = np.array([1.0, 2, 3, 9, 3, 2, 1, 1, 1, 12, 1, 1])
    l = h.copy()
    c = h.copy()
    s, k = structure_events(h, l, c, 3)
    assert (k == 2).sum() >= 1
    assert (k[k > 0][0] == 2)


def test_structure_alternation_is_choch():
    from qbot.research.choch import structure_events
    rng = np.random.default_rng(7)
    n = 4000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    h, l = c * 1.002, c * 0.998
    s, k = structure_events(h, l, c, 3)
    ev = s[k > 0]
    assert len(ev) > 20
    # 사건은 반드시 방향이 번갈아 난다 (같은 방향 연속은 세지 않으므로)
    assert (ev[1:] != ev[:-1]).all()
    assert (k[k > 0][1:] == 1).all()      # 첫 사건만 BOS, 나머지는 전부 CHoCH


def test_structure_no_lookahead():
    from qbot.research.choch import structure_events
    df = synth_5m(4000, seed=9)
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    s1, k1 = structure_events(h, l, c, 3)
    s2, k2 = structure_events(h[:2500], l[:2500], c[:2500], 3)
    assert np.array_equal(s1[:2500], s2) and np.array_equal(k1[:2500], k2)


def test_structure_needs_a_break():
    from qbot.research.choch import structure_events
    c = np.full(500, 100.0)
    s, k = structure_events(c + 0.5, c - 0.5, c, 3)
    assert (k == 0).all()
