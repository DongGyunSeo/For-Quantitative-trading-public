"""배리어 레이서 — 손계산 대조 + 치환 대조군."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import synth_5m
from qbot.research.barrier import control_expectancy, first_passage_r, permuted_entries


def _arr(*xs):
    return [np.asarray(x, float) for x in xs]


def test_clean_win():
    h, l = _arr([10, 10, 13, 10], [10, 10, 10, 10])
    r = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                        np.array([1]), tp_r=2.0, sl_r=1.0, horizon=10)
    assert r[0] == 2.0


def test_clean_loss():
    h, l = _arr([10, 10, 10, 10], [10, 10, 8.5, 10])
    r = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                        np.array([1]), tp_r=2.0, sl_r=1.0, horizon=10)
    assert r[0] == -1.0


def test_same_bar_is_zero():
    h, l = _arr([10, 13, 10], [10, 8, 10])
    r = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                        np.array([1]), tp_r=2.0, sl_r=1.0, horizon=10)
    assert r[0] == 0.0


def test_unresolved_is_zero():
    h = np.full(30, 10.01)
    l = np.full(30, 9.99)
    r = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                        np.array([1]), horizon=5)
    assert r[0] == 0.0


def test_skip_entry_bar_removes_fastest_loss():
    """진입봉을 빼면 가장 빠른 손실이 삭제된다 — 낙관 편향의 정체."""
    h = np.array([10.0, 10.0, 13.0, 10.0])
    l = np.array([8.5, 10.0, 10.0, 10.0])
    kw = dict(tp_r=2.0, sl_r=1.0, horizon=10)
    inc = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                          np.array([1]), skip_entry_bar=False, **kw)
    exc = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                          np.array([1]), skip_entry_bar=True, **kw)
    assert inc[0] == -1.0 and exc[0] == 2.0


def test_short_is_mirror():
    rng = np.random.default_rng(3)
    n = 5000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h, l = c * 1.002, c * 0.998
    bars = np.arange(10, 4000, 37)
    e = c[bars]
    rk = e * 0.02
    kw = dict(tp_r=1.0, sl_r=1.0, horizon=500, skip_entry_bar=False)
    lo = first_passage_r(h, l, bars, e, rk, np.ones(len(bars), int), **kw)
    sh = first_passage_r(h, l, bars, e, rk, -np.ones(len(bars), int), **kw)
    both = (lo != 0) & (sh != 0)
    assert both.sum() > 50
    assert (lo[both] == -sh[both]).all()


def test_zero_or_bad_risk_gives_zero():
    h, l = _arr([10, 13, 10], [10, 10, 10])
    r = first_passage_r(h, l, np.array([0, 0, 0]), np.array([10.0, 10.0, np.nan]),
                        np.array([0.0, -1.0, 1.0]), np.array([1, 1, 1]), horizon=5)
    assert (r == 0).all()


def test_side_zero_gives_zero():
    h, l = _arr([10, 13], [10, 10])
    r = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                        np.array([0]), horizon=5)
    assert r[0] == 0.0


def test_horizon_truncates():
    h = np.concatenate([np.full(10, 10.0), np.full(10, 20.0)])
    l = np.full(20, 10.0)
    near = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                           np.array([1]), tp_r=2.0, horizon=5)
    far = first_passage_r(h, l, np.array([0]), np.array([10.0]), np.array([1.0]),
                          np.array([1]), tp_r=2.0, horizon=19)
    assert near[0] == 0.0 and far[0] == 2.0


def test_tight_risk_produces_more_ambiguity():
    """규칙이 존재하는 이유. 1R 이 봉보다 작으면 모호가 폭증한다."""
    df = synth_5m(20_000, seed=6)
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    bars = np.arange(100, 15_000, 53)
    tight = first_passage_r(h, l, bars, c[bars], c[bars] * 0.0002,
                            np.ones(len(bars), int), horizon=500, skip_entry_bar=False)
    wide = first_passage_r(h, l, bars, c[bars], c[bars] * 0.02,
                           np.ones(len(bars), int), horizon=500, skip_entry_bar=False)
    assert (tight == 0).mean() > (wide == 0).mean()


def test_permuted_entries_stay_in_range():
    rng = np.random.default_rng(1)
    b = permuted_entries(np.arange(100), 10_000, rng, lo=500, horizon=2016)
    assert (b >= 500).all() and (b < 10_000 - 2016 - 1).all() and len(b) == 100


def test_permuted_entries_degenerate_range():
    rng = np.random.default_rng(1)
    b = permuted_entries(np.arange(5), 100, rng, lo=500, horizon=2016)
    assert (b == np.arange(5)).all()


def test_control_uses_given_risk_distribution():
    """대조군은 1R(bps) 분포를 그대로 쓴다 — 넓은 1R 이면 모호가 줄어든다."""
    df = synth_5m(30_000, seed=8)
    h, l, o = (df[k].to_numpy() for k in ("high", "low", "open"))
    bars = np.arange(1000, 20_000, 91)
    side = np.ones(len(bars), int)
    kw = dict(n_draw=3, tp_r=1.0, sl_r=1.0, horizon=500, skip_entry_bar=False)
    m_t, _ = control_expectancy(h, l, o, bars, np.full(len(bars), 2.0), side,
                                rng=np.random.default_rng(1), **kw)
    m_w, _ = control_expectancy(h, l, o, bars, np.full(len(bars), 200.0), side,
                                rng=np.random.default_rng(1), **kw)
    assert abs(m_t) < abs(m_w) + 1.0        # 둘 다 계산되고 유한
    assert np.isfinite(m_t) and np.isfinite(m_w)


def test_control_is_near_zero_on_symmetric_barriers():
    """대칭 배리어 무작위 진입의 기대값은 0 근처지만 정확히 0 은 아니다."""
    df = synth_5m(60_000, seed=9)
    h, l, o = (df[k].to_numpy() for k in ("high", "low", "open"))
    bars = np.arange(1000, 40_000, 37)
    m, draws = control_expectancy(h, l, o, bars, np.full(len(bars), 100.0),
                                  np.ones(len(bars), int), n_draw=5,
                                  rng=np.random.default_rng(3),
                                  tp_r=1.0, sl_r=1.0, horizon=2016,
                                  skip_entry_bar=False)
    assert abs(m) < 0.15 and len(draws) == 5


def test_control_draws_vary():
    df = synth_5m(30_000, seed=11)
    h, l, o = (df[k].to_numpy() for k in ("high", "low", "open"))
    bars = np.arange(1000, 20_000, 101)
    _, draws = control_expectancy(h, l, o, bars, np.full(len(bars), 50.0),
                                  np.ones(len(bars), int), n_draw=4,
                                  rng=np.random.default_rng(5),
                                  tp_r=2.0, sl_r=1.0, horizon=1000,
                                  skip_entry_bar=False)
    assert draws.std() > 0
