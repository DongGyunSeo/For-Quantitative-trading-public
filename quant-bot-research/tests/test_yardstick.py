"""공통 잣대 — 직교화와 심어둔 신호 탐지."""
from __future__ import annotations

import numpy as np

from conftest import synth_5m
from qbot.research import yardstick as Y


def test_build_shapes(df5):
    ys = Y.build(df5)
    n = len(df5)
    for a in (ys.atr, ys.fwd, ys.month, ys.ok, ys.cell, ys.ok_orth):
        assert len(a) == n
    assert ys.ok.sum() > 0.8 * n


def test_forward_return_uses_next_open(df5):
    ys = Y.build(df5)
    o = df5["open"].to_numpy(); c = df5["close"].to_numpy()
    i = 1000
    assert np.isclose(ys.fwd[i], (c[i + 1 + 48] - o[i + 1]) / ys.atr[i])


def test_cell_has_up_to_25_values(df5):
    ys = Y.build(df5)
    vals = np.unique(ys.cell[ys.ok])
    assert 1 < len(vals) <= 25


def test_spread_of_noise_is_near_zero(df5):
    rng = np.random.default_rng(1)
    ys = Y.build(df5)
    _, vv = ys.spread(rng.normal(size=len(df5)))
    assert abs(np.mean(vv)) < 0.3


def test_spread_detects_planted_signal(df5):
    ys = Y.build(df5)
    score = np.where(np.isfinite(ys.fwd), ys.fwd, 0.0)      # 완벽한 예지
    _, vv = ys.spread(score)
    assert np.mean(vv) > 1.0


def test_orthogonalize_removes_the_control(df5):
    ys = Y.build(df5)
    c = df5["close"].to_numpy()
    past = np.full(len(c), np.nan); past[288:] = c[:-288]
    mom = (c - past) / ys.atr                                # 통제 집합의 원소
    r = Y.orthogonalize(np.nan_to_num(mom), ys.ok_orth, ys._Q)
    assert np.nanmax(np.abs(r[ys.ok_orth])) < 1e-6


def test_orthogonalize_keeps_independent_signal(df5):
    rng = np.random.default_rng(4)
    ys = Y.build(df5)
    x = rng.normal(size=len(df5))
    r = Y.orthogonalize(x, ys.ok_orth, ys._Q)
    ok = ys.ok_orth
    assert np.corrcoef(r[ok], x[ok])[0, 1] > 0.98


def test_tstat():
    m, t, n = Y.tstat(np.ones(100))
    assert m == 1.0 and n == 100 and not np.isfinite(t)
    m, t, n = Y.tstat([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and t > 0 and n == 4
    assert Y.tstat([1.0])[2] == 1


def test_cluster_t_singletons_equal_plain_t():
    rng = np.random.default_rng(5)
    x = rng.normal(0.1, 1.0, 400)
    m, t, n, G = Y.cluster_t(x, np.arange(400))
    m2, t2, _ = Y.tstat(x)
    assert n == 400 and G == 400
    assert np.isclose(m, m2) and np.isclose(t, t2)


def test_cluster_t_shrinks_with_correlated_clusters():
    """같은 달 거래가 같이 움직이면 유효 표본이 줄어 t 가 작아져야 한다."""
    rng = np.random.default_rng(6)
    month = np.repeat(np.arange(40), 50)
    shock = rng.normal(0, 1.0, 40)[month]           # 달마다 공통 충격
    x = 0.1 + shock + rng.normal(0, 1.0, len(month))
    _, t_naive, _ = Y.tstat(x)
    _, t_cl, _, G = Y.cluster_t(x, month)
    assert G == 40 and abs(t_cl) < abs(t_naive)


def test_cluster_t_is_trade_weighted_not_month_weighted():
    """거래가 몰린 달이 나쁘면 월평균 t 와 부호가 갈릴 수 있다 — 이 함수는 거래 가중."""
    month = np.array([0] * 100 + [1] * 5 + [2] * 5 + [3] * 5)
    x = np.concatenate([np.full(100, -0.5), np.full(15, 1.0)])
    x = x + np.random.default_rng(1).normal(0, 0.01, len(x))
    m, _, _, _ = Y.cluster_t(x, month)
    mm = np.array([x[month == k].mean() for k in range(4)])
    assert m < 0 < mm.mean()
