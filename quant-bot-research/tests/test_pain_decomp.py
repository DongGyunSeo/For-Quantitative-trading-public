"""분해 엔진 — 원본 구현과 완전 일치해야 한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import daily_atr, synth_5m
from qbot.research.pain_decomp import (
    contributions, imbalance, monthly_spread, quintile,
)
from qbot.research.trader_pop import CANON, Archetype, population_state

STRUCT = ("ma_cross_struct", "rsi_band", "pivot_break")


@pytest.fixture(scope="module")
def synth():
    df = synth_5m(12_000, seed=7)
    archs = [Archetype(k, p, tf, stop=st, exit=ex, label=lb)
             for lb, k, p, st, ex, _ in CANON if lb in STRUCT
             for tf in ("5m", "15m", "1h")]
    return df, daily_atr(df), archs


@pytest.fixture(scope="module")
def cs(synth):
    df, atr, archs = synth
    return contributions(df, archs, atr)


def test_roster_shape(cs, synth):
    _, _, archs = synth
    assert len(cs) == len(archs)
    assert {c.name for c in cs} == {f"{a.label}@{a.tf}" for a in archs}


def test_pain_r3_matches_population_state(cs, synth):
    df, atr, archs = synth
    R, _ = population_state(df, archs, atr)
    assert np.allclose(imbalance(cs, axis="pain", cap=3.0), R.pain_imb_R3,
                       atol=1e-9, rtol=1e-9)


def test_pain_r_uncapped_matches(cs, synth):
    df, atr, archs = synth
    R, _ = population_state(df, archs, atr)
    assert np.allclose(imbalance(cs, axis="pain", cap=np.inf), R.pain_imb_R,
                       atol=1e-9, rtol=1e-9)


def test_uw_matches(cs, synth):
    df, atr, archs = synth
    R, _ = population_state(df, archs, atr)
    assert np.allclose(imbalance(cs, axis="uw"), R.uw_imb, atol=1e-9, rtol=1e-9)


def test_count_matches(cs, synth):
    df, atr, archs = synth
    _, cnt = population_state(df, archs, atr)
    assert np.allclose(imbalance(cs, axis="count"), cnt, atol=1e-9, rtol=1e-9)


def test_gain_matches(cs, synth):
    df, atr, archs = synth
    R, _ = population_state(df, archs, atr)
    ref = (R.gain_R_short - R.gain_R_long) / np.maximum(R.n_live, 1)
    assert np.allclose(imbalance(cs, axis="gain"), ref, atol=1e-9, rtol=1e-9)


def test_cap_is_monotone(cs):
    sds = [imbalance(cs, cap=k).std() for k in (0.5, 1.0, 2.0, 3.0, 6.0, np.inf)]
    assert all(b >= a - 1e-12 for a, b in zip(sds, sds[1:]))


def test_cap_infinite_equals_large_finite(cs):
    assert np.allclose(imbalance(cs, cap=1e9), imbalance(cs, cap=np.inf))


def test_subset_of_one_is_that_trader(cs):
    c = cs[0]
    v = np.minimum(c.loss, 3.0)
    d = np.where(c.side < 0, v, 0.0) - np.where(c.side > 0, v, 0.0)
    assert np.allclose(imbalance([c]),
                       d / np.maximum((c.side != 0).astype(float), 1.0))


def test_weights_differ_but_correlate(cs):
    a = imbalance(cs, weight="sum")
    for other in (imbalance(cs, weight="equal"), imbalance(cs, weight="z")):
        assert not np.allclose(a, other)
        ok = np.isfinite(a) & np.isfinite(other)
        assert 0.1 < np.corrcoef(a[ok], other[ok])[0, 1] < 0.999


def test_equal_weight_lifts_rare_trader(cs):
    rare = min(cs, key=lambda c: (c.side != 0).mean())
    busy = max(cs, key=lambda c: (c.side != 0).mean())
    assert (rare.side != 0).mean() < (busy.side != 0).mean()
    pair = [rare, busy]
    s = imbalance(pair, weight="sum")
    e = imbalance(pair, weight="equal")
    v = np.minimum(rare.loss, 3.0)
    d = np.where(rare.side < 0, v, 0.0) - np.where(rare.side > 0, v, 0.0)
    ok = np.isfinite(s) & np.isfinite(e) & np.isfinite(d)
    assert np.corrcoef(e[ok], d[ok])[0, 1] > np.corrcoef(s[ok], d[ok])[0, 1]


def test_unknown_axis_and_weight_raise(cs):
    with pytest.raises(ValueError):
        imbalance(cs, axis="없는축")
    with pytest.raises(ValueError):
        imbalance(cs, weight="없는가중")
    with pytest.raises(ValueError):
        imbalance([])


def test_quintile_is_balanced():
    rng = np.random.default_rng(3)
    x = rng.normal(size=50_000)
    q = quintile(x, np.ones(len(x), bool))
    cnt = np.bincount(q[q >= 0], minlength=5)
    assert cnt.min() > 0.19 * len(x) and cnt.max() < 0.21 * len(x)


def test_quintile_marks_excluded():
    x = np.arange(1000.0)
    ok = x < 500
    q = quintile(x, ok)
    assert (q[~ok] == -1).all() and (q[ok] >= 0).all()


def _ref_spread(month, fwd, a_q, cell, nq=5, min_cell=30):
    """원본 파이썬 이중 groupby 구현 그대로."""
    df = pd.DataFrame({"m": month, "f": fwd, "a": a_q, "c": cell})
    df = df[np.isfinite(df.f) & (df.a >= 0)]
    out = []
    for m, g in df.groupby("m", sort=True):
        num = den = 0.0
        for _, gc in g.groupby("c", sort=False):
            hi, lo = gc[gc.a == nq - 1].f, gc[gc.a == 0].f
            if len(hi) < min_cell or len(lo) < min_cell:
                continue
            w = min(len(hi), len(lo))
            num += w * (hi.mean() - lo.mean())
            den += w
        if den > 0:
            out.append((int(m), num / den))
    return out


def test_monthly_spread_matches_reference():
    rng = np.random.default_rng(11)
    n = 120_000
    month = np.repeat(np.arange(24), n // 24)
    cell = rng.integers(0, 25, n)
    score = rng.normal(size=n)
    fwd = 0.3 * score + rng.normal(size=n)
    fwd[rng.random(n) < 0.05] = np.nan
    q = quintile(score, np.isfinite(score))
    ref = _ref_spread(month, fwd, q, cell)
    mm, vv = monthly_spread(month, fwd, q, cell)
    assert len(ref) == len(mm)
    assert np.array_equal(np.array([r[0] for r in ref]), mm)
    assert np.allclose([r[1] for r in ref], vv)


def test_monthly_spread_respects_min_cell():
    rng = np.random.default_rng(13)
    n = 4_000
    month = np.zeros(n, np.int64)
    cell = np.arange(n) % 200
    q = quintile(rng.normal(size=n), np.ones(n, bool))
    mm, _ = monthly_spread(month, rng.normal(size=n), q, cell, min_cell=30)
    assert len(mm) == 0


def test_monthly_spread_detects_planted_edge():
    rng = np.random.default_rng(17)
    n = 200_000
    month = np.repeat(np.arange(20), n // 20)
    cell = rng.integers(0, 10, n)
    score = rng.normal(size=n)
    fwd = 0.5 * score + rng.normal(size=n) * 0.5
    _, vv = monthly_spread(month, fwd, quintile(score, np.ones(n, bool)), cell)
    assert 1.2 < vv.mean() < 1.6 and (vv > 0).all()
