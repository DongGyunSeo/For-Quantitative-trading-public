"""변화점 — 심어둔 단절을 찾고, 잡음에서는 유의하지 않아야 한다."""
from __future__ import annotations

import numpy as np

from qbot.research import changepoint as CP


def test_split_t_matches_two_sample_t():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    t = CP.split_t(x, 6)
    k = 17
    a, b = x[:k], x[k:]
    s2 = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / (len(x) - 2)
    ref = (a.mean() - b.mean()) / np.sqrt(s2 * (1 / len(a) + 1 / len(b)))
    assert np.isclose(t[k], ref)
    assert np.isnan(t[:6]).all() and np.isnan(t[-5:]).all()


def test_finds_planted_break():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(1.0, 1.0, 36), rng.normal(-1.0, 1.0, 27)])
    cps = CP.detect(x, n_perm=500)
    assert abs(cps[0].k - 36) <= 2
    assert cps[0].p < 0.01
    assert cps[0].before > 0 > cps[0].after


def test_noise_is_not_significant_on_average():
    ps = []
    for s in range(20):
        x = np.random.default_rng(100 + s).normal(size=60)
        ps.append(CP.detect(x, n_perm=300, second=False)[0].p)
    ps = np.array(ps)
    assert (ps < 0.05).mean() <= 0.15          # 명목 5% 근처 (표본 20개라 여유)
    assert np.median(ps) > 0.2


def test_max_over_k_is_corrected():
    """k 를 고른 뒤의 t 로 보통 t 검정을 하면 잡음에서도 자주 '유의' 가 나온다 — 교정 확인."""
    naive = 0
    corrected = 0
    for s in range(30):
        x = np.random.default_rng(200 + s).normal(size=60)
        cp = CP.detect(x, n_perm=300, second=False)[0]
        naive += abs(cp.t) > 1.96
        corrected += cp.p < 0.05
    assert naive > corrected


def test_second_break_only_when_first_significant():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20), rng.normal(-2, 1, 20)])
    cps = CP.detect(x, n_perm=400)
    assert len(cps) == 2
    ks = sorted(c.k for c in cps)
    assert abs(ks[0] - 20) <= 3 and abs(ks[1] - 40) <= 3
