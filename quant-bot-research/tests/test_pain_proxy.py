"""무상태 프록시 — 룩어헤드 차단과 부호 규약."""

from __future__ import annotations

import numpy as np

from qbot.research.pain_proxy import SCALES, trapped_imbalance, trend_side


def test_trend_side_is_backward_looking():
    s = trend_side(np.array([1.0, 2, 3, 2, 1, 2, 3]), 2)
    assert np.isnan(s[0]) and np.isnan(s[1])
    assert s[2] == 1.0 and s[4] == -1.0


def test_no_lookahead_truncation_invariant():
    rng = np.random.default_rng(3)
    n = 3000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    a = np.full(n, 0.3)
    full = trapped_imbalance(c, a, lookback=50, w=20)
    cut = trapped_imbalance(c[:2000], a[:2000], lookback=50, w=20)
    for d in SCALES:
        assert np.allclose(full[d][:2000], cut[d], equal_nan=True)


def _ramp_then_drop(drop, n_up=200, n_dn=60):
    """천천히 오르다(전원 롱) 한 번 떨어지고 계속 조금씩 내리는 경로.

    평평한 구간을 두면 ``trend_side`` 가 0 이 되어 아무도 안 세어진다 —
    테스트를 처음 그렇게 짰다가 전부 0 이 나왔다.
    """
    up = 100.0 + 0.01 * np.arange(n_up)
    dn = up[-1] - drop - 0.001 * np.arange(1, n_dn + 1)
    return np.concatenate([up, dn])


def test_sign_convention_longs_hurt_is_negative():
    c = _ramp_then_drop(0.5)
    out = trapped_imbalance(c, np.full(len(c), 1.0), lookback=80, w=1, scales=(1.0,))
    assert out[1.0][205] < 0            # 롱이 0.25~0.75 ATR 물렸다


def test_sign_convention_shorts_hurt_is_positive():
    c = -_ramp_then_drop(0.5)
    c = c - c.min() + 100.0             # 거울상: 전원 숏이 물린다
    out = trapped_imbalance(c, np.full(len(c), 1.0), lookback=80, w=1, scales=(1.0,))
    assert out[1.0][205] > 0


def test_band_excludes_outside():
    """손실이 5 ATR 이면 d=1 (띠 0.25~0.75) 밖이고 d=20 (띠 5~15) 안이다."""
    c = _ramp_then_drop(5.0)
    a = np.full(len(c), 1.0)
    assert np.allclose(trapped_imbalance(c, a, lookback=80, w=1, scales=(1.0,))[1.0][205], 0.0)
    assert abs(trapped_imbalance(c, a, lookback=80, w=1, scales=(20.0,))[20.0][205]) > 0


def test_scale_ladder_tracks_lookback():
    """어느 스케일이 걸리느냐는 룩백 동안의 움직임 크기가 정한다.

    짧은 룩백이면 손실이 작아 작은 d 가, 긴 룩백이면 큰 d 가 담당한다.
    "작은 스케일이 항상 더 자주 걸린다" 는 틀렸다 — 처음 그렇게 단정했다가
    ATR 대비 200봉 이동이 12 ATR 이라는 실측에 걸렸다.
    """
    rng = np.random.default_rng(11)
    n = 6000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    a = np.full(n, float(np.median(np.abs(np.diff(c)))))
    short = trapped_imbalance(c, a, lookback=10, w=5)
    long_ = trapped_imbalance(c, a, lookback=400, w=5)
    r_short = (short[1.0][600:] != 0).mean() / max((short[30.0][600:] != 0).mean(), 1e-9)
    r_long = (long_[1.0][600:] != 0).mean() / max((long_[30.0][600:] != 0).mean(), 1e-9)
    assert r_short > r_long


def test_bounded_by_one():
    rng = np.random.default_rng(9)
    n = 2000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    a = np.maximum(np.abs(rng.normal(0.4, 0.05, n)), 0.05)
    out = trapped_imbalance(c, a, lookback=100, w=30)
    for d in SCALES:
        assert np.nanmax(np.abs(out[d])) <= 1.0 + 1e-12


def test_flat_price_gives_zero():
    out = trapped_imbalance(np.full(500, 50.0), np.full(500, 1.0), lookback=100, w=20)
    for d in SCALES:
        assert np.allclose(out[d][150:], 0.0)


def test_zero_atr_rows_are_skipped():
    rng = np.random.default_rng(5)
    n = 300
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    a = np.full(n, 1.0); a[50:80] = 0.0
    out = trapped_imbalance(c, a, lookback=40, w=10)
    for d in SCALES:
        assert np.isfinite(out[d]).all()
