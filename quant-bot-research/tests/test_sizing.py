"""사이징 · 교차 증거금 — 손계산."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.research import sizing as S


def _df(rows, cols=("A", "B", "C")):
    return pd.DataFrame(rows, columns=list(cols), index=pd.RangeIndex(len(rows)))


def test_unit_weights_schemes():
    sig = _df([[1, -1, 1], [1, np.nan, -1]])
    sig_v = _df([[0.5, 1.0, 2.0], [0.5, 1.0, 1.0]])
    ew = S.unit_weights(sig, None, "ew")
    assert ew.iloc[0].tolist() == pytest.approx([1 / 3, -1 / 3, 1 / 3])
    assert ew.iloc[1, 0] == pytest.approx(0.5) and np.isnan(ew.iloc[1, 1])
    iv = S.unit_weights(sig, sig_v, "iv")
    a = np.array([2.0, 1.0, 0.5]) / 3.5
    assert iv.iloc[0].tolist() == pytest.approx([a[0], -a[1], a[2]])
    vp = S.unit_weights(sig, sig_v, "vp")
    assert vp.iloc[1, 0] == pytest.approx(1 / 3) and vp.iloc[1, 2] == pytest.approx(-2 / 3)
    for W in (ew, iv, vp):
        assert W.abs().sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_book_actual_turnover_by_hand():
    W = _df([[0.5, -0.5], [0.5, -0.5]], cols=("A", "B"))
    R = _df([[0.10, 0.10], [0.0, 0.0]], cols=("A", "B"))
    b = S.book(W, R, fee_mode="actual")
    # 1주: 처음 진입 거래 1.0 → 수수료 6bps, 수익 0.5·0.1 − 0.5·0.1 = 0
    assert b.fee.iloc[0] == pytest.approx(6e-4)
    net0 = 0 - 6e-4 - S.FUND_WK * 0.0
    assert b.ret.iloc[0] == pytest.approx(net0)
    # 2주: 흐른 노출 A 0.55/(1+net0), B −0.55/(1+net0) → 되맞춤 거래 = 2·|0.5 − 0.55/(1+net0)|
    drift = 0.55 / (1 + net0)
    assert b.turnover.iloc[1] == pytest.approx(2 * abs(0.5 - drift))
    flip = S.book(W, R, fee_mode="flip")
    assert flip.fee.iloc[1] == pytest.approx(0.0)


def test_book_gross_scales():
    W = _df([[1.0, 0.0], [-1.0, 0.0]], cols=("A", "B"))
    R = _df([[0.05, 0.0], [0.02, 0.0]], cols=("A", "B"))
    b = S.book(W, R, gross=pd.Series([2.0, 2.0]))
    assert b.gross_ret.tolist() == pytest.approx([0.10, -0.04])
    assert b.fund.iloc[0] == pytest.approx(2 * S.FUND_WK) and b.fund.iloc[1] == pytest.approx(-2 * S.FUND_WK)


def test_week_paths_and_liquidation_threshold():
    # 2코인, 주 1개, 봉 2개: 0행 = 주초 직전 봉
    close = np.array([[100, 100], [90, 110], [95, 100]], dtype=np.float32)
    high = np.array([[100, 100], [100, 120], [96, 111]], dtype=np.float32)
    low = np.array([[100, 100], [80, 105], [94, 99]], dtype=np.float32)
    W = np.array([[0.5, -0.5]])                               # A 롱, B 숏
    p = S.week_paths(close, high, low, W, bars=2)
    assert p["r"][0].tolist() == pytest.approx([0.5 * -0.1 - 0.5 * 0.1, 0.5 * -0.05 - 0])
    assert p["g"][0].tolist() == pytest.approx([0.5 * 0.9 + 0.5 * 1.1, 0.5 * 0.95 + 0.5 * 1.0])
    assert p["rc"][0][0] == pytest.approx(0.5 * -0.2 - 0.5 * 0.2)       # 롱 저가 80 · 숏 고가 120
    th = S.liq_threshold(p["r"], p["g"], np.array([0.0]), mmr=0.02)
    assert th[0] == pytest.approx(min(1 / (0.02 * 1.0 + 0.1), 1 / (0.02 * 0.975 + 0.025)))
    th2 = S.liq_threshold(p["r"], p["g"], np.array([0.0]), mmr=0.02, stress=2.0)
    assert th2[0] == pytest.approx(1 / (0.02 * 1.0 + 0.2))


def test_equity_path_and_mdd():
    net = np.array([0.10, -0.20, 0.05])
    rmin = np.array([-0.05, -0.30, 0.0])
    f = np.zeros(3)
    end, trough, liq = S.equity_path(net, rmin, f, np.ones(3), np.full(3, np.inf))
    assert end.tolist() == pytest.approx([1.1, 0.88, 0.924]) and liq == -1
    assert trough.tolist() == pytest.approx([0.95, 1.1 * 0.7, 0.88])
    assert S.mdd(end) == pytest.approx(0.2)
    assert S.mdd(end, trough) == pytest.approx(0.3)                 # 1.1 → 0.77
    end2, tr2, liq2 = S.equity_path(net, rmin, f, np.full(3, 3.0), np.array([np.inf, 2.5, np.inf]))
    assert liq2 == 1 and end2[1] == 0 and end2[2] == 0 and S.mdd(end2, tr2) == pytest.approx(1.0)


def test_book_cadence_weekly_equals_book_and_monthly_drift():
    W = _df([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]], cols=("A", "B"))
    R = _df([[0.20, 0.0], [-0.40, 0.0], [0.05, 0.0]], cols=("A", "B"))    # 2주차 큰 손실 → 자산이 처음보다 낮아진다
    G = 0.75
    ref = S.book(W, R, gross=pd.Series(G, index=W.index))
    wk, liq = S.book_cadence(W, R, G, [True, True, True])
    assert liq == -1 and np.allclose(wk.ret, ref.ret) and np.allclose(wk.G_eff, G)
    # 매월(첫 주만 갱신): 2주차 실효 G = G · 1 / E1
    mo, _ = S.book_cadence(W, R, G, [True, False, False])
    E1 = 1 + mo.ret.iloc[0]
    assert E1 > 1 and mo.G_eff.iloc[1] == pytest.approx(G / E1)
    assert mo.E_end.iloc[2] == pytest.approx((1 + mo.ret).prod())
    # 비대칭: 이익 뒤엔 키우지 않고(1주 뒤 G/E1), 손실로 E < B 가 되면 B = E → 실효 G = G
    asy, _ = S.book_cadence(W, R, G, [True, False, False], asym=True)
    assert asy.G_eff.iloc[1] == pytest.approx(G / E1)
    E2 = asy.E_end.iloc[1]
    assert E2 < 1.0 and asy.G_eff.iloc[2] == pytest.approx(G)
    # 갱신 안 함은 손실 뒤 실효 G 가 G 보다 커진다
    W2 = _df([[1.0, 0.0], [1.0, 0.0]], cols=("A", "B"))
    R2 = _df([[-0.20, 0.0], [0.0, 0.0]], cols=("A", "B"))
    nv, _ = S.book_cadence(W2, R2, G, [True, False])
    assert nv.G_eff.iloc[1] == pytest.approx(G / (1 + nv.ret.iloc[0])) and nv.G_eff.iloc[1] > G


def test_book_cadence_liquidation_and_trough():
    W = _df([[1.0, 0.0], [1.0, 0.0]], cols=("A", "B"))
    R = _df([[-0.10, 0.0], [0.0, 0.0]], cols=("A", "B"))
    df, liq = S.book_cadence(W, R, 2.0, [True, True], rmin=np.array([-0.2, 0.0]), gliq=np.array([np.inf, 1.5]))
    assert df.trough.iloc[0] == pytest.approx(1 - df.fee.iloc[0] + 2.0 * -0.2)
    assert liq == 1 and df.E_end.iloc[1] == 0.0
