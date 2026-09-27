"""뒤집기 보류 규칙 — 극한 경우(기준과 같음 · 항상 유지) · 손계산 · 주봉 ATR · 인과성."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.research import legacy as L
from qbot.research import tsmom_rules as R


def _panel(n_days=400, n=4, seed=1):
    idx = pd.date_range("2022-01-02", periods=n_days, freq="D", tz="UTC")     # 일요일 시작
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"C{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n_days))) for i in range(n)}, index=idx)


def _inputs(C):
    wk = C[C.index.dayofweek == 6]
    sig = np.sign(C / C.shift(28) - 1).reindex(wk.index)
    chg = wk - wk.shift(1)
    atr = (wk - wk.shift(1)).abs().rolling(4, min_periods=4).mean()          # 테스트용 간이 ATR
    fwd = wk.shift(-1) / wk - 1
    return sig, chg, atr, fwd


def test_no_hold_equals_legacy_base():
    C = _panel()
    sig, chg, atr, fwd = _inputs(C)
    pos = R.hold_positions(sig, chg, atr, None)
    a = R.book_from_positions(pos, fwd)
    b = L.tsmom_weekly(C, funding_bps_8h=0.137)
    common = a.index.intersection(b.index)
    assert len(common) > 40
    assert np.allclose(a.ret.loc[common], b.ret.loc[common])
    never = R.hold_positions(sig, chg, atr, -1e9)                           # 절대 보류 안 함 = 기준
    assert np.array_equal(never.fillna(9).to_numpy(), pos.fillna(9).to_numpy())


def test_infinite_k_never_flips_after_atr_available():
    C = _panel()
    sig, chg, atr, _ = _inputs(C)
    pos = R.hold_positions(sig, chg, atr, 1e9)
    for c in pos.columns:
        p = pos[c][atr[c].notna()].dropna()
        assert p.nunique() == 1


def test_hand_case_small_drop_holds_big_drop_flips():
    idx = pd.date_range("2024-01-07", periods=4, freq="7D", tz="UTC")
    sig = pd.DataFrame({"X": [1, -1, -1, -1]}, index=idx, dtype=float)
    chg = pd.DataFrame({"X": [np.nan, -2.0, -6.0, +1.0]}, index=idx)      # 주간 변화(가격)
    atr = pd.DataFrame({"X": [5.0, 5.0, 5.0, 5.0]}, index=idx)
    p = R.hold_positions(sig, chg, atr, 1.0)["X"].tolist()
    assert p == [1.0, 1.0, -1.0, -1.0]          # −2 < 5 → 유지 · −6 ≥ 5 → 숏 · 이후 신호와 같음
    p2 = R.hold_positions(sig, chg, atr, 0.5)["X"].tolist()
    assert p2 == [1.0, 1.0, -1.0, -1.0]         # 2 < 2.5 유지, 6 ≥ 2.5 뒤집기
    sig_s = pd.DataFrame({"X": [-1, 1, 1, 1]}, index=idx, dtype=float)
    chg_s = pd.DataFrame({"X": [np.nan, 3.0, 12.0, 0.0]}, index=idx)
    assert R.hold_positions(sig_s, chg_s, atr, 1.0)["X"].tolist() == [-1.0, -1.0, 1.0, 1.0]
    chg_f = pd.DataFrame({"X": [np.nan, +8.0, 0, 0]}, index=idx)            # 롱인데 올랐는데 신호만 숏 → 유지
    assert R.hold_positions(sig, chg_f, atr, 0.5)["X"].tolist()[:2] == [1.0, 1.0]


def test_weekly_atr_matches_hand_and_is_causal():
    idx = pd.date_range("2024-01-01", periods=16, freq="7D", tz="UTC")
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 3, 16))
    W = {"high": pd.DataFrame({"X": c + 2}, index=idx), "low": pd.DataFrame({"X": c - 2}, index=idx),
         "close": pd.DataFrame({"X": c}, index=idx)}
    a = R.weekly_atr(W, period=14)["X"]
    tr = np.maximum(4.0, np.maximum(np.abs(c[1:] + 2 - c[:-1]), np.abs(c[1:] - 2 - c[:-1])))
    tr = np.concatenate(([4.0], tr))
    assert np.isnan(a.iloc[12]) and a.iloc[13] == pytest.approx(tr[:14].mean()) and a.iloc[15] == pytest.approx(tr[2:16].mean())
    W2 = {k: v.copy() for k, v in W.items()}
    for k in W2:
        W2[k].iloc[15] = W2[k].iloc[15] * 3
    assert R.weekly_atr(W2, period=14)["X"].iloc[14] == a.iloc[14]


def test_positions_are_causal():
    C = _panel(seed=5)
    sig, chg, atr, _ = _inputs(C)
    p = R.hold_positions(sig, chg, atr, 1.0)
    C2 = C.copy()
    cut = C.index[300]
    C2.loc[C2.index > cut] *= 1.7
    s2, c2, a2, _ = _inputs(C2)
    p2 = R.hold_positions(s2, c2, a2, 1.0)
    keep = p.index <= cut
    assert np.array_equal(p[keep].fillna(9).to_numpy(), p2[keep].fillna(9).to_numpy())


def test_zero_signal_is_flat_then_takes_signal():
    idx = pd.date_range("2024-01-07", periods=4, freq="7D", tz="UTC")
    sig = pd.DataFrame({"X": [1, 0, -1, -1]}, index=idx, dtype=float)
    chg = pd.DataFrame({"X": [np.nan, 0.0, -0.1, 0.0]}, index=idx)
    atr = pd.DataFrame({"X": [5.0] * 4}, index=idx)
    assert R.hold_positions(sig, chg, atr, 2.0)["X"].tolist() == [1.0, 0.0, -1.0, -1.0]
