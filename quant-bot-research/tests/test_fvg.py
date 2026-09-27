"""FVG — 정의 · 1R · 측정해상도 규칙 · 룩어헤드."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.data.resample import resample_htf
from qbot.research.fvg import FVGConfig, atr5m, find_fvg, run_fvg


def _htf(rows):
    idx = pd.date_range("2022-01-01", periods=len(rows), freq="1h", tz="UTC")
    d = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    d["volume"] = 1.0
    d["close_ts"] = idx + pd.Timedelta("1h")
    d["n_ltf"] = 12
    return d


def test_bullish_fvg_detected():
    h = _htf([[1, 10, 5, 8], [1, 14, 9, 13], [1, 20, 12, 18]])
    f = find_fvg(h)
    assert len(f) == 1
    assert f.iloc[0]["side"] == 1 and f.iloc[0]["bottom"] == 10 and f.iloc[0]["top"] == 12


def test_bearish_fvg_detected():
    h = _htf([[1, 20, 12, 14], [1, 14, 9, 10], [1, 9, 4, 5]])
    f = find_fvg(h)
    assert len(f) == 1
    assert f.iloc[0]["side"] == -1 and f.iloc[0]["bottom"] == 9 and f.iloc[0]["top"] == 12


def test_no_gap_no_fvg():
    h = _htf([[1, 10, 5, 8], [1, 11, 6, 9], [1, 12, 7, 10]])
    assert len(find_fvg(h)) == 0


def test_zero_width_rejected():
    h = _htf([[1, 10, 5, 8], [1, 14, 9, 13], [1, 20, 10, 18]])   # low[i] == high[i-2]
    assert len(find_fvg(h)) == 0


def test_risk_is_half_the_gap_at_mid(df5):
    res = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, entry_level=0.5))
    assert len(res) > 10
    assert np.allclose(res.risk_bps * 2, res.size_bps, rtol=1e-9)


def test_entry_level_scales_risk(df5):
    shallow = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, entry_level=0.25))
    mid = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, entry_level=0.5))
    # 얕게 들어갈수록 반대편 끝까지가 멀다 -> 1R 이 크다
    assert np.median(shallow.risk_bps) > np.median(mid.risk_bps)


def test_resolution_rule_filters(df5):
    loose = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0))
    tight = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=2.0))
    assert len(tight) < len(loose)
    assert (tight.r_over_atr >= 2.0).all()
    assert np.median(tight.risk_bps) > np.median(loose.risk_bps)


def test_resolution_rule_kills_ambiguity(df5):
    loose = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0))
    tight = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=2.0))
    assert (tight.r == 0).mean() < (loose.r == 0).mean()


def test_entry_bar_is_touch_plus_one_when_skipping(df5):
    res = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, skip_entry_bar=True))
    assert (res.entry_bar == res.touch_bar + 1).all()
    res2 = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, skip_entry_bar=False))
    assert (res2.entry_bar == res2.touch_bar).all()


def test_no_lookahead_truncation(df5):
    """뒤쪽을 잘라도 앞쪽 이벤트가 바뀌면 안 된다."""
    cut = len(df5) // 2
    full = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0)).frame()
    part = run_fvg(df5.iloc[:cut], FVGConfig(htf="1h", min_r_over_atr=0.0)).frame()
    lim = cut - 2100          # 지평 안쪽만 비교
    a = full[full.touch_bar < lim].reset_index(drop=True)
    b = part[part.touch_bar < lim].reset_index(drop=True)
    assert len(a) == len(b)
    for col in ("touch_bar", "side", "entry", "risk", "r"):
        assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float))


def test_future_price_distortion_does_not_change_past(df5):
    """설계 문서가 지정한 가장 중요한 회귀 테스트."""
    cut = len(df5) // 2
    warped = df5.copy()
    warped.iloc[cut:] *= 3.0
    a = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0)).frame()
    b = run_fvg(warped, FVGConfig(htf="1h", min_r_over_atr=0.0)).frame()
    lim = cut - 2100
    a = a[a.touch_bar < lim].reset_index(drop=True)
    b = b[b.touch_bar < lim].reset_index(drop=True)
    assert len(a) == len(b) and np.allclose(a.r.to_numpy(), b.r.to_numpy())


def test_atr_modes_differ(df5):
    inst = atr5m(df5, 14, mode="inst")
    daily = atr5m(df5, 14, mode="daily")
    assert not np.allclose(inst, daily, equal_nan=True)
    with pytest.raises(ValueError):
        atr5m(df5, 14, mode="없는모드")


def test_daily_mode_is_the_smoother_denominator(df5):
    """분모가 어느 쪽이냐로 표본이 갈린다 — 합성 데이터에서는 변동성 군집이
    없어 차이가 작지만, 실데이터에서는 daily 가 1h 에서 1.4배 더 통과시킨다
    (터치봉 ATR 은 "가격이 방금 움직인 순간"이라 내생적으로 부푼다)."""
    inst = atr5m(df5, 14, mode="inst")
    daily = atr5m(df5, 14, mode="daily")
    fin = np.isfinite(inst) & np.isfinite(daily)
    assert np.nanstd(np.diff(daily[fin])) < np.nanstd(np.diff(inst[fin]))
    a = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=2.0, atr_mode="inst"))
    b = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=2.0, atr_mode="daily"))
    assert len(a) > 0 and len(b) > 0


def test_wait_window_limits_entries(df5):
    short = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, wait=2))
    long_ = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0, wait=50))
    assert len(short) < len(long_)


def test_long_and_short_both_present(df5):
    res = run_fvg(df5, FVGConfig(htf="1h", min_r_over_atr=0.0))
    assert (res.side > 0).any() and (res.side < 0).any()


def test_frame_roundtrip(df5):
    res = run_fvg(df5, FVGConfig(htf="4h", min_r_over_atr=0.0))
    f = res.frame()
    assert len(f) == len(res) and "r_over_atr" in f.columns
