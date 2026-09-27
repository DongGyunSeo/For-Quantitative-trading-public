"""주봉 · 요일 일봉 — 경계(UTC 월 · 뉴욕 자정 DST) · 캔들 모양 · 전주 고저 선도달 · σ 인과성."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.research import weekly as K


def _flat(start, days, price=100.0):
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC").as_unit("us")
    c = np.full(len(idx), price)
    return pd.DataFrame(dict(open=c, high=c, low=c, close=c, volume=1.0), index=idx)


def test_weekly_utc_starts_monday_and_matches_manual():
    df = synth_5m(288 * 21, seed=3, start="2024-01-01")        # 2024-01-01 은 월요일
    W = K.weekly_utc(df)
    assert (W.index.dayofweek == 0).all() and (W.index == W.index.normalize()).all()
    w1 = df.loc["2024-01-08":"2024-01-14 23:55"]
    row = W.loc[pd.Timestamp("2024-01-08", tz="UTC")]
    assert row.open == w1.open.iloc[0] and row.close == w1.close.iloc[-1]
    assert row.high == w1.high.max() and row.low == w1.low.min() and row.n == 2016


def test_prior_week_first_touch_hours():
    df = _flat("2024-01-01", 14)
    t = pd.Timestamp("2024-01-09 06:00", tz="UTC")               # 둘째 주 화 06:00 → 주 시작 후 30시간
    df.loc[t, "high"] = 101.0
    df.loc[pd.Timestamp("2024-01-10 12:00", tz="UTC"), "low"] = 99.0
    W = K.weekly_utc(df)
    r = W.iloc[1]
    assert r.up_first_h == pytest.approx(30.0) and r.dn_first_h == pytest.approx(60.0)
    assert np.isnan(W.iloc[0].up_first_h)


def test_daily_new_york_handles_dst():
    df = _flat("2024-03-08", 5)
    D = K.daily(df)
    day = D.loc[pd.Timestamp("2024-03-10", tz=K.ET)]            # 미국 서머타임 시작: 23시간
    assert day.n == 23 * 12
    nxt = D.index[D.index.get_loc(pd.Timestamp("2024-03-10", tz=K.ET)) + 1]
    assert nxt.tz_convert("UTC") == pd.Timestamp("2024-03-11 04:00", tz="UTC")
    assert D.loc[pd.Timestamp("2024-03-09", tz=K.ET)].n == 24 * 12
    U = K.daily(df, tz="UTC")
    assert (U.n.iloc[1:-1] == 288).all()


def test_candle_shape_parts_sum_to_one_and_hand_case():
    sh = K.candle_shape([100, 100, 105], [110, 104, 106], [95, 90, 95], [108, 91, 96])
    assert np.allclose(sh[["body", "upper", "lower"]].sum(axis=1), 1.0)
    assert sh.body.iloc[0] == pytest.approx(8 / 15) and sh.upper.iloc[0] == pytest.approx(2 / 15)
    assert sh.dir.tolist() == [1.0, -1.0, -1.0]


def test_shape_cell_rules():
    sh = K.candle_shape([100, 100, 100, 100], [110, 101, 104, 110], [99.5, 90, 96, 90], [109, 91, 101, 100])
    lab = K.shape_cell(sh).tolist()
    assert lab[0] == "양봉 · 몸통 큼 · 짧음"
    assert lab[1] == "음봉 · 몸통 큼 · 짧음"
    assert lab[2] == "양봉 · 몸통 작음 · 아랫꼬리 김"          # b 1/8 · u 3/8 · d 4/8
    assert pd.isna(lab[3])                                    # 도지(방향 0) 는 제외


def test_trailing_sigma_is_causal():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.03, 100))
    s = K.trailing_sigma(r)
    r2 = r.copy()
    r2.iloc[60] = 5.0
    s2 = K.trailing_sigma(r2)
    assert s.iloc[60] == s2.iloc[60] and s.iloc[61] != s2.iloc[61]


def test_day_class_thresholds():
    c = K.day_class(np.array([0.004, 0.01, 0.02, -0.02, -0.021, 0.021]), np.full(6, 0.02))
    assert c["mixed"].tolist() == [True, False, False, False, False, False]
    assert c["up_small"].tolist() == [True, True, True, False, False, False]
    assert c["big_dn"].tolist() == [False, False, False, True, True, False]
    assert c["big_up"].tolist() == [False, False, True, False, False, True]
