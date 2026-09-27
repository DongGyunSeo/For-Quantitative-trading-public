"""리샘플 — 확정 시각 규약이 핵심."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.data.resample import TF_ALIASES, build_ref_map, resample_htf


@pytest.mark.parametrize("tf", ["15m", "30m", "1h", "2h", "4h", "1d"])
def test_all_tfs_resample(df5, tf):
    h = resample_htf(df5, tf)
    assert len(h) > 0
    assert {"open", "high", "low", "close", "volume", "close_ts", "n_ltf"} <= set(h.columns)


def test_day_offset_does_not_break():
    """pd.Timedelta(to_offset("1D")) 가 Day 를 거부해 예전에 여기서 터졌다."""
    h = resample_htf(synth_5m(6000), "1d")
    assert len(h) > 0 and int(h["n_ltf"].median()) == 288


def test_aliases_cover_project_tfs():
    for tf in ("5m", "15m", "30m", "1h", "4h", "1d"):
        assert tf in TF_ALIASES


def test_close_ts_is_next_open(df5):
    h = resample_htf(df5, "1h")
    assert (h["close_ts"].to_numpy()[:-1] == h.index.to_numpy()[1:]).all()


def test_ohlc_aggregation_is_correct(df5):
    h = resample_htf(df5, "1h")
    row = h.iloc[5]
    seg = df5[(df5.index >= h.index[5]) & (df5.index < h["close_ts"].iloc[5])]
    assert row["high"] == seg["high"].max()
    assert row["low"] == seg["low"].min()
    assert row["open"] == seg["open"].iloc[0]
    assert row["close"] == seg["close"].iloc[-1]


def test_ref_map_never_references_unconfirmed(df5):
    """참조된 HTF 봉의 확정 시각은 반드시 그 5m 봉의 시각 이하여야 한다."""
    h = resample_htf(df5, "4h")
    ref = build_ref_map(df5.index, h)
    ok = ref >= 0
    ct = h["close_ts"].to_numpy()[ref[ok]]
    assert (ct <= df5.index.to_numpy()[ok]).all()


def test_ref_map_is_the_latest_confirmed(df5):
    h = resample_htf(df5, "1h")
    ref = build_ref_map(df5.index, h)
    ok = ref >= 0
    nxt = ref[ok] + 1
    inb = nxt < len(h)
    ct = h["close_ts"].to_numpy()
    assert (ct[nxt[inb]] > df5.index.to_numpy()[ok][inb]).all()


def test_first_bars_have_no_reference(df5):
    ref = build_ref_map(df5.index, resample_htf(df5, "1d"))
    assert ref[0] == -1


def test_short_htf_bars_dropped():
    df = synth_5m(1000)
    df = df.drop(df.index[100:200])          # 1h 봉 하나가 통째로 얇아진다
    h = resample_htf(df, "1h")
    assert (h["n_ltf"] >= 12).all()
