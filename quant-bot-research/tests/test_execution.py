"""체결 모델 — 손계산 대조 + 불변식."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synth_5m
from qbot.research.barrier import first_passage_r
from qbot.research.execution import (
    ExecConfig, limit_entry_trades, summarize_trades, timed_exit_trades,
)


def _arr(*xs):
    return [np.asarray(x, float) for x in xs]


CFG = ExecConfig(fill_window=3, tp_r=2.0, sl_r=1.0, horizon=50)


def test_no_fill_when_limit_never_touched():
    h, l, c = _arr([10.5] * 10, [10.2] * 10, [10.3] * 10)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([0.5]),
                           CFG, limit_px=np.array([9.0]))
    assert not t.filled[0] and t.outcome[0] == "미체결" and t.r_net[0] == 0.0


def test_fill_bar_is_first_touch_after_signal():
    h = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    l = np.array([10.0, 10.0, 9.0, 8.0, 10.0])
    c = np.full(5, 10.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           CFG, limit_px=np.array([9.5]))
    assert t.filled[0] and t.fill_bar[0] == 2


def test_signal_bar_itself_cannot_fill():
    """지정가는 신호봉 **종가에** 걸리므로 그 봉에서는 체결될 수 없다."""
    h = np.array([12.0, 10.0, 10.0])
    l = np.array([8.0, 10.0, 10.0])
    c = np.full(3, 10.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           CFG, limit_px=np.array([9.0]))
    assert not t.filled[0]


def test_fill_window_expires():
    n = 20
    h = np.full(n, 10.0)
    l = np.full(n, 10.0)
    l[10] = 8.0
    c = np.full(n, 10.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           ExecConfig(fill_window=3, horizon=5), limit_px=np.array([9.0]))
    assert not t.filled[0]
    t2 = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                            ExecConfig(fill_window=15, horizon=5), limit_px=np.array([9.0]))
    assert t2.filled[0] and t2.fill_bar[0] == 10


def test_win_pays_maker_exit():
    h = np.array([10.0, 10.0, 10.0, 13.0, 10.0])
    l = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    c = np.full(5, 10.0)
    cfg = ExecConfig(fill_window=3, tp_r=2.0, sl_r=1.0, horizon=50,
                     entry_bps=2.0, tp_bps=2.0, sl_bps=6.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           cfg, limit_px=np.array([10.0]))
    assert t.outcome[0] == "승" and t.r_gross[0] == 2.0
    assert t.fee_r[0] == pytest.approx((2.0 + 2.0) / (1.0 / 10.0 * 1e4))


def test_loss_pays_taker_exit():
    h = np.array([10.0, 10.0, 10.0, 10.0])
    l = np.array([10.0, 9.9, 8.5, 10.0])
    c = np.full(4, 10.0)
    cfg = ExecConfig(fill_window=3, tp_r=2.0, sl_r=1.0, horizon=50,
                     entry_bps=2.0, tp_bps=2.0, sl_bps=6.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           cfg, limit_px=np.array([10.0]))
    assert t.outcome[0] == "패" and t.r_gross[0] == -1.0
    assert t.fee_r[0] == pytest.approx((2.0 + 6.0) / 1000.0)


def test_same_bar_is_ambiguous():
    h = np.array([10.0, 13.0, 10.0])
    l = np.array([10.0, 8.0, 10.0])
    c = np.full(3, 10.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           CFG, limit_px=np.array([10.0]))
    assert t.outcome[0] == "모호" and t.r_gross[0] == 0.0 and t.fee_r[0] > 0


def test_unresolved_pays_taker_and_is_zero_gross():
    n = 30
    h = np.full(n, 10.01)
    l = np.full(n, 9.99)
    c = np.full(n, 10.0)
    t = limit_entry_trades(h, l, c, np.array([0]), np.array([1]), np.array([1.0]),
                           ExecConfig(fill_window=3, tp_r=2.0, sl_r=1.0, horizon=10),
                           limit_px=np.array([10.0]))
    assert t.outcome[0] == "미해결" and t.r_gross[0] == 0.0 and t.r_net[0] < 0


def test_short_mirrors_long():
    rng = np.random.default_rng(5)
    n = 8000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h, l = c * 1.002, c * 0.998
    bars = np.arange(50, 6000, 61)
    risk = c[bars] * 0.01
    cfg = ExecConfig(fill_window=6, tp_r=1.0, sl_r=1.0, horizon=400)
    lo = limit_entry_trades(h, l, c, bars, np.ones(len(bars), int), risk, cfg)
    sh = limit_entry_trades(h, l, c, bars, -np.ones(len(bars), int), risk, cfg)
    both = (lo.filled & sh.filled & (lo.fill_bar == sh.fill_bar)
            & lo.outcome.isin(["승", "패"]) & sh.outcome.isin(["승", "패"]))
    assert both.sum() > 20
    assert (lo.r_gross[both].to_numpy() == -sh.r_gross[both].to_numpy()).all()


def test_matches_barrier_racer_when_fill_is_immediate():
    rng = np.random.default_rng(9)
    n = 20_000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0025, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.0015, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.0015, n)))
    bars = np.arange(100, 15_000, 53)
    side = rng.choice([-1, 1], len(bars))
    risk = c[bars] * 0.02
    cfg = ExecConfig(fill_window=400, tp_r=2.0, sl_r=1.0, horizon=1000)
    t = limit_entry_trades(h, l, c, bars, side, risk, cfg)
    f = t[t.filled]
    mine = first_passage_r(h, l, f.fill_bar.to_numpy(), f.entry.to_numpy(),
                           risk[t.filled.to_numpy()], f.side.to_numpy(),
                           tp_r=2.0, sl_r=1.0, horizon=1000, skip_entry_bar=False)
    assert np.allclose(mine, f.r_gross.to_numpy())


def test_fee_r_shrinks_as_risk_grows():
    rng = np.random.default_rng(11)
    n = 12_000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h, l = c * 1.003, c * 0.997
    bars = np.arange(50, 9000, 47)
    side = np.ones(len(bars), int)
    cfg = ExecConfig(fill_window=8, horizon=600)
    tight = limit_entry_trades(h, l, c, bars, side, c[bars] * 0.003, cfg)
    wide = limit_entry_trades(h, l, c, bars, side, c[bars] * 0.05, cfg)
    assert tight[tight.filled].fee_r.mean() > wide[wide.filled].fee_r.mean() * 5


def test_summarize_reports_fill_rate():
    h = np.full(40, 10.0)
    l = np.full(40, 10.0)
    c = np.full(40, 10.0)
    bars = np.arange(0, 20, 5)
    t = limit_entry_trades(h, l, c, bars, np.ones(len(bars), int),
                           np.full(len(bars), 1.0), CFG, limit_px=np.full(len(bars), 9.0))
    s = summarize_trades(t)
    assert s["체결률"] == 0.0 and s["n"] == len(bars)


def test_summarize_counts_outcomes():
    df = synth_5m(20_000, seed=12)
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    bars = np.arange(100, 15_000, 57)
    t = limit_entry_trades(h, l, c, bars, np.ones(len(bars), int), c[bars] * 0.01,
                           ExecConfig(fill_window=12, horizon=800))
    s = summarize_trades(t)
    assert s["승"] + s["패"] + s["모호"] + s["미해결"] == s["체결"]
    assert 0 <= s["승률"] <= 1


def test_timed_exit_hand_calc():
    c = np.array([100.0, 100.0, 100.0, 101.0, 102.0])
    h, l = c + 0.5, c - 0.5
    cfg = ExecConfig(fill_window=3, entry_bps=2.0, sl_bps=6.0)
    t = timed_exit_trades(h, l, c, np.array([0]), np.array([1]), hold=3, cfg=cfg)
    assert t.filled[0] and t.fill_bar[0] == 1
    assert t.gross_bps[0] == pytest.approx((102.0 - 100.0) / 100.0 * 1e4)
    assert t.fee_bps[0] == pytest.approx(8.0)
    assert t.net_bps[0] == pytest.approx(200.0 - 8.0)


def test_timed_exit_short_flips_sign():
    c = np.array([100.0, 100.0, 100.0, 99.0, 98.0])
    h, l = c + 0.5, c - 0.5
    t = timed_exit_trades(h, l, c, np.array([0]), np.array([-1]), hold=3,
                          cfg=ExecConfig(fill_window=3))
    assert t.filled[0]
    assert t.gross_bps[0] == pytest.approx((100.0 - 98.0) / 100.0 * 1e4)


def test_timed_exit_r_units_use_given_risk():
    c = np.full(10, 100.0)
    c[5:] = 101.0
    h, l = c + 0.5, c - 0.5
    t = timed_exit_trades(h, l, c, np.array([0]), np.array([1]), hold=6,
                          cfg=ExecConfig(fill_window=3), risk=np.array([1.0]))
    assert t.risk_bps[0] == pytest.approx(100.0)
    assert t.r_gross[0] == pytest.approx(t.gross_bps[0] / 100.0)


def test_timed_exit_unfilled_is_zero():
    c = np.full(20, 100.0)
    h, l = c + 0.1, c - 0.1
    t = timed_exit_trades(h, l, c, np.array([0]), np.array([1]), hold=3,
                          cfg=ExecConfig(fill_window=3), limit_px=np.array([50.0]))
    assert not t.filled[0] and t.net_bps[0] == 0.0 and t.fee_bps[0] == 0.0
