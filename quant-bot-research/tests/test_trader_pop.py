"""트레이더 집단 — 세그먼트 원자 · 손절 규칙 · 룩어헤드 · 상태 집계."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import daily_atr, synth_5m
from qbot.research.trader_pop import (
    CANON, LEVELS, TFS, Archetype, ExitSpec, PopConfig, StopSpec,
    archetypes_for_level, canon_archetypes, first_hit_per_segment, population_state,
    run_trader, segment_cummax, segment_cummin, segment_cumor, signal_series,
)
from qbot.data.resample import resample_htf

STRUCT = ("ma_cross_struct", "rsi_band", "pivot_break")


def _ref_segcum(x, starts, fn):
    out = np.empty_like(x)
    b = list(starts) + [len(x)]
    for s, e in zip(b[:-1], b[1:]):
        out[s:e] = fn(x[s:e])
    return out


# ---------------------------------------------------------------- 세그먼트 원자


def test_segment_cummin_matches_reference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000) * 100
    starts = np.unique(np.concatenate([[0], rng.integers(1, 5000, 300)]))
    assert np.allclose(segment_cummin(x, starts),
                       _ref_segcum(x, starts, np.minimum.accumulate))


def test_segment_cummax_matches_reference():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000) * 100
    starts = np.unique(np.concatenate([[0], rng.integers(1, 5000, 300)]))
    assert np.allclose(segment_cummax(x, starts),
                       _ref_segcum(x, starts, np.maximum.accumulate))


def test_segment_cummax_offset_direction():
    """+off / −off 를 뒤집으면 구간 경계에서 값이 새어 나온다."""
    x = np.array([1.0, 5.0, 2.0, 0.5, 9.0])
    out = segment_cummax(x, np.array([0, 3]))
    assert np.allclose(out, [1, 5, 5, 0.5, 9])


def test_segment_cumor():
    m = np.array([0, 1, 0, 0, 0, 1, 0], bool)
    out = segment_cumor(m, np.array([0, 3]))
    assert list(out) == [False, True, True, False, False, True, True]


def test_segment_ops_handle_huge_values():
    x = np.array([1e12, -1e12, 5.0, 3.0])
    assert np.allclose(segment_cummin(x, np.array([0, 2])), [1e12, -1e12, 5.0, 3.0])


def test_first_hit_per_segment():
    m = np.array([0, 0, 1, 0, 1, 0, 0, 0], bool)
    starts = np.array([0, 3, 6])
    ends = np.array([3, 6, 8])
    assert list(first_hit_per_segment(m, starts, ends)) == [2, 4, -1]


def test_first_hit_respects_segment_end():
    m = np.array([0, 0, 0, 1], bool)
    assert first_hit_per_segment(m, np.array([0]), np.array([3]))[0] == -1


# ---------------------------------------------------------------- 신호


@pytest.mark.parametrize("kind,params", [
    ("ma_cross", (20, 60)), ("ma_side", (120,)), ("rsi50", (14,)),
    ("macd_osc", (12, 26, 9)), ("donchian", (20,)), ("rsi_band", (14, 30, 70)),
    ("rsi2_sma", (2, 10, 90, 200)), ("pivot_break", (3,)),
])
def test_signal_is_ternary(df5, kind, params):
    htf = resample_htf(df5, "1h")
    s = signal_series(htf, kind, params)
    assert len(s) == len(htf)
    assert set(np.unique(s)) <= {-1, 0, 1}


def test_unknown_signal_raises(df5):
    with pytest.raises(ValueError, match="알 수 없는 지표"):
        signal_series(resample_htf(df5, "1h"), "없는지표", ())


def test_signal_has_no_lookahead(df5):
    htf = resample_htf(df5, "1h")
    full = signal_series(htf, "pivot_break", (3,))
    cut = signal_series(htf.iloc[:len(htf) // 2], "pivot_break", (3,))
    assert np.array_equal(full[:len(cut)], cut)


# ---------------------------------------------------------------- 실행


@pytest.mark.parametrize("label", [c[0] for c in CANON])
def test_every_canon_archetype_runs(df5, label):
    lb, kind, params, st, ex, _ = next(c for c in CANON if c[0] == label)
    s = run_trader(df5, Archetype(kind, params, "1h", stop=st, exit=ex, label=lb))
    assert len(s.side5) == len(df5)
    if len(s) == 0:
        return
    assert (s.ep_end > s.ep_start).all()
    assert set(np.unique(s.ep_side)) <= {-1, 1}


def test_episodes_do_not_overlap(df5):
    s = run_trader(df5, Archetype("ma_cross", (20, 60), "1h",
                                  stop=StopSpec("ma", ma=60), exit=ExitSpec("flip")))
    assert (s.ep_start[1:] >= s.ep_end[:-1]).all()


def test_side5_matches_episodes(df5):
    s = run_trader(df5, Archetype("pivot_break", (3,), "1h",
                                  stop=StopSpec("pivot", lookback=3), exit=ExitSpec("flip")))
    live = np.zeros(len(df5), np.int8)
    for a, b, sd in zip(s.ep_start, s.ep_end, s.ep_side):
        live[a:b] = sd
    assert np.array_equal(live, s.side5)


def test_stop_never_on_wrong_side_of_entry(df5):
    """구조 손절이 진입가 반대편에 놓이면 전원 1봉 만에 털린다."""
    for lb in ("pivot_break", "rsi_band", "ma_cross_struct"):
        _, kind, params, st, ex, _ = next(c for c in CANON if c[0] == lb)
        s = run_trader(df5, Archetype(kind, params, "1h", stop=st, exit=ex, label=lb))
        if len(s) == 0:
            continue
        ok = np.isfinite(s.ep_stop0)
        bad = np.where(s.ep_side[ok] > 0,
                       s.ep_stop0[ok] > s.ep_entry[ok],
                       s.ep_stop0[ok] < s.ep_entry[ok])
        assert bad.mean() < 0.02, f"{lb}: 손절이 진입가 반대편 {bad.mean():.1%}"


def test_hold_through_path_is_stop_only(df5):
    """exit="none" 은 신호가 뒤집혀도 버틴다 — 손절이나 지평으로만 끝난다."""
    s = run_trader(df5, Archetype("macd_osc", (12, 26, 9), "1h",
                                  stop=StopSpec("atr_trail", mult=3.0, period=14),
                                  exit=ExitSpec("none")))
    assert len(s) > 0
    assert set(np.unique(s.ep_exit)) <= {0, 4}


def test_trailing_stop_is_a_ratchet(df5):
    s = run_trader(df5, Archetype("macd_osc", (12, 26, 9), "1h",
                                  stop=StopSpec("atr_trail", mult=3.0, period=14),
                                  exit=ExitSpec("none")))
    for a, b, sd in list(zip(s.ep_start, s.ep_end, s.ep_side))[:200]:
        seg = s.stop_px5[a:b]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 2:
            continue
        d = np.diff(seg)
        assert (d >= -1e-9).all() if sd > 0 else (d <= 1e-9).all()


def test_time_stop_caps_holding(df5):
    long_ = run_trader(df5, Archetype("rsi_band", (14, 30, 70), "1h",
                                      stop=StopSpec("none"), exit=ExitSpec("flip")))
    short = run_trader(df5, Archetype("rsi_band", (14, 30, 70), "1h",
                                      stop=StopSpec("none"),
                                      exit=ExitSpec("flip", time_bars=3)))
    assert (short.ep_end - short.ep_start).max() <= 3 * 12
    assert (short.ep_end - short.ep_start).max() <= (long_.ep_end - long_.ep_start).max()


def test_no_stop_means_no_stop_exits(df5):
    s = run_trader(df5, Archetype("ma_cross", (20, 60), "1h",
                                  stop=StopSpec("none"), exit=ExitSpec("flip")))
    assert (s.ep_exit != 0).all()


def test_htf_stop_is_wider_than_ltf(df5):
    """손절 재료를 5m 에서 만들면 HTF 트레이더가 즉사한다 — 재구축 중 실제로 그랬다."""
    a = run_trader(df5, Archetype("ma_cross", (20, 60), "5m",
                                  stop=StopSpec("ma", ma=60), exit=ExitSpec("flip")))
    b = run_trader(df5, Archetype("ma_cross", (20, 60), "1h",
                                  stop=StopSpec("ma", ma=60), exit=ExitSpec("flip")))
    assert np.median(b.ep_end - b.ep_start) > np.median(a.ep_end - a.ep_start)


def test_future_bars_cannot_change_past_episodes(df5):
    """설계 문서가 지정한 가장 중요한 회귀 테스트."""
    cut = len(df5) // 2
    warped = df5.copy()
    warped.iloc[cut:] *= 3.0
    arch = Archetype("pivot_break", (3,), "1h",
                     stop=StopSpec("pivot", lookback=3), exit=ExitSpec("flip"))
    a = run_trader(df5, arch)
    b = run_trader(warped, arch)
    ka = a.ep_end <= cut - 300
    kb = b.ep_end <= cut - 300
    assert ka.sum() > 20
    assert np.array_equal(a.ep_start[ka], b.ep_start[kb])
    assert np.array_equal(a.ep_end[ka], b.ep_end[kb])


def test_stop_on_entry_bar_flag(df5):
    arch = Archetype("pivot_break", (3,), "1h",
                     stop=StopSpec("pivot", lookback=3), exit=ExitSpec("flip"))
    a = run_trader(df5, arch, None, PopConfig(stop_on_entry_bar=True))
    b = run_trader(df5, arch, None, PopConfig(stop_on_entry_bar=False))
    assert (a.ep_exit == 0).mean() >= (b.ep_exit == 0).mean()


# ---------------------------------------------------------------- 카탈로그


def test_canon_has_13_and_unique_labels():
    assert len(CANON) == 13
    assert len({c[0] for c in CANON}) == 13


def test_canon_archetypes_product():
    a = canon_archetypes(("5m", "1h"))
    assert len(a) == 26
    a2 = canon_archetypes(TFS, only=("pivot_break",))
    assert len(a2) == 5 and all(x.label == "pivot_break" for x in a2)


@pytest.mark.parametrize("lvl", list(LEVELS) + ["L5"])
def test_levels_build(lvl):
    a = archetypes_for_level(lvl, ("5m", "1h"))
    assert len(a) > 0


def test_every_canon_has_a_source():
    for lb, _, _, _, _, src in CANON:
        assert len(src) > 10, lb


# ---------------------------------------------------------------- 집단 상태


def test_population_state_shapes(df5, atr5):
    archs = [Archetype(k, p, tf, stop=st, exit=ex, label=lb)
             for lb, k, p, st, ex, _ in CANON if lb in STRUCT for tf in ("5m", "1h")]
    R, cnt = population_state(df5, archs, atr5)
    n = len(df5)
    assert R.n_live.shape == (n,) and cnt.shape == (n,)
    assert (R.n_live <= len(archs)).all()
    assert (R.pain_R_long >= 0).all() and (R.gain_R_short >= 0).all()


def test_pain_cap_is_below_uncapped(df5, atr5):
    archs = canon_archetypes(("1h",), only=("pivot_break",))
    R, _ = population_state(df5, archs, atr5)
    assert (R.pain_R3_long <= R.pain_R_long + 1e-9).all()
    assert (R.pain_R3_short <= R.pain_R_short + 1e-9).all()


def test_pain_is_zero_when_nobody_is_live(df5, atr5):
    archs = canon_archetypes(("1h",), only=("rsi_band",))
    R, _ = population_state(df5, archs, atr5)
    dead = R.n_live == 0
    assert dead.any()
    assert np.allclose(R.pain_imb_R3[dead], 0.0)


def test_imbalance_sign_convention(df5, atr5):
    """양수 = 숏이 더 아프다."""
    archs = canon_archetypes(("1h",), only=("pivot_break",))
    R, _ = population_state(df5, archs, atr5)
    v = R.pain_imb_R3
    manual = (R.pain_R3_short - R.pain_R3_long) / np.maximum(R.n_live, 1)
    assert np.allclose(v, manual)


def test_empty_archetype_list(df5, atr5):
    R, cnt = population_state(df5, [], atr5)
    assert (R.n_live == 0).all() and np.allclose(cnt, 0.0)
