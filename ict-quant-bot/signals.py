# signals.py  v3.0
# ─────────────────────────────────────────────────────────────────
# 레벨별 진입/청산 로직 분기
#   Group A (asia_high/low, london_high/low, idm_high/low, pdl, pdh):
#     sweep → displacement → FVG 탐색
#       1) FVG 없음  → no entry
#       2) FVG 1개+  → 가장 넓은 FVG edge에 100%
#     SL = sweep_extreme
#     STANDBY timeout: 30봉 (2.5시간)
#
#   Group B (pwh, pwl, breakerblock, eql, eqh):
#     sweep → displacement → 5m choch/bos → OTE FVG 탐색 or OB fallback
#     SL = 1차 5m OB(30%) / 2차 sweep_extreme(70%, 메인 SL)
#     STANDBY timeout: 240봉 (20시간)
#
# HTF size 패널티:
#   Group A: size × (1 - 4H_mismatch×0.2 - 1H_mismatch×0.3)
#   Group B: size × (1 - 4H_mismatch×0.3 - 1H_mismatch×0.2)
#
# size_multiplier:
#   BASE=1.0 × HTF × PD position(0.6~1.4) × PWH/PWL bonus(1.3) × HUNT_VOL(1.0 or 0.8)
#   minimum size = 0.35
#
# Displacement 조건:
#   net_move >= 1.5 × ATR  AND  연속 1~3봉 body합 >= 1.2 × ATR
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from collections import deque

Bar = Union[dict, pd.Series]

# ──────────────────────────────────────────────────────────────────
# 레벨 그룹 분류
# ──────────────────────────────────────────────────────────────────
LEVEL_GROUP_A = {"asia_high", "asia_low", "london_high", "london_low", "pdl", "pdh",
                 "idm_high", "idm_low"}
LEVEL_GROUP_B = {"pwh", "pwl", "breakerblock", "eql", "eqh"}

# HTF 패널티 계수 (1H 미스매치, 4H 미스매치)
HTF_PENALTY = {
    "asia_high":    (0.20, 0.10),
    "asia_low":     (0.20, 0.10),
    "london_high":  (0.20, 0.10),
    "london_low":   (0.20, 0.10),
    "idm_high":     (0.20, 0.10),
    "idm_low":      (0.20, 0.10),
    "pdl":          (0.15, 0.15),
    "pdh":          (0.15, 0.15),
    "pwh":          (0.10, 0.20),
    "pwl":          (0.10, 0.20),
    "breakerblock": (0.10, 0.20),
    "eql":          (0.10, 0.20),
    "eqh":          (0.10, 0.20),
}
HTF_PENALTY_DEFAULT = (0.10, 0.15)  # fallback

# PWL/PWH 보너스
PW_BONUS = 1.3
# HUNT_VOL 기준 (볼륨 필터: 평균 부품 시 size 패널티)
HUNT_VOL_THRESHOLD = 1.3   # 스윗 평균 볼륨 × 1.3 이상 필요
HUNT_VOL_PENALTY   = 0.8   # 미충족 시 size ×0.8
HUNT_VOL_LOOKBACK  = 120
# 최소 size
SIZE_MIN = 0.35

# 세션 정보 (ML 피처용 — 차단하지 않음)
def _session_features(ts: pd.Timestamp) -> dict:
    """시간/요일 피처 계산 (차단 없음, ML이 판단)."""
    import numpy as np
    hour = ts.hour
    return {
        "hour_sin": round(np.sin(2 * np.pi * hour / 24), 4),
        "hour_cos": round(np.cos(2 * np.pi * hour / 24), 4),
        "dow": ts.weekday(),
    }

# OTE 피보나치
OTE_FIB_LOW   = 0.618
OTE_FIB_HIGH  = 0.790
OTE_FIB_IDEAL = 0.705

# Sweep 하드 차단
SWEEP_WICK_MIN = 0.15

# STANDBY timeout 레벨별 (5m봉 기준)
# Group A (asia_h/l, london_h/l, ny_h/l, pdl, pdh): 30봉 = 2.5시간
# Group B (pwh, pwl, eql, eqh, bb): 240봉 = 20시간
STANDBY_TIMEOUT_GROUP_A = 30   # 2.5시간 (5m봉 기준)
STANDBY_TIMEOUT_GROUP_B = 240  # 20시간 (5m봉 기준)

# 최소 Displacement
MIN_DISP_PCT         = 0.003
MIN_DISP_PCT_ASIA_HL = 0.004
MIN_DISP_ABS         = 0.001   # displacement 최소 0.1% (절대 필터)
WICK_DISP_FIB_PENALTY_MAX = 0.5  # 이 fib 위치 미만이면 size 패널티
STRONG_DISP_LV_TO_DISP_PCT = 0.0025  # level→disp 거리 0.25% 이상 = strong disp
                                      # → OTE 0.5 체크 스킵, 반대봉 2봉 유지

# ADX 필터
ADX_PERIOD    = 14
ADX_THRESHOLD = 25.0
ASIA_HL_ADX_FILTER = True

# 거래량 필터
VOL_LOOKBACK = 120
VOL_MULT     = 2.0
RECLAIM_VOL_CONFIRM_BARS = 2

# SL
SL_MIN_PCT    = 0.008
SL_ATR_LOOKBACK = 20


# ──────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────

def _get(bar: Bar, k: str, default=None):
    if isinstance(bar, dict):
        return bar.get(k, default)
    return bar.get(k, default)

def _as_ts(x) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp): return x
    return pd.to_datetime(x)

def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0


# ──────────────────────────────────────────────────────────────────
# 볼륨 유틸
# ──────────────────────────────────────────────────────────────────

def _vol_median(df_5m: pd.DataFrame, ref_idx: int,
                lookback: int = VOL_LOOKBACK) -> float:
    if "volume" not in df_5m.columns: return 0.0
    vols = df_5m["volume"].to_numpy(dtype=float)
    start = max(0, ref_idx - lookback)
    if start >= ref_idx: return 0.0
    return float(pd.Series(vols[start:ref_idx]).median())


def _vol_is_valid(df_5m: pd.DataFrame, sweep_bar_idx: int,
                  vol_mult: float = VOL_MULT,
                  lookback: int = VOL_LOOKBACK) -> bool:
    if "volume" not in df_5m.columns: return True
    vols = df_5m["volume"].to_numpy(dtype=float)
    n = len(vols)
    if n < lookback + 2: return True
    baseline = _vol_median(df_5m, sweep_bar_idx, lookback)
    if baseline <= 0: return True
    end = min(n, sweep_bar_idx + 2)
    sweep_avg = float(vols[sweep_bar_idx:end].mean())
    return sweep_avg >= baseline * vol_mult


def _vol_reclaim_ok(df_5m: pd.DataFrame, reclaim_bar_idx: int,
                    confirm_bars: int = RECLAIM_VOL_CONFIRM_BARS,
                    vol_mult: float = VOL_MULT,
                    lookback: int = VOL_LOOKBACK) -> bool:
    if "volume" not in df_5m.columns: return True
    vols = df_5m["volume"].to_numpy(dtype=float)
    n = len(vols)
    if n < lookback + 2 or reclaim_bar_idx < lookback: return True
    baseline = _vol_median(df_5m, reclaim_bar_idx, lookback)
    if baseline <= 0: return True
    end = min(n, reclaim_bar_idx + 1 + confirm_bars)
    confirm_avg = float(vols[reclaim_bar_idx:end].mean())
    return confirm_avg >= baseline * vol_mult


def _hunt_vol_ok(df_5m: pd.DataFrame, sweep_bar_idx: int,
                 reclaim_bar_idx: int,
                 threshold: float = HUNT_VOL_THRESHOLD,
                 lookback: int = HUNT_VOL_LOOKBACK) -> bool:
    """sweep 봉 ~ reclaim 봉까지 평균 볼륨이 baseline × threshold 이상이면 True"""
    if "volume" not in df_5m.columns: return True
    vols = df_5m["volume"].to_numpy(dtype=float)
    n = len(vols)
    baseline = _vol_median(df_5m, sweep_bar_idx, lookback)
    if baseline <= 0: return True
    start = max(0, sweep_bar_idx)
    end   = min(n, reclaim_bar_idx + 1)
    if start >= end: return True
    avg_vol = float(vols[start:end].mean())
    return avg_vol >= baseline * threshold


# ──────────────────────────────────────────────────────────────────
# ADX
# ──────────────────────────────────────────────────────────────────

def _calc_adx(df_1h: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    if df_1h is None or len(df_1h) < period * 2 + 1: return 0.0
    hi = df_1h["high"].to_numpy(dtype=float)
    lo = df_1h["low"].to_numpy(dtype=float)
    cl = df_1h["close"].to_numpy(dtype=float)
    n = len(hi)
    tr_arr, pdm_arr, ndm_arr = [], [], []
    for i in range(1, n):
        tr  = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        pdm = max(hi[i]-hi[i-1], 0.0) if (hi[i]-hi[i-1]) > (lo[i-1]-lo[i]) else 0.0
        ndm = max(lo[i-1]-lo[i], 0.0) if (lo[i-1]-lo[i]) > (hi[i]-hi[i-1]) else 0.0
        tr_arr.append(tr); pdm_arr.append(pdm); ndm_arr.append(ndm)
    def _wilder(arr, p):
        res = [0.0]*len(arr); res[p-1] = sum(arr[:p])
        for i in range(p, len(arr)): res[i] = res[i-1] - res[i-1]/p + arr[i]
        return res
    atr14 = _wilder(tr_arr, period)
    pdm14 = _wilder(pdm_arr, period)
    ndm14 = _wilder(ndm_arr, period)
    dx_arr = []
    for i in range(period-1, len(atr14)):
        if atr14[i] == 0: continue
        pdi = 100*pdm14[i]/atr14[i]; ndi = 100*ndm14[i]/atr14[i]
        denom = pdi+ndi
        if denom == 0: continue
        dx_arr.append(100*abs(pdi-ndi)/denom)
    if len(dx_arr) < period: return 0.0
    adx = sum(dx_arr[-period:]) / period
    for i in range(len(dx_arr)-period, len(dx_arr)-1):
        adx = (adx*(period-1) + dx_arr[i+1]) / period
    return float(adx)


# ──────────────────────────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────────────────────────

def _calc_atr_5m(df_5m: pd.DataFrame, lookback: int = SL_ATR_LOOKBACK) -> float:
    if df_5m is None or len(df_5m) < lookback + 1: return 0.0
    tail = df_5m.iloc[-(lookback+1):]
    hi = tail["high"].to_numpy(dtype=float)
    lo = tail["low"].to_numpy(dtype=float)
    cl = tail["close"].to_numpy(dtype=float)
    tr = [max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])) for i in range(1, len(hi))]
    return float(sum(tr)/len(tr)) if tr else 0.0


def _calc_atr_wilder(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder 지수평활 ATR. 단순평균보다 displacement 봉 1개에 덜 민감."""
    if df is None or len(df) < period + 1:
        return 0.0
    tail = df.iloc[-(period * 3 + 1):] if len(df) > period * 3 + 1 else df
    hi = tail["high"].to_numpy(dtype=float)
    lo = tail["low"].to_numpy(dtype=float)
    cl = tail["close"].to_numpy(dtype=float)
    if len(hi) < period + 1:
        return 0.0
    tr = np.array([max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
                   for i in range(1, len(hi))], dtype=float)
    # Wilder smoothing: 첫 ATR = 단순평균, 이후 (prev*(p-1)+tr)/p
    atr = float(tr[:period].mean())
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


def _calc_1h_ma_atr(df_1h: pd.DataFrame):
    """1h MA(5/20/60)와 1h Wilder ATR(14) 반환. 부족하면 None."""
    if df_1h is None or len(df_1h) < 60:
        return None
    cl = df_1h["close"].to_numpy(dtype=float)
    ma5 = float(cl[-5:].mean())
    ma20 = float(cl[-20:].mean())
    ma60 = float(cl[-60:].mean())
    atr_1h = _calc_atr_wilder(df_1h, 14)
    return {"ma5": ma5, "ma20": ma20, "ma60": ma60, "atr_1h": atr_1h}


def _bar_disp_quality(o, h, l, c, direction):
    """
    단일 봉의 방향성 displacement 품질 (0~1, 클수록 건강).
    long : 아랫꼬리 50%를 body로 인정 (sweep 흔적=긍정 신호).
    short: 윗꼬리 50%를 body로 인정.
    body / (body + quality_wick).
    """
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if direction == "long":
        quality_wick = lower_wick * 0.5 + upper_wick
    else:
        quality_wick = upper_wick * 0.5 + lower_wick
    denom = body + quality_wick
    if denom <= 1e-12:
        return 0.0
    return float(body / denom)


def _calc_disp_quality_15m(df_15m, sweep_time, disp_end_time, direction, min_bars=2):
    """
    sweep_time ~ disp_end_time 구간의 15m 봉들의 평균 displacement 품질.
    body 가중 평균 (큰 봉이 displacement를 더 대표).
    봉 수 < min_bars 이면 None (NaN sentinel).
    """
    if df_15m is None or len(df_15m) == 0 or sweep_time is None:
        return None
    if "time" not in df_15m.columns:
        return None
    t = pd.to_datetime(df_15m["time"])
    mask = (t >= sweep_time)
    if disp_end_time is not None:
        mask &= (t <= disp_end_time)
    seg = df_15m[mask]
    if len(seg) < min_bars:
        return None
    qs, ws = [], []
    for _, r in seg.iterrows():
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        q = _bar_disp_quality(o, h, l, c, direction)
        w = abs(c - o) + 1e-9  # body 가중
        qs.append(q); ws.append(w)
    qs = np.array(qs); ws = np.array(ws)
    return float(np.average(qs, weights=ws))




# ──────────────────────────────────────────────────────────────────
# Liquidity 강도 피처 (ML용)
# ──────────────────────────────────────────────────────────────────

def _calc_liquidity_features(
    lv: dict,
    sweep_depth_atr: float,
    sweep_ratio: float,
    all_levels: List[dict],
    atr: float,
    current_price: float,
    current_bar_idx: int,
    sweep_history: Optional['_SweepHistory'] = None,
    current_time: Optional[pd.Timestamp] = None,
) -> dict:
    """
    sweep된 liquidity level의 강도를 연속형 피처로 계산.

    반환 피처:
      liquidity_strength_score: 종합 강도 (0~1)
      level_size_atr:    레벨 폭 / ATR
      level_age_bars:    레벨 생성 후 경과 5m 봉 수
      is_fresh_level:    최근 50봉(~4시간) 이내 생성 여부
      recent_sweep_count: 이 레벨의 누적 sweep 횟수
      multi_sweep_flag:  2회 이상 sweep 이력 여부
      touches:           레벨에 닿은 횟수
      cluster_density:   ±2×ATR 내 다른 레벨 수
      opp_liq_dist_atr:  반대편 가장 가까운 레벨 거리/ATR
    """
    zlow  = float(lv.get("zone_low", 0))
    zhigh = float(lv.get("zone_high", 0))
    zone_mid = (zlow + zhigh) / 2 if (zlow + zhigh) > 0 else current_price

    # ── level_size_atr: 레벨 폭 / ATR

    # ── level_age_bars: ts_start → 현재 시각 차이 (5m봉 단위)
    level_age_bars = 0
    ts_start = lv.get("ts_start")
    if ts_start is not None and current_time is not None:
        try:
            _ts = pd.Timestamp(ts_start)
            if _ts.tzinfo is None:
                _ts = _ts.tz_localize("UTC")
            _ct = current_time
            if _ct.tzinfo is None:
                _ct = _ct.tz_localize("UTC")
            _delta_minutes = (_ct - _ts).total_seconds() / 60.0
            level_age_bars = max(0, int(_delta_minutes / 5))  # 5m봉 단위
        except Exception:
            level_age_bars = 0

    # ── is_fresh_level: 최근 50봉(~4시간) 이내 생성

    # ── recent_sweep_count: _SweepHistory에서 이 레벨의 sweep 이력 횟수 (30봉 window)
    recent_sweep_count = 0
    if sweep_history is not None:
        recs = sweep_history.get(lv)
        if current_time is not None:
            _window_minutes = 30 * 5  # 30봉 × 5분
            recs = [r for r in recs
                    if hasattr(r, 'timestamp') and r.timestamp is not None
                    and (current_time - pd.Timestamp(r.timestamp)).total_seconds() <= _window_minutes * 60]
        recent_sweep_count = len(recs)

    # ── multi_sweep_flag

    # ── touches

    # ── cluster_density: 현재 레벨 ±2×ATR 내 다른 레벨 수
    cluster_count = 0
    if atr > 0 and all_levels:
        cluster_range = 2.0 * atr
        for other_lv in all_levels:
            other_mid = (float(other_lv.get("zone_low", 0)) + float(other_lv.get("zone_high", 0))) / 2
            if abs(other_mid - zone_mid) < 1e-6:
                continue
            if abs(other_mid - zone_mid) <= cluster_range:
                cluster_count += 1

    # ── opp_liq_dist_atr: 반대편 가장 가까운 레벨까지 거리/ATR
    opp_dist_atr = 0.0
    if atr > 0 and all_levels:
        above = [lv2 for lv2 in all_levels
                 if (float(lv2.get("zone_low",0))+float(lv2.get("zone_high",0)))/2 > zone_mid + atr*0.5]
        below = [lv2 for lv2 in all_levels
                 if (float(lv2.get("zone_low",0))+float(lv2.get("zone_high",0)))/2 < zone_mid - atr*0.5]
        if above:
            nearest_above = min(above, key=lambda x: (float(x["zone_low"])+float(x["zone_high"]))/2)
            opp_dist_atr = ((float(nearest_above["zone_low"])+float(nearest_above["zone_high"]))/2 - zone_mid) / atr
        if below:
            nearest_below = max(below, key=lambda x: (float(x["zone_low"])+float(x["zone_high"]))/2)
            below_dist = (zone_mid - (float(nearest_below["zone_low"])+float(nearest_below["zone_high"]))/2) / atr
            if opp_dist_atr == 0 or below_dist < opp_dist_atr:
                opp_dist_atr = below_dist

    # ── liquidity_strength_score (종합 0.0~1.0)
    import math
    _depth_norm     = min(sweep_depth_atr / 2.0, 1.0)
    _sweep_cnt_norm = min(recent_sweep_count / 3.0, 1.0)
    _age_norm       = min(level_age_bars / 500.0, 1.0)  # 500봉(~42시간) 이상 = 1.0
    _ratio_norm     = min(sweep_ratio / 0.5, 1.0)

    liquidity_strength = round(
        _depth_norm * 0.4 + _sweep_cnt_norm * 0.25 + _age_norm * 0.2 + _ratio_norm * 0.15,
        3
    )

    # level_age_bars: log 변환 (스케일 압축)
    _level_age_log = round(math.log1p(level_age_bars), 3)

    return {
        "liquidity_strength_score": liquidity_strength,
        "level_age_bars":          _level_age_log,
        "recent_sweep_count":      recent_sweep_count,
        "opp_liq_dist_atr":        round(opp_dist_atr, 3),  # Oracle 피처
    }


def _calc_timeout_adjust(direction: str, htf_1h: str, htf_4h: str,
                          df_5m: Optional[pd.DataFrame] = None,
                          atr_avg_lookback: int = 120) -> tuple:
    """
    HTF bias + 변동성 기반 timeout 조정값 반환.
    반환: (standby_adj, armed_adj)

    HTF bias:
      둘 다 같은 방향 → +6/+4 (더 오래 기다림)
      둘 다 반대/unknown → -4/-2 (빨리 포기)

    vol_ratio = current_atr / avg_atr:
      >1.5 → -6/-4  (고변동 = 빨리 결정)
      >1.2 → -4/-2
      >0.8 → 0/0
      >0.6 → +4/+2  (저변동 = 더 기다림)
      else  → +6/+4
    """
    sb_adj, ar_adj = 0, 0

    # HTF bias
    htf_1h_aligned = (direction == "long" and htf_1h == "bull") or \
                     (direction == "short" and htf_1h == "bear")
    htf_4h_aligned = (direction == "long" and htf_4h == "bull") or \
                     (direction == "short" and htf_4h == "bear")
    htf_1h_against = (direction == "long" and htf_1h == "bear") or \
                     (direction == "short" and htf_1h == "bull")
    htf_4h_against = (direction == "long" and htf_4h == "bear") or \
                     (direction == "short" and htf_4h == "bull")

    if htf_1h_aligned and htf_4h_aligned:
        sb_adj += 6; ar_adj += 4
    elif (htf_1h_against or htf_1h == "unknown") and (htf_4h_against or htf_4h == "unknown"):
        sb_adj -= 4; ar_adj -= 2

    # vol_ratio
    if df_5m is not None and len(df_5m) > atr_avg_lookback + 20:
        cur_atr = _calc_atr_5m(df_5m, lookback=20)
        avg_atr = _calc_atr_5m(df_5m, lookback=atr_avg_lookback)
        if avg_atr > 0:
            vol_ratio = cur_atr / avg_atr
            if vol_ratio > 1.5:
                sb_adj -= 6; ar_adj -= 4
            elif vol_ratio > 1.2:
                sb_adj -= 4; ar_adj -= 2
            elif vol_ratio > 0.8:
                pass  # 0/0
            elif vol_ratio > 0.6:
                sb_adj += 4; ar_adj += 2
            else:
                sb_adj += 6; ar_adj += 4

    return sb_adj, ar_adj


# ──────────────────────────────────────────────────────────────────
# Sweep 품질
# ──────────────────────────────────────────────────────────────────

def _sweep_wick_ratio(o: float, h: float, l: float, c: float, direction: str) -> float:
    rng = h - l
    if rng < 1e-9: return 0.0
    if direction == "long": return (min(o, c) - l) / rng
    return (h - max(o, c)) / rng

def _sweep_is_valid(ratio: float) -> bool:
    return ratio >= SWEEP_WICK_MIN


# ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════
# FVG Scoring — fib WR / LVN density / size composite
# ══════════════════════════════════════════════════════════

def fib_wr_score(fib: float) -> float:
    """Empirical WR multiplier — linear interpolation from 23K trades.
    1.0 = baseline WR, >1.0 = 좋은 fib 위치, <1.0 = 나쁜 fib 위치.
    """
    _FIBS   = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375,
               0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775,
               0.825, 0.875, 0.925, 0.975, 1.025]
    _SCORES = [0.948, 0.662, 0.810, 0.854, 0.855, 1.063, 1.041, 1.033,
               1.082, 1.196, 1.062, 1.119, 0.995, 1.092, 0.929, 0.802,
               0.928, 0.746, 0.559, 0.805, 0.606]
    return float(np.clip(np.interp(fib, _FIBS, _SCORES), 0.55, 1.22))


def lvn_score(density_percentile: float) -> float:
    """LVN quality score. density가 낮을수록(LVN) 점수 높음.
    density_percentile: 0~100 (0 = 가장 Low Volume)
    """
    score = 9 * np.exp(-0.045 * density_percentile) + 3.0
    return float(np.clip(score, 3.0, 9.0))


def size_score(size_percentile: float) -> float:
    """FVG size score. 크기가 클수록 점수 높음.
    size_percentile: 0~100 (현재 displacement 구간 내 FVG 크기 백분위)
    """
    score = 2.0 + 4.0 * (size_percentile / 100) ** 1.45
    return float(np.clip(score, 2.0, 6.0))


def fvg_composite_score(fib: float, density_percentile: float,
                        size_percentile: float) -> float:
    """FVG 종합 점수. 곱셈 방식 — 세 조건이 모두 갖춰져야 높은 점수.
    비중 fib:lvn:size = 4:3:2. 범위 1~60."""
    fib_norm = (fib_wr_score(fib) - 0.55) / (1.22 - 0.55)
    lvn_norm = (lvn_score(density_percentile) - 3.0) / 6.0
    size_norm = (size_score(size_percentile) - 2.0) / 4.0
    return (1 + fib_norm * 4) * (1 + lvn_norm * 3) * (1 + size_norm * 2)


def fib_tail_score(fib: float) -> float:
    """Empirical TAIL(8R+) multiplier — 29,100 trades.
    1.0 = baseline P(8R+), >1.0 = 꼬리(러너) 잘나는 fib(깊은 reclaim/OTE).
    fib_wr_score(WR용)와 병존: WR=entry filter(SL회피), tail=trail/사이징.
    fib 깊을수록(→1.0) 단조 증가 — 컨벡시티 엔진과 정렬.
    검증(verify_fib_tail): Q5/Q1 P8배수 2.67 (WR곡선 1.24 대비 우수)."""
    _FIBS   = [0.025, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.025]
    _SCORES = [0.545, 0.545, 0.493, 0.773, 0.928, 0.979, 1.228, 1.414, 1.635, 1.727, 1.766, 1.766]
    return float(np.clip(np.interp(fib, _FIBS, _SCORES), 0.49, 1.77))


def fvg_composite_score_tail(fib: float, density_percentile: float,
                             size_percentile: float) -> float:
    """[폐기 — 미사용] 꼬리 우대 composite. _best_fvg_fib(FVG후보 fib)을 쓰면
    sweep_to_disp_fib(검증 엣지)과 역상관이라 composite가 거꾸로 나옴(가지7 검증).
    fib_tail_score(sweep_to_disp_fib 직접 적용)로 대체. density 실주입+sweep_fib
    기반 재설계 전까지 호출 안 함. 보존만."""
    fib_norm = (fib_tail_score(fib) - 0.49) / (1.77 - 0.49)
    lvn_norm = (lvn_score(density_percentile) - 3.0) / 6.0
    size_norm = (size_score(size_percentile) - 2.0) / 4.0
    return (1 + fib_norm * 4) * (1 + lvn_norm * 3) * (1 + size_norm * 2)


# OTE 구간 — v2.0: sweep_extreme ~ disp_extreme 기준
# ──────────────────────────────────────────────────────────────────

def _calc_ote_zone(direction: str,
                   sweep_extreme: float,
                   disp_extreme: float) -> Optional[Dict[str, float]]:
    """
    v2.0: sweep_extreme ~ disp_extreme 기준 61.8~79% 피보나치 되돌림.
    long:  sweep_extreme(저점) ~ disp_extreme(고점)
    short: sweep_extreme(고점) ~ disp_extreme(저점)
    """
    rng = abs(disp_extreme - sweep_extreme)
    if rng < 1e-9: return None
    if direction == "long":
        return {
            "low":   disp_extreme - rng * OTE_FIB_HIGH,
            "high":  disp_extreme - rng * OTE_FIB_LOW,
            "ideal": disp_extreme - rng * OTE_FIB_IDEAL,
        }
    return {
        "low":   disp_extreme + rng * OTE_FIB_LOW,
        "high":  disp_extreme + rng * OTE_FIB_HIGH,
        "ideal": disp_extreme + rng * OTE_FIB_IDEAL,
    }


def _ote_fib_pos(price: float, sweep_extreme: float,
                 disp_extreme: float, direction: str) -> Optional[float]:
    rng = abs(disp_extreme - sweep_extreme)
    if rng < 1e-9: return None
    if direction == "long": return (disp_extreme - price) / rng
    return (price - disp_extreme) / rng


def _in_ote(price: float, ote_zone: dict) -> bool:
    return ote_zone["low"] <= price <= ote_zone["high"]


def _calc_lv_fib_pos(direction: str,
                     sweep_extreme: float,
                     disp_extreme: float,
                     level_price: float) -> float:
    """
    sweep_extreme ~ disp_extreme 구간에서 level_price의 피보나치 위치 반환 (0.0~1.0).
    sweep_extreme = 0.0, disp_extreme = 1.0 기준.

    long:  sweep_extreme(저점=0) → disp_extreme(고점=1)
           level_price가 sweep에 가까울수록 0, disp에 가까울수록 1
    short: sweep_extreme(고점=0) → disp_extreme(저점=1)
           level_price가 sweep에 가까울수록 0, disp에 가까울수록 1

    0.5 미만 = displacement 전 절반 구간에 level이 있음
            → 가격이 level 너머까지 제대로 이동했는지 불확실 → size 패널티
    """
    rng = abs(disp_extreme - sweep_extreme)
    if rng < 1e-9:
        return 1.0  # 구간 없으면 패널티 없음
    if direction == "long":
        # sweep=저점(0), disp=고점(1): level이 sweep으로부터 얼마나 멀리 있는가
        pos = (level_price - sweep_extreme) / rng
    else:
        # sweep=고점(0), disp=저점(1): 아래 방향이므로 반전
        pos = (sweep_extreme - level_price) / rng
    return max(0.0, min(1.0, pos))



# ──────────────────────────────────────────────────────────────────
# FVG 선택 — Group A (단계별)
# ──────────────────────────────────────────────────────────────────

def _find_fvgs_after_disp(fvgs_df: pd.DataFrame, direction: str,
                           trigger_time: pd.Timestamp,
                           sweep_extreme: float = 0.0,
                           disp_extreme: float = 0.0) -> List[dict]:
    """
    displacement 이후 생성된 방향 일치 FVG 목록 반환.

    가격 범위 필터 (핵심):
      sweep_extreme ~ disp_extreme 구간 안에 FVG mid가 있어야 함.
      - long:  sweep_extreme(저점) ~ disp_extreme(고점) 사이
      - short: disp_extreme(저점) ~ sweep_extreme(고점) 사이
      sweep_extreme/disp_extreme이 0이면 필터 스킵 (하위 호환).
    """
    if fvgs_df is None or fvgs_df.empty: return []
    want = "bull" if direction == "long" else "bear"
    f = fvgs_df[fvgs_df["fvg_type"] == want].copy()
    if f.empty: return []
    if "created_time" in f.columns:
        f = f[pd.to_datetime(f["created_time"]) >= trigger_time]

    # 가격 범위 경계
    price_lo = min(sweep_extreme, disp_extreme)
    price_hi = max(sweep_extreme, disp_extreme)
    use_range = (price_lo > 0 and price_hi > price_lo)

    result = []
    for _, r in f.iterrows():
        zl, zh = float(r["zone_low"]), float(r["zone_high"])
        mid = _mid(zl, zh)
        # ── FVG mid가 sweep~disp 구간 밖이면 제외
        if use_range and not (price_lo <= mid <= price_hi):
            continue
        result.append({
            "fvg_type": r["fvg_type"],
            "zone_low": zl, "zone_high": zh,
            "mid": mid,
            "size": float(r.get("size", zh - zl)),
        })
    return result


# ── FVG 크기 기반 진입가 결정 ──
# 거대 FVG: mid와 edge의 중앙 진입 (백테스트 단일가 근사)
# 일반 FVG: edge 진입 (체결 확률 우선)
FVG_HUGE_NET_RATIO = 0.25   # net_move 대비 25% 이상이면 거대
FVG_HUGE_ATR_RATIO = 0.5    # 5m ATR 대비 0.5배 이상이면 거대

def _fvg_entry_price(fvg: dict, direction: str,
                     atr: float = 0.0, net_move: float = 0.0) -> float:
    """
    FVG 크기에 따른 진입가 결정.
    거대 FVG (ATR×0.5 이상 or net_move×25% 이상):
      → mid와 edge의 중앙 (실전 50:50 분할의 백테스트 근사)
    일반 FVG:
      → edge 진입 (long: zone_high, short: zone_low)
    """
    fvg_size = fvg["zone_high"] - fvg["zone_low"]
    is_huge_relative = (net_move > 0 and fvg_size > net_move * FVG_HUGE_NET_RATIO)
    is_huge_absolute = (atr > 0 and fvg_size > atr * FVG_HUGE_ATR_RATIO)

    edge = fvg["zone_high"] if direction == "long" else fvg["zone_low"]
    mid  = fvg["mid"]

    if is_huge_relative or is_huge_absolute:
        return (mid + edge) / 2.0   # mid~edge 중앙
    return edge


def _pick_fvg_group_a(fvgs_df: pd.DataFrame, direction: str,
                       trigger_time: pd.Timestamp,
                       sweep_extreme: float, disp_extreme: float,
                       ote_zone: Optional[dict],
                       atr: float = 0.0, net_move: float = 0.0) -> Optional[dict]:
    """
    Group A FVG 단계 선택 — 가장 넓은 FVG 1개.
    거대 FVG → mid~edge 중앙 진입, 일반 → edge 진입.
    """
    fvgs = _find_fvgs_after_disp(fvgs_df, direction, trigger_time,
                                  sweep_extreme, disp_extreme)
    if len(fvgs) == 0:
        return None

    widest = max(fvgs, key=lambda f: f["size"])
    ep = _fvg_entry_price(widest, direction, atr=atr, net_move=net_move)
    in_ote_flag = _in_ote(widest["mid"], ote_zone) if ote_zone else False
    return {
        "zone_low": widest["zone_low"], "zone_high": widest["zone_high"],
        "mid": widest["mid"], "size": widest["size"],
        "entry_price": ep, "in_ote": in_ote_flag,
        "orders": [{"type":"limit","side":direction,"price":ep,"weight":1.0,"tag":"fvg_widest"}],
    }


# ──────────────────────────────────────────────────────────────────
# FVG 선택 — Group B (OTE 우선)
# ──────────────────────────────────────────────────────────────────

def _pick_fvg_group_b(fvgs_df: pd.DataFrame, direction: str,
                       trigger_time: pd.Timestamp,
                       sweep_extreme: float, disp_extreme: float,
                       ote_zone: Optional[dict]) -> Optional[dict]:
    if fvgs_df is None or fvgs_df.empty: return None
    if ote_zone is None: return None
    want = "bull" if direction == "long" else "bear"
    f = fvgs_df[fvgs_df["fvg_type"] == want].copy()
    if f.empty: return None
    if "created_time" in f.columns:
        f_after = f[pd.to_datetime(f["created_time"]) >= trigger_time]
        pool = f_after if not f_after.empty else f
    else:
        pool = f
    ideal_px = ote_zone["ideal"]
    best, best_dist = None, float("inf")
    for _, r in pool.iterrows():
        zl, zh = float(r["zone_low"]), float(r["zone_high"])
        m = _mid(zl, zh)
        if not _in_ote(m, ote_zone): continue
        fib = _ote_fib_pos(m, sweep_extreme, disp_extreme, direction)
        dist = abs(m - ideal_px)
        if dist < best_dist:
            best_dist = dist
            best = {"fvg_type":r["fvg_type"],"zone_low":zl,"zone_high":zh,
                    "mid":m,"size":zh-zl,"fib_pos":fib,"in_ote":True}
    return best


def _pick_widest_fvg(fvgs_df: pd.DataFrame, direction: str,
                      trigger_time: pd.Timestamp,
                      sweep_extreme: float, disp_extreme: float) -> Optional[dict]:
    """
    sweep~disp 구간 내 가장 넓은 FVG 선택 (OTE 무관).
    Group B용.
    """
    fvgs = _find_fvgs_after_disp(fvgs_df, direction, trigger_time,
                                  sweep_extreme, disp_extreme)
    if not fvgs:
        return None
    widest = max(fvgs, key=lambda f: f["size"])
    return widest


# ──────────────────────────────────────────────────────────────────
# OB 선택
# ──────────────────────────────────────────────────────────────────

def _pick_best_ob(obs_df: pd.DataFrame, t_now: pd.Timestamp,
                  ob_type: str, ote_zone: Optional[dict]) -> Optional[dict]:
    if obs_df is None or obs_df.empty: return None
    o = obs_df[obs_df["formed_at"] <= t_now].copy()
    if o.empty: return None
    if "invalidated" in o.columns:
        o = o[o["invalidated"] == False]
        if o.empty: return None
    o = o[o["ob_type"] == ob_type].sort_values("formed_at")
    if o.empty: return None
    def _row(r, in_ote_flag): return dict(r) | {"in_ote": in_ote_flag}
    if ote_zone is not None:
        def _ob_in_ote(r): return _in_ote(_mid(float(r["zone_low"]),float(r["zone_high"])), ote_zone)
        o_in = o[o.apply(_ob_in_ote, axis=1)]
        if not o_in.empty: return _row(o_in.iloc[-1], True)
    # OTE 없거나 OTE 안에 OB 없으면 → 가장 최근 valid OB 반환
    return _row(o.iloc[-1], False)


# ──────────────────────────────────────────────────────────────────
# SL 계산
# ──────────────────────────────────────────────────────────────────

# SL 최소 손절 비율: 유동성 레벨 ~ 진입가 기준
SL_MIN_GROUP_A = 0.0015   # Group A: 0.15%
SL_MIN_GROUP_B = 0.003    # Group B: 0.3%

def _apply_min_sl(sl: float, entry_price: float, direction: str,
                  group: str = "B") -> float:
    """진입가 대비 최소 SL 거리 보장 (Group A: 0.15%, Group B: 0.3%)"""
    if entry_price <= 0: return sl
    min_pct = SL_MIN_GROUP_A if group == "A" else SL_MIN_GROUP_B
    if direction == "long":
        min_sl = entry_price * (1 - min_pct)
        return min(sl, min_sl)   # 더 낮은(넓은) 쪽
    else:
        min_sl = entry_price * (1 + min_pct)
        return max(sl, min_sl)   # 더 높은(넓은) 쪽


def _sl_min_ok(entry_price: float, sl: float, direction: str, group: str = "B") -> bool:
    """SL이 최소 거리 이상인지 확인"""
    if entry_price <= 0: return True
    min_pct = SL_MIN_GROUP_A if group == "A" else SL_MIN_GROUP_B
    dist_pct = abs(entry_price - sl) / entry_price
    return dist_pct >= min_pct


def _calc_sl_group_a(direction: str, sweep_extreme: float,
                     entry_price: float, buf: float) -> float:
    """Group A SL = sweep_extreme, 최소 0.15% 보장"""
    if direction == "long":
        sl = sweep_extreme * (1 - buf)
    else:
        sl = sweep_extreme * (1 + buf)
    return _apply_min_sl(sl, entry_price, direction, group="A")


def _calc_sl_group_b(direction: str, sweep_extreme: float,
                     entry_price: float, buf: float) -> float:
    """Group B 메인 SL = sweep_extreme, 최소 0.3% 보장"""
    if direction == "long":
        sl = sweep_extreme * (1 - buf)
    else:
        sl = sweep_extreme * (1 + buf)
    return _apply_min_sl(sl, entry_price, direction, group="B")


def _calc_ob_sl(direction: str, ob: dict, buf: float, entry_price: float) -> float:
    """Group B OB SL (1차 30%)"""
    if direction == "long":
        sl = float(ob["zone_low"]) * (1 - buf)
    else:
        sl = float(ob["zone_high"]) * (1 + buf)
    return _apply_min_sl(sl, entry_price, direction)


# ──────────────────────────────────────────────────────────────────
# HTF size multiplier — v3.0
# ──────────────────────────────────────────────────────────────────

def _calc_htf_multiplier(level_type: str, htf_1h: str, htf_4h: str,
                          direction: str) -> float:
    """
    Group A: size × (1 - 4H_mismatch×0.2 - 1H_mismatch×0.3)
    Group B: size × (1 - 4H_mismatch×0.3 - 1H_mismatch×0.2)
    """
    is_group_a = level_type in LEVEL_GROUP_A
    if is_group_a:
        p4h, p1h = 0.2, 0.3
    else:
        p4h, p1h = 0.3, 0.2

    pen_1h = p1h if (htf_1h == "bull" and direction == "short") or \
                    (htf_1h == "bear" and direction == "long") else 0.0
    pen_4h = p4h if (htf_4h == "bull" and direction == "short") or \
                    (htf_4h == "bear" and direction == "long") else 0.0
    return max(0.0, 1.0 - pen_1h - pen_4h)


def _calc_pd_multiplier(pd_pos: float, direction: str) -> float:
    """
    PD position 기반 사이즈 배율.
    pd_pos: 0.0(저점) ~ 1.0(고점)
    long:  pd_pos 낮을수록(discount) 유리 → 1 - pd_pos → 0.6~1.4 스케일
    short: pd_pos 높을수록(premium) 유리 → pd_pos → 0.6~1.4 스케일
    """
    if pd_pos is None:
        return 1.0
    if direction == "long":
        # pd_pos=0 → score=1.0 → mult=1.4, pd_pos=1 → score=0.0 → mult=0.6
        score = 1.0 - pd_pos
    else:
        # pd_pos=1 → score=1.0 → mult=1.4, pd_pos=0 → score=0.0 → mult=0.6
        score = pd_pos
    # 0.6 ~ 1.4 범위로 매핑
    return round(0.6 + score * 0.8, 2)


def _calc_size_multiplier_v2(level_type: str, htf_1h: str, htf_4h: str,
                               direction: str, sweep_vol_ok: bool,
                               pd_pos: float = None) -> float:
    """
    ML 전환: 항상 1.0 반환. 기존 로직은 feature_dict로 전달.
    (피처 계산은 _calc_size_features()에서 수행)
    """
    return 1.0


def _calc_size_features(level_type: str, htf_1h: str, htf_4h: str,
                         direction: str, sweep_vol_ok: bool,
                         pd_pos: float = None,
                         df_5m: Optional[pd.DataFrame] = None,
                         sweep_bar_idx: int = 0) -> dict:
    """기존 size_multiplier 구성요소를 ML 피처로 반환."""
    # HTF alignment (레거시 — htf_4h_aligned에서만 사용)
    def _htf_align(htf, d):
        if htf == "unknown": return 0
        if (d == "long" and htf == "bull") or (d == "short" and htf == "bear"):
            return 1
        return -1

    htf_4h_aligned = _htf_align(htf_4h, direction)

    # Directional Efficiency Ratio (방향성 효율 비율)
    # = 방향별 순이동 / 총 이동경로 (Perry Kaufman ER의 방향 확장)
    # long: (close[-1] - close[-period]) / sum(|close[i] - close[i-1]|)
    # short: (close[-period] - close[-1]) / sum(|close[i] - close[i-1]|)
    # 범위: -1 ~ +1, 양수면 진입 방향과 추세 일치
    directional_er = 0.0
    _er_period = 144  # 12시간 (5m × 144봉)
    if df_5m is not None and "close" in df_5m.columns and len(df_5m) > _er_period + 1:
        _c = df_5m["close"].values.astype(float)
        _net = _c[-1] - _c[-_er_period - 1]
        _diffs = np.abs(np.diff(_c[-_er_period - 1:]))
        _path = float(_diffs.sum())
        # 거래소 점검 등으로 close가 동일한 봉이 연속되면 path=0
        # 또는 데이터 누락으로 NaN/Inf 발생 가능
        if _path > 1e-10 and np.isfinite(_net) and np.isfinite(_path):
            _raw_er = _net / _path  # -1 ~ +1
            if direction == "short":
                _raw_er = -_raw_er
            directional_er = round(float(np.clip(_raw_er, -1.0, 1.0)), 4)

    # 볼륨 비율
    vol_ratio = 1.0
    if df_5m is not None and "volume" in df_5m.columns and len(df_5m) > HUNT_VOL_LOOKBACK + 2:
        reclaim_idx = len(df_5m) - 1
        baseline = _vol_median(df_5m, sweep_bar_idx, HUNT_VOL_LOOKBACK)
        if baseline > 0:
            start = max(0, sweep_bar_idx)
            end = min(len(df_5m), reclaim_idx + 1)
            if start < end:
                vols = df_5m["volume"].to_numpy(dtype=float)
                avg_vol = float(vols[start:end].mean())
                vol_ratio = avg_vol / baseline

    return {
        "directional_er": directional_er,
        "htf_4h_aligned": htf_4h_aligned,
        "vol_ratio": round(vol_ratio, 3),
    }



# ──────────────────────────────────────────────────────────────────
# TP 탐색
# ──────────────────────────────────────────────────────────────────

def _find_next_liquidity_target(levels: List[dict], px: float,
                                 direction: str, sl: float,
                                 min_rr: float = 1.5, max_rr: float = 4.0,
                                 min_tp_pct: float = 0.003,
                                 tp_buffer_pct: float = 0.0015) -> Optional[float]:
    if not levels: return None
    risk = abs(px - sl)
    min_dist = px * min_tp_pct
    if direction == "long":
        cands = []
        for lv in levels:
            tp_cand = float(lv["zone_low"]) if float(lv["zone_low"]) > px else float(lv["zone_high"])
            if tp_cand <= px: continue
            tp_cand *= (1 - tp_buffer_pct)  # TP buffer: 레벨 약간 아래
            dist = tp_cand - px; rr = dist/risk if risk>0 else 0
            if dist >= min_dist and min_rr <= rr <= max_rr: cands.append(tp_cand)
        return float(min(cands, key=lambda v: v-px)) if cands else None
    if direction == "short":
        cands = []
        for lv in levels:
            tp_cand = float(lv["zone_high"]) if float(lv["zone_high"]) < px else float(lv["zone_low"])
            if tp_cand >= px: continue
            tp_cand *= (1 + tp_buffer_pct)  # TP buffer: 레벨 약간 위
            dist = px-tp_cand; rr = dist/risk if risk>0 else 0
            if dist >= min_dist and min_rr <= rr <= max_rr: cands.append(tp_cand)
        return float(min(cands, key=lambda v: px-v)) if cands else None
    return None


def _find_tp_fallback(fvgs_5m: pd.DataFrame, obs_5m: pd.DataFrame,
                      ref_px: float, direction: str, sl: float,
                      pivots_15m: Optional[pd.DataFrame] = None,
                      min_rr: float = 1.2, max_rr: float = 5.0,
                      min_tp_pct: float = 0.003) -> Optional[float]:
    """
    no_tp fallback 순서:
    1) 직전 15m pivot swing H(short) or L(long) — RR 1:1 이상, 보수적으로 가장 가까운 것
    2) 반대방향 5m FVG mid
    3) 반대방향 5m OB zone_low/high
    """
    risk = abs(ref_px - sl)
    min_dist = ref_px * min_tp_pct
    def _ok(cand):
        d = abs(cand - ref_px)
        rr = d / risk if risk > 0 else 0
        return d >= min_dist and rr >= min_rr

    # ── 1순위: 직전 15m pivot swing (보수적 = 가장 가까운 것)
    if pivots_15m is not None and not pivots_15m.empty:
        if direction == "long":
            # 직전 swing high: 진입가 위에 있는 것 중 가장 가까운(낮은) 것
            highs = pivots_15m[pivots_15m["kind"] == "H"]
            cands = [float(r["price"]) for _, r in highs.iterrows()
                     if float(r["price"]) > ref_px and _ok(float(r["price"]))]
            if cands:
                return float(min(cands))
        else:
            # 직전 swing low: 진입가 아래에 있는 것 중 가장 가까운(높은) 것
            lows = pivots_15m[pivots_15m["kind"] == "L"]
            cands = [float(r["price"]) for _, r in lows.iterrows()
                     if float(r["price"]) < ref_px and _ok(float(r["price"]))]
            if cands:
                return float(max(cands))

    # ── 2순위: 반대방향 5m FVG
    opp_fvg_t = "bear" if direction == "long" else "bull"
    if fvgs_5m is not None and not fvgs_5m.empty:
        ff = fvgs_5m[fvgs_5m["fvg_type"] == opp_fvg_t]
        cands = []
        for _, r in ff.iterrows():
            m = _mid(float(r["zone_low"]), float(r["zone_high"]))
            if direction=="long" and m>ref_px and _ok(m): cands.append(m)
            elif direction=="short" and m<ref_px and _ok(m): cands.append(m)
        if cands: return float(min(cands)) if direction=="long" else float(max(cands))

    # ── 3순위: 반대방향 5m OB
    opp_ob_t = "bear_ob" if direction == "long" else "bull_ob"
    if obs_5m is not None and not obs_5m.empty:
        oo = obs_5m[obs_5m["ob_type"] == opp_ob_t].copy()
        if "invalidated" in oo.columns: oo = oo[oo["invalidated"]==False]
        cands = []
        for _, r in oo.iterrows():
            zl, zh = float(r["zone_low"]), float(r["zone_high"])
            if direction=="long" and zl>ref_px and _ok(zl): cands.append(zl)
            elif direction=="short" and zh<ref_px and _ok(zh): cands.append(zh)
        if cands: return float(min(cands)) if direction=="long" else float(max(cands))

    return None



# ──────────────────────────────────────────────────────────────────
# 레벨 방향 유효성
# ──────────────────────────────────────────────────────────────────

def _level_dir_valid(level_type: str, sweep_dir: str) -> bool:
    """
    레벨 본래 방향과 sweep 방향의 정합성 검증.
    저점 레벨(pdl/pwl/asia_low 등)은 lower sweep(long)만 허용.
    고점 레벨(pdh/pwh/asia_high 등)은 upper sweep(short)만 허용.
    S/R flip(breakerblock)은 반대 방향 허용.
    """
    # 저점 레벨 → long(lower sweep)만
    LOW_LEVELS  = {"pdl", "pwl", "asia_low", "london_low",
                   "idm_low", "eql", "swing_low"}
    # 고점 레벨 → short(upper sweep)만
    HIGH_LEVELS = {"pdh", "pwh", "asia_high", "london_high",
                   "idm_high", "eqh", "swing_high"}
    if level_type in LOW_LEVELS:  return sweep_dir == "long"
    if level_type in HIGH_LEVELS: return sweep_dir == "short"
    return True  # breakerblock 등 방향 무관


# ──────────────────────────────────────────────────────────────────
# Multi-Sweep History (레벨별 sweep 이력 관리)
# ──────────────────────────────────────────────────────────────────

class _SweepRecord:
    """개별 sweep 기록"""
    __slots__ = ("timestamp", "depth", "sweep_extreme", "disp_net_move",
                 "direction", "sl_hit")
    def __init__(self, timestamp, depth: float, sweep_extreme: float,
                 disp_net_move: float, direction: str):
        self.timestamp = timestamp
        self.depth = depth
        self.sweep_extreme = sweep_extreme
        self.disp_net_move = disp_net_move
        self.direction = direction
        self.sl_hit = False  # 나중에 SL 결과 기록


class _SweepHistory:
    """
    레벨별 sweep 이력 관리.
    key = (level_type, zone_mid_rounded)
    """
    def __init__(self):
        self._history: Dict[str, List[_SweepRecord]] = {}
        self._opposite_flags: Dict[str, bool] = {}  # 반대 sweep 감지 플래그
        self._opposite_cooldown: Dict[str, int] = {}  # 반대 sweep 후 쿨다운

    def _key(self, level: dict) -> str:
        lt = level.get("level_type", "?")
        mid = (float(level.get("zone_low", 0)) + float(level.get("zone_high", 0))) / 2
        return f"{lt}_{mid:.0f}"

    def add(self, level: dict, record: _SweepRecord):
        k = self._key(level)
        if k not in self._history:
            self._history[k] = []
        self._history[k].append(record)
        # 최대 3개 유지
        if len(self._history[k]) > 3:
            self._history[k] = self._history[k][-3:]

    def get(self, level: dict) -> List[_SweepRecord]:
        return self._history.get(self._key(level), [])

    def mark_sl_hit(self, level: dict):
        k = self._key(level)
        recs = self._history.get(k, [])
        if recs:
            recs[-1].sl_hit = True

    def mark_opposite_sweep(self, level: dict, opp_depth: float):
        """반대 방향 sweep 발생 시 호출"""
        k = self._key(level)
        self._opposite_flags[k] = True
        self._opposite_cooldown[k] = 24  # 24봉 쿨다운

        # Group A 안전장치: 반대 sweep depth가 기존의 70% 이상이면 history 축소
        recs = self._history.get(k, [])
        if recs:
            lt = level.get("level_type", "")
            if lt in LEVEL_GROUP_A:
                last_depth = recs[-1].depth
                if opp_depth >= last_depth * 0.7:
                    # 가장 최근 1개만 남기고 clear
                    self._history[k] = [recs[-1]]

    def tick_cooldown(self):
        """매 봉 호출: 반대 sweep 쿨다운 감소"""
        for k in list(self._opposite_cooldown.keys()):
            self._opposite_cooldown[k] -= 1
            if self._opposite_cooldown[k] <= 0:
                del self._opposite_cooldown[k]
                self._opposite_flags.pop(k, None)

    def can_reenter(self, level: dict, new_depth: float,
                    new_disp_net_move: float) -> tuple:
        """
        재진입 가능 여부 판단.
        반환: (allowed: bool, reason: str)
        """
        k = self._key(level)
        lt = level.get("level_type", "")
        is_group_a = lt in LEVEL_GROUP_A
        recs = self._history.get(k, [])

        # 이력 없으면 무조건 허용
        if not recs:
            return True, "no_history"

        # SL hit 된 기록만 카운트
        sl_hits = [r for r in recs if r.sl_hit]
        if not sl_hits:
            return True, "no_sl_hit"

        last = sl_hits[-1]

        # 최대 재진입 횟수
        max_reentry = 2 if is_group_a else 3
        if len(sl_hits) >= max_reentry:
            return False, f"max_reentry({max_reentry})"

        # 반대 sweep 쿨다운 중
        if k in self._opposite_cooldown and is_group_a:
            # Group A: 반대 sweep 후 쿨다운 중에는 더 깊은 sweep만 허용
            depth_mult = 1.4  # 더 엄격
            if new_depth < last.depth * depth_mult:
                return False, f"opp_cooldown(need {depth_mult}×)"

        # depth 비교: 단순히 1.0×ATR 이상이면 재진입 허용
        if new_depth < 1.0:
            return False, f"depth({new_depth:.2f}<1.0)"

        # displacement net move 비교
        if last.disp_net_move > 0:
            min_ratio = 0.80 if is_group_a else 0.75
            if new_disp_net_move < last.disp_net_move * min_ratio:
                return False, f"disp_weak({new_disp_net_move:.0f}<{last.disp_net_move*min_ratio:.0f})"

        return True, "ok"

    def clear_level(self, level: dict):
        k = self._key(level)
        self._history.pop(k, None)
        self._opposite_flags.pop(k, None)
        self._opposite_cooldown.pop(k, None)


# ──────────────────────────────────────────────────────────────────
# 상태머신 슬롯
# ──────────────────────────────────────────────────────────────────

class _StandbySlot:
    __slots__ = ("sweep_dir","level","sweep_extreme","sweep_ratio",
                 "bar_count","run_high","run_low","swept_this_bar",
                 "reclaim_bar_idx","sweep_bar_idx","timeout_bars",
                 "sweep_depth_score","sweep_extreme_chain","sweep_time",
                 "sl_hits_count","cooldown_left","liq_features")

    def __init__(self, sweep_dir: str, level: dict, sweep_extreme: float,
                 h: float, l: float, o: float, c: float, df_len: int = 0,
                 sweep_depth_score: float = 1.0,
                 sweep_time: Optional[pd.Timestamp] = None,
                 sl_hits_count: int = 0, cooldown_left: int = 0,
                 liq_features: Optional[dict] = None):
        self.sweep_dir      = sweep_dir
        self.level          = level
        self.sweep_extreme  = sweep_extreme
        self.bar_count      = 0
        self.swept_this_bar = True
        self.run_high       = h
        self.run_low        = l
        self.reclaim_bar_idx: Optional[int] = None
        self.sweep_bar_idx:   int = max(0, df_len - 1)
        rng = h - l
        if sweep_dir == "short":
            self.sweep_ratio = (h - max(o, c)) / rng if rng > 1e-9 else 0.0
        else:
            self.sweep_ratio = (min(o, c) - l) / rng if rng > 1e-9 else 0.0
        # 레벨 그룹별 timeout
        lv_t = level.get("level_type", "")
        self.timeout_bars = STANDBY_TIMEOUT_GROUP_A if lv_t in LEVEL_GROUP_A else STANDBY_TIMEOUT_GROUP_B
        self.sweep_depth_score = sweep_depth_score
        self.sweep_extreme_chain = [sweep_extreme]  # 연속 sweep SL 체인
        self.sweep_time = sweep_time  # sweep 발생 시각
        self.sl_hits_count = sl_hits_count    # ML 피처: 최근 SL 횟수
        self.cooldown_left = cooldown_left    # ML 피처: 쿨다운 잔여 봉
        self.liq_features = liq_features or {}  # ML 피처: liquidity 강도 피처

    def update_ratio(self, h: float, l: float, c: float):
        self.run_high = max(self.run_high, h)
        self.run_low  = min(self.run_low,  l)
        rng = self.run_high - self.run_low
        if rng > 1e-9:
            if self.sweep_dir == "short":
                self.sweep_ratio = (self.run_high - c) / rng
            else:
                self.sweep_ratio = (c - self.run_low) / rng


# ──────────────────────────────────────────────────────────────────
# 메인 상태머신
# ──────────────────────────────────────────────────────────────────

class SweepReclaimOTE:
    """signals.py v2.0"""

    def __init__(self, reclaim_seconds: int = 1800, reclaim_bars: int = 6,
                 orders_ttl_bars: int = 15, sl_buffer_pct: float = 0.001,
                 ob_wait_bars: int = 6, disp_flat_bars: int = 2):
        self.reclaim_bars    = max(1, reclaim_seconds//300) if reclaim_seconds!=1800 else int(reclaim_bars)
        self.orders_ttl_bars = int(orders_ttl_bars)
        self.sl_buffer_pct   = float(sl_buffer_pct)
        self.ob_wait_bars    = int(ob_wait_bars)
        self.disp_flat_bars  = int(disp_flat_bars)
        self.armed_timeout_bars: int = int(reclaim_bars)

        self.state               = "IDLE"
        self._standbys: List[_StandbySlot] = []
        self.sweep_dir:          Optional[str]          = None
        self.level:              Optional[dict]         = None
        self.sweep_extreme:      Optional[float]        = None
        self.sweep_ratio:        float                  = 0.0
        self.sweep_bar_idx:      int                    = 0
        self.choch_price:        Optional[float]        = None
        self.trigger_time:       Optional[pd.Timestamp] = None
        self.sweep_time:         Optional[pd.Timestamp] = None  # sweep 발생 시각
        self.trigger_type:       str                    = "bos"
        self.run_high:           Optional[float]        = None
        self.run_low:            Optional[float]        = None
        self._choch_candle_ref:  Optional[float]        = None
        self._disp_flat_count:   int  = 0
        self._disp_trigger_time: Optional[pd.Timestamp] = None
        self.entry_plan:         Optional[dict]         = None
        self.plan_time:          Optional[pd.Timestamp] = None
        self.plan_bar_count:     int  = 0
        self._armed_bar_count:   int  = 0
        self._armed_df_len:      int  = 0
        self._armed_start_price: Optional[float] = None  # reclaim 시점 종가
        self._sweep_depth_score: float = 1.0  # ATR 기반 sweep 깊이 스코어
        self._sweep_extreme_chain: list = []  # 연속 sweep extreme 체인
        self._ob_wait_count:     int  = 0
        self._armed_features:    dict = {}  # ARMED→DISP_RUNNING 피처 전달용
        self._liq_features:      dict = {}  # liquidity 강도 피처
        self._sl_hits_count_armed: int = 0
        self._cooldown_left_armed: int = 0
        # ── FVG_WATCHING 상태 필드
        self._watching_fvg:      Optional[dict] = None  # 최고 점수 FVG (best candidate)
        self._watching_entry_px: float = 0.0            # best candidate 진입가
        self._watching_sl:       float = 0.0
        self._watching_tp:       Optional[float] = None
        self._watching_bars:     int   = 0              # bars_since_fvg_created
        self._watching_disp_extreme: float = 0.0
        self._watching_group:    str   = "A"            # 원래 그룹
        self._fvg_pool:          list  = []             # FVG candidate pool
        self._cooldown_level:    Optional[dict] = None
        self._cooldown_count:    int  = 0
        self._cooldown_bars:     int  = 12
        self._last_sl_depth_score: float = 0.0  # 이중 sweep 재진입 비교용
        self._pending_disables:  List[dict] = []
        self._sweep_history = _SweepHistory()  # Multi-Sweep History
        # ── 전역 시간순 sweep 큐 (반대편 유동성 선행회수 추적용; 재진입 _sweep_history와 별개).
        #    진입 직전 FVG 반대편 유동성을 먼저 쓸었나(ICT 반대편 회수 후 반전).
        #    셋업 sweep 말고 그 직전 독립 sweep을 봄.
        self._sweep_queue: deque = deque(maxlen=12)
        self._opp_sweep_feats: dict = {}

    def reset(self, exit_reason: str = ""):
        # Multi-Sweep History: SL hit 기록
        if exit_reason in ("sl", "trail_sl") and self.level is not None:
            self._sweep_history.mark_sl_hit(self.level)

        self._cooldown_level    = self.level
        # 반대 방향 감지를 위해 마지막 sweep 방향 기록
        if self._cooldown_level is not None and self.sweep_dir:
            self._cooldown_level["_last_sweep_dir"] = self.sweep_dir
        self._cooldown_count    = 0
        self._cooldown_bars     = 24 if exit_reason in ("sl","trail_sl") else 12
        self._last_sl_depth_score = self._sweep_depth_score if exit_reason in ("sl","trail_sl") else 0.0
        self.state              = "IDLE"
        self._standbys          = []
        self.sweep_dir          = None
        self.level              = None
        self.sweep_extreme      = None
        self.sweep_ratio        = 0.0
        self.sweep_bar_idx      = 0
        self.choch_price        = None
        self.trigger_time       = None
        self.sweep_time         = None
        self.trigger_type       = "bos"
        self.run_high           = None
        self.run_low            = None
        self._choch_candle_ref  = None
        self._disp_flat_count   = 0
        self._disp_trigger_time = None
        self.entry_plan         = None
        self.plan_time          = None
        self.plan_bar_count     = 0
        self._armed_bar_count   = 0
        self._armed_df_len      = 0
        self._armed_start_price = None
        self._sweep_depth_score = 1.0
        self._sweep_extreme_chain = []
        self._ob_wait_count     = 0
        self._armed_features    = {}
        self._liq_features      = {}
        self._sl_hits_count_armed = 0
        self._cooldown_left_armed = 0
        self._watching_fvg      = None
        # ── _sweep_queue / _opp_sweep_feats는 reset에서 비우지 않음(의도적):
        #    전역 시간순 sweep 이력은 트레이드 경계를 넘어 유지돼야 "직전 반대
        #    sweep" 조회가 성립. 심볼간 누수는 strategy가 심볼당 새 인스턴스라 없음.
        self._opp_sweep_feats   = {}
        self._watching_entry_px = 0.0
        self._watching_sl       = 0.0
        self._watching_tp       = None
        self._watching_bars     = 0
        self._watching_disp_extreme = 0.0
        self._watching_group    = "A"
        self._fvg_pool          = []

    def drain_pending_disables(self) -> List[dict]:
        """timeout된 레벨 목록 반환 후 초기화. backtest에서 liq_pool.disable_level() 호출용."""
        out = self._pending_disables
        self._pending_disables = []
        return out

    def _query_opposite_sweep(self, setup_dir: str, setup_time: Any,
                              setup_extreme: float = None,
                              df_5m=None) -> dict:
        """셋업 sweep 직전, 반대방향 독립 sweep이 있었나 (ICT 반대편 선행회수 = 가격 전달).
        진입 셋업 sweep(setup_dir/time) 말고 그보다 '이전' 시각 큐 항목 중
        '반대방향' 가장 가까운(직전) 1건. extreme/level은 원시 — atr 정규화는 backtest.

        '전달' 의도 정밀화 3종 (재수집 필수 — collect 시점 큐/구간가격 의존):
          opp_sweep_is_last     : 큐의 '바로 직전' 이벤트가 반대방향인가 (직접 전달).
                                  사이에 동방향 sweep 끼면 0 → 단순 연속추세 배제.
          opp_sweep_unviolated  : 반대 sweep 후 셋업까지 가격이 그 extreme을 재침범 안함
                                  (딜링레인지 보존; long이면 opp_high 위로 안 넘음).
          opp_sweep_range_raw   : |opp_extreme - 셋업 extreme| 원시 (backtest서 atr 정규화).
        """
        out = {"opp_sweep_present": 0.0, "opp_sweep_to_setup_bars": -1.0,
               "opp_sweep_extreme": np.nan, "opp_sweep_level_mid": np.nan,
               "opp_sweep_is_last": 0.0, "opp_sweep_unviolated": np.nan,
               "opp_sweep_range_raw": np.nan}
        if setup_dir is None or setup_time is None or not self._sweep_queue:
            return out
        opp = "short" if setup_dir == "long" else "long"
        st = pd.Timestamp(setup_time)
        # 셋업 이전 큐 항목만, 시간순 (deque는 append 순 = 시간순)
        prior = [r for r in self._sweep_queue if r["time"] < st]
        if not prior:
            return out
        # best = 가장 가까운(직전) 반대방향
        best = None
        for rec in prior:
            if rec["dir"] != opp:
                continue
            if best is None or rec["time"] > best["time"]:
                best = rec
        if best is None:
            return out
        out["opp_sweep_present"] = 1.0
        out["opp_sweep_to_setup_bars"] = float(round((st - best["time"]).total_seconds() / 300.0))
        out["opp_sweep_extreme"] = float(best["extreme"])
        out["opp_sweep_level_mid"] = float(best["level_mid"])

        # ── is_last: prior의 마지막(=셋업 직전) 이벤트가 반대방향인가
        out["opp_sweep_is_last"] = 1.0 if prior[-1]["dir"] == opp else 0.0

        # ── range_raw: 반대 extreme ~ 셋업 extreme 절대거리 (전달 레인지 높이)
        if setup_extreme is not None and np.isfinite(setup_extreme):
            out["opp_sweep_range_raw"] = abs(float(best["extreme"]) - float(setup_extreme))

        # ── unviolated: best 시각~셋업 사이 가격이 opp extreme 재침범 안했는가.
        #    long 셋업 → 반대(short) sweep high를 가격이 다시 넘었으면 레인지 깨짐.
        if df_5m is not None and len(df_5m) > 0:
            try:
                _t = df_5m["time"] if "time" in df_5m else df_5m.index
                _mask = (_t > best["time"]) & (_t < st)
                seg = df_5m[_mask.values] if hasattr(_mask, "values") else df_5m[_mask]
                if len(seg) > 0:
                    if setup_dir == "long":
                        # opp는 상방 sweep → high가 opp extreme 위로 재돌파 시 침범
                        out["opp_sweep_unviolated"] = (
                            0.0 if seg["high"].max() > best["extreme"] else 1.0)
                    else:
                        out["opp_sweep_unviolated"] = (
                            0.0 if seg["low"].min() < best["extreme"] else 1.0)
            except Exception:
                pass  # df 구조 예외 시 NaN 유지
        return out

    # ── tick (실시간)
    def on_tick_sweep_check(self, t_now: Any, wick_high: float, wick_low: float,
                             bar_open: float, bar_close: float,
                             liquidity_levels: List[dict]) -> Dict[str, Any]:
        if self.state not in ("IDLE","STANDBY"):
            return {"action":"none","state":self.state,"reason":"not idle/standby"}
        h,l,o,c = float(wick_high),float(wick_low),float(bar_open),float(bar_close)
        added = []
        for lv in liquidity_levels:
            if lv.get("flipped",False): continue
            zlow,zhigh = float(lv["zone_low"]),float(lv["zone_high"])
            new_dir = None
            if h>zhigh: new_dir="short"
            elif l<zlow: new_dir="long"
            if new_dir is None: continue
            already = any(s.sweep_dir==new_dir and
                          abs((s.level["zone_low"]+s.level["zone_high"])/2-(zlow+zhigh)/2)<500
                          for s in self._standbys)
            if not already:
                slot = _StandbySlot(new_dir,lv,h if new_dir=="short" else l,h,l,o,c,
                                    sweep_time=pd.Timestamp(t_now))
                self._standbys.append(slot); added.append(new_dir)
        if added:
            self.state = "STANDBY"
            return {"action":"standby","state":self.state,"direction":",".join(added),"reason":"sweep"}
        return {"action":"none","state":self.state,"reason":"no sweep"}

    # ── 5m close 메인
    def step_on_5m_close(self, bar_5m: Bar, liquidity_levels: List[dict],
                          df_5m_struct: pd.DataFrame, fvgs_5m: pd.DataFrame,
                          obs_5m: pd.DataFrame,
                          htf_state: str = "unknown",
                          tp_levels: Optional[List[dict]] = None,
                          in_position: bool = False, position_closed: bool = False,
                          orders_filled: bool = False,
                          df_5m: Optional[pd.DataFrame] = None,
                          df_1h: Optional[pd.DataFrame] = None,
                          htf_4h: str = "unknown",
                          pivots_15m: Optional[pd.DataFrame] = None,
                          df_15m_struct: Optional[pd.DataFrame] = None,
                          exit_reason: str = "",
                          ) -> Dict[str, Any]:
        notify: List[str] = []
        t = _as_ts(_get(bar_5m,"time") or _get(bar_5m,"t"))
        h = float(_get(bar_5m,"high"))
        l = float(_get(bar_5m,"low"))
        c = float(_get(bar_5m,"close"))
        o = float(_get(bar_5m,"open"))

        liq_no_flip = [lv for lv in liquidity_levels if not lv.get("flipped",False)]
        tp_src = tp_levels if tp_levels is not None else liquidity_levels
        levels_for_tp = list(tp_src)

        # Multi-Sweep History: 반대 sweep 쿨다운 tick
        self._sweep_history.tick_cooldown()

        def _any_since(col: str, n: int) -> bool:
            if col not in df_5m_struct.columns: return False
            return bool(df_5m_struct[col].iloc[-max(1,n):].any())
        def _first_close_since(col: str, n: int) -> Optional[float]:
            if col not in df_5m_struct.columns: return None
            tail = df_5m_struct.iloc[-max(1,n):]
            hit = tail[tail[col]==True]
            if hit.empty or "close" not in hit.columns: return None
            return float(hit.iloc[-1]["close"])

        htf_1h = htf_state  # 호환성

        # ── IN_POSITION: 멀티 포지션 모드에서는 차단하지 않음
        # entry_plan이 체결되면 strategy를 reset하여 새 셋업 감지 재개
        # (backtest.py에서 can_open으로 동시진입 필터)

        # ── ORDERS_PLANNED
        if self.state == "ORDERS_PLANNED":
            self.plan_bar_count += 1
            if orders_filled:
                # 체결 완료 → reset하여 새 셋업 감지 재개
                self.reset()
                return {"action":"filled_and_reset","state":self.state,"entry_plan":None,"notify":notify}
            if self.plan_bar_count >= self.orders_ttl_bars:
                plan = self.entry_plan; self.reset()
                return {"action":"cancel_unfilled","state":self.state,"entry_plan":plan,"notify":notify}
            return {"action":"none","state":self.state,"entry_plan":self.entry_plan,"notify":notify}

        # ─────────────────────────
        # 공통 sweep 감지 헬퍼
        # ─────────────────────────
        def _detect_sweeps_and_add_slots(cur_df_len: int):
            # ATR for sweep depth scoring
            _atr_sweep = _calc_atr_5m(df_5m) if df_5m is not None else 0.0

            bar_sweeps: dict = {}
            for lv in liq_no_flip:
                # 쿨다운 중 같은 레벨은 아래 RE-ENTRY 로직에서 depth 비교 후 결정
                zlow,zhigh = float(lv["zone_low"]),float(lv["zone_high"])
                lv_type = lv.get("level_type","")
                zone_mid = (zlow+zhigh)/2
                if abs(zone_mid-c)/c > 0.04: continue
                new_dir = None
                if h>zhigh: new_dir="short"
                elif l<zlow: new_dir="long"
                if new_dir is None: continue
                if not _level_dir_valid(lv_type, new_dir): continue
                if df_5m is not None:
                    if not _vol_is_valid(df_5m, cur_df_len-1): continue

                # ── sweep 깊이 ATR 스코어링
                # sweep_depth = wick이 레벨을 넘은 거리
                if new_dir == "short":
                    sweep_depth = h - zhigh
                else:
                    sweep_depth = zlow - l

                sweep_depth_n = sweep_depth / _atr_sweep if _atr_sweep > 0 else 0.0

                # ML 전환: skip 제거, 수치만 기록 (ML이 판단)
                _sweep_score = sweep_depth_n  # 원시 ATR 배수 그대로 전달
                _sweep_wick_r = _sweep_wick_ratio(o, h, l, c, new_dir)  # wick ratio 계산
                if sweep_depth_n < 0.8:
                    # notify.append(
                        # f"SWEEP_SHALLOW_NOTE: {new_dir} [{lv_type}] depth={sweep_depth:.1f} "
                        # f"({sweep_depth_n:.2f}×ATR) — ML이 판단"
                    # )

                    pass
                # ML 전환: Multi-Sweep History — 차단 제거, 피처로 전달
                _sl_hits_count = 0
                _cooldown_left = 0
                if self._cooldown_level is not None and self._cooldown_count < self._cooldown_bars:
                    cl = self._cooldown_level
                    lv_mid_chk = (zlow+zhigh)/2
                    cl_mid_chk = (float(cl["zone_low"])+float(cl["zone_high"]))/2
                    is_same_lv = (lv_type == cl.get("level_type","") and
                                  abs(lv_mid_chk-cl_mid_chk)/max(cl_mid_chk,1) < 0.005)
                    if is_same_lv:
                        # 반대 방향 sweep 감지 (이력 기록은 유지)
                        _last_dir = cl.get("_last_sweep_dir", "")
                        if _last_dir and new_dir != _last_dir:
                            self._sweep_history.mark_opposite_sweep(lv, _sweep_score)
                        # 피처 수집
                        recs = self._sweep_history.get(lv)
                        _sl_hits_count = sum(1 for r in recs if r.sl_hit)
                        _cooldown_left = max(0, self._cooldown_bars - self._cooldown_count)
                        # notify.append(
                            # f"RE-ENTRY_NOTE: [{lv_type}] sl_hits={_sl_hits_count} cooldown_left={_cooldown_left} — ML이 판단"
                        # )
                        self._cooldown_level = None
                        self._cooldown_count = 0

                strength = float(lv.get("strength",1.0))
                # liquidity 피처 계산
                _liq_feats = _calc_liquidity_features(
                    lv=lv, sweep_depth_atr=_sweep_score, sweep_ratio=_sweep_wick_r,
                    all_levels=liq_no_flip, atr=_atr_sweep,
                    current_price=c, current_bar_idx=cur_df_len - 1,
                    sweep_history=self._sweep_history,
                    current_time=t,
                )
                if new_dir not in bar_sweeps or strength > bar_sweeps[new_dir][0]:
                    bar_sweeps[new_dir] = (strength, lv, _sweep_score, _sl_hits_count, _cooldown_left, _liq_feats)
            for new_dir,(strength,lv,_sw_score,_sl_hits,_cd_left,_liq_f) in bar_sweeps.items():
                zlow=float(lv["zone_low"]); zhigh=float(lv["zone_high"])
                lv_type=lv.get("level_type","")
                lv_mid_new = (zlow+zhigh)/2
                # 같은 방향+같은 레벨타입+가격 0.5% 이내 기존 슬롯 찾기
                existing_slot = None
                for s in self._standbys:
                    s_type = s.level.get("level_type","")
                    if s.sweep_dir == new_dir and s_type == lv_type:
                        s_mid = (float(s.level["zone_low"])+float(s.level["zone_high"]))/2
                        if abs(s_mid - lv_mid_new) / max(lv_mid_new, 1) < 0.005:
                            existing_slot = s
                            break

                new_extreme = h if new_dir == "short" else l
                if existing_slot is not None:
                    # 더 깊은 sweep이면 기존 슬롯 업데이트
                    old_extreme = existing_slot.sweep_extreme
                    deeper = (new_dir == "short" and new_extreme > old_extreme) or \
                             (new_dir == "long"  and new_extreme < old_extreme)
                    if deeper:
                        existing_slot.sweep_extreme_chain.append(new_extreme)
                        existing_slot.sweep_extreme = new_extreme
                        existing_slot.sweep_depth_score = _sw_score
                        existing_slot.bar_count = 0  # 타이머 리셋
                        existing_slot.sweep_bar_idx = max(0, cur_df_len - 1)
                        # 전역 sweep 큐: 더 깊은 sweep 갱신도 sweep 이벤트
                        self._sweep_queue.append({
                            "time": pd.Timestamp(t), "dir": new_dir,
                            "extreme": new_extreme,
                            "level_mid": (zlow + zhigh) / 2.0,
                        })
                        # notify.append(
                            # f"STANDBY_UPDATE: {sw} deeper sweep [{ex_type}]  "
                            # f"time={t}  extreme={new_extreme:,.1f}  depth={_sw_score:.1f}×"
                        # )
                else:
                    slot = _StandbySlot(new_dir,lv,new_extreme,
                                        h,l,o,c,df_len=cur_df_len,
                                        sweep_depth_score=_sw_score,
                                        sweep_time=t,
                                        sl_hits_count=_sl_hits,
                                        cooldown_left=_cd_left,
                                        liq_features=_liq_f)
                    self._standbys.append(slot)
                    # Multi-Sweep History: 기록 추가
                    self._sweep_history.add(lv, _SweepRecord(
                        timestamp=t, depth=_sw_score,
                        sweep_extreme=new_extreme,
                        disp_net_move=0.0,
                        direction=new_dir
                    ))
                    # 전역 sweep 큐: 셋업 성사 무관 모든 sweep 발견 시간순 기록
                    # (반대편 유동성 선행회수 추적). zlow/zhigh는 이 스코프 정의됨.
                    self._sweep_queue.append({
                        "time": pd.Timestamp(t), "dir": new_dir,
                        "extreme": new_extreme,
                        "level_mid": (zlow + zhigh) / 2.0,
                    })
                    # notify.append(
                        # f"STANDBY+: {sw} sweep [{lv_type}]  "
                        # f"time={t}  extreme={new_extreme:,.1f}  "
                        # f"depth={_sw_score:.1f}×  (총{len(self._standbys)}개)"
                    # )

        # ── IDLE
        if self.state == "IDLE":
            if self._cooldown_level is not None:
                self._cooldown_count += 1
                if self._cooldown_count >= self._cooldown_bars:
                    self._cooldown_level = None; self._cooldown_count = 0
            # ML 전환: 세션 차단 제거 — 시간 피처는 feature_dict에서 전달
            cur_df_len = len(df_5m) if df_5m is not None else 0
            _detect_sweeps_and_add_slots(cur_df_len)
            if self._standbys:
                self.state = "STANDBY"
                return {"action":"standby","state":self.state,
                        "direction":",".join(s.sweep_dir for s in self._standbys),
                        "entry_plan":None,"notify":notify}
            return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

        # ── STANDBY
        if self.state == "STANDBY":
            # ML 전환: 세션 차단 제거
            cur_df_len2 = len(df_5m) if df_5m is not None else 0
            _detect_sweeps_and_add_slots(cur_df_len2)

            armed_slot = None
            expired = []
            # 동적 timeout 조정 계산 (봉마다 1회)
            _sb_adj_cache = {}
            for slot in self._standbys:
                if slot.swept_this_bar:
                    slot.swept_this_bar = False; continue
                slot.bar_count += 1

                # 동적 timeout: HTF + vol 기반 조정
                _adj_key = slot.sweep_dir
                if _adj_key not in _sb_adj_cache:
                    _sb_a, _ = _calc_timeout_adjust(
                        slot.sweep_dir, htf_1h, htf_4h, df_5m)
                    _sb_adj_cache[_adj_key] = _sb_a
                _dyn_timeout = max(6, slot.timeout_bars + _sb_adj_cache[_adj_key])

                if slot.bar_count > _dyn_timeout:
                    expired.append(slot)
                    # timeout된 레벨 → backtest가 liq_pool에서 삭제
                    if slot.level is not None:
                        self._pending_disables.append({
                            "level_type":   slot.level.get("level_type", ""),
                            "price_center": float((slot.level["zone_low"] + slot.level["zone_high"]) / 2),
                        })
                    # notify.append(f"STANDBY timeout({_dyn_timeout}봉) [{slot.sweep_dir}/{lv_type_s_to}] → 유동성 삭제")
                    continue
                slot.update_ratio(h,l,c)
                zlow_s = float(slot.level["zone_low"]); zhigh_s = float(slot.level["zone_high"])
                _buf = c*0.001
                reclaimed = (c<=zhigh_s+_buf) if slot.sweep_dir=="short" else (c>=zlow_s-_buf)
                if reclaimed:
                    if armed_slot is None:
                        slot.reclaim_bar_idx = len(df_5m)-1 if df_5m is not None else 0
                        armed_slot = slot

            for s in expired:
                if s in self._standbys: self._standbys.remove(s)

            if armed_slot is not None:
                self._standbys.remove(armed_slot)
                self.sweep_dir     = armed_slot.sweep_dir
                self.level         = armed_slot.level
                self.sweep_extreme = armed_slot.sweep_extreme
                self.sweep_ratio   = armed_slot.sweep_ratio
                self.sweep_bar_idx = armed_slot.sweep_bar_idx
                self._sweep_depth_score = armed_slot.sweep_depth_score
                # 연속 sweep SL 체인: 가장 먼 extreme을 SL용으로 저장
                self._sweep_extreme_chain = armed_slot.sweep_extreme_chain
                if len(self._sweep_extreme_chain) > 1:
                    if self.sweep_dir == "long":
                        self.sweep_extreme = min(self._sweep_extreme_chain)  # 가장 낮은 = 가장 먼 SL
                    else:
                        self.sweep_extreme = max(self._sweep_extreme_chain)  # 가장 높은 = 가장 먼 SL
                self.state = "ARMED"; self._armed_bar_count = 0
                self._armed_df_len = len(df_5m_struct)
                self._armed_start_price = c  # reclaim 시점 종가 기록
                self.trigger_time = armed_slot.sweep_time  # sweep 시점으로 설정 (FVG 시간 필터용)
                self.sweep_time   = armed_slot.sweep_time  # sweep 발생 시각 (피처 계산용)
                # 반대편 유동성 선행회수 조회 (entry_plan emit용)
                self._opp_sweep_feats = self._query_opposite_sweep(
                    setup_dir=self.sweep_dir, setup_time=self.sweep_time,
                    setup_extreme=self.sweep_extreme, df_5m=df_5m_struct)
                self.run_high = armed_slot.run_high
                self.run_low  = armed_slot.run_low
                self._ob_wait_count = 0  # FVG 대기 카운터 초기화
                # liquidity 피처 임시 저장 (entry_plan에 포함용)
                self._liq_features = armed_slot.liq_features
                self._sl_hits_count_armed = armed_slot.sl_hits_count
                self._cooldown_left_armed = armed_slot.cooldown_left
                # notify.append(f"ARMED: reclaim✓  {htf_note}  [{lv_type_arm}]{remaining}")

                return {"action":"armed","state":self.state,"direction":self.sweep_dir,"entry_plan":None,"notify":notify}


            if not self._standbys:
                self.state = "IDLE"
                return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}
            return {"action":"none","state":self.state,"direction":",".join(s.sweep_dir for s in self._standbys),"entry_plan":None,"notify":notify}

        # ── ARMED
        if self.state == "ARMED":
            self._armed_bar_count += 1

            # 동적 ARMED timeout: HTF + vol 기반 조정 + Group A 추가 6봉
            _, _ar_adj = _calc_timeout_adjust(
                self.sweep_dir, htf_1h, htf_4h, df_5m)
            lv_type_armed = self.level.get("level_type","") if self.level else ""
            is_group_a = lv_type_armed in LEVEL_GROUP_A
            _group_a_bonus = 5 if is_group_a else 0
            _dyn_armed_timeout = max(6, self.armed_timeout_bars + _ar_adj + _group_a_bonus)

            if self._armed_bar_count > _dyn_armed_timeout:
                # notify.append(f"ARMED timeout({_dyn_armed_timeout}봉) → reset")
                self.reset()
                return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

            sweep_extreme_armed = float(self.sweep_extreme)

            # ── 공통 displacement 감지 (Group A/B 통합)
            # 극값 추적
            if self.sweep_dir == "long":
                self.run_high = max(self.run_high or h, h)
            else:
                self.run_low = min(self.run_low or l, l)

            disp_ext = float(self.run_high if self.sweep_dir == "long" else self.run_low)

            # (1) Net move (피처용 — 차단 안 함)
            atr = _calc_atr_5m(df_5m) if df_5m is not None else 0.0
            net_move = abs(disp_ext - sweep_extreme_armed)

            # (2) FVG 탐색: sweep_extreme 시점부터 현재까지 (하드 필터)

            fvg_now = None
            if df_5m_struct is not None and len(df_5m_struct) >= 3:
                # sweep_bar_idx 이후부터 탐색
                _search_start = max(2, len(df_5m_struct) - self._armed_bar_count - 1)
                _search_start = max(2, _search_start)  # 최소 인덱스 2 (i-2 참조 필요)

                _fvg_candidates = []
                for _fi in range(_search_start, len(df_5m_struct)):
                    _c_bar = df_5m_struct.iloc[_fi]
                    _p2_bar = df_5m_struct.iloc[_fi - 2]

                    if self.sweep_dir == "long":
                        _fl = float(_p2_bar["high"]); _fh = float(_c_bar["low"])
                        if _fh > _fl:
                            _edge = _fh  # long: 진입 edge = FVG upper
                            # edge가 유동성 zone 안에 있어야 함 (sweep_extreme ~ lv_zh)
                            disp_extreme_a = float(self.run_high)
                            if sweep_extreme_armed <= _edge <= disp_extreme_a:
                                _fvg_candidates.append({
                                    "zone_low": _fl, "zone_high": _fh,
                                    "mid": (_fl+_fh)/2, "entry_price": _fh,
                                    "size": _fh-_fl, "in_ote": False,
                                    "bar_idx": _fi
                                })
                    else:
                        _fh2 = float(_p2_bar["low"]); _fl2 = float(_c_bar["high"])
                        if _fl2 < _fh2:
                            _edge = _fl2  # short: 진입 edge = FVG lower
                            # edge가 유동성 zone 안에 있어야 함 (lv_zl ~ sweep_extreme)
                            disp_extreme_a = float(self.run_low)
                            if disp_extreme_a <= _edge <= sweep_extreme_armed:
                                _fvg_candidates.append({
                                    "zone_low": _fl2, "zone_high": _fh2,
                                    "mid": (_fl2+_fh2)/2, "entry_price": _fl2,
                                    "size": _fh2-_fl2, "in_ote": False,
                                    "bar_idx": _fi
                                })

                if _fvg_candidates:
                    fvg_now = max(_fvg_candidates, key=lambda f: f["size"])

            # ── 판정: ML 전환
            # 하드 필터: FVG 발생 = displacement 증명
            # net_move_ok, candle_ok는 피처로만 전달 (차단 안 함)
            _net_move_atr_ratio = net_move / atr if atr > 0 else 0.0

            # ════════════════════════════════════════════════════════════
            # 신규 feature 블록 (건강함 측정)
            # ════════════════════════════════════════════════════════════
            # (A) net_move ATR 재정규화 변형 — sweet spot 흩어짐의 ATR 원인 검증용
            _atr50 = _calc_atr_5m(df_5m, lookback=50) if df_5m is not None else 0.0
            _atr_wilder = _calc_atr_wilder(df_5m, 14) if df_5m is not None else 0.0
            _ma_atr_1h = _calc_1h_ma_atr(df_1h)
            _atr_1h = _ma_atr_1h["atr_1h"] if _ma_atr_1h else 0.0
            _net_move_atr50 = net_move / _atr50 if _atr50 > 0 else 0.0
            _net_move_atr_wilder = net_move / _atr_wilder if _atr_wilder > 0 else 0.0
            _net_move_atr1h = net_move / _atr_1h if _atr_1h > 0 else 0.0

            # (2) reclaim_quality — 거부 강도 / 체류 페널티
            #   reclaim_move = |현재종가 - sweep_extreme(가장 먼 극점)|
            #   self.sweep_extreme 은 이미 체인 최댓/최솟값(가장 먼 SL)으로 설정됨 (line 1668)
            _cur_close = float(c)
            _reclaim_move = abs(_cur_close - float(self.sweep_extreme))
            _reclaim_move_atr = _reclaim_move / atr if atr > 0 else 0.0
            # sweep_depth_atr 은 아래에서 채워지지만, 여기선 _sweep_depth_score 사용 (ATR정규화 깊이)
            _sw_depth_atr = float(self._sweep_depth_score) if self._sweep_depth_score else 0.0
            _bars_since_sweep = max(0, (len(df_5m_struct) - 1 - self.sweep_bar_idx)
                                    if df_5m_struct is not None else 0)
            if _sw_depth_atr > 1e-9:
                _reclaim_ratio = _reclaim_move_atr / _sw_depth_atr
            else:
                _reclaim_ratio = 0.0
            _reclaim_quality = _reclaim_ratio / (np.sqrt(_bars_since_sweep) + 1.0)

            # candle body sum 최대값 계산 (피처용)
            _max_body_sum_atr = 0.0
            if df_5m_struct is not None and len(df_5m_struct) >= 1 and atr > 0:
                _n_since_armed = len(df_5m_struct) - self._armed_df_len + 2
                _n_since_armed = max(1, min(_n_since_armed, len(df_5m_struct)))
                _armed_tail = df_5m_struct.iloc[-_n_since_armed:]
                for _win in (1, 2, 3):
                    if len(_armed_tail) < _win: continue
                    for _start in range(len(_armed_tail) - _win + 1):
                        _window = _armed_tail.iloc[_start:_start + _win]
                        _body_sum = sum(abs(float(r["close"]) - float(r["open"]))
                                        for _, r in _window.iterrows())
                        _max_body_sum_atr = max(_max_body_sum_atr, _body_sum / atr)

            has_fvg = fvg_now is not None


            # Multi-Sweep History: displacement net_move 기록
            if has_fvg and self.level is not None:
                recs = self._sweep_history.get(self.level)
                if recs:
                    recs[-1].disp_net_move = net_move

            if has_fvg:
                # 하드 필터 충족: FVG 존재 = displacement 증명 → DISP_RUNNING
                self.trigger_type = "displacement"
                self.choch_price  = c
                self.state        = "DISP_RUNNING"
                self.trigger_time = t
                self._disp_trigger_time = t
                self._ob_wait_count = 0
                self._disp_flat_count = 0

                # ── choch / displacement 피처 계산
                # choch_break_atr_ratio: disp_extreme이 유동성 레벨(self.level)을
                #   얼마나 넘어섰는지 / ATR  (= CHoCH 돌파 깊이)
                #   long:  sweep이 level 하단을 깸 → displacement가 level 상단을 돌파
                #          choch_depth = disp_ext(고점) - level_zone_high
                #   short: sweep이 level 상단을 깸 → displacement가 level 하단을 돌파
                #          choch_depth = level_zone_low - disp_ext(저점)
                #   net_move(전체 이동 거리)와 다름: 레벨을 넘어선 '초과분'만 측정
                if self.level and self.sweep_dir == "long":
                    _level_ref = float(self.level.get("zone_high", sweep_extreme_armed))
                    _choch_break_depth = max(0.0, disp_ext - _level_ref)
                elif self.level and self.sweep_dir == "short":
                    _level_ref = float(self.level.get("zone_low", sweep_extreme_armed))
                    _choch_break_depth = max(0.0, _level_ref - disp_ext)
                else:
                    _choch_break_depth = 0.0
                _choch_break_atr_ratio = round(_choch_break_depth / atr, 3) if atr > 0 else 0.0

                # choch_is_close_beyond: 마지막 봉 close가 sweep_extreme을 0.15 ATR 이상 넘어섰는지
                _choch_margin = atr * 0.15 if atr > 0 else 0.0
                if self.sweep_dir == "long":
                    _choch_is_close_beyond = 1 if c > (sweep_extreme_armed - _choch_margin) else 0
                else:
                    _choch_is_close_beyond = 1 if c < (sweep_extreme_armed + _choch_margin) else 0

                # displacement_body_wick_ratio: displacement 구간 body합 / wick합
                _disp_body_sum = 0.0
                _disp_wick_sum = 0.0
                if df_5m_struct is not None and len(df_5m_struct) >= 1:
                    _n_bars = len(df_5m_struct) - self._armed_df_len + 2
                    _n_bars = max(1, min(_n_bars, len(df_5m_struct)))
                    _disp_tail = df_5m_struct.iloc[-_n_bars:]
                    for _, _db in _disp_tail.iterrows():
                        _d_o = float(_db["open"]); _d_h = float(_db["high"])
                        _d_l = float(_db["low"]);  _d_c = float(_db["close"])
                        _d_body = abs(_d_c - _d_o)
                        _d_range = _d_h - _d_l
                        _d_wick = max(0.0, _d_range - _d_body)
                        _disp_body_sum += _d_body
                        _disp_wick_sum += _d_wick
                _disp_body_wick_ratio = round(
                    min(_disp_body_sum / _disp_wick_sum, 10.0) if _disp_wick_sum > 0 else 10.0, 3
                )

                # displacement_consistency: 방향 일치 봉 비율 (0~1)
                #   long: close > open인 봉 / 전체 displacement 봉
                #   short: close < open인 봉 / 전체 displacement 봉
                _disp_consistency = 0.0
                if df_5m_struct is not None and len(df_5m_struct) >= 1:
                    _dc_n = len(df_5m_struct) - self._armed_df_len + 2
                    _dc_n = max(1, min(_dc_n, len(df_5m_struct)))
                    _dc_tail = df_5m_struct.iloc[-_dc_n:]
                    if len(_dc_tail) > 0:
                        if self.sweep_dir == "long":
                            _dc_match = sum(1 for _, _r in _dc_tail.iterrows()
                                            if float(_r["close"]) > float(_r["open"]))
                        else:
                            _dc_match = sum(1 for _, _r in _dc_tail.iterrows()
                                            if float(_r["close"]) < float(_r["open"]))
                        _disp_consistency = round(_dc_match / len(_dc_tail), 3)

                # strong_displacement_flag: 종합 강도

                # ── Reclaim 피처 계산 (sweep → displacement 구간)
                _reclaim_vol_spike = 0.0
                _reclaim_speed = 0.0
                _reclaim_vol_profile = 0.0

                if df_5m_struct is not None and len(df_5m_struct) >= 2:
                    _reclaim_n = len(df_5m_struct) - self._armed_df_len + 2
                    _reclaim_n = max(1, min(_reclaim_n, len(df_5m_struct)))
                    _reclaim_bars = df_5m_struct.iloc[-_reclaim_n:]
                    _reclaim_vols = _reclaim_bars["volume"].astype(float).values

                    # 1) reclaim_vol_spike: 되돌림 평균 거래량 / 직전 동일 길이 baseline
                    _baseline_start = max(0, len(df_5m_struct) - _reclaim_n * 2)
                    _baseline_end = len(df_5m_struct) - _reclaim_n
                    if _baseline_end > _baseline_start:
                        _baseline_bars = df_5m_struct.iloc[_baseline_start:_baseline_end]
                        _baseline_vol = float(_baseline_bars["volume"].astype(float).mean())
                        if _baseline_vol > 0:
                            _reclaim_avg_vol = float(_reclaim_vols.mean())
                            _reclaim_vol_spike = round(min(10.0, _reclaim_avg_vol / _baseline_vol), 3)

                    # 2) reclaim_speed: 되돌림 가격 거리(ATR 정규화) / 봉 수
                    if _reclaim_n > 0 and atr > 0:
                        _reclaim_price_dist = abs(
                            float(_reclaim_bars.iloc[-1]["close"]) - float(_reclaim_bars.iloc[0]["close"])
                        )
                        _reclaim_speed = round(min(5.0, (_reclaim_price_dist / atr) / _reclaim_n), 3)

                    # 3) reclaim_vol_profile: 최대 거래량 봉의 비중 (단일 스파이크 감지)
                    if len(_reclaim_vols) > 0:
                        _total_vol = float(_reclaim_vols.sum())
                        if _total_vol > 0:
                            _max_vol = float(_reclaim_vols.max())
                            _reclaim_vol_profile = round(min(1.0, _max_vol / _total_vol), 3)

                # ════════════════════════════════════════════════════════
                # 신규 feature (2nd 배치): absorption, trend_structure, disp_quality_15m
                # ════════════════════════════════════════════════════════
                # (3) reclaim_absorption: 강한 vol + 느린 이동(흡수) = 강한 반대주문
                #     vol_spike 클수록 / speed 작을수록(천천히=흡수) absorption 큼
                _reclaim_absorption = _reclaim_vol_spike / (_reclaim_speed + 0.1)

                # (4) trend_structure + ma_alignment (1h MA 구조, adx 대체)
                _trend_structure = 0.0
                _ma_alignment = 0.0
                if _ma_atr_1h is not None and _ma_atr_1h["atr_1h"] > 1e-9:
                    _m = _ma_atr_1h
                    _s1 = (_m["ma5"] - _m["ma20"]) / _m["atr_1h"]
                    _s2 = (_m["ma20"] - _m["ma60"]) / _m["atr_1h"]
                    _trend_structure = (_s1 + _s2) / 2.0
                    # 배열 정렬도: 완전정렬 +1 / 역정렬 -1 / 엉킴 0
                    if _m["ma5"] > _m["ma20"] > _m["ma60"]:
                        _ma_alignment = 1.0
                    elif _m["ma5"] < _m["ma20"] < _m["ma60"]:
                        _ma_alignment = -1.0
                    # 방향 보정: short면 "내 방향과 일치하는 추세"가 +가 되도록 반전
                    if self.sweep_dir == "short":
                        _trend_structure = -_trend_structure
                        _ma_alignment = -_ma_alignment

                # (1) disp_quality_15m: sweep~disp 구간 15m 봉 품질 (ATR 무관)
                _disp_quality_15m = _calc_disp_quality_15m(
                    df_15m_struct, self.sweep_time, None, self.sweep_dir, min_bars=2)
                if _disp_quality_15m is None:
                    _disp_quality_15m = np.nan  # NaN sentinel (XGBoost native)

                # 피처 임시 저장 (DISP_RUNNING에서 entry_plan에 포함)
                self._armed_features = {
                    "net_move_atr_ratio": round(_net_move_atr_ratio, 3),
                    "net_move_atr50": round(_net_move_atr50, 3),
                    "net_move_atr_wilder": round(_net_move_atr_wilder, 3),
                    "net_move_atr1h": round(_net_move_atr1h, 3),
                    "candle_body_sum_atr": round(_max_body_sum_atr, 3),
                    "bars_since_sweep": self._armed_bar_count,
                    "choch_break_atr_ratio": _choch_break_atr_ratio,
                    "choch_is_close_beyond": _choch_is_close_beyond,
                    "displacement_body_wick_ratio": _disp_body_wick_ratio,
                    "displacement_consistency": _disp_consistency,
                    "reclaim_vol_spike": _reclaim_vol_spike,
                    "reclaim_speed": _reclaim_speed,
                    "reclaim_vol_profile": _reclaim_vol_profile,
                    "reclaim_quality": round(_reclaim_quality, 4),
                    "reclaim_absorption": round(_reclaim_absorption, 4),
                    "trend_structure": round(_trend_structure, 4),
                    "ma_alignment": _ma_alignment,
                    "disp_quality_15m": (round(_disp_quality_15m, 4)
                                         if _disp_quality_15m == _disp_quality_15m else np.nan),
                }
                # notify.append(
                    # f"DISP_OK({_grp_label}/{lv_type_armed}): net_atr={_net_move_atr_ratio:.2f} "
                    # f"body_atr={_max_body_sum_atr:.2f}  {_fvg_note} → DISP_RUNNING"
                # )
                return {"action":"disp_running","state":self.state,
                        "direction":self.sweep_dir,"entry_plan":None,"notify":notify}

            # FVG 미발견 → ARMED 유지 (timeout까지 대기)
            return {"action":"none","state":self.state,
                    "direction":self.sweep_dir,"entry_plan":None,"notify":notify}

        # ── DISP_RUNNING
        if self.state == "DISP_RUNNING":
            lv_type_disp  = self.level.get("level_type", "") if self.level else ""
            is_group_a_disp = lv_type_disp in LEVEL_GROUP_A
            sweep_extreme = float(self.sweep_extreme)

            # ══════════════════════════════════════════
            # Group A: 최소 4봉 ~ 최대 8봉 FVG 탐색
            # ══════════════════════════════════════════
            if is_group_a_disp:
                disp_extreme_a = float(self.run_high if self.sweep_dir == "long" else self.run_low)

                # ML 전환: FVG_WAIT timeout 제거 — 봉 수만 카운트
                self._ob_wait_count += 1

                # ── FVG 탐색: sweep_bar_idx부터 현재까지, edge가 유동성 zone 안
                _fvg_disp = None
                if df_5m_struct is not None and len(df_5m_struct) >= 3:
                    _s_start = max(2, len(df_5m_struct) - self._armed_bar_count - self._ob_wait_count - 1)
                    _s_start = max(2, _s_start)
                    _fvg_cands = []
                    for _fi in range(_s_start, len(df_5m_struct)):
                        _c_bar = df_5m_struct.iloc[_fi]
                        _p2_bar = df_5m_struct.iloc[_fi - 2]
                        if self.sweep_dir == "long":
                            _fl = float(_p2_bar["high"]); _fh = float(_c_bar["low"])
                            if _fh > _fl:
                                _edge = _fh
                                if sweep_extreme <= _edge <= disp_extreme_a:
                                    _fvg_cands.append({"zone_low":_fl,"zone_high":_fh,
                                        "mid":(_fl+_fh)/2,"entry_price":_fh,
                                        "size":_fh-_fl,"in_ote":False,"bar_idx":_fi})
                        else:
                            _fh2 = float(_p2_bar["low"]); _fl2 = float(_c_bar["high"])
                            if _fl2 < _fh2:
                                _edge = _fl2
                                if disp_extreme_a <= _edge <= sweep_extreme:
                                    _fvg_cands.append({"zone_low":_fl2,"zone_high":_fh2,
                                        "mid":(_fl2+_fh2)/2,"entry_price":_fl2,
                                        "size":_fh2-_fl2,"in_ote":False,"bar_idx":_fi})

                    if _fvg_cands:
                        _fvg_disp = max(_fvg_cands, key=lambda f: f["size"])

                # fallback: 기존 FVG 풀
                _atr_disp = _calc_atr_5m(df_5m) if df_5m is not None else 0.0
                _net_move_a = abs(disp_extreme_a - sweep_extreme)
                if _fvg_disp is None:
                    _fvg_trigger_time = self.trigger_time or t
                    _fvg_disp = _pick_fvg_group_a(
                        fvgs_5m, self.sweep_dir, _fvg_trigger_time,
                        sweep_extreme, disp_extreme_a, None,
                        atr=_atr_disp, net_move=_net_move_a
                    )

                # 하드 필터: FVG 존재 필수 (displacement 증명)
                if _fvg_disp is None:
                    # notify.append(f"FVG_WAIT(A): {self._ob_wait_count}봉  FVG 미발견 — 대기")
                    # 안전장치: 최대 20봉까지만 대기 (과도한 대기 방지)
                    if self._ob_wait_count > 20:
                        # notify.append(f"FVG_WAIT_HARD_LIMIT(A): 20봉 초과 → reset")
                        self.reset()
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

                # ── FVG size 피처

                # ── 진입가 계산: (mid + edge) / 2
                _edge = _fvg_disp["zone_high"] if self.sweep_dir == "long" else _fvg_disp["zone_low"]
                entry_px = (_fvg_disp["mid"] + _edge) / 2.0
                sl = _calc_sl_group_a(self.sweep_dir, sweep_extreme, entry_px, self.sl_buffer_pct)

                if (self.sweep_dir == "long" and sl >= entry_px) or (self.sweep_dir == "short" and sl <= entry_px):
                    # notify.append(f"SL_ERR(A): sl={sl:,.2f} ref={entry_px:,.2f} → reset"); self.reset()
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

                _tp_src = tp_levels if tp_levels is not None else liquidity_levels
                tp = _find_next_liquidity_target(list(_tp_src), entry_px, self.sweep_dir, sl)
                if tp is None:
                    tp = _find_tp_fallback(fvgs_5m, obs_5m, entry_px, self.sweep_dir, sl, pivots_15m=pivots_15m)

                # ── FVG_WATCHING 전환 (체결 대기가 아니라 mitigation 감시)
                _fvg_disp["departed"] = False
                _fvg_disp["edge"] = _fvg_disp["zone_high"] if self.sweep_dir == "long" else _fvg_disp["zone_low"]
                self._watching_fvg      = _fvg_disp
                self._watching_entry_px = entry_px
                self._watching_sl       = sl
                self._watching_tp       = tp
                self._watching_bars     = 0
                self._watching_disp_extreme = disp_extreme_a
                self._watching_group    = "A"
                self._fvg_pool          = [_fvg_disp]  # pool 초기화 + 첫 FVG
                self.state = "FVG_WATCHING"

                # notify.append(
                    # f"FVG_WATCHING [{lv_type_disp.upper()}|A]  "
                    # f"FVG={_fvg_disp['zone_low']:,.0f}~{_fvg_disp['zone_high']:,.0f}  "
                    # f"entry={(entry_px):,.0f}  SL={sl:,.0f}  TP={tp_str}"
                # )
                return {"action":"fvg_watching","state":self.state,
                        "direction":self.sweep_dir,"entry_plan":None,"notify":notify}

            # ══════════════════════════════════════════
            # Group B: FVG 우선 + OB fallback (ML 전환)
            # ══════════════════════════════════════════
            else:
                disp_extreme = float(self.run_high if self.sweep_dir == "long" else self.run_low)

                # ML 전환: timeout 제거 — 봉 수만 카운트
                self._ob_wait_count += 1
                # 안전장치: 최대 30봉까지만 대기
                if self._ob_wait_count > 30:
                    # notify.append(f"FVG_WAIT_HARD_LIMIT(B): 30봉 초과 → reset")
                    self.reset()
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

                # ML 전환: size_multiplier = 1.0 (ML이 결정)

                _fvg_trigger_time_b = self.trigger_time or t
                ob_type = "bull_ob" if self.sweep_dir == "long" else "bear_ob"

                # ── 1순위: FVG 탐색 (sweep_bar_idx부터, edge가 유동성 zone 안)
                _fvg_b = None
                if df_5m_struct is not None and len(df_5m_struct) >= 3:
                    _s_start_b = max(2, len(df_5m_struct) - self._armed_bar_count - self._ob_wait_count - 1)
                    _s_start_b = max(2, _s_start_b)
                    _fvg_cands_b = []
                    for _fi in range(_s_start_b, len(df_5m_struct)):
                        _c_bar = df_5m_struct.iloc[_fi]
                        _p2_bar = df_5m_struct.iloc[_fi - 2]
                        if self.sweep_dir == "long":
                            _fl = float(_p2_bar["high"]); _fh = float(_c_bar["low"])
                            if _fh > _fl:
                                _edge = _fh
                                if sweep_extreme <= _edge <= disp_extreme:
                                    _fvg_cands_b.append({"zone_low":_fl,"zone_high":_fh,
                                        "mid":(_fl+_fh)/2,"entry_price":_fh,
                                        "size":_fh-_fl,"bar_idx":_fi})
                        else:
                            _fh2 = float(_p2_bar["low"]); _fl2 = float(_c_bar["high"])
                            if _fl2 < _fh2:
                                _edge = _fl2
                                if disp_extreme <= _edge <= sweep_extreme:
                                    _fvg_cands_b.append({"zone_low":_fl2,"zone_high":_fh2,
                                        "mid":(_fl2+_fh2)/2,"entry_price":_fl2,
                                        "size":_fh2-_fl2,"bar_idx":_fi})

                    if _fvg_cands_b:
                        _fvg_b = max(_fvg_cands_b, key=lambda f: f["size"])

                # fallback: 기존 FVG 풀
                if _fvg_b is None:
                    _fvg_pool = _pick_widest_fvg(
                        fvgs_5m, self.sweep_dir, _fvg_trigger_time_b,
                        sweep_extreme, disp_extreme
                    )
                    if _fvg_pool:
                        _fvg_b = _fvg_pool

                # ── 2순위: OB fallback (FVG 없을 때)
                ob = None
                if _fvg_b is None:
                    ob = _pick_best_ob(obs_5m, t_now=t, ob_type=ob_type, ote_zone=None)
                    if ob is not None:
                        _ob_mid = (float(ob["zone_low"]) + float(ob["zone_high"])) / 2
                        if c > 0 and abs(_ob_mid - c) / c > 0.02:
                            ob = None
                        elif self.sweep_dir == "short" and _ob_mid < c:
                            ob = None
                        elif self.sweep_dir == "long" and _ob_mid > c:
                            ob = None
                        elif ob is not None:
                            if self.sweep_dir == "long" and _ob_mid < sweep_extreme:
                                ob = None
                            elif self.sweep_dir == "short" and _ob_mid > sweep_extreme:
                                ob = None

                # 둘 다 없으면 대기
                if _fvg_b is None and ob is None:
                    # notify.append(f"FVG/OB_WAIT(B): {self._ob_wait_count}봉 대기 중")
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

                # ── 진입가/오더 결정
                best_fvg = _fvg_b  # FVG 또는 None

                if _fvg_b is not None:
                    _edge_b = _fvg_b["zone_high"] if self.sweep_dir == "long" else _fvg_b["zone_low"]
                    entry_px = (_fvg_b["mid"] + _edge_b) / 2.0
                elif ob is not None:
                    ob_low = float(ob["zone_low"]); ob_high = float(ob["zone_high"])
                    entry_px = ob_high if self.sweep_dir == "long" else ob_low
                    # OB를 FVG-like dict로 래핑 (FVG_WATCHING에서 통일 처리)
                    best_fvg = {
                        "zone_low": ob_low, "zone_high": ob_high,
                        "mid": (ob_low + ob_high) / 2, "size": ob_high - ob_low,
                        "_is_ob": True, "_ob_data": ob,
                    }
                else:
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

                sl = _calc_sl_group_b(self.sweep_dir, sweep_extreme, entry_px, self.sl_buffer_pct)

                if (self.sweep_dir == "long" and sl >= entry_px) or (self.sweep_dir == "short" and sl <= entry_px):
                    # notify.append(f"SL_ERR(B): sl={sl:,.2f} ref={entry_px:,.2f} → reset"); self.reset()
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

                _tp_src2 = tp_levels if tp_levels is not None else liquidity_levels
                tp = _find_next_liquidity_target(list(_tp_src2), entry_px, self.sweep_dir, sl)
                if tp is None:
                    tp = _find_tp_fallback(fvgs_5m, obs_5m, entry_px, self.sweep_dir, sl, pivots_15m=pivots_15m)

                # ── FVG_WATCHING 전환
                best_fvg["departed"] = False
                best_fvg["edge"] = best_fvg["zone_high"] if self.sweep_dir == "long" else best_fvg["zone_low"]
                self._watching_fvg      = best_fvg
                self._watching_entry_px = entry_px
                self._watching_sl       = sl
                self._watching_tp       = tp
                self._watching_bars     = 0
                self._watching_disp_extreme = disp_extreme
                self._watching_group    = "B"
                self._fvg_pool          = [best_fvg]  # pool 초기화 + 첫 FVG
                self.state = "FVG_WATCHING"

                # notify.append(
                    # f"FVG_WATCHING [{lv_type_disp.upper()}|B|{_src}]  "
                    # f"zone={best_fvg['zone_low']:,.0f}~{best_fvg['zone_high']:,.0f}  "
                    # f"entry={entry_px:,.0f}  SL={sl:,.0f}  TP={tp_str}"
                # )
                return {"action":"fvg_watching","state":self.state,
                        "direction":self.sweep_dir,"entry_plan":None,"notify":notify}

        # ── FVG_WATCHING: FVG candidate pool 관리 + mitigation 감시
        if self.state == "FVG_WATCHING":
            self._watching_bars += 1
            _w_atr = _calc_atr_5m(df_5m) if df_5m is not None else 0.0
            sweep_extreme_w = float(self.sweep_extreme)
            lv_type_w = self.level.get("level_type", "") if self.level else ""
            _disp_ext_w = self._watching_disp_extreme

            # ── (A) 새 FVG 탐색 → pool에 추가
            if df_5m_struct is not None and len(df_5m_struct) >= 3:
                _search_start_w = max(2, len(df_5m_struct) - self._armed_bar_count - self._ob_wait_count - self._watching_bars - 1)
                _search_start_w = max(2, _search_start_w)

                for _fi in range(_search_start_w, len(df_5m_struct)):
                    _c_bar = df_5m_struct.iloc[_fi]
                    _p2_bar = df_5m_struct.iloc[_fi - 2]
                    _new_fvg = None
                    if self.sweep_dir == "long":
                        _fl = float(_p2_bar["high"]); _fh = float(_c_bar["low"])
                        if _fh > _fl and sweep_extreme_w <= _fh <= _disp_ext_w:
                            _new_fvg = {"zone_low":_fl, "zone_high":_fh,
                                "mid":(_fl+_fh)/2, "size":_fh-_fl, "bar_idx":_fi,
                                "departed": False, "edge": _fh}
                    else:
                        _fh2 = float(_p2_bar["low"]); _fl2 = float(_c_bar["high"])
                        if _fl2 < _fh2 and _disp_ext_w <= _fl2 <= sweep_extreme_w:
                            _new_fvg = {"zone_low":_fl2, "zone_high":_fh2,
                                "mid":(_fl2+_fh2)/2, "size":_fh2-_fl2, "bar_idx":_fi,
                                "departed": False, "edge": _fl2}

                    if _new_fvg is not None:
                        # 중복 체크 (mid 기준)
                        _is_dup = any(abs(f["mid"] - _new_fvg["mid"]) < _w_atr * 0.05
                                      for f in self._fvg_pool)
                        if not _is_dup:
                            self._fvg_pool.append(_new_fvg)

            # ── (B) Departed 체크: 가격이 FVG zone을 벗어났는지
            for _fc in self._fvg_pool:
                if _fc["departed"]:
                    continue
                if self.sweep_dir == "long":
                    # long: close가 zone_high 위로 올라가면 departed
                    if c > _fc["zone_high"]:
                        _fc["departed"] = True
                else:
                    # short: close가 zone_low 아래로 내려가면 departed
                    if c < _fc["zone_low"]:
                        _fc["departed"] = True

            # ── (C) Fib 범위 체크 → 0~1 벗어나면 pool에서 제거
            _rng_w = abs(_disp_ext_w - sweep_extreme_w)
            _valid_pool = []
            for _fc in self._fvg_pool:
                if _rng_w > 0:
                    if self.sweep_dir == "long":
                        _fib_mid = (_fc["mid"] - sweep_extreme_w) / _rng_w
                        _fib_edge = (_fc["edge"] - sweep_extreme_w) / _rng_w
                    else:
                        _fib_mid = (sweep_extreme_w - _fc["mid"]) / _rng_w
                        _fib_edge = (sweep_extreme_w - _fc["edge"]) / _rng_w
                    # fib 0~1.1 범위 안이면 유지 (약간의 여유)
                    if 0 <= _fib_mid <= 1.1 or 0 <= _fib_edge <= 1.1:
                        _fc["_fib_mid"] = _fib_mid
                        _fc["_fib_edge"] = _fib_edge
                        _valid_pool.append(_fc)
                else:
                    _valid_pool.append(_fc)
            self._fvg_pool = _valid_pool

            # ── (D) 점수 계산 — departed된 FVG만 점수 대상
            # size percentile: pool 내 상대 크기
            _sizes = [f["size"] for f in self._fvg_pool]
            _max_size = max(_sizes) if _sizes else 1.0
            _min_size = min(_sizes) if _sizes else 0.0
            _size_range = _max_size - _min_size if _max_size > _min_size else 1.0

            _departed_candidates = []
            for _fc in self._fvg_pool:
                if not _fc["departed"]:
                    continue

                _fib_mid = _fc.get("_fib_mid", 0.5)
                _fib_edge = _fc.get("_fib_edge", 0.5)

                # size percentile (0~100)
                _size_pct = ((_fc["size"] - _min_size) / _size_range) * 100 if _size_range > 0 else 50.0

                # LVN density percentile — 현재는 50(기본값) 사용
                # backtest.py에서 VP profile을 주입하면 실제 density 사용 가능
                _lvn_pct = 50.0

                # mid entry candidate
                _score_mid = fvg_composite_score(_fib_mid, _lvn_pct, _size_pct)
                _departed_candidates.append({
                    "fvg": _fc,
                    "entry_type": "mid",
                    "entry_px": _fc["mid"],
                    "fib": _fib_mid,
                    "score": _score_mid,
                })

                # edge entry candidate
                _score_edge = fvg_composite_score(_fib_edge, _lvn_pct, _size_pct)
                _departed_candidates.append({
                    "fvg": _fc,
                    "entry_type": "edge",
                    "entry_px": _fc["edge"],
                    "fib": _fib_edge,
                    "score": _score_edge,
                })

            # ── (E) Invalidation 체크
            # E-1: 시간 초과 (15봉) — 10→15 완화
            FVG_WATCH_MAX_BARS = 15
            if self._watching_bars > FVG_WATCH_MAX_BARS:
                self.reset()
                return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

            # E-2: 가격 이탈 (3 ATR)
            FVG_WATCH_MAX_DIST_ATR = 3.0
            if _w_atr > 0 and self._fvg_pool:
                _pool_mid = np.mean([f["mid"] for f in self._fvg_pool])
                _dist = abs(c - _pool_mid)
                if _dist > FVG_WATCH_MAX_DIST_ATR * _w_atr:
                    self.reset()
                    return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

            # E-3: departed candidate가 없으면 대기 계속
            if not _departed_candidates:
                return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

            # ── (F) Best candidate 선택 (최고 점수)
            _best = max(_departed_candidates, key=lambda x: x["score"])
            _best_fvg = _best["fvg"]
            _best_entry_px = _best["entry_px"]
            self._best_fvg_score = _best["score"]  # feature_dict용

            # 현재 best를 _watching 필드에 반영
            self._watching_fvg = _best_fvg
            self._watching_entry_px = _best_entry_px

            # SL/TP 계산
            _sl_func = _calc_sl_group_a if self._watching_group == "A" else _calc_sl_group_b
            self._watching_sl = _sl_func(self.sweep_dir, sweep_extreme_w,
                                          _best_entry_px, self.sl_buffer_pct)
            _tp_new = _find_next_liquidity_target(
                levels_for_tp, _best_entry_px,
                self.sweep_dir, self._watching_sl)
            if _tp_new is None:
                _tp_new = _find_tp_fallback(
                    fvgs_5m, obs_5m, _best_entry_px,
                    self.sweep_dir, self._watching_sl,
                    pivots_15m=pivots_15m)
            self._watching_tp = _tp_new

            # ── (G) Mitigation 체크: 가격이 best FVG 영역 터치
            _mitigated = False
            if self.sweep_dir == "long":
                if l <= _best_fvg["zone_high"]:
                    _mitigated = True
            else:
                if h >= _best_fvg["zone_low"]:
                    _mitigated = True

            if not _mitigated:
                return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}

            # ══════════════════════════════════════════
            # MITIGATION 발동 → entry_plan 생성 + ML 호출
            # ══════════════════════════════════════════
            # notify.append(
                # f"FVG_MITIGATED! [{lv_type_w}|{self._watching_group}]  "
                # f"bar={self._watching_bars}  px={c:,.0f} → entry_plan"
            # )

            _entry_px = self._watching_entry_px
            _sl = self._watching_sl
            _tp = self._watching_tp
            _disp_ext_final = self._watching_disp_extreme

            # ── TP sanity check: 1.0R 미만이면 1.0R로 클램프
            if _tp is not None:
                _tp_risk = abs(_entry_px - _sl)
                _min_tp_dist = _tp_risk * 1.0
                if self.sweep_dir == "long":
                    _tp_dist = _tp - _entry_px
                else:
                    _tp_dist = _entry_px - _tp
                if _tp_risk > 0 and _tp_dist / _tp_risk < 1.0:
                    if self.sweep_dir == "long":
                        _tp = _entry_px + _min_tp_dist
                    else:
                        _tp = _entry_px - _min_tp_dist

            orders = [{"type":"limit","side":self.sweep_dir,
                       "price":_entry_px,"weight":1.0,
                       "tag":f"fvg_mitigated_{self._watching_group.lower()}"}]

            # ── feature_dict 구성 (mitigation 시점 기준)
            _size_feats_m = _calc_size_features(
                level_type=lv_type_w, htf_1h=htf_1h, htf_4h=htf_4h,
                direction=self.sweep_dir, sweep_vol_ok=True,
                pd_pos=None, df_5m=df_5m, sweep_bar_idx=self.sweep_bar_idx
            )
            _armed_feats_m = getattr(self, '_armed_features', {})

            # ── liq_features 재계산 (mitigation 시점 기준)
            #    기존: sweep 시점의 stale 데이터 사용
            #    변경: mitigation 시점의 최신 liquidity 상태 반영
            _cur_df_len_m = len(df_5m) if df_5m is not None else 0
            _liq_features_fresh = _calc_liquidity_features(
                lv=self.level if self.level else {},
                sweep_depth_atr=float(self._sweep_depth_score),
                sweep_ratio=float(self.sweep_ratio),
                all_levels=liq_no_flip,
                atr=_w_atr,
                current_price=c,
                current_bar_idx=_cur_df_len_m - 1,
                sweep_history=self._sweep_history,
                current_time=t,
            )

            _fvg_size_atr_m = 0.0
            if _w_atr > 0:
                _fvg_size_atr_m = min(_best_fvg["size"] / _w_atr, 1.5)  # ★ 1.5 ATR 캡 (뉴스 노이즈 방지)

            # displacement_consistency 기본값 0.2 (0 방지)
            _disp_consistency_m = max(_armed_feats_m.get("displacement_consistency", 0.0), 0.2)

            # ── sweep_duration_norm 계산
            #    sweep 발생부터 현재(mitigation)까지 봉 수 / 평균 swing 주기
            _sweep_duration_bars = _armed_feats_m.get("bars_since_sweep", 10)
            _avg_swing_duration = 12.0  # 평균 swing 주기 (5m 기준 ~1시간)
            if pivots_15m is not None and len(pivots_15m) >= 3:
                _swing_diffs = pivots_15m["idx"].diff().dropna()
                if len(_swing_diffs) > 0:
                    _avg_swing_duration = max(float(_swing_diffs.mean()) * 3, 3.0)  # 15m→5m 변환 (×3)
            _sweep_duration_norm = round(_sweep_duration_bars / _avg_swing_duration, 4) if _avg_swing_duration > 0 else 1.0

            # ── liq_imbalance 계산
            #    weighted_bsl = Σ(strength / distance^1.3) for buy-side liquidity (위쪽)
            #    weighted_ssl = Σ(strength / distance^1.3) for sell-side liquidity (아래쪽)
            _weighted_bsl = 0.0
            _weighted_ssl = 0.0
            _HIGH_LIQ = {"eqh", "swing_high", "pdh", "pwh", "asia_high", "london_high", "ny_high"}
            _LOW_LIQ = {"eql", "swing_low", "pdl", "pwl", "asia_low", "london_low", "ny_low"}
            for _lv in liq_no_flip:
                _lt = _lv.get("level_type", "")
                _lv_mid = (float(_lv.get("zone_low", 0)) + float(_lv.get("zone_high", 0))) / 2
                _dist = abs(_lv_mid - c)
                if _dist < 1e-6:
                    _dist = 1e-6
                _str = float(_lv.get("strength", 1.0))
                _w = _str / (_dist ** 1.3)
                if _lt in _HIGH_LIQ:
                    _weighted_bsl += _w
                elif _lt in _LOW_LIQ:
                    _weighted_ssl += _w
            _liq_total = _weighted_bsl + _weighted_ssl
            _liq_imbalance = round(_weighted_bsl / _liq_total, 4) if _liq_total > 0 else 0.5

            # ── asia_range_pos & bars_since_asia_break 계산
            _asia_range_pos = 0.5  # 기본값 (아시아 레인지 정보 없을 때)
            _bars_since_asia_break = -1  # 기본값 (벗어나지 않았거나 정보 없음)
            _asia_high_px = None
            _asia_low_px = None
            for _lv in liq_no_flip:
                _lt = _lv.get("level_type", "")
                if _lt == "asia_high":
                    _asia_high_px = (float(_lv["zone_low"]) + float(_lv["zone_high"])) / 2
                elif _lt == "asia_low":
                    _asia_low_px = (float(_lv["zone_low"]) + float(_lv["zone_high"])) / 2
            if _asia_high_px is not None and _asia_low_px is not None:
                _asia_rng = _asia_high_px - _asia_low_px
                if _asia_rng > 0:
                    _asia_range_pos = round(max(0.0, min(1.0, (c - _asia_low_px) / _asia_rng)), 4)
                    # bars_since_asia_break: 최근 N봉 역추적
                    if c > _asia_high_px or c < _asia_low_px:
                        _bars_since_asia_break = 0  # 현재 벗어나 있음
                        if df_5m is not None and len(df_5m) > 1:
                            for _ab_i in range(1, min(len(df_5m), 100)):
                                _ab_c = float(df_5m["close"].iloc[-1 - _ab_i])
                                if _asia_low_px <= _ab_c <= _asia_high_px:
                                    _bars_since_asia_break = _ab_i
                                    break
                    # else: -1 (아직 레인지 안에 있음)

            # ── fake_break_intensity
            _fake_break_intensity = round(_asia_range_pos / (_sweep_duration_bars + 1.0), 4)

            # ── trap_alignment
            _trap_alignment = round((_liq_imbalance - 0.5) * (1 if self.sweep_dir == "long" else -1), 4)

            # ── effective_fvg_strength
            #    displacement_consistency(범주형: 0, 1, 2 수준) → 가중치로 변환
            #    0(더러움) → 0.2 / 보통(0~0.5) → 0.6 / 깔끔함(0.5~1.0) → 1.0
            if _disp_consistency_m < 0.33:
                _quality_multiplier = 0.2
            elif _disp_consistency_m < 0.66:
                _quality_multiplier = 0.6
            else:
                _quality_multiplier = 1.0
            _effective_fvg_strength = round(_fvg_size_atr_m * _quality_multiplier, 4)

            feature_dict = {
                # ── A. Displacement & Structure Quality (3)
                "net_move_atr_ratio":   _armed_feats_m.get("net_move_atr_ratio", 0.0),
                "sweep_depth_atr":     round(float(self._sweep_depth_score), 3),
                # strong_displacement_score: 아래에서 계산 후 삽입

                # ── B. Liquidity & Volume (2)
                # liquidity_strength_score, vol_ratio — merge에서 추가

                # ── C. Market Context (2) — _size_feats_m에서 merge
                # htf_1h_aligned, pd_pos
            }

            feature_dict.update({
                "directional_er": _size_feats_m.get("directional_er", 0.0),
                "htf_4h_aligned": _size_feats_m.get("htf_4h_aligned", 0),
                "range_pos_30d": 0.5,   # backtest.py에서 5m 캔들 기반으로 덮어씀
                "range_pos_7d":  0.5,    # backtest.py에서 5m 캔들 기반으로 덮어씀
                "vol_ratio": _size_feats_m.get("vol_ratio", 1.0),
            })
            feature_dict["liquidity_strength_score"] = _liq_features_fresh.get("liquidity_strength_score", 0.0)

            # ── 반대편 유동성 선행회수 (sweep 추적): extreme/level_mid 원시 →
            #    backtest.py에서 atr 정규화 (opp_sweep_dist_atr).
            _opp = getattr(self, "_opp_sweep_feats", {}) or {}
            feature_dict["opp_sweep_present"] = _opp.get("opp_sweep_present", 0.0)
            feature_dict["opp_sweep_to_setup_bars"] = _opp.get("opp_sweep_to_setup_bars", -1.0)
            feature_dict["opp_sweep_extreme"] = _opp.get("opp_sweep_extreme", np.nan)
            feature_dict["opp_sweep_level_mid"] = _opp.get("opp_sweep_level_mid", np.nan)
            # 전달 정밀화 3종 (직접성/레인지보존/레인지높이) — backtest서 range 정규화
            feature_dict["opp_sweep_is_last"] = _opp.get("opp_sweep_is_last", 0.0)
            feature_dict["opp_sweep_unviolated"] = _opp.get("opp_sweep_unviolated", np.nan)
            feature_dict["opp_sweep_range_raw"] = _opp.get("opp_sweep_range_raw", np.nan)
            # 셋업 sweep 레벨 type (진단 전용, 학습 제외): london 등 레벨 오염을
            #   사후 식별 가능하게 보존. opp_sweep_extreme_raw와 같은 보험 논리.
            feature_dict["setup_level_type"] = (getattr(self, "level", {}) or {}).get("level_type", "")

            # ── is_killzone: NY Killzone (02:00~05:00 ET, DST 반영)
            try:
                from zoneinfo import ZoneInfo
                _et = t.astimezone(ZoneInfo("America/New_York"))
                _et_hour = _et.hour
                _is_kz = 1.0 if (2 <= _et_hour < 5) else 0.0
            except Exception:
                # fallback: UTC 기준 근사 (EDT: UTC-4 → 06~09, EST: UTC-5 → 07~10)
                _utc_hour = t.hour
                _is_kz = 1.0 if (6 <= _utc_hour < 10) else 0.0
            feature_dict["is_killzone"] = _is_kz

            # ── strong_displacement_score (soft version)
            _nmr = _armed_feats_m.get("net_move_atr_ratio", 0.0)
            _ccb = _armed_feats_m.get("choch_is_close_beyond", 0)
            _dbwr = _armed_feats_m.get("displacement_body_wick_ratio", 0.0)

            _sds = (
                min(_nmr / 2.0, 1.0) *
                (1.0 if _ccb else 0.3) *
                min(_dbwr / 1.5, 1.0)
            )
            feature_dict["strong_displacement_score"] = round(_sds, 4)

            # ── D. sweep_to_disp_fib: 진입가의 fib 위치 (sweep↔disp 구간)
            _stdf = _ote_fib_pos(_entry_px, sweep_extreme_w, _disp_ext_final, self.sweep_dir)
            feature_dict["sweep_to_disp_fib"] = round(_stdf, 4) if _stdf is not None else 0.5

            # ── D2. FVG composite score 기록 (best candidate의 점수)
            feature_dict["fvg_composite_score"] = round(
                getattr(self, '_best_fvg_score', 0.0), 4)

            # ── D2-tail. fib_tail_score (sweep_to_disp_fib 기반 꼬리 multiplier)
            #   러너/사이징 신호. composite_tail은 _best_fvg_fib(≠sweep_to_disp_fib)을
            #   써서 검증곡선과 불일치(역상관)하므로 폐기. fib_tail_score만 유지.
            feature_dict["fib_tail_score"] = round(
                fib_tail_score(_stdf), 4) if _stdf is not None else 1.0

            # ── F. Interaction 피처
            _liq_score = feature_dict.get("liquidity_strength_score", 0.0)
            feature_dict["choch_x_liquidity"] = round(_sds * _liq_score, 4)

            # ── G. Sweep & Trap 피처
            feature_dict["sweep_duration_norm"] = _sweep_duration_norm
            feature_dict["fake_break_intensity"] = _fake_break_intensity
            feature_dict["trap_alignment"] = _trap_alignment

            # ── H. Asia Session 피처
            feature_dict["asia_range_pos"] = _asia_range_pos
            feature_dict["bars_since_asia_break"] = _bars_since_asia_break

            # ── I. Liquidity Imbalance
            feature_dict["liq_imbalance"] = _liq_imbalance

            # ── J. FVG Quality
            feature_dict["effective_fvg_strength"] = _effective_fvg_strength

            # ── K. v7 복원 피처 (_armed_feats_m에서 추출)
            feature_dict["choch_break_atr_ratio"] = round(
                float(_armed_feats_m.get("choch_break_atr_ratio", 0.0)), 4)
            feature_dict["displacement_consistency"] = round(
                float(_armed_feats_m.get("displacement_consistency", 0.0)), 4)
            feature_dict["bars_since_sweep"] = int(
                _armed_feats_m.get("bars_since_sweep", 10))

            # ── K2. Reclaim 피처 (_armed_feats_m에서 추출)
            feature_dict["reclaim_vol_spike"] = round(
                float(_armed_feats_m.get("reclaim_vol_spike", 0.0)), 4)
            feature_dict["reclaim_speed"] = round(
                float(_armed_feats_m.get("reclaim_speed", 0.0)), 4)
            feature_dict["reclaim_vol_profile"] = round(
                float(_armed_feats_m.get("reclaim_vol_profile", 0.0)), 4)

            # ── K3. 신규 건강함 피처 (_armed_feats_m에서 추출)
            feature_dict["net_move_atr50"] = round(
                float(_armed_feats_m.get("net_move_atr50", 0.0)), 4)
            feature_dict["net_move_atr_wilder"] = round(
                float(_armed_feats_m.get("net_move_atr_wilder", 0.0)), 4)
            feature_dict["net_move_atr1h"] = round(
                float(_armed_feats_m.get("net_move_atr1h", 0.0)), 4)
            feature_dict["reclaim_quality"] = round(
                float(_armed_feats_m.get("reclaim_quality", 0.0)), 4)
            feature_dict["reclaim_absorption"] = round(
                float(_armed_feats_m.get("reclaim_absorption", 0.0)), 4)
            feature_dict["trend_structure"] = round(
                float(_armed_feats_m.get("trend_structure", 0.0)), 4)
            feature_dict["ma_alignment"] = float(_armed_feats_m.get("ma_alignment", 0.0))
            _dq15 = _armed_feats_m.get("disp_quality_15m", np.nan)
            feature_dict["disp_quality_15m"] = (round(float(_dq15), 4)
                                                if _dq15 == _dq15 else np.nan)

            # ── L. Interaction 피처 (기존 피처 조합)
            _sda = feature_dict.get("sweep_depth_atr", 0.0)
            _sds = feature_dict.get("strong_displacement_score", 0.0)
            _efs = feature_dict.get("effective_fvg_strength", 0.0)
            _bss = feature_dict.get("bars_since_sweep", 10)
            _sdn = feature_dict.get("sweep_duration_norm", 0.0)
            _lvn = feature_dict.get("lvn_proximity", np.nan)
            _rp7 = feature_dict.get("range_pos_7d", 0.5)

            feature_dict["sweep_x_disp"] = round(_sda * _sds, 4)
            feature_dict["disp_x_fvg"] = round(_sds * _efs, 4)
            feature_dict["sweep_velocity"] = round(_sda / (_bss + 1.0), 4)
            feature_dict["rangepos_x_sweep_dur"] = round(_rp7 * _sdn, 4)
            feature_dict["rangepos_x_sweep_depth"] = round(_rp7 * _sda, 4)
            # v3: lvn이 NaN이면 lvn_x_fvg도 NaN (XGBoost native)
            _lxf = _lvn * _efs if not (isinstance(_lvn, float) and np.isnan(_lvn)) else np.nan
            feature_dict["lvn_x_fvg"] = round(_lxf, 4) if not (isinstance(_lxf, float) and np.isnan(_lxf)) else np.nan
            feature_dict["killzone_x_disp"] = round(_is_kz * _sds, 4)
            feature_dict["consistency_x_vol"] = round(
                feature_dict.get("displacement_consistency", 0.0) * feature_dict.get("vol_ratio", 1.0), 4)

            # ── L. Nasdaq (backtest.py에서 주입, 여기서는 0 초기화)
            feature_dict["nasdaq_session_divergence_pct"] = 0.0

            # ── L. Derivatives → 삭제됨 (소급 불가)

            # ── M. Reclaim 피처 — 위(2381~2386)에서 _armed_feats_m 기반 주입 완료
            # (여기서 다시 0으로 초기화하면 안 됨)

            # ── N. Spot-Futures signed 괴리 (backtest.py에서 주입, stage7b/7c 검증판)
            #     구 5종(abs z-of-z/slope/combined)은 방향소거+노이즈증폭으로 폐기
            feature_dict["spot_div_signed_z"] = 0.0
            # ── 방향 플래그 (backtest.py에서 ep["direction"]으로 덮어씀)
            feature_dict["is_long"] = 0.0

            # ── O. Basis (backtest.py에서 bisect lookup 주입)
            feature_dict["basis_z"] = 0.0
            feature_dict["basis_momentum"] = 0.0
            feature_dict["spot_lead"] = 0.0

            # ── P. Liquidation Heatmap (backtest.py에서 주입)
            feature_dict["sweep_liq_consumed"] = 0.0
            feature_dict["liq_imbalance_hm"] = 0.0

            # VP 피처 (lvn_proximity, lvn_available) — backtest.py에서 주입
            # v3: 3.0 default 제거. backtest.py가 df_5m 접근 후 갱신.
            #     주입 실패 시 lvn_available=0으로 학습 단에서 제외 가능.
            feature_dict["lvn_proximity"] = np.nan
            feature_dict["lvn_available"] = 0.0

            entry_plan = {
                "direction":       self.sweep_dir,
                "trigger_time":    self.trigger_time,
                "sweep_extreme":   sweep_extreme_w,
                "choch_price":     float(self.choch_price) if self.choch_price else sweep_extreme_w,
                "disp_extreme":    _disp_ext_final,
                "ote_zone":        None,
                "swept_level":     self.level,
                "selected_fvg":    _best_fvg if not _best_fvg.get("_is_ob") else None,
                "selected_ob":     _best_fvg.get("_ob_data") if _best_fvg.get("_is_ob") else None,
                "orders":          orders,
                "sl":              float(_sl),
                "tp":              float(_tp) if _tp is not None else None,
                "size_multiplier": 1.0,
                "reason_trigger":  "fvg_mitigation",
                "htf_state":       htf_1h,
                "htf_4h":          htf_4h,
                "sweep_ratio":     float(self.sweep_ratio),
                "sweep_vol_ok":    True,
                "level_group":     self._watching_group,
                "_notional":       0.0,
                "feature_dict":    feature_dict,
                "bars_since_sweep": _armed_feats_m.get("bars_since_sweep", 10),  # ★ timeout 계산용
            }
            self.entry_plan = entry_plan; self.plan_time = t
            self.plan_bar_count = 0; self.state = "ORDERS_PLANNED"
            return {"action":"enter_plan","state":self.state,
                    "entry_plan":entry_plan,"notify":notify}

        return {"action":"none","state":self.state,"entry_plan":None,"notify":notify}