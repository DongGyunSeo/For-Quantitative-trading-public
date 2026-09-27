"""
volume_profile.py — Volume Profile 기반 피처 계산
═══════════════════════════════════════════════════════════
POC (Point of Control) 및 LVN (Low Volume Node) 탐지.

사용처:
  - backtest.py: Entry 피처 주입 (vp_confluence, lvn_proximity)
  - tp_selector.py: TP 피처 주입 (distance_to_poc_atr, poc_miss_ratio 등)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────
# Rolling POC (Point of Control)
# ──────────────────────────────────────────────────────────

def calculate_rolling_poc(
    df_5m: pd.DataFrame,
    current_idx: int,
    lookback_bars: int = 288,       # 24시간 = 288 × 5m
    n_bins: int = 50,
) -> float:
    """
    현재 캔들 직전까지의 rolling Volume Profile에서 POC(최대 거래량 가격) 반환.

    Parameters
    ----------
    df_5m : DataFrame with 'high', 'low', 'close', 'volume' columns
    current_idx : int — 현재 캔들 인덱스 (이 캔들 직전까지 사용)
    lookback_bars : int — lookback 윈도우 (기본 288 = 24h)
    n_bins : int — 가격 구간 수

    Returns
    -------
    float — POC 가격 (가격 bin의 중심값). 데이터 부족 시 0.0
    """
    start = max(0, current_idx - lookback_bars)
    end = current_idx  # 현재 캔들 직전까지 (data leakage 방지)

    if end - start < 10:
        return 0.0

    highs = df_5m["high"].iloc[start:end].values
    lows = df_5m["low"].iloc[start:end].values
    volumes = df_5m["volume"].iloc[start:end].values

    price_min = float(np.nanmin(lows))
    price_max = float(np.nanmax(highs))
    if price_max <= price_min or price_max <= 0:
        return 0.0

    # 가격 bin 생성
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    vol_profile = np.zeros(n_bins, dtype=np.float64)

    # 각 캔들의 거래량을 해당 가격 range에 분배
    for i in range(len(highs)):
        h, l, v = highs[i], lows[i], volumes[i]
        if np.isnan(h) or np.isnan(l) or np.isnan(v) or v <= 0:
            continue
        # 캔들이 걸치는 bin 범위
        lo_bin = np.searchsorted(bin_edges, l, side="right") - 1
        hi_bin = np.searchsorted(bin_edges, h, side="right") - 1
        lo_bin = max(0, min(lo_bin, n_bins - 1))
        hi_bin = max(0, min(hi_bin, n_bins - 1))
        n_overlap = hi_bin - lo_bin + 1
        if n_overlap > 0:
            vol_per_bin = v / n_overlap
            vol_profile[lo_bin:hi_bin + 1] += vol_per_bin

    if vol_profile.sum() <= 0:
        return 0.0

    poc_idx = int(np.argmax(vol_profile))
    return float(bin_centers[poc_idx])


# ──────────────────────────────────────────────────────────
# LVN (Low Volume Node) 탐지
# ──────────────────────────────────────────────────────────

def find_nearest_lvn(
    df_5m: pd.DataFrame,
    current_idx: int,
    reference_price: float,
    atr: float,
    lookback_bars: int = 288,
    n_bins: int = 50,
    lvn_percentile: float = 20.0,
    max_dist_atr: float = 5.0,
) -> Tuple[Optional[float], Optional[float]]:
    """
    reference_price에서 가장 가까운 LVN의 가격과 ATR 정규화 거리 반환.

    LVN = Volume Profile 히스토그램에서 하위 lvn_percentile%에 해당하는 bin.

    Parameters
    ----------
    reference_price : float — 기준 가격 (보통 진입가 또는 FVG mid)
    max_dist_atr : float — LVN 탐색 최대 거리 (ATR 배수)

    Returns
    -------
    (lvn_price, lvn_dist_atr) or (None, None) if no LVN found
    """
    start = max(0, current_idx - lookback_bars)
    end = current_idx

    if end - start < 10 or atr <= 0:
        return None, None

    highs = df_5m["high"].iloc[start:end].values
    lows = df_5m["low"].iloc[start:end].values
    volumes = df_5m["volume"].iloc[start:end].values

    price_min = float(np.nanmin(lows))
    price_max = float(np.nanmax(highs))
    if price_max <= price_min:
        return None, None

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    vol_profile = np.zeros(n_bins, dtype=np.float64)

    for i in range(len(highs)):
        h, l, v = highs[i], lows[i], volumes[i]
        if np.isnan(h) or np.isnan(l) or np.isnan(v) or v <= 0:
            continue
        lo_bin = max(0, min(np.searchsorted(bin_edges, l, side="right") - 1, n_bins - 1))
        hi_bin = max(0, min(np.searchsorted(bin_edges, h, side="right") - 1, n_bins - 1))
        n_overlap = hi_bin - lo_bin + 1
        if n_overlap > 0:
            vol_profile[lo_bin:hi_bin + 1] += v / n_overlap

    if vol_profile.sum() <= 0:
        return None, None

    # LVN: 하위 percentile 이하인 bin들
    threshold = np.percentile(vol_profile[vol_profile > 0], lvn_percentile)
    lvn_mask = (vol_profile > 0) & (vol_profile <= threshold)

    if not lvn_mask.any():
        return None, None

    lvn_centers = bin_centers[lvn_mask]
    distances = np.abs(lvn_centers - reference_price)

    # max_dist_atr 내의 LVN만
    within_range = distances <= (max_dist_atr * atr)
    if not within_range.any():
        return None, None

    # 가장 가까운 LVN
    filtered_centers = lvn_centers[within_range]
    filtered_dists = distances[within_range]
    nearest_idx = int(np.argmin(filtered_dists))

    lvn_price = float(filtered_centers[nearest_idx])
    lvn_dist_atr = float(filtered_dists[nearest_idx]) / atr

    return lvn_price, lvn_dist_atr


# ──────────────────────────────────────────────────────────
# Entry 피처: VP Confluence + LVN Proximity
# ──────────────────────────────────────────────────────────

def compute_vp_entry_features(
    df_5m: pd.DataFrame,
    current_idx: int,
    fvg_high: float,
    fvg_low: float,
    entry_price: float,
    atr: float,
    lookback_bars: int = 288,
) -> dict:
    """
    Entry ML용 VP 피처 계산.

    v3 변경 (LVN edge case):
      - LVN 미발견 또는 atr<=0 → lvn_proximity = NaN (XGBoost native handling)
      - lvn_available: 신규. 0.0 = 정보 없음, 1.0 = LVN 발견.
      - vp_confluence: LVN 미발견 시 NaN.

    Returns: {vp_confluence, lvn_proximity, lvn_available}
    """
    fvg_center = (fvg_high + fvg_low) / 2.0
    fvg_size = fvg_high - fvg_low

    # LVN 탐지
    lvn_price, lvn_dist_atr = find_nearest_lvn(
        df_5m, current_idx, fvg_center, atr, lookback_bars,
    )

    # LVN 발견 여부 (atr<=0이면 거리 계산 자체가 불가하므로 정보 없음으로 처리)
    lvn_found = (lvn_price is not None) and (atr > 0)

    if lvn_found:
        # vp_confluence: FVG + LVN 중첩도 (0~1)
        if fvg_size > 0:
            overlap_ratio = max(0.0, 1.0 - abs(fvg_center - lvn_price) / fvg_size)
            proximity_bonus = max(0.0, 1.0 - abs(fvg_center - lvn_price) / (2.0 * atr))
            vp_confluence = overlap_ratio * 0.6 + proximity_bonus * 0.4
        else:
            vp_confluence = 0.0

        lvn_proximity = abs(entry_price - lvn_price) / atr
        lvn_available = 1.0

        return {
            "vp_confluence": round(vp_confluence, 4),
            "lvn_proximity": round(lvn_proximity, 4),
            "lvn_available": lvn_available,
        }
    else:
        # LVN 미발견 → NaN sentinel (XGBoost가 자동으로 branch 처리)
        return {
            "vp_confluence": np.nan,
            "lvn_proximity": np.nan,
            "lvn_available": 0.0,
        }


# ──────────────────────────────────────────────────────────
# TP 피처: POC 관련
# ──────────────────────────────────────────────────────────

def compute_vp_tp_features(
    df_5m: pd.DataFrame,
    current_idx: int,
    candidate_price: float,
    entry_price: float,
    atr: float,
    strong_displacement_score: float = 0.0,
    lookback_bars: int = 288,
    direction: str = "",
) -> dict:
    """
    TP ML용 VP 피처 계산.

    Returns: {distance_to_poc_atr, poc_miss_ratio, poc_direction,
              poc_proximity_x_displacement}

    v2 추가: poc_direction — POC가 후보 방향에 있는지 (거리 독립적)
        +1.0: POC가 entry→candidate 경로 너머에 있음 (추가 가격 이동 지지)
        -1.0: POC가 entry→candidate 경로 사이에 있음 (가격 이동 방해)
         0.0: 판단 불가
    """
    poc_price = calculate_rolling_poc(df_5m, current_idx, lookback_bars)

    if poc_price <= 0 or atr <= 0:
        return {
            "distance_to_poc_atr": 0.0,
            "poc_miss_ratio": 1.0,
            "poc_direction": 0.0,
            "poc_proximity_x_displacement": 0.0,
        }

    dist_to_poc_atr = abs(candidate_price - poc_price) / atr
    dist_to_candidate_atr = abs(candidate_price - entry_price) / atr

    poc_miss_ratio = dist_to_poc_atr / max(dist_to_candidate_atr, 0.25)
    poc_proximity = 1.0 / (dist_to_poc_atr + 0.15)
    poc_prox_x_disp = poc_proximity * strong_displacement_score

    # v2: poc_direction (거리 독립적)
    # POC가 entry와 candidate 사이에 있으면 방해(-1), 너머에 있으면 지지(+1)
    if direction == "long":
        # long: entry < candidate
        if entry_price < poc_price < candidate_price:
            poc_dir = -1.0   # POC가 경로 중간 → 저항
        elif poc_price >= candidate_price:
            poc_dir = 1.0    # POC가 candidate 너머 → 가격 끌어당김
        else:
            poc_dir = 0.0    # POC가 entry 아래 → 무관
    elif direction == "short":
        # short: entry > candidate
        if candidate_price < poc_price < entry_price:
            poc_dir = -1.0   # POC가 경로 중간 → 지지(방해)
        elif poc_price <= candidate_price:
            poc_dir = 1.0    # POC가 candidate 너머 → 가격 끌어당김
        else:
            poc_dir = 0.0
    else:
        poc_dir = 0.0

    return {
        "distance_to_poc_atr": round(dist_to_poc_atr, 4),
        "poc_miss_ratio": round(poc_miss_ratio, 4),
        "poc_direction": round(poc_dir, 4),
        "poc_proximity_x_displacement": round(poc_prox_x_disp, 4),
    }


# ──────────────────────────────────────────────────────────
# VP Scoring: density profile 빌더 + bar-by-bar 누적 계산
# ──────────────────────────────────────────────────────────

class VPDensityProfile:
    """
    Rolling Volume Profile의 density map을 보관.
    bar-by-bar로 가격이 통과한 구간의 VP density를 누적하는 데 사용.

    사용법:
        vp = VPDensityProfile.build(df_5m, current_idx, lookback_bars=288, n_bins=50)
        density = vp.density_at(price)
        total = vp.density_between(price_lo, price_hi)
    """

    def __init__(self, bin_edges: np.ndarray, vol_profile: np.ndarray,
                 mean_density: float):
        self.bin_edges = bin_edges        # (n_bins+1,) — bin 경계
        self.vol_profile = vol_profile    # (n_bins,) — bin별 거래량
        self.mean_density = mean_density  # 평균 bin 거래량
        self.n_bins = len(vol_profile)

    @classmethod
    def build(cls, df_5m: pd.DataFrame, current_idx: int,
              lookback_bars: int = 288, n_bins: int = 50,
              end_idx: Optional[int] = None,
              atr: float = 0.0,
              bin_size_atr: float = 0.0,
              n_bins_min: int = 30,
              n_bins_max: int = 200) -> "VPDensityProfile":
        """현재 봉 직전까지의 rolling VP density profile 생성.

        Parameters
        ----------
        current_idx : int
            현재 캔들 인덱스. lookback의 종료점 기본값으로 사용.
        end_idx : Optional[int]
            VP 계산의 종료 인덱스(exclusive). None이면 current_idx 사용.
            sweep_idx로 잘라서 contamination 회피하는 용도.
            (lookback_bars는 end_idx 기준이 아닌 current_idx 기준 — start만
             end_idx가 너무 작을 때 자동 조정됨)
        atr : float
            동적 bin 폭 계산용. 0이면 고정 n_bins 사용.
        bin_size_atr : float
            1 bin의 ATR 단위 폭. 0이면 고정 n_bins 사용.
            권장값: 0.1 (10 bins = 1 ATR).
            이 값이 설정되면 n_bins는 price_range에 따라 동적으로 계산.
        n_bins_min, n_bins_max : int
            동적 n_bins의 하한/상한.
        """
        if end_idx is None:
            end_idx = current_idx
        # lookback은 current_idx 기준으로 잡되, end_idx 이전까지만
        start = max(0, current_idx - lookback_bars)
        end = max(start, end_idx)  # end_idx가 start보다 작으면 빈 윈도우

        empty = cls(np.array([0, 1]), np.array([0.0]), 1.0)

        if end - start < 10:
            return empty

        highs = df_5m["high"].iloc[start:end].values
        lows = df_5m["low"].iloc[start:end].values
        volumes = df_5m["volume"].iloc[start:end].values

        price_min = float(np.nanmin(lows))
        price_max = float(np.nanmax(highs))
        if price_max <= price_min or price_max <= 0:
            return empty

        # ── 동적 bin: ATR-normalized
        if bin_size_atr > 0 and atr > 0:
            bin_width = bin_size_atr * atr
            dyn_bins = int(round((price_max - price_min) / bin_width))
            n_bins = max(n_bins_min, min(dyn_bins, n_bins_max))

        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        vol_profile = np.zeros(n_bins, dtype=np.float64)

        for i in range(len(highs)):
            h, l, v = highs[i], lows[i], volumes[i]
            if np.isnan(h) or np.isnan(l) or np.isnan(v) or v <= 0:
                continue
            lo_bin = np.searchsorted(bin_edges, l, side="right") - 1
            hi_bin = np.searchsorted(bin_edges, h, side="right") - 1
            lo_bin = max(0, min(lo_bin, n_bins - 1))
            hi_bin = max(0, min(hi_bin, n_bins - 1))
            n_overlap = hi_bin - lo_bin + 1
            if n_overlap > 0:
                vol_per_bin = v / n_overlap
                vol_profile[lo_bin:hi_bin + 1] += vol_per_bin

        total = vol_profile.sum()
        mean_d = total / n_bins if n_bins > 0 and total > 0 else 1.0

        return cls(bin_edges, vol_profile, mean_d)

    def density_at(self, price: float) -> float:
        """특정 가격의 VP density (정규화: 1.0 = 평균 거래량 수준)."""
        if self.n_bins <= 1 or self.mean_density <= 0:
            return 0.0
        idx = np.searchsorted(self.bin_edges, price, side="right") - 1
        idx = max(0, min(idx, self.n_bins - 1))
        return float(self.vol_profile[idx] / self.mean_density)

    def density_between(self, price_lo: float, price_hi: float) -> float:
        """두 가격 사이 구간의 VP density 합 (정규화)."""
        if price_lo > price_hi:
            price_lo, price_hi = price_hi, price_lo
        if self.n_bins <= 1 or self.mean_density <= 0:
            return 0.0
        lo_bin = np.searchsorted(self.bin_edges, price_lo, side="right") - 1
        hi_bin = np.searchsorted(self.bin_edges, price_hi, side="right") - 1
        lo_bin = max(0, min(lo_bin, self.n_bins - 1))
        hi_bin = max(0, min(hi_bin, self.n_bins - 1))
        return float(self.vol_profile[lo_bin:hi_bin + 1].sum() / self.mean_density)


def compute_vp_topology_features(
    df_5m: pd.DataFrame,
    current_idx: int,
    entry_price: float,
    candidate_price: float,
    atr: float,
    fuel_a: float = 0.0,
    fuel_b: float = 0.0,
    sweep_idx: Optional[int] = None,
    lookback_bars: int = 288,
    bin_size_atr: float = 0.1,
    eps: float = 0.1,
) -> dict:
    """
    TP ML용 VP topology Tier 1 피처 (4개).

    Parameters
    ----------
    df_5m : DataFrame with 'high', 'low', 'volume'
    current_idx : int — 현재 봉 인덱스(추론 시점)
    entry_price : float
    candidate_price : float — TP 후보 가격
    atr : float
    fuel_a : float — sweep_depth_atr × displacement_consistency × vol_ratio
    fuel_b : float — strong_displacement_score
    sweep_idx : Optional[int] — sweep이 발생한 봉 인덱스. None이면 contamination 회피 안 함.
                                 설정 시 VP는 [start, sweep_idx) 구간만 사용.
    lookback_bars : int — 기본 288 (24h)
    bin_size_atr : float — 동적 bin 폭 (ATR 단위). 0이면 고정 n_bins=50.
    eps : float — frr 분모에 더하는 안정화 상수 (normalized density 단위)

    Returns
    -------
    dict with:
        path_vp_integral : float
            entry→candidate 구간의 VP density 적분.
            VPDensityProfile.density_between() 결과 (정규화: 1.0=평균 density × 1 bin).
        max_density_on_path : float
            경로상 단일 bin의 최대 normalized density (가장 큰 wall).
        density_at_candidate : float
            candidate_price 자체의 normalized density (도달 시 정지 가능성).
        frr_a : float — fuel_a / (path_vp_integral + eps)
        frr_b : float — fuel_b / (path_vp_integral + eps)

    Notes
    -----
    - sweep_idx 주입 시: VP는 sweep 이전까지만 계산 → fresh wick volume 오염 제거
    - 모든 density 값은 mean_density 정규화 (코인간 비교 가능)
    """
    # ── VP 빌드: sweep_idx 이전까지만 (contamination 회피)
    end_idx = sweep_idx if sweep_idx is not None else current_idx
    vp = VPDensityProfile.build(
        df_5m, current_idx,
        lookback_bars=lookback_bars,
        end_idx=end_idx,
        atr=atr,
        bin_size_atr=bin_size_atr,
    )

    # 빈 profile 가드
    if vp.n_bins <= 1 or vp.mean_density <= 0:
        return {
            "path_vp_integral": 0.0,
            "max_density_on_path": 0.0,
            "density_at_candidate": 0.0,
            "frr_a": 0.0,
            "frr_b": 0.0,
        }

    # ── 1. path_vp_integral — entry→candidate 적분 (정규화)
    path_integral = vp.density_between(entry_price, candidate_price)

    # ── 2. max_density_on_path — 경로 내 max bin density
    p_lo = min(entry_price, candidate_price)
    p_hi = max(entry_price, candidate_price)
    lo_bin = int(np.searchsorted(vp.bin_edges, p_lo, side="right") - 1)
    hi_bin = int(np.searchsorted(vp.bin_edges, p_hi, side="right") - 1)
    lo_bin = max(0, min(lo_bin, vp.n_bins - 1))
    hi_bin = max(0, min(hi_bin, vp.n_bins - 1))
    if hi_bin >= lo_bin:
        seg = vp.vol_profile[lo_bin:hi_bin + 1]
        max_d = float(seg.max() / vp.mean_density) if len(seg) > 0 else 0.0
    else:
        max_d = 0.0

    # ── 3. density_at_candidate
    density_at = vp.density_at(candidate_price)

    # ── 4. FRR — fuel / (resistance + eps)
    frr_a = fuel_a / (path_integral + eps)
    frr_b = fuel_b / (path_integral + eps)

    return {
        "path_vp_integral": round(path_integral, 4),
        "max_density_on_path": round(max_d, 4),
        "density_at_candidate": round(density_at, 4),
        "frr_a": round(frr_a, 4),
        "frr_b": round(frr_b, 4),
    }


def compute_vp_score(
    vp_consumption: float,
    distance_atr: float,
    max_adverse_atr: float,
) -> dict:
    """
    VP 기반 진입 품질 score 계산.

    Parameters
    ----------
    vp_consumption : float — Σ(통과한 VP density) / mean(VP density). 높을수록 heavy zone 소화.
    distance_atr : float — 진입→최대 유리 가격까지 ATR 정규화 거리.
    max_adverse_atr : float — 진입→최대 불리 가격까지 ATR 정규화 거리 (adversity).

    Returns
    -------
    dict with: vp_score, vp_consumption, vp_efficiency, vp_adversity
    """
    efficiency = distance_atr / (1 + vp_consumption) if vp_consumption >= 0 else distance_atr
    adversity = max_adverse_atr

    # score = (distance × (1 + consumption) × (1 + efficiency)) / (1 + adversity)
    score = (distance_atr * (1 + vp_consumption) * (1 + efficiency)) / (1 + adversity)

    return {
        "vp_score": round(score, 4),
        "vp_consumption": round(vp_consumption, 4),
        "vp_efficiency": round(efficiency, 4),
        "vp_adversity": round(adversity, 4),
    }
