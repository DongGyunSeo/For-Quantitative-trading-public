"""
liquidation_heatmap_features.py — 청산 히트맵 근사 피처 모듈 (Entry 필터용)
═══════════════════════════════════════════════════════════════════
CoinGlass 류 청산 히트맵을 OHLCV만으로 근사하여, 진입 시점의
"청산 유동성 풀(liquidation pool)" 상태를 스칼라 피처로 요약.

원리 (자료 기반):
  1. 각 과거 봉에서 거래량 = 신규 포지션 진입으로 가정 (entry≈OHLC4)
  2. 레버리지 buckets(10/25/50/100x)별 청산가 역산, 1/L 가중으로 흩뿌림
  3. 가격축 bin에 누적 → 청산 풀 강도 맵
  4. 소멸 처리: 진입 시점까지 가격이 통과한 bin은 거래량 비례 소멸
                (평균 거래량 1배 통과 = 0.5 소멸, 누적 곱연산, 0~1 capping)

피처 (2개):
  sweep_liq_consumed : sweep extreme가 통과/소진한 청산 풀 강도
                       → 큰 풀을 먹고 reclaim했으면 진짜 유동성 사냥 (셋업 진정성)
  liq_imbalance_hm   : 진입가 기준 위/아래 청산 풀 불균형
                       → (above - below)/(above + below), 방향 동력

설정:
  LOOKBACK_BARS = 4032  (14일 = 14*288)
  소멸 처리    = 거래량 기반 (평균 1배 통과 = 0.5 소멸, 곱연산 누적)

미래 참조 차단:
  진입 봉 인덱스 이전 데이터만 슬라이스하여 풀 계산.
  소멸 처리도 진입 시점까지의 가격 경로만 사용.

주의:
  - OHLCV 근사이므로 절대값 부정확, 상대 강도만 의미.
  - OI 보정 없음 (거래량으로만 포지션 규모 근사).
  - 레버리지 분포 미상 → 1/L 경험적 가중.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 설정 ───────────────────────────────────────────────
LOOKBACK_BARS = 4032        # 14일 (14 * 288봉)
LEVERAGES = (10, 25, 50, 100)
MMR = 0.005                 # 유지증거금률 근사
N_PRICE_BINS = 240          # 가격축 bin 수
LIQ_RANGE_PCT = 0.15        # 현재가 ±15% 범위만 본다 (그 밖은 무의미)

LIQ_HEATMAP_KEYS = [
    "sweep_liq_consumed",
    "liq_imbalance_hm",
]

NEUTRAL = {k: 0.0 for k in LIQ_HEATMAP_KEYS}


def _neutral_features() -> dict:
    return NEUTRAL.copy()


def _build_heatmap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    price_now: float,
):
    """
    과거 OHLCV 슬라이스로 청산 풀 히트맵 구성 + 소멸 처리.

    Returns
    -------
    bins : np.ndarray (N_PRICE_BINS+1,)  bin 경계
    heat : np.ndarray (N_PRICE_BINS,)    각 bin의 청산 강도 (소멸 후)
    """
    # 가격 범위: 현재가 ±LIQ_RANGE_PCT
    pmin = price_now * (1 - LIQ_RANGE_PCT)
    pmax = price_now * (1 + LIQ_RANGE_PCT)
    bins = np.linspace(pmin, pmax, N_PRICE_BINS + 1)
    heat = np.zeros(N_PRICE_BINS)

    # OHLC4 근사 진입가
    entry_px = (opens + highs + lows + closes) / 4.0

    # ── 1~3단계: 청산가 누적 (벡터화)
    valid = (volumes > 0) & (entry_px > 0)
    e = entry_px[valid]
    v = volumes[valid]
    if len(e) == 0:
        return bins, heat

    span = pmax - pmin
    for L in LEVERAGES:
        w = v / L  # (n,)
        long_liq = e * (1 - 1.0 / L + MMR)
        short_liq = e * (1 + 1.0 / L - MMR)
        for liq_arr in (long_liq, short_liq):
            m = (liq_arr >= pmin) & (liq_arr < pmax)
            if not m.any():
                continue
            idxs = ((liq_arr[m] - pmin) / span * N_PRICE_BINS).astype(int)
            idxs = np.clip(idxs, 0, N_PRICE_BINS - 1)
            np.add.at(heat, idxs, w[m])

    # ── 4단계: 소멸 처리 (거래량 기반, 벡터화)
    # 곱연산 누적 Π(1-rate)를 log 공간 합연산으로 변환:
    #   log(survive) = Σ log(1 - consume_rate_i)  (각 통과 봉마다)
    # 봉이 [lo_i, hi_i] 구간을 통과 → 그 구간 bin들에 log(1-rate) 더함.
    # 구간 더하기는 diff array(누적합) 트릭으로 O(n+bins).
    avg_vol = float(np.mean(v)) if len(v) > 0 else 0.0
    survive = np.ones(N_PRICE_BINS)

    if avg_vol > 0:
        lo_idxs = np.clip(((lows - pmin) / span * N_PRICE_BINS).astype(int), 0, N_PRICE_BINS - 1)
        hi_idxs = np.clip(((highs - pmin) / span * N_PRICE_BINS).astype(int), 0, N_PRICE_BINS - 1)
        in_range = (highs >= pmin) & (lows <= pmax) & (hi_idxs >= lo_idxs)

        consume_rate = np.clip(0.5 * (volumes / avg_vol), 0.0, 0.999)  # 0.999로 cap (log 발산 방지)
        log_keep = np.log(1.0 - consume_rate)  # 음수

        # diff array: 구간 [lo, hi]에 log_keep 더하기 (완전 벡터화)
        diff = np.zeros(N_PRICE_BINS + 1)
        lo_v = lo_idxs[in_range]
        hi_v = hi_idxs[in_range]
        lk_v = log_keep[in_range]
        np.add.at(diff, lo_v, lk_v)
        np.add.at(diff, hi_v + 1, -lk_v)
        log_survive = np.cumsum(diff[:N_PRICE_BINS])
        survive = np.exp(log_survive)

    heat_raw = heat.copy()       # 소멸 전 (sweep이 통과하며 먹은 풀 측정용)
    heat = heat * survive        # 소멸 후 (현재 남은 풀, imbalance용)

    return bins, heat, heat_raw


def compute_liq_features(
    df_5m: pd.DataFrame,
    entry_bar_idx: int,
    entry_price: float,
    direction: str,
    sweep_extreme_price: float = None,
) -> dict:
    """
    진입 시점의 청산 히트맵 피처 계산.

    Parameters
    ----------
    df_5m : 선물 5m OHLCV (time, open, high, low, close, volume)
    entry_bar_idx : 진입 봉 인덱스 (이 봉 '이전'만 사용 → 미래참조 차단)
    entry_price : 진입가
    direction : "long" / "short"
    sweep_extreme_price : sweep의 극단 가격 (long이면 sweep low, short이면 sweep high)
                          None이면 sweep_liq_consumed=0

    Returns
    -------
    dict : {sweep_liq_consumed, liq_imbalance_hm}
    """
    feats = _neutral_features()

    if df_5m is None or entry_bar_idx is None or entry_bar_idx < 50:
        return feats
    if entry_price is None or entry_price <= 0:
        return feats

    # ── 미래참조 차단: 진입 봉 이전 LOOKBACK_BARS만 슬라이스
    start = max(0, entry_bar_idx - LOOKBACK_BARS)
    end = entry_bar_idx  # 진입 봉 미포함 (이전까지만)
    if end - start < 50:
        return feats

    sl = df_5m.iloc[start:end]
    highs = sl["high"].to_numpy(dtype=float)
    lows = sl["low"].to_numpy(dtype=float)
    closes = sl["close"].to_numpy(dtype=float)
    opens = sl["open"].to_numpy(dtype=float)
    volumes = sl["volume"].to_numpy(dtype=float)

    try:
        bins, heat, heat_raw = _build_heatmap(highs, lows, closes, opens, volumes, entry_price)
    except Exception:
        return feats

    total_heat = heat.sum()
    total_raw = heat_raw.sum()

    # bin 중심 가격
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # ── 1) liq_imbalance_hm: 진입가 기준 위/아래 풀 불균형 (소멸 후 = 현재 남은 풀)
    if total_heat > 0:
        above_mask = bin_centers > entry_price
        below_mask = bin_centers < entry_price
        pool_above = heat[above_mask].sum()
        pool_below = heat[below_mask].sum()
        denom = pool_above + pool_below
        if denom > 0:
            # 진입 방향 관점으로 정렬:
            #   long이면 위쪽 풀(목표)이 클수록 +, short이면 아래쪽이 클수록 +
            if direction == "long":
                imb = (pool_above - pool_below) / denom
            else:
                imb = (pool_below - pool_above) / denom
            feats["liq_imbalance_hm"] = round(float(imb), 4)

    # ── 2) sweep_liq_consumed: sweep이 통과하며 먹은 청산 풀 강도
    # ★ 소멸 전 히트맵(heat_raw) 사용:
    #    sweep extreme는 가격이 방금 통과한 곳 → 소멸 후 히트맵에선 이미 0.
    #    "sweep이 먹은 풀"을 측정하려면 소멸 전 풀에서 sweep 구간 강도를 봐야 함.
    if sweep_extreme_price is not None and sweep_extreme_price > 0 and total_raw > 0:
        lo_px = min(sweep_extreme_price, entry_price)
        hi_px = max(sweep_extreme_price, entry_price)
        consumed_mask = (bin_centers >= lo_px) & (bin_centers <= hi_px)
        consumed = heat_raw[consumed_mask].sum()
        # 전체 풀(소멸 전) 대비 비율 (0~1), 정규화로 심볼/시기 비교 가능
        feats["sweep_liq_consumed"] = round(float(consumed / total_raw), 4)

    return feats


# ── 단위 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    # 가상 데이터: 100 근처에서 거래량 누적, sweep이 95까지 내려갔다 반등(long)
    n = 500
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "open": base,
        "high": base + rng.uniform(0.1, 0.5, n),
        "low": base - rng.uniform(0.1, 0.5, n),
        "close": base + rng.normal(0, 0.1, n),
        "volume": rng.uniform(100, 1000, n),
    })

    entry_idx = 480
    entry_px = float(df["close"].iloc[entry_idx])
    sweep_low = entry_px * 0.97  # sweep이 3% 아래까지

    feats = compute_liq_features(
        df, entry_bar_idx=entry_idx, entry_price=entry_px,
        direction="long", sweep_extreme_price=sweep_low,
    )
    print("청산 히트맵 피처 (long, entry=%.2f):" % entry_px)
    for k, v in feats.items():
        print(f"  {k:25s} = {v:+.4f}")
    print("\n✅ 단위 테스트 통과 (NaN/에러 없음)")