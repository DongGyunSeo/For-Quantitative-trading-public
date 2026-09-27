# swings.py  v2.1 — Fractal + Dynamic Lookback (ZigZag 제거)
# ─────────────────────────────────────────────────────────────────
# 변경 이력:
#   v1.0: 기본 fractal pivot 감지 (lookback 고정)
#   v2.0: ZigZag deviation 필터 결합 + ATR 기반 dynamic lookback
#   v2.1: ZigZag 비활성화 (compress_pivots과 중복 + tail 호출 시 비결정적)
#         dynamic lookback만 사용, TF별 lb_low/lb_high 분리
#
# 구조:
#   1단계: ATR 기반 dynamic lookback 결정
#   2단계: fractal pivot 감지
#   (ZigZag 교대 필터는 compress_pivots(mode="zigzag")에 위임)
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════
# (1) 기존 Fractal Pivot 감지 (변경 없음)
# ══════════════════════════════════════════════════════════════════

def find_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Mark confirmed pivot/fractal swings on OHLCV DataFrame.

    swing_high at i:
      high[i] >= max(high[i-lookback : i]) AND high[i] > max(high[i+1 : i+1+lookback])

    swing_low at i:
      low[i] <= min(low[i-lookback : i]) AND low[i] < min(low[i+1 : i+1+lookback])

    Notes:
      - Left: >= (같거나 높으면 통과), Right: > (엄격)
      - Edges cannot be evaluated: i in [lookback, n-lookback-1]
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("df must contain 'high' and 'low' columns")

    out = df.copy()
    out["swing_high"] = False
    out["swing_low"] = False

    highs = out["high"].astype(float).to_list()
    lows = out["low"].astype(float).to_list()
    n = len(out)

    if n < (2 * lookback + 1):
        return out

    for i in range(lookback, n - lookback):
        left_high_max = max(highs[i - lookback : i])
        right_high_max = max(highs[i + 1 : i + 1 + lookback])

        left_low_min = min(lows[i - lookback : i])
        right_low_min = min(lows[i + 1 : i + 1 + lookback])

        if highs[i] >= left_high_max and highs[i] > right_high_max:
            out.at[out.index[i], "swing_high"] = True

        if lows[i] <= left_low_min and lows[i] < right_low_min:
            out.at[out.index[i], "swing_low"] = True

    return out


# ══════════════════════════════════════════════════════════════════
# (2) ATR 기반 Dynamic Lookback
# ══════════════════════════════════════════════════════════════════

def _calc_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    True Range 기반 ATR 시리즈 계산.
    rolling mean 사용 (Wilder 스무딩 대비 연산 빠름, 차이 미미).
    """
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    cl = df["close"].astype(float)

    tr1 = hi - lo
    tr2 = (hi - cl.shift(1)).abs()
    tr3 = (lo - cl.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


def _dynamic_lookback(atr_current: float, atr_mean: float,
                      lb_low: int = 2, lb_high: int = 4) -> int:
    """
    ATR 기반 lookback 동적 결정.
      고변동 (ATR > 평균 120%): lb_low (빠른 반응, 스윙 빨리 확정)
      저변동 (ATR < 평균  80%): lb_high (느린 반응, 노이즈 필터)
      중간:                      (lb_low + lb_high) // 2

    기본값 lb_low=2, lb_high=4 → 평균 lb≈3 (5m/15m용)
    1h/4h는 lb_low=3, lb_high=5 → 평균 lb≈4
    """
    if atr_mean <= 0:
        return lb_low
    ratio = atr_current / atr_mean
    if ratio > 1.2:
        return lb_low
    elif ratio < 0.8:
        return lb_high
    else:
        return (lb_low + lb_high) // 2


# ══════════════════════════════════════════════════════════════════
# (3) 통합 함수 — 메인 엔트리포인트
# ══════════════════════════════════════════════════════════════════

def find_swings_enhanced(
    df: pd.DataFrame,
    lookback: int = 3,
    dynamic_lb: bool = True,
    lb_low: int = 2,
    lb_high: int = 4,
    atr_period: int = 14,
    deviation_pct: float = 0.0,   # v2.1: 기본 비활성화, 하위 호환용 유지
) -> pd.DataFrame:
    """
    Fractal Pivot + Dynamic Lookback 통합.

    v2.1 변경:
      - ZigZag deviation 기본 비활성화 (compress_pivots과 중복,
        incremental tail 호출 시 비결정적 결과 발생)
      - lb_low/lb_high 기본값 변경: 5m/15m용 평균 lb≈3

    파라미터:
      lookback:       기본 lookback (dynamic_lb=False일 때 고정값)
      dynamic_lb:     True면 ATR 기반 lookback 동적 조정
      lb_low:         고변동 시 lookback (기본 2)
      lb_high:        저변동 시 lookback (기본 4)
      atr_period:     ATR 계산 기간 (기본 14봉)
      deviation_pct:  ZigZag 최소 변화율 (기본 0 = 비활성화)

    TF별 권장 설정:
      5m/15m: lb_low=2, lb_high=4 → 평균 lb≈3
      1h/4h:  lb_low=3, lb_high=5 → 평균 lb≈4

    반환:
      df + swing_high, swing_low 컬럼 (Boolean)
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("df must contain 'high' and 'low' columns")

    # ── Step 1: Lookback 결정
    actual_lb = lookback
    if dynamic_lb and "close" in df.columns and len(df) > atr_period + 2:
        atr_series = _calc_atr_series(df, period=atr_period)
        atr_current = atr_series.iloc[-1]
        atr_mean = atr_series.dropna().mean()
        if not (np.isnan(atr_current) or np.isnan(atr_mean)):
            actual_lb = _dynamic_lookback(
                float(atr_current), float(atr_mean),
                lb_low=lb_low, lb_high=lb_high
            )

    # ── Step 2: Fractal Pivot 감지
    out = find_swings(df, lookback=actual_lb)

    # ── Step 3: ZigZag Deviation 필터 (기본 비활성화)
    if deviation_pct > 0:
        out = _apply_zigzag_filter(out, deviation_pct=deviation_pct)

    return out


# ══════════════════════════════════════════════════════════════════
# (4) ZigZag Deviation 필터 (하위 호환용 유지, 기본 비활성화)
# ══════════════════════════════════════════════════════════════════

def _apply_zigzag_filter(df: pd.DataFrame, deviation_pct: float = 0.001) -> pd.DataFrame:
    """
    fractal swing 결과에 ZigZag deviation 필터 적용.

    ⚠️ v2.1: 기본 비활성화. compress_pivots(mode="zigzag")이 동일 역할을 수행하며,
    incremental tail 호출 시 첫 스윙 시작점이 달라져 비결정적 결과 발생.

    원리: "직전 확정 스윙"으로부터의 가격 변화가 deviation_pct 미만이면
          노이즈로 간주하고 swing 마크를 제거.
    """
    out = df.copy()
    highs = out["high"].astype(float).values
    lows = out["low"].astype(float).values
    sh = out["swing_high"].values.copy()
    sl = out["swing_low"].values.copy()
    n = len(out)

    candidates = []
    for i in range(n):
        if sh[i]:
            candidates.append((i, "H", highs[i]))
        if sl[i]:
            candidates.append((i, "L", lows[i]))

    if len(candidates) < 2:
        return out

    confirmed = []
    last_type = None
    last_price = 0.0

    for idx, kind, price in candidates:
        if last_type is None:
            confirmed.append((idx, kind, price))
            last_type = kind
            last_price = price
            continue

        if kind == last_type:
            prev_idx, prev_kind, prev_price = confirmed[-1]
            if kind == "H" and price > prev_price:
                confirmed[-1] = (idx, kind, price)
                last_price = price
            elif kind == "L" and price < prev_price:
                confirmed[-1] = (idx, kind, price)
                last_price = price
        else:
            if last_price > 0:
                change = abs(price - last_price) / last_price
                if change < deviation_pct:
                    continue

            confirmed.append((idx, kind, price))
            last_type = kind
            last_price = price

    confirmed_indices_h = set()
    confirmed_indices_l = set()
    for idx, kind, price in confirmed:
        if kind == "H":
            confirmed_indices_h.add(idx)
        else:
            confirmed_indices_l.add(idx)

    for i in range(n):
        out.iat[i, out.columns.get_loc("swing_high")] = (i in confirmed_indices_h)
        out.iat[i, out.columns.get_loc("swing_low")]  = (i in confirmed_indices_l)

    return out


# ══════════════════════════════════════════════════════════════════
# (5) 유틸 — extract_swings (기존 호환)
# ══════════════════════════════════════════════════════════════════

def extract_swings(out: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows where swing_high or swing_low is True.
    """
    if "swing_high" not in out.columns or "swing_low" not in out.columns:
        raise ValueError("Run find_swings() first.")

    cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in out.columns]
    return out[out["swing_high"] | out["swing_low"]][cols + ["swing_high", "swing_low"]]
