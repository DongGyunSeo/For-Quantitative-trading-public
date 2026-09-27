# timeframes.py
import pandas as pd
from typing import Tuple


def resample_5m_to_15m_1h(
    df_5m: pd.DataFrame,
    time_col: str = "time",
    tz: str = "UTC",
    drop_incomplete: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    5분봉(보통 2000개)을 15분봉 / 1시간봉 OHLCV로 리샘플링해서 반환.

    요구 컬럼:
      - time_col (기본 "time")
      - open, high, low, close, volume

    반환:
      (df_15m, df_1h)

    변경 이력:
      v0.2 — pandas 2.2+ deprecation 대응
        "15T" -> "15min"
        "1H"  -> "1h"
    """
    required = {time_col, "open", "high", "low", "close", "volume"}
    missing = required - set(df_5m.columns)
    if missing:
        raise ValueError(f"df_5m missing columns: {sorted(missing)}")

    x = df_5m.copy()

    # time 컬럼을 datetime으로 변환
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col])

    # 타임존 정리: naive면 로컬라이즈, aware면 tz로 변환
    if x[time_col].dt.tz is None:
        x[time_col] = x[time_col].dt.tz_localize(tz)
    else:
        x[time_col] = x[time_col].dt.tz_convert(tz)

    # 시간순 정렬 후 index 설정
    x = x.sort_values(time_col).set_index(time_col)

    # 중복 타임스탬프 제거
    x = x[~x.index.duplicated(keep="last")]

    def _resample(rule: str, expected_count: int) -> pd.DataFrame:
        agg = x.resample(
            rule,
            label="left",
            closed="left",
            origin="start_day",
        ).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        })

        if drop_incomplete:
            counts = x["close"].resample(
                rule, label="left", closed="left", origin="start_day"
            ).size()
            valid = counts == expected_count
            if not valid.empty:
                valid.iloc[:-1] = True  # 과거 데이터는 API 통신 등에 의한 손실 틱이 있어도 강제 보존
            agg = agg[valid]

        agg = agg.dropna(subset=["open", "high", "low", "close"])
        agg = agg.reset_index().rename(columns={agg.index.name: time_col})
        return agg

    # ── pandas 2.2+ 호환 offset 문자열 사용 ──
    df_15m = _resample("15min", expected_count=3)   # 구: "15T"
    df_1h  = _resample("1h",    expected_count=12)  # 구: "1H"

    return df_15m, df_1h


def resample_5m_to_4h(
    df_5m: pd.DataFrame,
    time_col: str = "time",
    tz: str = "UTC",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    5분봉을 4시간봉 OHLCV로 리샘플링.
    4H봉 1개 = 5m봉 48개 (완성봉만 반환).
    """
    required = {time_col, "open", "high", "low", "close", "volume"}
    missing = required - set(df_5m.columns)
    if missing:
        raise ValueError(f"df_5m missing columns: {sorted(missing)}")

    x = df_5m.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col])
    if x[time_col].dt.tz is None:
        x[time_col] = x[time_col].dt.tz_localize(tz)
    else:
        x[time_col] = x[time_col].dt.tz_convert(tz)
    x = x.sort_values(time_col).set_index(time_col)
    x = x[~x.index.duplicated(keep="last")]

    agg = x.resample("4h", label="left", closed="left", origin="start_day").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })
    if drop_incomplete:
        counts = x["close"].resample("4h", label="left", closed="left", origin="start_day").size()
        valid = counts == 48
        if not valid.empty:
            valid.iloc[:-1] = True  # 과거 데이터는 손실 틱이 있어도 보존
        agg = agg[valid]
    agg = agg.dropna(subset=["open","high","low","close"])
    agg = agg.reset_index().rename(columns={agg.index.name: time_col})
    return agg