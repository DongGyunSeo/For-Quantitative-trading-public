"""OHLCV CSV 로더 + 무결성 리포트.

컬럼 규약: ``time,open,high,low,close,volume``. ``time`` 은 **오픈 타임**(UTC).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DataQuality", "load_ohlcv", "symbol_files"]

COLS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class DataQuality:
    n: int
    start: pd.Timestamp
    end: pd.Timestamp
    #: 기대 간격 대비 빠진 봉 수
    gaps: int
    #: 가장 긴 공백 (봉 수)
    max_gap: int
    dupes: int
    #: high < low 같은 말이 안 되는 봉
    bad_ohlc: int
    zero_volume: int

    def __str__(self) -> str:
        return (f"{self.n:,}행 {self.start:%Y-%m-%d}~{self.end:%Y-%m-%d} "
                f"공백 {self.gaps} (최장 {self.max_gap}) 중복 {self.dupes} "
                f"이상봉 {self.bad_ohlc} 무거래 {self.zero_volume}")


def symbol_files(cache: str | Path, pattern: str = "*_5m_futures") -> dict[str, Path]:
    """캐시 폴더 -> {심볼: 경로}. ``.csv`` 와 ``.csv.gz`` 를 모두 찾고,
    둘 다 있으면 압축본을 쓴다(수집기가 gz 로 내보내므로).
    """
    cache = Path(cache)
    out: dict[str, Path] = {}
    for ext in (".csv", ".csv.gz"):
        for f in sorted(cache.glob(pattern + ext)):
            out[f.name.split("_")[0]] = f
    return dict(sorted(out.items()))


def load_ohlcv(path: str | Path, *, step: str = "5min") -> tuple[pd.DataFrame, DataQuality]:
    """CSV -> (시각 인덱스 DataFrame, 무결성 리포트).

    인덱스는 tz-aware UTC 이고 **마이크로초 해상도**로 고정한다.
    ``index.view("int64")`` 를 저장했다가 되읽을 때 단위가 흔들리면
    1970년이 나온다 — 한 번 겪었으므로 여기서 못 박는다.
    """
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"'time' 컬럼이 없다: {list(df.columns)}")
    ts = pd.to_datetime(df["time"], utc=True).dt.as_unit("us")
    df = df.drop(columns=["time"])
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 누락: {missing}")
    df = df[list(COLS)].astype(np.float64)
    df.index = pd.DatetimeIndex(ts, name="time")

    dupes = int(df.index.duplicated().sum())
    if dupes:
        df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    d = df.index.to_series().diff().dropna()
    unit = pd.Timedelta(step)
    steps = (d / unit).round().astype("int64")
    gaps = int((steps - 1).clip(lower=0).sum())
    max_gap = int((steps - 1).max()) if len(steps) else 0

    bad = int(((df.high < df.low) | (df.high < df.open) | (df.high < df.close)
               | (df.low > df.open) | (df.low > df.close)).sum())
    q = DataQuality(len(df), df.index[0], df.index[-1], gaps, max_gap, dupes,
                    bad, int((df.volume <= 0).sum()))
    return df, q
