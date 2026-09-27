"""주봉 캔들 · 요일 일봉 — `claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`.

- 주봉: UTC 월 00:00 → 다음 월 00:00
- 일봉: 뉴욕 자정(DST 반영) 또는 UTC 자정
- 캔들 특징: 몸통 비율 b, 윗꼬리 u, 아랫꼬리 d (b + u + d = 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ET", "weekly_utc", "daily", "candle_shape", "shape_cell", "day_class", "trailing_sigma"]

ET = "America/New_York"


def _ohlc(df: pd.DataFrame, key: pd.Index) -> pd.DataFrame:
    g = df.groupby(key)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    out["n"] = g.size()
    return out


def weekly_utc(df: pd.DataFrame) -> pd.DataFrame:
    """UTC 월요일 00:00 시작 주봉. 인덱스 = 주 시작(UTC). 이번 주가 전주 고점/저점을 처음 넘은 시각(시간)도 붙인다."""
    idx = df.index
    wk = (idx.normalize() - pd.to_timedelta(idx.dayofweek, unit="D"))
    W = _ohlc(df, wk)
    W.index.name = "week"
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    pos = np.searchsorted(wk.as_unit("ns").asi8, W.index.as_unit("ns").asi8, side="left")
    end = np.append(pos[1:], len(df))
    up_h, dn_h = np.full(len(W), np.nan), np.full(len(W), np.nan)
    ph, pl = W["high"].shift(1).to_numpy(), W["low"].shift(1).to_numpy()
    t0 = idx.as_unit("ns").asi8
    for i in range(1, len(W)):
        s, e = pos[i], end[i]
        if e <= s or not np.isfinite(ph[i]):
            continue
        a = h[s:e] > ph[i]
        b = l[s:e] < pl[i]
        if a.any():
            up_h[i] = (t0[s + int(a.argmax())] - t0[s]) / 3.6e12
        if b.any():
            dn_h[i] = (t0[s + int(b.argmax())] - t0[s]) / 3.6e12
    W["up_first_h"], W["dn_first_h"] = up_h, dn_h
    return W


def daily(df: pd.DataFrame, tz: str = ET) -> pd.DataFrame:
    """tz 자정 기준 일봉. 인덱스 = 그 날짜(자정, tz). DST 로 23/25시간인 날도 그 날 그대로."""
    loc = df.index.tz_convert(tz)
    D = _ohlc(df.set_axis(loc), loc.normalize())
    D.index.name = "day"
    D["dow"] = D.index.dayofweek
    D["ret"] = D["close"] / D["close"].shift(1) - 1
    return D


def candle_shape(o, h, l, c) -> pd.DataFrame:
    """몸통 b · 윗꼬리 u · 아랫꼬리 d (범위 대비). 범위 0 이면 NaN."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.abs(c - o) / rng
        u = (h - np.maximum(o, c)) / rng
        d = (np.minimum(o, c) - l) / rng
    bad = ~(rng > 0)
    b[bad] = u[bad] = d[bad] = np.nan
    return pd.DataFrame(dict(dir=np.sign(c - o), body=b, upper=u, lower=d))


def shape_cell(sh: pd.DataFrame) -> pd.Series:
    """사전등록 14칸 라벨. 방향 0 이거나 범위 0 이면 None."""
    body = np.where(sh.body >= 2 / 3, "큼", np.where(sh.body >= 1 / 3, "중간", "작음"))
    wick = np.where((sh.upper >= 1 / 3) & (sh.upper > sh.lower), "윗꼬리 김",
                    np.where((sh.lower >= 1 / 3) & (sh.lower > sh.upper), "아랫꼬리 김", "짧음"))
    d = np.where(sh.dir > 0, "양봉", np.where(sh.dir < 0, "음봉", ""))
    lab = pd.Series([f"{a} · 몸통 {b} · {w}" if a else None for a, b, w in zip(d, body, wick)], index=sh.index)
    lab[sh.body.isna()] = None
    return lab


def trailing_sigma(ret: pd.Series, days: int = 30) -> pd.Series:
    """그날 **전까지** 30일 일간 수익 표준편차 (인과, 그날 수익은 안 쓴다)."""
    return ret.rolling(days, min_periods=days // 2).std().shift(1)


def day_class(r: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    """사전등록 말 → 불리언. r 일간 수익, s 그 기준 σ."""
    z = r / s
    return dict(mixed=np.abs(z) < 0.5, up_small=(z > 0) & (z <= 1), dn_small=(z < 0) & (z >= -1),
                big_dn=z <= -1, big_up=z >= 1)
