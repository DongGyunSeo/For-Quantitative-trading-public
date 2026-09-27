"""범용 배리어 레이서 + 치환 대조군.

임의의 (진입봉, 진입가, 1R, 방향) 목록에 대해 TP/SL 레이스를 돌린다.
어떤 신호든 **같은 잣대**로 재기 위한 최소 원자.

두 가지 규약
------------
1. 같은 봉에서 TP·SL 이 모두 걸리면 **모호 → 0 R**. 승리로 세지 않는다.
   (봉 내부 순서는 5m OHLC 로 관측 불가능하다.)
2. 지평 안에 아무것도 안 걸리면 **미해결 → 0 R**.

치환 대조군
-----------
실제 거래의 **1R(bps) 분포·방향 구성·심볼은 그대로** 두고 진입 시점만 섞는다.
"대칭 배리어의 승률은 50% 가 아니다" — 로그정규 기하 때문에 숏 50.3~50.7% /
롱 49.3~49.7% 에서 출발한다. 이 대조군이 그 기준선을 데이터로 만들어 준다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["first_passage_r", "permuted_entries", "control_expectancy"]


def first_passage_r(
    high: np.ndarray,
    low: np.ndarray,
    bars: np.ndarray,
    entry: np.ndarray,
    risk: np.ndarray,
    side: np.ndarray,
    *,
    tp_r: float = 2.0,
    sl_r: float = 1.0,
    horizon: int = 2016,
    skip_entry_bar: bool = True,
) -> np.ndarray:
    """거래별 R 결과.

    Parameters
    ----------
    bars
        진입봉 인덱스. ``skip_entry_bar`` 면 레이스는 그 **다음** 봉부터.
    risk
        1R 을 **가격 단위**로. 0 이하나 비유한이면 결과는 0.
    """
    n = len(high)
    m = len(bars)
    out = np.zeros(m)
    bars = np.asarray(bars, np.int64)
    entry = np.asarray(entry, float)
    risk = np.asarray(risk, float)
    side = np.asarray(side, np.int64)

    for i in range(m):
        sd = int(side[i])
        rk = float(risk[i])
        e = float(entry[i])
        if sd == 0 or not np.isfinite(rk) or rk <= 0 or not np.isfinite(e):
            continue
        s = int(bars[i]) + (1 if skip_entry_bar else 0)
        t = min(n, s + horizon)
        if s >= t:
            continue
        hi = np.maximum.accumulate(high[s:t])
        lo = np.minimum.accumulate(low[s:t])
        tp_px = e + sd * tp_r * rk
        sl_px = e - sd * sl_r * rk
        span = t - s
        if sd > 0:
            i_tp = np.searchsorted(hi, tp_px - 1e-12, "left")
            i_sl = np.searchsorted(-lo, -sl_px - 1e-12, "left")
        else:
            i_tp = np.searchsorted(-lo, -tp_px - 1e-12, "left")
            i_sl = np.searchsorted(hi, sl_px - 1e-12, "left")
        if i_tp >= span and i_sl >= span:
            continue                      # 미해결
        if i_tp == i_sl:
            continue                      # 모호
        out[i] = tp_r if i_tp < i_sl else -sl_r
    return out


def permuted_entries(bars: np.ndarray, n_bars: int, rng: np.random.Generator, *,
                     lo: int = 500, horizon: int = 2016) -> np.ndarray:
    """같은 개수만큼 **진입 시점만** 새로 뽑는다. 앞뒤 여유는 남긴다."""
    hi = n_bars - horizon - 2
    if hi <= lo:
        return np.asarray(bars, np.int64)
    return rng.integers(lo, hi, len(bars)).astype(np.int64)


def control_expectancy(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    bars: np.ndarray,
    risk_bps: np.ndarray,
    side: np.ndarray,
    *,
    n_draw: int = 5,
    rng: np.random.Generator | None = None,
    **race_kw,
) -> tuple[float, np.ndarray]:
    """치환 대조군의 기대 R.

    1R 은 **bps 로 받아** 새 진입가에 맞춰 가격 단위로 되돌린다. 그래야
    "같은 상대 위험, 다른 시점" 이 된다. 방향 구성도 원본 그대로 재사용한다.

    Returns
    -------
    (평균 R, 추첨별 평균 R 배열)
    """
    rng = rng or np.random.default_rng(0)
    n = len(high)
    horizon = int(race_kw.get("horizon", 2016))
    means = np.empty(n_draw)
    for d in range(n_draw):
        b = permuted_entries(bars, n, rng, horizon=horizon)
        e = np.asarray(open_, float)[b]
        r = e * np.asarray(risk_bps, float) / 1e4
        means[d] = float(first_passage_r(high, low, b, e, r, side, **race_kw).mean())
    return float(means.mean()), means
