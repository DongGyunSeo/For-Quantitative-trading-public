"""스윙(프랙탈) 탐지 — 구조 분석의 최소 원자.

``lb`` 개 봉이 **양옆에** 있어야 스윙이 확정된다. 즉 i 번 봉의 스윙 여부는
``i + lb`` 시점에야 알 수 있다. 이 지연을 쓰는 쪽에서 반드시 shift 로 넣어야 한다
(`swing_points` 자체는 지연을 넣지 않는다 — 판정만 한다).
"""

from __future__ import annotations

import numpy as np

__all__ = ["swing_points", "confirmed_levels"]


def swing_points(high: np.ndarray, low: np.ndarray, lb: int = 3
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(스윙고 마스크, 스윙저 마스크). 양옆 ``lb`` 봉보다 높거나(낮거나) 같지 않다."""
    n = len(high)
    if lb < 1:
        raise ValueError("lb >= 1")
    is_h = np.zeros(n, bool)
    is_l = np.zeros(n, bool)
    if n < 2 * lb + 1:
        return is_h, is_l
    core = slice(lb, n - lb)
    h = high[core]
    l = low[core]
    ok_h = np.ones(len(h), bool)
    ok_l = np.ones(len(l), bool)
    for k in range(1, lb + 1):
        ok_h &= (h > high[lb - k: n - lb - k]) & (h > high[lb + k: n - lb + k])
        ok_l &= (l < low[lb - k: n - lb - k]) & (l < low[lb + k: n - lb + k])
    is_h[core] = ok_h
    is_l[core] = ok_l
    return is_h, is_l


def confirmed_levels(high: np.ndarray, low: np.ndarray, lb: int = 3
                     ) -> tuple[np.ndarray, np.ndarray]:
    """각 시점에서 **이미 확정된** 마지막 스윙 고/저 가격.

    프랙탈은 뒤 ``lb`` 봉이 있어야 확정되므로 ``shift(lb)`` 로 지연을 구조적으로 넣는다.
    """
    import pandas as pd

    is_h, is_l = swing_points(high, low, lb)
    hv = pd.Series(np.where(is_h, high, np.nan)).shift(lb).ffill().to_numpy()
    lv = pd.Series(np.where(is_l, low, np.nan)).shift(lb).ffill().to_numpy()
    return hv, lv


def structure_events(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     lb: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """구조 전환(CHoCH)과 구조 돌파(BOS).

    확정 스윙 레벨을 종가로 돌파할 때 사건이 난다. 그 방향이

        현재 추세와 **반대**  ->  CHoCH (change of character) : 반환값 1
        현재 추세와 **같음**  ->  BOS   (break of structure)  : 반환값 2

    추세 상태는 마지막 돌파 방향으로 정의한다. 첫 돌파는 추세가 없으므로 BOS 로 센다.

    Returns
    -------
    (부호, 종류) — 부호 ∈ {−1,0,+1}, 종류 ∈ {0=없음, 1=CHoCH, 2=BOS}

    확정 스윙은 이미 ``lb`` 만큼 지연돼 있으므로 룩어헤드가 없다.
    """
    n = len(close)
    hv, lv = confirmed_levels(high, low, lb)
    sign = np.zeros(n, np.int8)
    kind = np.zeros(n, np.int8)
    trend = 0
    for i in range(n):
        up = np.isfinite(hv[i]) and close[i] > hv[i]
        dn = np.isfinite(lv[i]) and close[i] < lv[i]
        if up and not dn:
            s = 1
        elif dn and not up:
            s = -1
        else:
            continue
        if s == trend:
            continue                      # 같은 방향 연속 돌파는 한 번만 센다
        sign[i] = s
        kind[i] = 1 if trend != 0 else 2
        trend = s
    return sign, kind
