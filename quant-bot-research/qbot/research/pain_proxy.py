"""무상태 프록시 — 트레이더 시뮬레이션 없이 "손절 임박 불균형" 을 재기.

분해에서 밝혀진 것: `painR3` 의 정보는 |미실현손실| 이 **0.25R ~ 0.75R** 인 구간에만
있다. 1R 을 넘으면 정보가 0 이다. 즉 재고 있던 것은 "고통의 총량" 이 아니라
**손절까지 가까워졌지만 아직 닿지 않은 포지션의 방향 불균형** 이었다.

필요한 재료는 셋뿐이고 전부 가격과 ATR 로 만들 수 있다.

    (1) 최근에 누가 어느 방향으로 들어갔는가   -> 그때의 추세 부호
    (2) 진입가 대비 지금 얼마나 물려 있는가    -> 가격 차 / 그때의 ATR
    (3) 손절폭이 얼마였는가                    -> 거리 스케일 d (ATR 배수)

진입·청산 로직도, 에피소드 경계도, 손절 규칙도 없다.
트레이더 집단 기계장치의 정직한 대조군이다.

룩어헤드 없음: ``t`` 의 값은 ``c[t]`` 와 ``c[t-j] (j>=1)``, ``atr[t-j]`` 만 쓴다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trend_side", "trapped_imbalance", "SCALES"]

#: 손절폭 후보 (ATR 배수). pivot_break 5 TF 의 실측 1R/ATR 은 2.9 ~ 55.6 이었다.
SCALES = (1.0, 2.0, 5.0, 12.0, 30.0)


def trend_side(close: np.ndarray, w: int) -> np.ndarray:
    """``w`` 봉 전 대비 부호. 진입 시점에 "그 사람이 롱이었나 숏이었나"의 대리."""
    n = len(close)
    past = np.full(n, np.nan)
    past[w:] = close[:-w]
    return np.sign(close - past)


def trapped_imbalance(close: np.ndarray, atr: np.ndarray, *, lookback: int,
                      w: int, scales=SCALES, lo: float = 0.25, hi: float = 0.75
                      ) -> dict[float, np.ndarray]:
    """거리 스케일마다 "띠 안에 물린" 방향 불균형.

    ``t`` 에서 ``j in [1..lookback]`` 을 훑으며, ``t-j`` 에 그 방향으로 들어간
    사람이 지금 ``lo*d ~ hi*d`` (ATR 단위) 만큼 **손실** 중이면 센다.
    반환값은 ``(숏 물림 − 롱 물림) / lookback``.

    부호 규약은 `pain_imb_R3` 와 같다 — 양수 = 숏이 더 물렸다.
    """
    n = len(close)
    side = trend_side(close, w)
    out = {d: np.zeros(n) for d in scales}
    cnt = np.zeros(n)
    for j in range(1, lookback + 1):
        ep = np.full(n, np.nan); ep[j:] = close[:-j]          # 진입가
        ea = np.full(n, np.nan); ea[j:] = atr[:-j]            # 진입 시점 ATR
        sd = np.full(n, np.nan); sd[j:] = side[:-j]
        ok = np.isfinite(ep) & np.isfinite(ea) & (ea > 0) & np.isfinite(sd) & (sd != 0)
        loss = np.where(ok, np.maximum(0.0, (ep - close) * sd)
                        / np.where(ea > 0, ea, 1.0), np.nan)
        cnt += ok
        for d in scales:
            inb = ok & (loss >= lo * d) & (loss <= hi * d)
            out[d] += np.where(inb & (sd < 0), 1.0, 0.0) - np.where(inb & (sd > 0), 1.0, 0.0)
    for d in scales:
        out[d] /= np.maximum(cnt, 1.0)
    return out
