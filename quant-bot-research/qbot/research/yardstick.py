"""공통 잣대 — 모든 상태량 연구가 같은 자로 재도록.

(모멘텀1d × 봉폭) 고정 5분위 스프레드 @ 4시간, 단위 = 일간중앙 5m ATR,
클러스터 = (심볼 × 월). 예전에는 이 코드가 `examples/` 여기저기 복사돼 있어
어떤 문서가 직교화한 값이고 어떤 게 raw 인지 헷갈렸다 — 한 곳으로 모은다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pain_decomp import monthly_spread, quintile

__all__ = ["Yardstick", "build", "orthogonalize", "tstat", "cluster_t"]

#: 통제 집합에 쓰는 추세 룩백 (5m 봉): 1일 · 7일 · 30일
MOMS = (288, 2016, 8640)
#: 평가 지평 (5m 봉). 4시간.
H = 48


class Yardstick:
    """한 심볼의 잣대 재료. 통제 직교기저를 한 번만 만들어 재사용한다."""

    __slots__ = ("atr", "fwd", "month", "ok", "cell", "ok_orth", "_Q")

    def __init__(self, atr, fwd, month, ok, cell, ok_orth, Q):
        self.atr, self.fwd, self.month = atr, fwd, month
        self.ok, self.cell, self.ok_orth, self._Q = ok, cell, ok_orth, Q

    def spread(self, score: np.ndarray, *, orth: bool = False
               ) -> tuple[np.ndarray, np.ndarray]:
        """점수 -> (월, 스프레드). ``orth`` 면 통제 집합의 열공간을 먼저 뺀다."""
        s, m = (orthogonalize(score, self.ok_orth, self._Q), self.ok_orth) if orth \
            else (score, self.ok)
        return monthly_spread(self.month, self.fwd, quintile(s, m), self.cell)


def build(df: pd.DataFrame) -> Yardstick:
    n = len(df)
    o, c, h, l = (df[k].to_numpy(np.float64) for k in ("open", "close", "high", "low"))
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a14 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    a = pd.Series(a14).rolling(288, min_periods=48).median().to_numpy()
    a = np.maximum(a, 5e-4 * c)
    a = np.where(np.isfinite(a) & (a > 0), a, np.nan)

    entry = np.full(n, np.nan); entry[:-1] = o[1:]
    ex = np.full(n, np.nan)
    j = np.arange(n) + 1 + H
    v = j < n
    ex[v] = c[j[v]]
    fwd = (ex - entry) / a
    rng_ = (h - l) / a
    past = np.full(n, np.nan); past[288:] = c[:-288]
    mom = (c - past) / a
    month = df.index.year.to_numpy() * 12 + df.index.month.to_numpy()
    ok = np.isfinite(fwd) & np.isfinite(rng_) & np.isfinite(mom)

    ctrl = []
    for k in MOMS:
        p = np.full(n, np.nan); p[k:] = c[:-k]
        m_ = (c - p) / a
        ctrl += [m_, np.sign(m_) * m_ ** 2]
    ctrl.append(rng_)
    ok_o = ok.copy()
    for cc in ctrl:
        ok_o &= np.isfinite(cc)
    A = np.column_stack([np.ones(int(ok_o.sum()))] + [cc[ok_o] for cc in ctrl])
    Q, _ = np.linalg.qr(A)

    cell = quintile(mom, ok).astype(np.int64) * 10 + quintile(rng_, ok)
    return Yardstick(a, fwd, month, ok, cell, ok_o, Q)


def orthogonalize(score: np.ndarray, ok_o: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """통제 집합의 열공간을 뺀 잔차. ``Q`` 는 미리 구한 직교기저."""
    y = score[ok_o]
    out = np.full(len(score), np.nan)
    out[ok_o] = y - Q @ (Q.T @ y)
    return out


def tstat(x) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    sd = x.std(ddof=1)
    m = float(x.mean())
    return m, (m / (sd / np.sqrt(len(x))) if sd > 0 else np.nan), len(x)


def cluster_t(x, groups) -> tuple[float, float, int, int]:
    """**거래 가중 평균**의 t — 군집(월·분기) 강건 표준오차.

    `tstat(월평균)` 은 달마다 거래 수가 달라도 **달을 같은 무게로** 센다. 그래서
    보고한 평균(거래 가중)과 t 가 서로 다른 것을 재고 있었다 — 신호가 몰리는
    달(변동성 폭발)이 나쁘면 월평균은 좋아 보이고 거래 가중 평균은 나빠진다.
    봇은 신호마다 같은 위험을 걸므로 손익은 거래 가중이다. 이 t 가 그 평균의 t 다.

    var = G/(G−1) · Σ_g (Σ_{i∈g} (x_i − x̄))² / N²   (군집이 전부 1개짜리면 보통 t 와 같다)

    Returns (평균, t, 거래 수, 군집 수)
    """
    x = np.asarray(x, float)
    g = np.asarray(groups)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n, 0
    m = float(x.mean())
    s = pd.Series(x - m).groupby(g).sum().to_numpy()
    G = len(s)
    if G < 3:
        return m, float("nan"), n, G
    var = (G / (G - 1)) * float((s ** 2).sum()) / n ** 2
    return m, (m / np.sqrt(var) if var > 0 else float("nan")), n, G
