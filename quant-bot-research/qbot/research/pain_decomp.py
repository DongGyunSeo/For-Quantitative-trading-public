"""painR3 분해 — 1인당 기여를 한 번만 계산하고 임의로 재조합한다.

`population_state` 는 15명을 한 덩어리로 합쳐 버려서 "누가 얼마나 기여했나" 를
되물을 수 없다. 여기서는 트레이더마다 **원재료**(봉당 |미실현손실| · 미실현이익 ·
연속수중봉 · 방향)를 남겨 두고, 부분집합 · 캡 · 가중을 사후에 자유롭게 바꾼다.

원재료만 남기면 캡을 바꿔도 재실행이 필요 없다 — ``min(loss, cap)`` 은 사후 연산이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trader_pop import PopConfig, _expand, run_trader

__all__ = ["TraderContrib", "contributions", "imbalance", "quintile", "monthly_spread"]


@dataclass(slots=True)
class TraderContrib:
    """트레이더 1명의 봉당 기여. 부호는 방향으로만 나뉜다."""

    label: str
    tf: str
    #: +1 롱 생존 / −1 숏 생존 / 0 미생존
    side: np.ndarray          # int8
    #: |미실현손실| (R). 미생존이면 0. float64 — 부분집합 차이를 재는 데
    #: float32 반올림(상대 3e-8)이 섞이면 leave-one-out 델타가 흐려진다.
    loss: np.ndarray
    #: 미실현이익 (R). 미생존이면 0
    gain: np.ndarray
    #: 연속 수중 봉 수. 미생존이면 0
    uw: np.ndarray
    #: 1R / 일간중앙ATR 의 중앙값 — 측정해상도 판정용
    risk_atr: float
    n_ep: int

    @property
    def name(self) -> str:
        return f"{self.label}@{self.tf}"


def contributions(ltf: pd.DataFrame, archs, atr: np.ndarray,
                  cfg: PopConfig = PopConfig()) -> list[TraderContrib]:
    """아키타입 목록 -> 1인당 기여. `population_state` 와 같은 재료를 쓴다."""
    n = len(ltf)
    close = ltf["close"].to_numpy(np.float64)
    a = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    ar = np.arange(n)
    cache: dict = {}
    out: list[TraderContrib] = []

    for arch in archs:
        st = run_trader(ltf, arch, cache, cfg)
        s, e = st.ep_start, st.ep_end
        z = np.zeros(n, np.float64)
        if len(s) == 0:
            out.append(TraderContrib(arch.label or arch.kind, arch.tf,
                                     np.zeros(n, np.int8), z, z.copy(), z.copy(),
                                     np.nan, 0))
            continue
        risk = np.abs(st.ep_entry - st.ep_stop0)
        bad = ~np.isfinite(risk) | (risk <= 0)
        risk = np.where(bad, 2.0 * a[s], risk)
        risk5 = _expand(risk, s, e, n)
        ent5 = _expand(st.ep_entry, s, e, n)
        live = st.side5 != 0
        pnl_R = np.nan_to_num(np.where(live, (close - ent5) * st.side5 / risk5, 0.0))

        uw_mask = live & (pnl_R < 0)
        last_ok = np.maximum.accumulate(np.where(~uw_mask, ar, -1))
        dur = np.where(uw_mask, ar - last_ok, 0)

        out.append(TraderContrib(
            arch.label or arch.kind, arch.tf,
            st.side5.astype(np.int8),
            np.maximum(0.0, -pnl_R),
            np.maximum(0.0, pnl_R),
            dur.astype(np.float64),
            float(np.nanmedian(risk / a[s])), int(len(s))))
    return out


def imbalance(cs: list[TraderContrib], *, axis: str = "pain", cap: float = 3.0,
              weight: str = "sum") -> np.ndarray:
    """부분집합 -> 봉당 불균형 (숏 − 롱).

    axis
        ``pain`` |미실현손실| · ``gain`` 미실현이익 · ``uw`` 연속수중봉 ·
        ``count`` 수중 인원수
    cap
        ``pain`` 축에만 적용. ``np.inf`` 면 무제한(= painR).
    weight
        ``sum``   Σ기여 / Σ생존자  — 현재 painR3 의 정의
        ``equal`` 트레이더마다 자기 생존율로 나눈 뒤 평균 (가동시간 지분 제거)
        ``z``     트레이더마다 자기 표준편차로 나눈 뒤 평균
    """
    if not cs:
        raise ValueError("빈 부분집합")
    n = len(cs[0].side)
    acc = np.zeros(n, np.float64)
    live_tot = np.zeros(n, np.float64)
    parts = []

    for c in cs:
        lg, sh = c.side > 0, c.side < 0
        if axis == "pain":
            v = np.minimum(c.loss, cap) if np.isfinite(cap) else c.loss
        elif axis == "gain":
            v = c.gain
        elif axis == "uw":
            v = c.uw
        elif axis == "count":
            v = (c.loss > 0).astype(np.float64)
        else:
            raise ValueError(f"알 수 없는 축: {axis}")
        d = np.where(sh, v, 0.0) - np.where(lg, v, 0.0)
        live = (c.side != 0)
        live_tot += live
        if weight == "sum":
            acc += d
        elif weight == "equal":
            r = float(live.mean())
            parts.append(d / r if r > 0 else np.zeros(n))
        elif weight == "z":
            sd = float(d.std())
            parts.append(d / sd if sd > 0 else np.zeros(n))
        else:
            raise ValueError(f"알 수 없는 가중: {weight}")

    if weight == "sum":
        return acc / np.maximum(live_tot, 1.0)
    return np.mean(parts, axis=0)


def quintile(x: np.ndarray, ok: np.ndarray, nq: int = 5) -> np.ndarray:
    """전역 분위. 결측·제외는 −1."""
    v = np.where(ok, x, np.nan)
    fin = np.isfinite(v)
    out = np.full(len(x), -1, np.int32)
    if fin.sum() < nq * 10:
        return out
    qs = np.nanquantile(v[fin], np.linspace(0, 1, nq + 1)[1:-1])
    out[fin] = np.searchsorted(qs, v[fin]).astype(np.int32)
    return out


def monthly_spread(month: np.ndarray, fwd: np.ndarray, q: np.ndarray,
                   cell: np.ndarray, *, nq: int = 5, min_cell: int = 30
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(모멘텀 × 봉폭) 셀 고정 상·하위 분위 스프레드를 월별로.

    셀마다 상위분위 평균 − 하위분위 평균을 구하고 ``min(n_hi, n_lo)`` 로 가중평균한다.
    파이썬 이중 groupby 를 bincount 로 바꾼 것이라 결과가 완전히 같아야 한다.
    """
    sel = np.isfinite(fwd) & (q >= 0) & ((q == 0) | (q == nq - 1))
    if not sel.any():
        return np.empty(0, np.int64), np.empty(0)
    m = month[sel]
    f = fwd[sel].astype(np.float64)
    c = cell[sel]
    hi = (q[sel] == nq - 1).astype(np.int64)

    um, mi = np.unique(m, return_inverse=True)
    uc, ci = np.unique(c, return_inverse=True)
    code = (mi * len(uc) + ci) * 2 + hi
    size = len(um) * len(uc) * 2
    cnt = np.bincount(code, minlength=size).reshape(len(um), len(uc), 2)
    tot = np.bincount(code, weights=f, minlength=size).reshape(len(um), len(uc), 2)
    n_lo, n_hi = cnt[..., 0], cnt[..., 1]
    ok = (n_lo >= min_cell) & (n_hi >= min_cell)
    if not ok.any():
        return np.empty(0, np.int64), np.empty(0)
    mean_lo = np.divide(tot[..., 0], np.maximum(n_lo, 1))
    mean_hi = np.divide(tot[..., 1], np.maximum(n_hi, 1))
    w = np.where(ok, np.minimum(n_lo, n_hi), 0).astype(np.float64)
    num = (w * (mean_hi - mean_lo)).sum(1)
    den = w.sum(1)
    keep = den > 0
    return um[keep], num[keep] / den[keep]
