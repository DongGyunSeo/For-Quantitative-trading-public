"""월별 손익의 평균 변화점 — 수익 곡선의 기울기가 꺾인 곳.

수익 곡선(누적 손익)의 기울기 = 그 구간의 월 평균 손익이다. 그래서 "변곡점" 은
월별 손익 수열에서 **평균이 바뀐 지점**으로 찾는다.

탐지
  모든 분할점 k (양쪽 최소 ``min_seg`` 개월)에서 두 구간 평균 차이의 t (합동분산)를 구하고
  |t| 최대인 k 를 고른다. 정규화 CUSUM 과 같은 추정량이다.

유의성
  k 를 **고른 뒤의** t 는 과대평가된다(여러 k 중 최대). 그래서 귀무분포를
  같은 절차로 만든다 — 3개월 블록 단위로 순서를 섞어(달 사이 상관 보존) 최대 |t| 를
  다시 구한다. p = 섞은 쪽 최대 |t| ≥ 관측 최대 |t| 의 비율.

두 번째 변화점은 이진 분할: 첫 변화점이 유의하면(p < alpha) 더 긴 쪽 구간에서 한 번 더.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["split_t", "best_split", "block_perm_p", "ChangePoint", "detect"]


def split_t(x: np.ndarray, min_seg: int = 6) -> np.ndarray:
    """분할점 k 마다 합동분산 t (앞 구간 평균 − 뒤 구간 평균). 불가능한 k 는 NaN."""
    x = np.asarray(x, float)
    T = len(x)
    out = np.full(T, np.nan)
    if T < 2 * min_seg:
        return out
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    for k in range(min_seg, T - min_seg + 1):
        n1, n2 = k, T - k
        m1 = cs[k - 1] / n1
        m2 = (cs[-1] - cs[k - 1]) / n2
        ss1 = cs2[k - 1] - n1 * m1 * m1
        ss2 = (cs2[-1] - cs2[k - 1]) - n2 * m2 * m2
        s2 = (ss1 + ss2) / (T - 2)
        if s2 <= 0:
            continue
        out[k] = (m1 - m2) / np.sqrt(s2 * (1 / n1 + 1 / n2))
    return out


def best_split(x: np.ndarray, min_seg: int = 6) -> tuple[int, float]:
    t = split_t(x, min_seg)
    if not np.isfinite(t).any():
        return -1, np.nan
    k = int(np.nanargmax(np.abs(t)))
    return k, float(t[k])


def block_perm_p(x: np.ndarray, stat: float, *, min_seg: int = 6, block: int = 3,
                 n_perm: int = 2000, rng: np.random.Generator | None = None) -> float:
    """블록 순서 섞기로 만든 최대 |t| 분포에서 관측값 이상의 비율."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    T = len(x)
    nb = int(np.ceil(T / block))
    blocks = [x[i * block:(i + 1) * block] for i in range(nb)]
    hit = 0
    for _ in range(n_perm):
        order = rng.permutation(nb)
        y = np.concatenate([blocks[i] for i in order])
        t = split_t(y, min_seg)
        if np.isfinite(t).any() and np.nanmax(np.abs(t)) >= abs(stat) - 1e-12:
            hit += 1
    return (hit + 1) / (n_perm + 1)


@dataclass
class ChangePoint:
    k: int                 # 변화 후 첫 구간의 위치
    t: float
    p: float
    before: float          # 앞 구간 평균
    after: float           # 뒤 구간 평균


def detect(x: np.ndarray, *, min_seg: int = 6, alpha: float = 0.05, n_perm: int = 2000,
           seed: int = 0, second: bool = True) -> list[ChangePoint]:
    """최대 두 개의 변화점. 첫 번째는 유의하지 않아도 보고한다(그래야 '없음' 을 말할 수 있다)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    k, t = best_split(x, min_seg)
    if k < 0:
        return []
    p = block_perm_p(x, t, min_seg=min_seg, n_perm=n_perm, rng=rng)
    out = [ChangePoint(k, t, p, float(x[:k].mean()), float(x[k:].mean()))]
    if second and p < alpha:
        lo, hi = (0, k) if k >= len(x) - k else (k, len(x))
        seg = x[lo:hi]
        k2, t2 = best_split(seg, min_seg)
        if k2 >= 0:
            p2 = block_perm_p(seg, t2, min_seg=min_seg, n_perm=n_perm, rng=rng)
            if p2 < alpha:
                out.append(ChangePoint(lo + k2, t2, p2, float(seg[:k2].mean()), float(seg[k2:].mean())))
    return out
