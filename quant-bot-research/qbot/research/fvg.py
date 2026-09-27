"""FVG (3봉 임밸런스) 되돌림 진입.

정의
----
HTF 봉 ``i`` 가 확정될 때 ``i-2, i-1, i`` 세 봉을 본다.

    상승 FVG   low[i] > high[i-2]   ->  구간 [high[i-2], low[i]]   ->  롱
    하락 FVG   high[i] < low[i-2]   ->  구간 [high[i], low[i-2]]   ->  숏

진입은 5m 가격이 구간의 ``entry_level`` 지점(기본 mid)을 **처음 터치**할 때.
**1R = 진입 지점 ~ 구간 반대편 끝** 이다. mid 진입이면 곧 구간 크기의 절반.

측정해상도 규칙
---------------
``1R / ATR5m >= min_r_over_atr`` 을 만족하지 않는 이벤트는 **결과를 못 믿는다**.
1R 이 5분봉보다 작으면 TP 와 SL 이 같은 봉 안에 들어가 승부가 관측 불가능한
봉 내부에서 갈린다. 이 규칙은 원본 FVG 연구(`FVG-mid진입-검증결과.md`)가
층화 + 진입봉 포함/제외 괴리 검사로 만들어낸 산물이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.resample import build_ref_map, resample_htf
from .barrier import first_passage_r

__all__ = ["FVGConfig", "FVGResult", "find_fvg", "run_fvg", "atr5m"]


@dataclass(frozen=True, slots=True)
class FVGConfig:
    htf: str = "4h"
    #: 되돌림을 기다리는 HTF 봉 수
    wait: int = 25
    #: 구간 안의 진입 지점. 0 = 가까운 끝(즉시), 0.5 = mid, 1 = 먼 끝
    entry_level: float = 0.5
    #: 손절을 먼 끝에서 **갭 크기의 몇 배** 더 밀지. 0 이면 정확히 먼 끝.
    #: TF 를 올리지 않고 1R 을 키우는 레버.
    stop_ext: float = 0.0
    #: 사전 1R 하한 (bps). 측정해상도 규칙과 **별개**로, 비용 기준 게이트다.
    #: 사후에 1R 분위를 골라내는 것과 구분할 것 — 이건 규칙의 일부여야 한다.
    min_risk_bps: float = 0.0
    tp_r: float = 2.0
    sl_r: float = 1.0
    #: 5m 봉 단위 추적 지평 (기본 7일)
    horizon: int = 2016
    skip_entry_bar: bool = True
    #: 측정해상도 규칙. 0 이면 끈다.
    min_r_over_atr: float = 2.0
    atr_period: int = 14
    #: 규칙의 분모. ``"daily"`` = ATR14 의 일간(288봉) 중앙값 — 이 프로젝트의
    #: 표준 정규화 스케일이고 **외생적**이다. ``"inst"`` = 터치봉의 ATR14 자체로,
    #: 터치봉이 원래 변동이 큰 순간이라 분모가 내생적으로 부풀어 표본을 깎는다.
    #: 두 값의 차이는 1h 에서 통과 건수 1.5배 — 문서에 반드시 어느 쪽인지 적을 것.
    atr_mode: str = "daily"


@dataclass(slots=True)
class FVGResult:
    #: 진입봉 (``skip_entry_bar`` 면 ``touch_bar + 1``)
    entry_bar: np.ndarray
    touch_bar: np.ndarray
    entry: np.ndarray
    risk: np.ndarray
    side: np.ndarray
    size_bps: np.ndarray
    risk_bps: np.ndarray
    r_over_atr: np.ndarray
    htf_idx: np.ndarray
    r: np.ndarray

    def __len__(self) -> int:
        return len(self.entry_bar)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({k: getattr(self, k) for k in
                             ("htf_idx", "touch_bar", "entry_bar", "side", "entry",
                              "risk", "size_bps", "risk_bps", "r_over_atr", "r")})


def atr5m(df: pd.DataFrame, period: int = 14, *, mode: str = "inst") -> np.ndarray:
    """5m ATR. ``mode="daily"`` 면 일간(288봉) 중앙값으로 안정화한다.

    ATR14 자체는 저변동 구간에서 0 에 가까워져 비율을 폭발시킨다. 이 프로젝트의
    다른 모듈(트레이더 집단·pain 계열)은 전부 일간 중앙값을 쓴다.
    """
    h, l = df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    if mode == "inst":
        return a
    if mode != "daily":
        raise ValueError(f"알 수 없는 mode: {mode}")
    a = pd.Series(a).rolling(288, min_periods=48).median().to_numpy()
    return np.maximum(a, 5e-4 * c)


def find_fvg(htf: pd.DataFrame) -> pd.DataFrame:
    """HTF 봉 -> 확정된 FVG 목록. ``i`` 는 세 번째(확정) 봉의 위치."""
    h = htf["high"].to_numpy(np.float64)
    l = htf["low"].to_numpy(np.float64)
    n = len(htf)
    if n < 3:
        return pd.DataFrame(columns=["i", "side", "bottom", "top"])
    i = np.arange(2, n)
    up = l[2:] > h[:-2]
    dn = h[2:] < l[:-2]
    rows = []
    if up.any():
        k = i[up]
        rows.append(pd.DataFrame({"i": k, "side": 1, "bottom": h[k - 2], "top": l[k]}))
    if dn.any():
        k = i[dn]
        rows.append(pd.DataFrame({"i": k, "side": -1, "bottom": h[k], "top": l[k - 2]}))
    if not rows:
        return pd.DataFrame(columns=["i", "side", "bottom", "top"])
    out = pd.concat(rows, ignore_index=True).sort_values("i").reset_index(drop=True)
    return out[out.top > out.bottom].reset_index(drop=True)


def run_fvg(df: pd.DataFrame, cfg: FVGConfig = FVGConfig()) -> FVGResult:
    """5m OHLCV -> FVG 되돌림 진입 결과."""
    htf = resample_htf(df, cfg.htf)
    fv = find_fvg(htf)
    n5 = len(df)
    hi5 = df["high"].to_numpy(np.float64)
    lo5 = df["low"].to_numpy(np.float64)
    a5 = atr5m(df, cfg.atr_period, mode=cfg.atr_mode)

    # HTF 확정 시각 -> 그 시각 이후 첫 5m 봉 위치
    ct = htf["close_ts"].to_numpy()
    start5 = np.searchsorted(df.index.to_numpy(), ct, side="left")
    ratio = max(int(np.median(htf["n_ltf"])), 1)
    win = cfg.wait * ratio

    m = len(fv)
    touch = np.full(m, -1, np.int64)
    ent = np.full(m, np.nan)
    rsk = np.full(m, np.nan)
    ii = fv["i"].to_numpy(np.int64)
    sd = fv["side"].to_numpy(np.int64)
    bot = fv["bottom"].to_numpy(np.float64)
    top = fv["top"].to_numpy(np.float64)

    # 진입 지점: 롱은 위에서 내려오므로 top 쪽이 가깝다. level 0 = 가까운 끝.
    lv = cfg.entry_level
    px = np.where(sd > 0, top - lv * (top - bot), bot + lv * (top - bot))
    gap = top - bot
    far = np.where(sd > 0, bot - cfg.stop_ext * gap, top + cfg.stop_ext * gap)
    risk0 = np.abs(px - far)

    for j in range(m):
        s = int(start5[ii[j]])
        e = min(n5, s + win)
        if s >= e:
            continue
        hit = (lo5[s:e] <= px[j]) if sd[j] > 0 else (hi5[s:e] >= px[j])
        if not hit.any():
            continue
        t = s + int(hit.argmax())
        touch[j] = t
        ent[j] = px[j]
        rsk[j] = risk0[j]

    ok = touch >= 0
    touch, ent, rsk, sd2, ii2 = touch[ok], ent[ok], rsk[ok], sd[ok], ii[ok]
    size_bps = np.abs(top[ok] - bot[ok]) / ent * 1e4
    risk_bps = rsk / ent * 1e4
    roa = np.where(a5[touch] > 0, rsk / a5[touch], np.nan)

    keep = np.ones(len(touch), bool)
    if cfg.min_r_over_atr > 0:
        keep &= np.isfinite(roa) & (roa >= cfg.min_r_over_atr)
    if cfg.min_risk_bps > 0:
        keep &= risk_bps >= cfg.min_risk_bps
    touch, ent, rsk, sd2, ii2 = touch[keep], ent[keep], rsk[keep], sd2[keep], ii2[keep]
    size_bps, risk_bps, roa = size_bps[keep], risk_bps[keep], roa[keep]

    entry_bar = touch + (1 if cfg.skip_entry_bar else 0)
    r = first_passage_r(hi5, lo5, entry_bar, ent, rsk, sd2,
                        tp_r=cfg.tp_r, sl_r=cfg.sl_r, horizon=cfg.horizon,
                        skip_entry_bar=False)
    return FVGResult(entry_bar, touch, ent, rsk, sd2, size_bps, risk_bps, roa, ii2, r)
