"""FVG 관통 이후 — 반등은 어디서 일어나는가.

기존 FVG 셋업은 **"갭 관통 = 손절"** 을 전제로 손절을 먼 끝에 둔다.
그런데 실거래 관찰은 반대다 — 갭이 뚫린 **아래에서** 유동성을 쓸고(sweep),
CHoCH 가 나온 뒤에 반등하는 경우가 많다. 그렇다면 먼 끝 손절은
**반등이 시작되는 바로 그 자리에서 털리는** 규칙이 된다.

이 모듈은 두 가지를 한다.

1. `excursion` — 진입 규칙 없이 **기하만** 잰다. 갭 첫 터치 이후 얼마나 깊이
   뚫고 내려갔다가(``depth``) 다시 갭 위로 돌아오는가(``recovered``).
2. `sweep_choch` — 뚫린 뒤 CHoCH 확정 시점에 진입하고 **스윕 최저**에 손절을
   두는 대안 진입.

깊이 단위
---------
갭 크기 ``g`` 배수로 잰다. ``depth = (먼끝 − 최저) / g`` 이므로

    depth = −1  가까운 끝에서 바로 반등 (갭 안으로 들어오지도 않음)
    depth =  0  갭을 정확히 다 채움 (먼 끝 도달)
    depth = +1  갭 크기만큼 더 뚫고 내려감
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.resample import build_ref_map, resample_htf
from .choch import structure_events
from .fvg import FVGConfig, atr5m, find_fvg

__all__ = ["ExcursionResult", "excursion", "sweep_choch", "choch_on_tf"]


@dataclass(slots=True)
class ExcursionResult:
    touch_bar: np.ndarray
    side: np.ndarray
    near: np.ndarray            # 가까운 끝 (롱이면 top)
    far: np.ndarray             # 먼 끝 (롱이면 bottom)
    gap: np.ndarray
    gap_bps: np.ndarray
    #: **회복 직전까지**의 최대 역행 / 갭. −1 = 갭에 들어오지도 않음, 0 = 정확히 채움.
    #: 회복하지 못하면 지평 전체의 극단. 지평 전체 최저를 쓰면 7일 드리프트가
    #: 깊이로 잡혀 "스윕" 과 구분이 안 된다 — 그래서 회복 시점으로 자른다.
    depth: np.ndarray
    depth_atr: np.ndarray
    bar_min: np.ndarray         # 최저(최고)점 봉
    #: 그 이후 가까운 끝을 종가로 회복했는가
    recovered: np.ndarray
    bars_to_recover: np.ndarray
    #: 회복 이후 갭 크기 배수로 얼마나 더 갔나 (회복 못 하면 NaN)
    run_after: np.ndarray

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({k: getattr(self, k) for k in
                             ("touch_bar", "side", "gap_bps", "depth", "depth_atr",
                              "bar_min", "recovered", "bars_to_recover", "run_after")})


def excursion(df: pd.DataFrame, cfg: FVGConfig = FVGConfig(), *,
              horizon: int = 2016, wait: int | None = None) -> ExcursionResult:
    """갭 첫 터치 이후의 기하. 진입 규칙도 비용도 없다 — 순수 기술통계."""
    htf = resample_htf(df, cfg.htf)
    fv = find_fvg(htf)
    n = len(df)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    a5 = atr5m(df, cfg.atr_period, mode=cfg.atr_mode)
    ct = htf["close_ts"].to_numpy()
    start5 = np.searchsorted(df.index.to_numpy(), ct, side="left")
    ratio = max(int(np.median(htf["n_ltf"])), 1)
    win = (cfg.wait if wait is None else wait) * ratio

    ii = fv["i"].to_numpy(np.int64)
    sd = fv["side"].to_numpy(np.int64)
    bot = fv["bottom"].to_numpy(np.float64)
    top = fv["top"].to_numpy(np.float64)
    near = np.where(sd > 0, top, bot)
    far = np.where(sd > 0, bot, top)
    gap = top - bot

    m = len(fv)
    tb = np.full(m, -1, np.int64)
    depth = np.full(m, np.nan)
    bmin = np.full(m, -1, np.int64)
    rec = np.zeros(m, bool)
    btr = np.full(m, -1, np.int64)
    run = np.full(m, np.nan)

    for j in range(m):
        s = int(start5[ii[j]])
        e = min(n, s + win)
        if s >= e:
            continue
        hit = (lo[s:e] <= near[j]) if sd[j] > 0 else (hi[s:e] >= near[j])
        if not hit.any():
            continue
        t0 = s + int(hit.argmax())
        tb[j] = t0
        f = min(n, t0 + horizon)
        if t0 + 2 >= f:
            continue
        # **회복을 먼저 찾고, 그 전까지의 최대 역행을 깊이로 삼는다.**
        # 지평 전체의 최저를 쓰면 7일 드리프트가 깊이로 잡혀 "스윕" 과 구분이 안 된다.
        if sd[j] > 0:
            broke = np.flatnonzero(lo[t0:f] < far[j])
            if len(broke) == 0:
                k = t0 + int(np.argmin(lo[t0:f]))
                depth[j] = (far[j] - lo[k]) / gap[j]
                bmin[j] = k
                rec[j] = bool((cl[t0:f] > near[j]).any())
                if rec[j]:
                    btr[j] = int(np.flatnonzero(cl[t0:f] > near[j])[0])
                continue
            b0 = t0 + int(broke[0])
            back = np.flatnonzero(cl[b0:f] > near[j])
            end = (b0 + int(back[0])) if len(back) else f
            k = t0 + int(np.argmin(lo[t0:end + 1]))
            depth[j] = (far[j] - lo[k]) / gap[j]
            if len(back):
                rec[j] = True
                btr[j] = end - k
                ext = np.max(hi[end:f]) - near[j]
                run[j] = ext / gap[j]
        else:
            broke = np.flatnonzero(hi[t0:f] > far[j])
            if len(broke) == 0:
                k = t0 + int(np.argmax(hi[t0:f]))
                depth[j] = (hi[k] - far[j]) / gap[j]
                bmin[j] = k
                rec[j] = bool((cl[t0:f] < near[j]).any())
                if rec[j]:
                    btr[j] = int(np.flatnonzero(cl[t0:f] < near[j])[0])
                continue
            b0 = t0 + int(broke[0])
            back = np.flatnonzero(cl[b0:f] < near[j])
            end = (b0 + int(back[0])) if len(back) else f
            k = t0 + int(np.argmax(hi[t0:end + 1]))
            depth[j] = (hi[k] - far[j]) / gap[j]
            if len(back):
                rec[j] = True
                btr[j] = end - k
                ext = near[j] - np.min(lo[end:f])
                run[j] = ext / gap[j]
        bmin[j] = k

    ok = tb >= 0
    d_atr = np.where(ok, depth * gap / np.where(a5[np.maximum(tb, 0)] > 0,
                                                a5[np.maximum(tb, 0)], np.nan), np.nan)
    return ExcursionResult(tb[ok], sd[ok], near[ok], far[ok], gap[ok],
                           (gap / near)[ok] * 1e4, depth[ok], d_atr[ok],
                           bmin[ok], rec[ok], btr[ok], run[ok])


def choch_on_tf(df: pd.DataFrame, tf: str, lb: int = 3, cache: dict | None = None
                ) -> np.ndarray:
    """``tf`` 에서 잡은 CHoCH 를 5m 축으로. 값은 그 봉에서 확정된 CHoCH 부호."""
    cache = {} if cache is None else cache
    key = ("choch", tf, lb)
    if key in cache:
        return cache[key]
    if tf.strip().lower() == "5m":
        htf, ref = df, np.arange(len(df))
    else:
        htf = resample_htf(df, tf)
        ref = build_ref_map(df.index, htf)
    s, k = structure_events(htf["high"].to_numpy(np.float64),
                            htf["low"].to_numpy(np.float64),
                            htf["close"].to_numpy(np.float64), lb)
    ev = np.where(k == 1, s, 0).astype(np.int8)      # CHoCH 만
    # HTF 사건은 그 HTF 봉이 **확정될 때** 5m 축에 나타난다
    out = np.zeros(len(df), np.int8)
    chg = np.flatnonzero(np.diff(ref, prepend=-1) != 0)
    for p in chg:
        h = ref[p]
        if h >= 0 and ev[h] != 0:
            out[p] = ev[h]
    cache[key] = out
    return out


def sweep_choch(df: pd.DataFrame, ex: ExcursionResult, choch5: np.ndarray, *,
                min_depth: float = 0.0, max_wait: int = 576,
                buffer_atr: float = 0.0, atr: np.ndarray | None = None
                ) -> pd.DataFrame:
    """갭이 ``min_depth`` 이상 뚫린 뒤 **첫 CHoCH** 에서 진입.

    손절은 그 시점까지의 **스윕 극단**(+버퍼). 1R = 진입가 − 손절가.
    진입 신호봉만 낸다 — 체결·레이스는 `execution` / `barrier` 가 맡는다.
    """
    n = len(df)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    rows = []
    for j in range(len(ex.touch_bar)):
        if not np.isfinite(ex.depth[j]) or ex.depth[j] < min_depth:
            continue
        s = int(ex.touch_bar[j])
        sd = int(ex.side[j])
        # 먼 끝을 뚫은 첫 봉부터 찾는다 (사후의 최저점을 쓰지 않는다)
        f = min(n, s + max_wait)
        brk = (lo[s:f] < ex.far[j]) if sd > 0 else (hi[s:f] > ex.far[j])
        if not brk.any():
            continue
        b0 = s + int(brk.argmax())
        e = min(n, b0 + max_wait)
        cand = np.flatnonzero(choch5[b0:e] == sd)
        if not len(cand):
            continue
        ent_bar = b0 + int(cand[0])
        swept = np.min(lo[s:ent_bar + 1]) if sd > 0 else np.max(hi[s:ent_bar + 1])
        buf = 0.0 if atr is None else buffer_atr * atr[ent_bar]
        stop = swept - buf if sd > 0 else swept + buf
        entry = cl[ent_bar]
        risk = (entry - stop) if sd > 0 else (stop - entry)
        if not np.isfinite(risk) or risk <= 0:
            continue
        rows.append(dict(bar=ent_bar, side=sd, entry=entry, stop=stop, risk=risk,
                         risk_bps=risk / entry * 1e4, depth=float(ex.depth[j]),
                         gap_bps=float(ex.gap_bps[j]), touch_bar=s, break_bar=b0,
                         wait=ent_bar - b0))
    if not rows:
        return pd.DataFrame(columns=["bar", "side", "entry", "stop", "risk", "risk_bps",
                                     "depth", "gap_bps", "touch_bar", "break_bar", "wait"])
    out = pd.DataFrame(rows)
    # **한 봉에 여러 FVG 가 같은 신호를 낼 수 있다.** 트레이더는 한 번만 들어가므로
    # (봉, 방향) 으로 묶고 **가장 넓은 손절**(가장 깊은 스윕)을 남긴다. 1h 에서
    # 중복이 35% 나 돼서 그냥 두면 표본이 부풀고 t 가 과장된다.
    out = (out.sort_values("risk", ascending=False)
              .drop_duplicates(subset=["bar", "side"], keep="first")
              .sort_values("bar").reset_index(drop=True))
    return out
