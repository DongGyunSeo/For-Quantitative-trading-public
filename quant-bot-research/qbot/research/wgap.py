"""주말 갭 × 4H 본장정렬 × 15m CISD — `claude/주말갭-4H정렬-CISD-사전등록.md` 그대로.

기호 (문서와 같다)::

    F   금 16:00 ET 에 끝나는 5m 봉(ts 15:55 ET)의 종가
    O   월 09:30 ET 5m 봉의 시가
    G   |O − F|            d = −sign(O − F)  (메움 방향, 위로 갭이면 −1 = 숏)
    φ   월 개장 이후 가격이 처음 F 에 닿은 5m 봉 (메움)

레이스는 `legacy.race_levels` 규약(같은 봉에서 양쪽 = 모호 0R), 만료는 전부 MTM(시장가).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.resample import build_ref_map, resample_htf
from .choch import structure_events
from .legacy import MAKER, TAKER, race_levels, trade_frame

__all__ = ["ET", "HOLD", "PATH_HOURS", "weekday_mask", "weekend_events", "htf_trend", "ltf_bars",
           "cisd_scan", "first_touch", "base_trades", "cisd_entry_trades", "postfill"]

ET = "America/New_York"
HOLD = 1440                                   # 5일 — 기존 주말 갭 그대로
BPH = 12                                      # 5m 봉 / 시간
PATH_HOURS = (1, 2, 4, 8, 12, 24, 48, 72, 96)


# ------------------------------------------------------------------ 본장 / 주말


def weekday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """본장 봉이면 True. 주말 창 [금 16:00 ET, 월 09:30 ET) 에 속하면 False (DST 는 tz_convert)."""
    et = index.tz_convert(ET)
    dow = et.dayofweek.to_numpy()
    hm = (et.hour * 60 + et.minute).to_numpy()
    weekend = (((dow == 4) & (hm >= 16 * 60)) | (dow == 5) | (dow == 6)
               | ((dow == 0) & (hm < 9 * 60 + 30)))
    return ~weekend


def weekend_events(df: pd.DataFrame, *, min_gap_bps: float = 200.0, hold: int = HOLD) -> pd.DataFrame:
    """|갭| ≥ min_gap_bps 인 주말마다 한 줄.

    fi · mi = 금 15:55 ET · 월 09:30 ET 봉 위치, wend = 그 주 금 16:00 ET 봉 위치(배타 끝),
    hend = min(n, mi + hold) — 기존 5일 보유의 끝(배타).
    """
    idx = df.index
    et = idx.tz_convert(ET)
    dow = et.dayofweek.to_numpy()
    hm = (et.hour * 60 + et.minute).to_numpy()
    fri = np.flatnonzero((dow == 4) & (hm == 15 * 60 + 55))
    mon = np.flatnonzero((dow == 0) & (hm == 9 * 60 + 30))
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    n = len(df)
    rows = []
    for fi in fri:
        k = np.searchsorted(mon, fi)
        if k >= len(mon):
            break
        mi = int(mon[k])
        if mi - fi > 3 * 288:                      # 다음 월요일이 아니면 버린다 (legacy 와 같다)
            continue
        F, O = float(c[fi]), float(o[mi])
        gap = O / F - 1
        if abs(gap) * 1e4 < min_gap_bps:
            continue
        fri16 = et[mi].normalize() + pd.Timedelta(days=4, hours=16)   # 주중엔 DST 전환이 없다(일요일)
        wend = int(idx.searchsorted(fri16.tz_convert("UTC"), side="left"))
        rows.append(dict(fi=int(fi), mi=mi, F=F, O=O, gap_bps=gap * 1e4, d=-1 if gap > 0 else 1,
                         G=abs(O - F), wend=wend, hend=min(n, mi + hold), monday=idx[mi]))
    cols = ["fi", "mi", "F", "O", "gap_bps", "d", "G", "wend", "hend", "monday"]
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------ HTF 추세


def htf_trend(df: pd.DataFrame, *, tf: str = "4h", lb: int = 2, weekday_only: bool = True) -> np.ndarray:
    """5m 봉마다, 그 시점에 **확정된** 마지막 HTF 봉의 구조 추세(+1/−1/0).

    ``weekday_only`` 면 주말 창의 5m 봉을 빼고 HTF 를 만든다 — 경계의 부분 봉도 남긴다.
    추세 = 확정 스윙을 종가로 돌파한 마지막 방향(`structure_events`, lb 2).
    """
    if weekday_only:
        H = resample_htf(df[weekday_mask(df.index)], tf, min_ltf_bars=1)
    else:
        H = resample_htf(df, tf)
    s, _ = structure_events(H["high"].to_numpy(np.float64), H["low"].to_numpy(np.float64),
                            H["close"].to_numpy(np.float64), lb)
    tr = pd.Series(np.where(s != 0, s, np.nan)).ffill().fillna(0).to_numpy()
    ref = build_ref_map(df.index, H)
    return np.where(ref >= 0, tr[np.maximum(ref, 0)], 0).astype(np.int8)


# ------------------------------------------------------------------ 15m · CISD


def ltf_bars(df: pd.DataFrame, tf: str = "15min") -> tuple[pd.DataFrame, np.ndarray]:
    """5m → LTF(UTC 구간). ``p_end[t]`` = 구간 t 가 끝난 뒤 첫 5m 봉 위치(행동 가능한 첫 봉)."""
    g = df.resample(tf, closed="left", label="left", origin="epoch")
    M = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"))
    M = M[g.size().to_numpy() > 0]
    p_end = df.index.searchsorted(M.index + pd.Timedelta(tf), side="left")
    return M, np.asarray(p_end, np.int64)


def _series_open(o: np.ndarray, c: np.ndarray, m: int, cap: int) -> float:
    """m 이하 마지막 양봉에서 거슬러 오른 연속 양봉 묶음의 첫 시가. 없으면 nan."""
    lim = max(0, m - cap)
    j = m
    while j >= lim and not (c[j] > o[j]):
        j -= 1
    if j < lim:
        return np.nan
    s = j
    while s - 1 >= lim and c[s - 1] > o[s - 1]:
        s -= 1
    return float(o[s])


def cisd_scan(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, start: int, stop: int,
              direction: int, *, cap: int = 96) -> tuple[int, float, float]:
    """start..stop−1 의 첫 CISD — '연속 캔들 첫 시가' 정의.

    direction −1 = 하락 CISD(숏 확인): start 이후 최고가 봉 m 까지의 마지막 연속 양봉 묶음의 첫 시가를
    종가로 하향 이탈. +1 = 가격을 뒤집은 대칭. Returns (봉, 레벨, 그때까지의 극단) · 없으면 (−1, nan, nan).
    """
    if direction > 0:
        t, lv, ex = cisd_scan(-o, -l, -h, -c, start, stop, -1, cap=cap)
        return t, -lv, -ex
    best, L = -np.inf, np.nan
    for t in range(start, stop):
        if h[t] >= best:                           # 동률이면 가장 최근 봉
            best = h[t]
            L = _series_open(o, c, t, cap)
        if np.isfinite(L) and c[t] < L:
            return int(t), float(L), float(best)
    return -1, np.nan, np.nan


def first_touch(h: np.ndarray, l: np.ndarray, start: int, stop: int, level: float, d: int) -> int:
    """start..stop−1 에서 처음 level 에 닿은 5m 봉(d −1 이면 저가 ≤ level, +1 이면 고가 ≥ level). 없으면 −1."""
    if stop <= start:
        return -1
    hit = (l[start:stop] <= level) if d < 0 else (h[start:stop] >= level)
    return int(start + hit.argmax()) if hit.any() else -1


# ------------------------------------------------------------------ 거래


def _ohlc(df: pd.DataFrame):
    return tuple(df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))


def _frame(df, rows, *, entry_bps, extra_cols=("monday",)):
    cols = ["bar", "side", "entry", "tp", "sl", "g", "kind", "xb", *extra_cols]
    if not rows:
        t = trade_frame(df, [], [], [], [], [], [], [], entry_bps=entry_bps)
        for k in ("exit_bar", *extra_cols):
            t[k] = pd.Series(dtype=object)
        return t
    a = pd.DataFrame(rows, columns=cols)
    extra = {k: a[k].to_numpy() for k in extra_cols}
    extra["exit_bar"] = a["xb"].to_numpy(np.int64)
    return trade_frame(df, a["bar"].to_numpy(np.int64), a["side"].to_numpy(np.int64), a["entry"], a["tp"],
                       a["sl"], a["g"], a["kind"], entry_bps=entry_bps, extra=extra)


def base_trades(df: pd.DataFrame, ev: pd.DataFrame, *, mtm: bool = True, hold: int = HOLD) -> pd.DataFrame:
    """B0 — 월 개장 시장가, TP F, SL 대칭, 5일. ``mtm`` False 면 예전 규약(미해결 0R)."""
    _, h, l, c = _ohlc(df)
    if ev.empty:
        return _frame(df, [], entry_bps=TAKER)
    bars = ev["mi"].to_numpy(np.int64)
    sd = ev["d"].to_numpy(np.int64)
    ent = ev["O"].to_numpy(np.float64)
    tp = ev["F"].to_numpy(np.float64)
    sl = ent - sd * ev["G"].to_numpy(np.float64)
    g, kind, xb = race_levels(h, l, c, bars, ent, tp, sl, sd, horizon=hold, mtm=mtm)
    rows = list(zip(bars, sd, ent, tp, sl, g, kind, xb, ev["monday"]))
    return _frame(df, rows, entry_bps=TAKER)


def cisd_entry_trades(df: pd.DataFrame, ev: pd.DataFrame, M: pd.DataFrame, p_end: np.ndarray, *,
                      cap: int = 96) -> pd.DataFrame:
    """C — 월 개장 뒤, 메우기 전에 나온 첫 메움 방향 15m CISD 다음 5m 시가에 시장가.

    SL = 월 개장 이후 CISD 봉까지의 극단, TP = F, 만료 = B0 와 같은 시각(hend), MTM.
    """
    o5, h5, l5, c5 = _ohlc(df)
    mo, mh, ml, mc = _ohlc(M)
    rows, ends = [], []
    for e in ev.itertuples(index=False):
        phi = first_touch(h5, l5, e.mi, e.hend, e.F, e.d)
        limit = min(phi if phi >= 0 else e.hend - 1, e.hend - 1)    # 진입봉 p 는 메움봉 이하
        r = int(M.index.searchsorted(df.index[e.mi], side="left"))
        stop = int(np.searchsorted(p_end, limit, side="right"))
        t, _, ex = cisd_scan(mo, mh, ml, mc, r, stop, e.d, cap=cap)
        if t < 0:
            continue
        p = int(p_end[t])
        E = o5[p]
        ok = (E > e.F and ex > E) if e.d < 0 else (E < e.F and ex < E)
        if not ok:
            continue
        rows.append([p, e.d, E, e.F, ex, 0.0, 0, -1, e.monday])
        ends.append(e.hend)
    if not rows:
        return _frame(df, [], entry_bps=TAKER)
    a = np.array([r[:5] for r in rows], dtype=float)
    g, kind, xb = race_levels(h5, l5, c5, a[:, 0].astype(np.int64), a[:, 2], a[:, 3], a[:, 4],
                              a[:, 1].astype(np.int64), end=np.array(ends, np.int64), mtm=True)
    for i, r in enumerate(rows):
        r[5], r[6], r[7] = g[i], kind[i], xb[i]
    return _frame(df, rows, entry_bps=TAKER)


def postfill(df: pd.DataFrame, ev: pd.DataFrame, M: pd.DataFrame, p_end: np.ndarray, *,
             cap: int = 96) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """D — 같은 주 금 16:00 ET 전에 메운 주만.

    Returns (주별 표, D1 반전·지정가, D2 연속·역지정가, D-CISD 반전·CISD 확인).
    D0 레이스: F 에서 연속 목표 F + d·G vs 반전 목표 O. φ 봉에서는 메움 방향 연장만 인정한다.
    """
    o5, h5, l5, c5 = _ohlc(df)
    mo, mh, ml, mc = _ohlc(M)
    recs, r1, r2, rc, rc_end = [], [], [], [], []
    for e in ev.itertuples(index=False):
        d, F, O, G = e.d, e.F, e.O, e.G
        phi = first_touch(h5, l5, e.mi, e.wend, F, d)
        rec = dict(monday=e.monday, phi=phi, filled=phi >= 0)
        if phi < 0:
            recs.append(rec)
            continue
        cont = F + d * G
        ext = ((F - l5[phi]) if d < 0 else (h5[phi] - F)) / G
        if ext >= 1:
            g0, k0, x0 = 1.0, 1, phi
        elif phi + 1 >= e.wend:
            g0, k0, x0 = d * (c5[phi] - F) / G, 2, phi
        else:
            g, k, x = race_levels(h5, l5, c5, np.array([phi + 1]), np.array([F]), np.array([cont]),
                                  np.array([O]), np.array([d]), end=np.array([e.wend]), mtm=True)
            g0, k0, x0 = float(g[0]), int(k[0]), int(x[0])
        rec.update(fill_h=(phi - e.mi) / BPH, fill_ext=ext, d0_g=g0, d0_kind=k0, d0_exit=x0,
                   fill_dow=int(df.index[phi].tz_convert(ET).dayofweek))
        for hh in PATH_HOURS:
            q = phi + BPH * hh
            rec[f"p{hh}"] = d * (c5[q] - F) / G if q < e.wend else np.nan
        rec["p_fri"] = d * (c5[e.wend - 1] - F) / G          # 금 15:55 ET 봉 종가
        # D2 연속(d, 역지정가) / D1 반전(−d, 지정가) — 같은 레이스의 양면
        r2.append([phi, d, F, cont, O, g0, k0, x0, e.monday])
        r1.append([phi, -d, F, O, cont, -g0, {1: -1, -1: 1}.get(k0, k0), x0, e.monday])
        # D-CISD — 메운 봉이 속한 15m 부터 반전 방향 CISD
        s0 = int(M.index.searchsorted(df.index[phi], side="right")) - 1
        stop = int(np.searchsorted(p_end, e.wend - 1, side="right"))
        t, _, ex = cisd_scan(mo, mh, ml, mc, max(s0, 0), stop, -d, cap=cap)
        rec["dc_bar"] = -1
        if t >= 0:
            p = int(p_end[t])
            E = o5[p]
            ok = (E < O and ex < E) if d < 0 else (E > O and ex > E)
            if ok:
                rc.append([p, -d, E, O, ex, 0.0, 0, -1, e.monday])
                rc_end.append(e.wend)
                rec["dc_bar"] = p
        recs.append(rec)
    if rc:
        a = np.array([r[:5] for r in rc], dtype=float)
        g, kind, xb = race_levels(h5, l5, c5, a[:, 0].astype(np.int64), a[:, 2], a[:, 3], a[:, 4],
                                  a[:, 1].astype(np.int64), end=np.array(rc_end, np.int64), mtm=True)
        for i, r in enumerate(rc):
            r[5], r[6], r[7] = g[i], kind[i], xb[i]
    return (pd.DataFrame(recs), _frame(df, r1, entry_bps=MAKER), _frame(df, r2, entry_bps=TAKER),
            _frame(df, rc, entry_bps=TAKER))
