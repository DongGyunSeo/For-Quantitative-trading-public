"""예전 신호군 재구축 — 2026-09-18 코드 유실로 사라진 것들을 문서 정의대로 다시 짠다.

각 함수는 한 심볼의 5m OHLCV 를 받아 **거래 목록**(표준 스키마)을 낸다::

    bar      진입(체결) 5m 봉 위치
    side     +1 롱 / −1 숏
    entry    진입가
    risk_bps 1R (bps)
    tp_r     익절 거리 (R)
    gross    수수료 전 R   (익절 +tp_r / 손절 −1 / 모호·미해결 0, 만료는 시가평가)
    fee      수수료 R      (진입 bps + 청산 bps) / risk_bps
    net      gross − fee
    kind     1 익절 / −1 손절 / 0 모호 / 2 미해결·만료

비용 규약 (프로젝트 표준, 2026-09-18 확정)
  지정가 = maker 2bps · 시장가/스톱 = taker 5 + 슬리피지 1 = 6bps.
  신호마다 진입 방식이 다르므로 진입 비용을 인자로 받는다.

측정해상도 규칙 ``1R / 일간중앙 ATR14 ≥ 2`` 는 `apply_rule` 로 거는데, 문서 숫자와
대조할 때는 끈다(규칙이 생기기 전 문서들이 많다).

정의의 출처는 함수마다 docstring 에 문서 이름으로 적었다. 문서가 모호한 곳은
**어떻게 정했는지** 적었다 — 나중에 문서와 숫자가 안 맞으면 거기부터 본다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.resample import build_ref_map, resample_htf
from .choch import confirmed_levels, structure_events
from .fvg import atr5m

__all__ = [
    "MAKER", "TAKER", "race_levels", "trade_frame", "apply_rule", "tr_atr_rank_5m",
    "htf_rank", "htf_fib_trades", "level_break_trades", "big_bar_choch_trades",
    "cascade_events", "liq_cascade_trades", "trend_fvg_trades", "weekend_gap_trades",
    "tsmom_weekly",
]

MAKER = 2.0
TAKER = 6.0


# ------------------------------------------------------------------ 레이서


def race_levels(high: np.ndarray, low: np.ndarray, close: np.ndarray, bars: np.ndarray,
                entry: np.ndarray, tp: np.ndarray, sl: np.ndarray, side: np.ndarray, *,
                horizon: int = 2016, include_entry_bar: bool = True,
                end: np.ndarray | None = None, mtm: bool = False
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """임의의 익절/손절 **가격**으로 레이스. R 단위 = |entry − sl|.

    같은 봉에서 둘 다 걸리면 모호(0R). ``end`` 가 있으면 거래마다 그 봉(배타) 전에 만료.
    ``mtm`` 이면 만료·미해결을 마지막 봉 종가로 평가한다(시장가 청산).

    Returns (gross_R, kind, exit_bar)
    """
    n = len(high)
    m = len(bars)
    g = np.zeros(m)
    kind = np.full(m, 2, np.int8)
    xb = np.full(m, -1, np.int64)
    for i in range(m):
        sd = int(side[i])
        e = float(entry[i])
        risk = abs(e - float(sl[i]))
        if sd == 0 or not np.isfinite(risk) or risk <= 0:
            kind[i] = 0
            continue
        s = int(bars[i]) + (0 if include_entry_bar else 1)
        f = min(n, s + horizon) if end is None else min(n, int(end[i]))
        if s >= f:
            continue
        hi = high[s:f]
        lo = low[s:f]
        if sd > 0:
            hit_tp = hi >= tp[i]
            hit_sl = lo <= sl[i]
        else:
            hit_tp = lo <= tp[i]
            hit_sl = hi >= sl[i]
        a = int(hit_tp.argmax()) if hit_tp.any() else f - s
        b = int(hit_sl.argmax()) if hit_sl.any() else f - s
        span = f - s
        if a >= span and b >= span:
            xb[i] = f - 1
            if mtm:
                g[i] = sd * (close[f - 1] - e) / risk
            continue
        if a == b:
            kind[i] = 0
            xb[i] = s + a
            continue
        if a < b:
            kind[i] = 1
            g[i] = abs(float(tp[i]) - e) / risk
            xb[i] = s + a
        else:
            kind[i] = -1
            g[i] = -1.0
            xb[i] = s + b
    return g, kind, xb


def trade_frame(df: pd.DataFrame, bars, side, entry, tp, sl, gross, kind, *,
                entry_bps: float, tp_bps: float = MAKER, sl_bps: float = TAKER,
                extra: dict | None = None) -> pd.DataFrame:
    """표준 스키마로 묶고 수수료를 R 로 환산한다. 익절만 maker, 나머지 청산은 taker."""
    bars = np.asarray(bars, np.int64)
    entry = np.asarray(entry, float)
    risk = np.abs(entry - np.asarray(sl, float))
    risk_bps = risk / entry * 1e4
    tp_r = np.abs(np.asarray(tp, float) - entry) / np.where(risk > 0, risk, np.nan)
    fee = (entry_bps + np.where(np.asarray(kind) == 1, tp_bps, sl_bps)) / risk_bps
    out = pd.DataFrame(dict(bar=bars, time=df.index[bars], side=np.asarray(side, np.int8),
                            entry=entry, risk_bps=risk_bps, tp_r=tp_r,
                            gross=np.asarray(gross, float), fee=fee,
                            net=np.asarray(gross, float) - fee, kind=np.asarray(kind, np.int8)))
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def apply_rule(t: pd.DataFrame, atr_daily: np.ndarray, k: float = 2.0) -> pd.DataFrame:
    """측정해상도 규칙: 1R(가격) / 일간중앙 ATR14(진입봉) ≥ k."""
    if t.empty:
        return t
    risk_px = t.entry.to_numpy() * t.risk_bps.to_numpy() / 1e4
    a = atr_daily[t.bar.to_numpy()]
    ok = np.isfinite(a) & (a > 0) & (risk_px / np.where(a > 0, a, np.nan) >= k)
    return t[ok].reset_index(drop=True)


# ------------------------------------------------------------------ 봉 크기 순위


def tr_atr_rank_5m(df: pd.DataFrame, *, period: int = 14, window: int = 2016,
                   min_periods: int = 288) -> np.ndarray:
    """5m ``TR / ATR14`` 의 **인과적** 롤링 백분위 (현재 봉 포함, 과거 window 봉).

    `장대봉-CHoCH-검증결과.md` · `청산이벤트-거래량복귀-검증결과.md` 의 후보봉 정의.
    문서는 창 길이를 적지 않았다 → 7일(2016봉)로 정했다.
    """
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    x = pd.Series(np.where(atr > 0, tr / atr, np.nan))
    return x.rolling(window, min_periods=min_periods).rank(pct=True).to_numpy()


def htf_rank(htf: pd.DataFrame, *, period: int = 15, window: int = 500,
             min_periods: int = 100) -> np.ndarray:
    """HTF 봉 ``TR / ATR15`` 롤링 백분위. ATR 은 **직전 15봉** 평균(현재 봉 제외).

    `ATR-봉선별-검증결과.md`: atr = 직전 15봉 TR 평균(shift 1), 창 500봉, 최소 100.
    """
    h, l, c = (htf[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).shift(1).rolling(period, min_periods=period).mean().to_numpy()
    x = pd.Series(np.where(atr > 0, tr / atr, np.nan))
    return x.rolling(window, min_periods=min_periods).rank(pct=True).to_numpy()


def _htf_entry_bar(df: pd.DataFrame, htf: pd.DataFrame) -> np.ndarray:
    """HTF 봉이 확정된 뒤 첫 5m 봉 위치 (없으면 len)."""
    return np.searchsorted(df.index.to_numpy(), htf["close_ts"].to_numpy(), side="left")


# ------------------------------------------------------------------ 1·2. HTF 피보 / ATR 봉선별


def htf_fib_trades(df: pd.DataFrame, *, tf: str = "4h", sl_lv: float = 0.0,
                   tp_lv: float = 1.27, rank_min: float | None = None,
                   horizon: int = 2016, no_overlap: bool = True) -> pd.DataFrame:
    """HTF 봉 확정 즉시 **봉 방향으로** 시장가 진입 (continuation), 피보 좌표 TP/SL.

    `실데이터-4H-검증결과.md` 의 ``con sl0.00 tp1.27`` 과 `ATR-봉선별-검증결과.md`.
    좌표는 SPATIAL(0 = 저가, 1 = 고가)이고 숏은 거울: SL = 고가 − sl_lv·폭, TP = 고가 − tp_lv·폭.
    진입 = 다음 5m 봉 시가(taker). 문서 엔진의 보유 한도는 기록이 없어 7일로 둔다.
    ``rank_min`` 이면 `htf_rank` ≥ rank_min 인 봉만 (ATR 상위 선별).
    ``no_overlap`` — 문서 엔진은 심볼당 포지션 하나라 보유 중 신호를 건너뛰었다(기본 켬).
    """
    htf = resample_htf(df, tf)
    o, h, l, c = (htf[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    rk = htf_rank(htf)
    eb = _htf_entry_bar(df, htf)
    n = len(df)
    side = np.sign(c - o).astype(np.int64)
    ok = (side != 0) & (eb < n - 1) & (h > l)
    if rank_min is not None:
        ok &= np.isfinite(rk) & (rk >= rank_min)
    idx = np.flatnonzero(ok)
    rng = h[idx] - l[idx]
    sd = side[idx]
    bars = eb[idx]
    ent = df["open"].to_numpy(np.float64)[bars]
    sl = np.where(sd > 0, l[idx] + sl_lv * rng, h[idx] - sl_lv * rng)
    tp = np.where(sd > 0, l[idx] + tp_lv * rng, h[idx] - tp_lv * rng)
    good = np.where(sd > 0, (ent > sl) & (tp > ent), (ent < sl) & (tp < ent))
    bars, sd, ent, sl, tp = bars[good], sd[good], ent[good], sl[good], tp[good]
    hi, lo, cl = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    g, kind, xb = race_levels(hi, lo, cl, bars, ent, tp, sl, sd, horizon=horizon)
    keep = np.ones(len(bars), bool)
    if no_overlap:
        # 문서의 엔진은 심볼당 포지션 하나였다 — 보유 중 신호는 건너뛴다
        busy = -1
        for i in range(len(bars)):
            if bars[i] <= busy:
                keep[i] = False
                continue
            busy = xb[i] if xb[i] >= 0 else bars[i]
    t = trade_frame(df, bars[keep], sd[keep], ent[keep], tp[keep], sl[keep], g[keep], kind[keep],
                    entry_bps=TAKER, extra=dict(rank=rk[idx][good][keep]))
    return t


# ------------------------------------------------------------------ 3. 피보 레벨 돌파


def level_break_trades(df: pd.DataFrame, *, tf: str = "4h", up: float = 0.9, dist: float = 0.8,
                       rank_min: float | None = 0.75, inside_only: bool = True,
                       include_entry_bar: bool = True) -> pd.DataFrame:
    """직전 HTF 봉 레인지의 0.9 를 **안쪽에서 위로** 뚫으면 롱(스톱 주문), 0.1 을 아래로 뚫으면 숏.

    `피보레벨-진입승률-분석.md` §3·§6: ``0.9 → 1.7 롱`` / ``0.1 → −0.7 숏``, 대칭 거리 0.8,
    보유 = 다음 HTF 봉 마감까지, ATR 상위 25%, 안쪽에서 돌파(= taker 필수).
    TP 1.7 · SL 0.1 (롱), 만료는 시장가 청산(시가평가). 레퍼런스 봉당 방향별 한 번.
    """
    htf = resample_htf(df, tf)
    h, l = htf["high"].to_numpy(np.float64), htf["low"].to_numpy(np.float64)
    rk = htf_rank(htf)
    eb = _htf_entry_bar(df, htf)
    n = len(df)
    o5, h5, l5, c5 = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    rows = []
    for i in range(len(htf) - 1):
        if rank_min is not None and not (np.isfinite(rk[i]) and rk[i] >= rank_min):
            continue
        s, f = int(eb[i]), int(eb[i + 1])
        if s < 1 or s >= f or f > n:
            continue
        rg = h[i] - l[i]
        if rg <= 0:
            continue
        for sd, lv, tp_lv, sl_lv in ((1, up, up + dist, up - dist),
                                     (-1, 1 - up, 1 - up - dist, 1 - up + dist)):
            L = l[i] + lv * rg
            hit = (h5[s:f] >= L) if sd > 0 else (l5[s:f] <= L)
            if not hit.any():
                continue
            t = s + int(hit.argmax())
            prev = c5[t - 1]
            inside = (prev < L) if sd > 0 else (prev > L)
            if inside_only and not inside:
                continue
            ent = max(L, o5[t]) if sd > 0 else min(L, o5[t])     # 스톱 주문: 갭이면 시가
            rows.append((t, sd, ent, l[i] + tp_lv * rg, l[i] + sl_lv * rg, f, rk[i]))
    if not rows:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=TAKER)
    a = np.array(rows, dtype=float)
    bars, sd, ent, tp, sl, end = (a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2],
                                  a[:, 3], a[:, 4], a[:, 5].astype(np.int64))
    g, kind, _ = race_levels(h5, l5, c5, bars, ent, tp, sl, sd, end=end, mtm=True,
                             include_entry_bar=include_entry_bar)
    return trade_frame(df, bars, sd, ent, tp, sl, g, kind, entry_bps=TAKER,
                       extra=dict(rank=a[:, 6]))


# ------------------------------------------------------------------ 4. 장대봉 → CHoCH


def big_bar_choch_trades(df: pd.DataFrame, *, top: float = 0.20, wait: int = 24, lb: int = 3,
                         tp_r: float = 2.0, rank: np.ndarray | None = None,
                         choch: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """장대 양봉 → 하방 CHoCH → 숏 (장대 음봉 → 상방 CHoCH → 롱). 다음 봉 시가 시장가.

    `장대봉-CHoCH-검증결과.md`: 장대봉 = TR/ATR14 인과 백분위 상위 ``top``, 대기 24봉,
    swing lb 3, SL = 장대봉~CHoCH 구간 극단. TP 는 현재 표준 2R.
    같은 CHoCH 에 여러 장대봉이 걸리면 **가장 이른 장대봉**(가장 넓은 손절) 하나만.
    """
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    n = len(df)
    rk = tr_atr_rank_5m(df) if rank is None else rank
    sg, kd = structure_events(h, l, c, lb) if choch is None else choch
    ch = np.where(kd == 1, sg, 0).astype(np.int8)
    big = np.flatnonzero(np.isfinite(rk) & (rk >= 1 - top) & (c != o))
    pos = {1: np.flatnonzero(ch == 1), -1: np.flatnonzero(ch == -1)}
    best: dict = {}
    for b in big:
        d = 1 if c[b] > o[b] else -1
        want = -d                                   # 양봉 → 하방 CHoCH
        arr = pos[want]
        k = np.searchsorted(arr, b, side="right")
        if k >= len(arr) or arr[k] > b + wait:
            continue
        j = int(arr[k])
        if j + 1 >= n:
            continue
        key = (j, want)
        if key in best:
            continue                                # 더 이른 장대봉이 이미 잡았다
        best[key] = b
    if not best:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=TAKER)
    js = np.array([k[0] for k in best], np.int64)
    sd = np.array([k[1] for k in best], np.int64)
    bs = np.array(list(best.values()), np.int64)
    ent = o[js + 1]
    sl = np.array([h[b:j + 1].max() if s < 0 else l[b:j + 1].min() for b, j, s in zip(bs, js, sd)])
    risk = np.where(sd < 0, sl - ent, ent - sl)
    good = risk > 0
    js, sd, bs, ent, sl, risk = js[good], sd[good], bs[good], ent[good], sl[good], risk[good]
    tp = ent + sd * tp_r * risk
    g, kind, _ = race_levels(h, l, c, js + 1, ent, tp, sl, sd)
    return trade_frame(df, js + 1, sd, ent, tp, sl, g, kind, entry_bps=TAKER,
                       extra=dict(big_bar=bs))


# ------------------------------------------------------------------ 5. 청산 캐스케이드


def cascade_events(df: pd.DataFrame, *, top: float = 0.20, gap: int = 6,
                   rank: np.ndarray | None = None) -> pd.DataFrame:
    """후보봉(TR/ATR14 인과 백분위 상위 ``top``)을 **방향별로** 간격 ≤ gap 봉이면 묶는다.

    `청산이벤트-거래량복귀-검증결과.md` §구현 1·3. 방향 = 봉 색(도지 제외).
    peak = 이벤트 구간의 극단(하락이면 최저가). 소진량(히트맵)은 쓰지 않는다 —
    그 필터는 세 번의 실험에서 기여 0 이었고, 원본 함수도 없다.
    """
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    rk = tr_atr_rank_5m(df) if rank is None else rank
    cand = np.isfinite(rk) & (rk >= 1 - top) & (c != o)
    rows = []
    for d in (1, -1):
        idx = np.flatnonzero(cand & ((c > o) if d > 0 else (c < o)))
        if len(idx) == 0:
            continue
        brk = np.flatnonzero(np.diff(idx) > gap)
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk, [len(idx) - 1]))
        for a, b in zip(starts, ends):
            s, e = int(idx[a]), int(idx[b])
            peak = h[s:e + 1].max() if d > 0 else l[s:e + 1].min()
            rows.append((s, e, d, peak, o[s]))
    ev = pd.DataFrame(rows, columns=["start", "end", "dir", "peak", "open0"])
    return ev.sort_values("end").reset_index(drop=True)


def liq_cascade_trades(df: pd.DataFrame, *, direction: str = "reversal", tp_r: float = 2.0,
                       wait: int = 96, vol_lo: float = 0.40, vol_hi: float = 0.60,
                       size_top: float | None = None, atr_daily: np.ndarray | None = None,
                       events: pd.DataFrame | None = None) -> pd.DataFrame:
    """청산 캐스케이드 → 거래량 복귀(백분위 0.40~0.60) → 다음 봉 시가 시장가.

    ``reversal``     하락 캐스케이드 → 롱, SL = peak (원 가설)
    ``continuation`` 하락 캐스케이드 → 숏, 1R = |진입 − peak|, SL = 진입 + 1R (극단 재테스트)

    ``size_top`` 이면 이벤트 크기 상위만 — **히트맵 소진량의 대리**다:
    크기 = |첫 후보봉 시가 − peak| / 일간중앙 ATR, 직전 5,000 이벤트 대비 인과 순위.
    원본(`청산-좁힌컷-...`)은 히트맵 소진량 순위였고 그 함수가 없으므로 같은 신호가 아니다.
    """
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    n = len(df)
    ev = cascade_events(df) if events is None else events
    if ev.empty:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=TAKER)
    vp = pd.Series(v).rolling(288, min_periods=144).rank(pct=True).to_numpy()
    okv = np.isfinite(vp) & (vp >= vol_lo) & (vp <= vol_hi)
    trig = np.flatnonzero(okv)
    if size_top is not None:
        a = atr5m(df, 14, mode="daily") if atr_daily is None else atr_daily
        size = np.abs(ev.open0.to_numpy() - ev.peak.to_numpy()) / a[ev.start.to_numpy()]
        r = pd.Series(size).rolling(5000, min_periods=1000).rank(pct=True).to_numpy()
        ev = ev[np.isfinite(r) & (r >= 1 - size_top)]
    best: dict = {}
    for e, d, peak in zip(ev.end.to_numpy(), ev.dir.to_numpy(), ev.peak.to_numpy()):
        k = np.searchsorted(trig, e, side="right")
        if k >= len(trig) or trig[k] > e + wait or trig[k] + 1 >= n:
            continue
        t = int(trig[k]) + 1
        ent = o[t]
        risk = (ent - peak) if d < 0 else (peak - ent)       # peak 가 진입가 반대편이어야
        if risk <= 0:
            continue
        sd = (1 if d < 0 else -1) if direction == "reversal" else (-1 if d < 0 else 1)
        key = (t, sd)
        if key in best and best[key][1] >= risk:
            continue
        best[key] = (peak, risk)
    if not best:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=TAKER)
    bars = np.array([k[0] for k in best], np.int64)
    sd = np.array([k[1] for k in best], np.int64)
    risk = np.array([x[1] for x in best.values()])
    ent = o[bars]
    sl = ent - sd * risk
    tp = ent + sd * tp_r * risk
    order = np.argsort(bars, kind="stable")
    bars, sd, ent, sl, tp = bars[order], sd[order], ent[order], sl[order], tp[order]
    g, kind, _ = race_levels(h, l, c, bars, ent, tp, sl, sd)
    return trade_frame(df, bars, sd, ent, tp, sl, g, kind, entry_bps=TAKER)


# ------------------------------------------------------------------ 6. 추세추종 FVG (fib 되돌림)


def trend_fvg_trades(df: pd.DataFrame, *, htf: str = "4h", mtf: str = "15m", lb: int = 2,
                     fib: float = 0.7, tp_r: float = 3.0, wait: int = 96,
                     align: int = 1) -> pd.DataFrame:
    """4H 구조 방향과 같은 방향의 MTF CHoCH 뒤, 레그의 fib 되돌림에 지정가 진입.

    `추세추종-FVG-사전등록.md` (15m) · `1H레그-4H정렬-사전등록.md` (1h, wait 384).
    - 4H 방향 = 마지막 확정 구조 돌파(BOS/CHoCH)의 부호 (lb 2)
    - MTF CHoCH (lb 2) 가 확정된 5m 봉부터 셋업 시작. ``align`` 1 = 정렬, −1 = 역추세 미러
    - 기원(origin) = CHoCH 시점의 마지막 확정 MTF 스윙 극단 = 손절
    - E = CHoCH 봉 고가에서 시작해 5m 고가로 트레일. 봉 t 의 지정가는 **t−1 까지의 E** 로
      L = E − fib·(E − origin) (롱). 체결은 L (갭이어도 L — 보수적)
    - 대기 ``wait`` 봉 안에 미체결이면 폐기. 1R = L − origin, TP = L + tp_r·1R
    - 체결봉 포함 레이스 (체결 모델 표준)
    """
    o5, h5, l5, c5 = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    n = len(df)
    H = resample_htf(df, htf)
    s4, k4 = structure_events(H["high"].to_numpy(np.float64), H["low"].to_numpy(np.float64),
                              H["close"].to_numpy(np.float64), lb)
    trend4 = pd.Series(np.where(s4 != 0, s4, np.nan)).ffill().fillna(0).to_numpy()
    ref4 = build_ref_map(df.index, H)
    M = resample_htf(df, mtf)
    mh, ml, mc = (M[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    sm, km = structure_events(mh, ml, mc, lb)
    hv, lv = confirmed_levels(mh, ml, lb)
    p5 = np.searchsorted(df.index.to_numpy(), M["close_ts"].to_numpy(), side="left")
    rows = []
    for k in np.flatnonzero(km == 1):
        s = int(sm[k])
        p = int(p5[k])
        if p >= n - 1:
            continue
        r4 = ref4[p]
        if r4 < 0 or trend4[r4] * align != s or trend4[r4] == 0:
            continue
        origin = lv[k] if s > 0 else hv[k]
        if not np.isfinite(origin):
            continue
        E = mh[k] if s > 0 else ml[k]
        if (s > 0 and E <= origin) or (s < 0 and E >= origin):
            continue
        f = min(n, p + wait)
        for t in range(p, f):
            L = E - fib * (E - origin)                 # 롱이면 E>origin, 숏이면 E<origin 이라 같은 식
            if (s > 0 and l5[t] <= L) or (s < 0 and h5[t] >= L):
                risk = abs(L - origin)
                rows.append((t, s, L, L + s * tp_r * risk, origin))
                break
            E = max(E, h5[t]) if s > 0 else min(E, l5[t])
    if not rows:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=MAKER)
    a = np.array(rows, dtype=float)
    bars, sd = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)
    ent, tp, sl = a[:, 2], a[:, 3], a[:, 4]
    g, kind, _ = race_levels(h5, l5, c5, bars, ent, tp, sl, sd)
    return trade_frame(df, bars, sd, ent, tp, sl, g, kind, entry_bps=MAKER)


# ------------------------------------------------------------------ 7. 주말 갭


def weekend_gap_trades(df: pd.DataFrame, *, min_gap_bps: float = 200.0,
                       hold: int = 1440) -> pd.DataFrame:
    """금 16:00 ET 종가 → 월 09:30 ET 시가 갭을 **메우는** 방향으로 시장가, 1:1 대칭.

    `주말갭-사전등록-및-IS결과.md`: 금 종가 = ts 15:55 ET 봉의 종가, 월 개장 = 09:30 ET 봉 시가.
    DST 는 tz_convert 로. |갭| ≥ 200bps, 보유 1440봉(5일), TP = 금 종가 레벨, SL 대칭.
    """
    et = df.index.tz_convert("America/New_York")
    dow = et.dayofweek.to_numpy()
    hm = (et.hour * 60 + et.minute).to_numpy()
    fri = np.flatnonzero((dow == 4) & (hm == 15 * 60 + 55))
    mon = np.flatnonzero((dow == 0) & (hm == 9 * 60 + 30))
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    rows = []
    for fi in fri:
        k = np.searchsorted(mon, fi)
        if k >= len(mon):
            break
        mi = int(mon[k])
        if mi - fi > 3 * 288:                         # 다음 월요일이 아니면 버린다
            continue
        fc, mo = c[fi], o[mi]
        gap = mo / fc - 1
        if abs(gap) * 1e4 < min_gap_bps:
            continue
        sd = -1 if gap > 0 else 1                     # 위로 갭 → 숏으로 메움
        risk = abs(mo - fc)
        rows.append((mi, sd, mo, fc, mo - sd * risk))
    if not rows:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=TAKER)
    a = np.array(rows, dtype=float)
    bars, sd = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)
    g, kind, _ = race_levels(h, l, c, bars, a[:, 2], a[:, 3], a[:, 4], sd, horizon=hold)
    return trade_frame(df, bars, sd, a[:, 2], a[:, 3], a[:, 4], g, kind, entry_bps=TAKER)


# ------------------------------------------------------------------ 8. 시계열 추세 (주간)


def tsmom_weekly(closes: pd.DataFrame, *, lookback_days: int = 28, fee_bps: float = TAKER,
                 funding_bps_8h: float = 0.137) -> pd.DataFrame:
    """심볼별 ``sign(28일 수익)`` 로 롱/숏, 일요일 종가(UTC)에 주간 리밸런스, 동일가중.

    `시계열추세-주간리밸런스-사전등록.md`. 입력은 **UTC 일 종가** 표(열 = 심볼).
    수수료는 포지션 변화분에만, 펀딩은 실측 평균 +0.137bps/8h 를 롱이 지급.
    반환: 주별 ``ret`` (전략, 비용 후) · ``gross`` (전략, 수수료·펀딩 전) · ``bh`` (매수보유) ·
    ``n`` (심볼 수) · ``long_frac``.
    """
    d = closes.sort_index()
    wk = d[d.index.dayofweek == 6]                     # 일요일 종가
    past = d.shift(lookback_days)
    sig = np.sign(d / past - 1).reindex(wk.index)
    fwd = wk.shift(-1) / wk - 1
    pos = sig.where(fwd.notna())
    dpos = pos.fillna(0).diff().abs()
    dpos.iloc[0] = pos.iloc[0].abs()
    fund = funding_bps_8h * 3 * 7 / 1e4
    r = pos * fwd - dpos * fee_bps / 1e4 - pos * fund
    out = pd.DataFrame(dict(ret=r.mean(axis=1), gross=(pos * fwd).mean(axis=1),
                            bh=fwd.mean(axis=1), n=pos.notna().sum(axis=1),
                            long_frac=(pos > 0).sum(axis=1) / pos.notna().sum(axis=1).clip(lower=1)))
    return out[out.n > 0]


# ------------------------------------------------------------------ 9·10. CHoCH 지정가 계열


def limit_signal_trades(df: pd.DataFrame, sig: pd.DataFrame, atr_daily: np.ndarray, *,
                        tp_r: float = 2.0, fill_window: int = 12, horizon: int = 2016,
                        rule: float = 2.0) -> pd.DataFrame:
    """신호봉 목록(bar, side, risk) → 신호봉 종가 지정가 진입 → 표준 스키마.

    `FVG관통-스윕CHoCH-기각-및-엣지부재.md` 의 `evaluate` 와 같은 순서:
    규칙(1R/일간ATR ≥ 2) → 끝단 제거 → `limit_entry_trades` (maker 2 / maker 2 / taker 6).
    """
    from .execution import ExecConfig, limit_entry_trades
    if sig is None or len(sig) == 0:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=MAKER)
    bars = sig.bar.to_numpy(np.int64)
    risk = sig.risk.to_numpy(float)
    roa = risk / atr_daily[bars]
    keep = np.isfinite(roa) & (roa >= rule) & (bars < len(df) - horizon - fill_window - 2)
    sig = sig[keep]
    if len(sig) == 0:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=MAKER)
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    cfg = ExecConfig(entry_bps=MAKER, tp_bps=MAKER, sl_bps=TAKER, fill_window=fill_window,
                     tp_r=tp_r, sl_r=1.0, horizon=horizon)
    tr = limit_entry_trades(h, l, c, sig.bar.to_numpy(np.int64), sig.side.to_numpy(np.int64),
                            sig.risk.to_numpy(float), cfg)
    f = tr[tr.filled].reset_index(drop=True)
    risk_f = sig.risk.to_numpy(float)[tr.filled.to_numpy()]
    kind = f.outcome.map({"승": 1, "패": -1, "모호": 0, "미해결": 2}).to_numpy(np.int8)
    ent = f.entry.to_numpy(float)
    sd = f.side.to_numpy(np.int64)
    return pd.DataFrame(dict(bar=f.fill_bar.to_numpy(np.int64), time=df.index[f.fill_bar.to_numpy()],
                             side=sd.astype(np.int8), entry=ent, risk_bps=risk_f / ent * 1e4,
                             tp_r=tp_r, gross=f.r_gross.to_numpy(float), fee=f.fee_r.to_numpy(float),
                             net=f.r_net.to_numpy(float), kind=kind))


def plain_choch_signals(df: pd.DataFrame, choch5: np.ndarray, *, lookback: int = 144) -> pd.DataFrame:
    """FVG 맥락 없는 CHoCH 전부. 손절 = 직전 ``lookback`` 봉 극단 (스윕CHoCH 문서의 대조군 B)."""
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    bars = np.flatnonzero(choch5 != 0)
    bars = bars[bars > lookback]
    if len(bars) == 0:
        return pd.DataFrame(columns=["bar", "side", "entry", "risk"])
    sd = choch5[bars].astype(np.int64)
    lo_r = pd.Series(l).rolling(lookback).min().to_numpy()[bars]
    hi_r = pd.Series(h).rolling(lookback).max().to_numpy()[bars]
    stop = np.where(sd > 0, lo_r, hi_r)
    ent = c[bars]
    risk = np.where(sd > 0, ent - stop, stop - ent)
    ok = np.isfinite(risk) & (risk > 0)
    return pd.DataFrame(dict(bar=bars[ok], side=sd[ok], entry=ent[ok], risk=risk[ok]))


# ------------------------------------------------------------------ 11. painR3 꼬리 (시간 청산)


def pain_tail_trades(df: pd.DataFrame, score: np.ndarray, atr_daily: np.ndarray, *,
                     q: float = 0.01, hold: int = 48, min_months: int = 3,
                     fill_window: int = 12) -> pd.DataFrame:
    """painR3 상위 q → 롱, 하위 q → 숏. 임계는 **그 달 이전 데이터만**으로 (워크포워드).

    `체결모델확정-및-painR3-3종검정.md` §1 2차: 지정가 진입(maker 2) + 48봉 시간 청산(taker 6),
    겹치는 거래 제거(보유 중 신호 무시). R 환산용 1R = 2 × 일간중앙 ATR (문서의 배리어판과 같음).
    """
    from .execution import ExecConfig, timed_exit_trades
    idx = df.index
    month = idx.year * 12 + idx.month
    ms = pd.Series(score, index=idx)
    ok = np.isfinite(score)
    hi_thr = np.full(len(df), np.nan)
    lo_thr = np.full(len(df), np.nan)
    um = np.unique(month)
    for i, m in enumerate(um):
        if i < min_months:
            continue
        past = score[(month < m) & ok]
        if len(past) < 10_000:
            continue
        a, b = np.quantile(past, [q, 1 - q])
        sel = month == m
        lo_thr[sel] = a
        hi_thr[sel] = b
    sig = np.zeros(len(df), np.int8)
    sig[ok & (score >= hi_thr)] = 1
    sig[ok & (score <= lo_thr)] = -1
    cand = np.flatnonzero(sig != 0)
    take = []
    busy = -1
    for b in cand:
        if b <= busy:
            continue
        take.append(b)
        busy = b + fill_window + hold                   # 보수적으로 최대 점유 구간
    if not take:
        return trade_frame(df, [], [], [], [], [], [], [], entry_bps=MAKER)
    bars = np.array(take, np.int64)
    bars = bars[bars < len(df) - hold - fill_window - 2]
    h, l, c = (df[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    risk = 2.0 * atr_daily[bars]
    cfg = ExecConfig(entry_bps=MAKER, tp_bps=MAKER, sl_bps=TAKER, fill_window=fill_window)
    tr = timed_exit_trades(h, l, c, bars, sig[bars].astype(np.int64), hold, cfg, risk=risk)
    f = tr[tr.filled].reset_index(drop=True)
    ent = f.entry.to_numpy(float)
    rb = f.risk_bps.to_numpy(float)
    return pd.DataFrame(dict(bar=f.fill_bar.to_numpy(np.int64), time=df.index[f.fill_bar.to_numpy()],
                             side=f.side.to_numpy(np.int8), entry=ent, risk_bps=rb, tp_r=np.nan,
                             gross=f.r_gross.to_numpy(float),
                             fee=(f.fee_bps.to_numpy(float) / rb),
                             net=f.r_net.to_numpy(float), kind=np.full(len(f), 2, np.int8),
                             gross_bps=f.gross_bps.to_numpy(float)))
