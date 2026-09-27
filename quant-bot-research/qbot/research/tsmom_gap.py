"""시계열 추세 — 금 종가 신호 · 월 장 시가 체결 · 주말갭 지정가 진입.

`claude/시계열추세-금신호-월시가-주말갭지정가-사전등록.md` 그대로.

- F = 금 16:00 ET 종가 (ts 15:55 ET 5m 봉), O = 월 09:30 ET 시가, O' = 다음 월 09:30 ET 시가
- 신호 = sign(F / 4주 전 F − 1)
- V2: 포지션이 바뀌면 O 에 시장가
- V3: 새 방향에 유리한 갭이면 지정가 L (mid = (O+F)/2 · edge = F), 그 주 안에 닿으면 L 에 체결, 안 닿으면 그 주 패스.
      불리한 갭 · 0 갭이면 O 에 시장가
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .wgap import ET, first_touch, weekend_events

__all__ = ["et_week", "week_table", "backtest", "TAKER", "MAKER", "FUND_WK"]

TAKER, MAKER = 6e-4, 2e-4
FUND_WK = 0.137 * 3 * 7 / 1e4


def et_week(ts) -> pd.DatetimeIndex:
    """월 09:30 ET 시각 → 그 월요일의 ET 날짜(tz 없음). DST 주에도 주 간격이 정확히 7일."""
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(ET).tz_localize(None).normalize()


def week_table(df: pd.DataFrame, pen_bps: float = 1.0) -> pd.DataFrame:
    """코인 하나의 5m → 주(월요일)별 F · O · O' · 4주 전 F · 갭 쪽 mid/edge 첫 닿음(봉 수, 없으면 −1).

    닿음 = 매수 지정가면 저가 ≤ L, 매도면 고가 ≥ L (사전등록 체결 모델).
    ``*_p`` = 관통 민감도: L 을 ``pen_bps`` 이상 넘어서야 체결(서술용).
    """
    ev = weekend_events(df, min_gap_bps=0.0).reset_index(drop=True)
    if ev.empty:
        return pd.DataFrame()
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    mon = pd.DatetimeIndex(pd.to_datetime(ev["monday"], utc=True))
    wk = et_week(mon)
    at = {w: i for i, w in enumerate(wk)}
    d7, d28 = pd.Timedelta(days=7), pd.Timedelta(days=28)
    rows = []
    for i in range(len(ev)):
        s, O, F = int(ev.mi.iloc[i]), float(ev.O.iloc[i]), float(ev.F.iloc[i])
        j, k = at.get(wk[i] + d7), at.get(wk[i] - d28)
        stop = int(ev.mi.iloc[j]) if j is not None else -1
        mid = (O + F) / 2
        t = dict(t_mid=-1, t_edge=-1, t_mid_p=-1, t_edge_p=-1)
        if stop > s and O != F:
            side = -1 if O > F else 1                      # 위로 갭 → 아래 매수(저가), 아래로 갭 → 위 매도(고가)
            q = pen_bps / 1e4
            for name, lv in (("mid", mid), ("edge", F)):
                a = first_touch(h, l, s, stop, lv, side)
                b = first_touch(h, l, s, stop, lv * (1 - q) if side < 0 else lv * (1 + q), side)
                t[f"t_{name}"] = a - s if a >= 0 else -1
                t[f"t_{name}_p"] = b - s if b >= 0 else -1
        rows.append(dict(monday=mon[i], wk=wk[i], F=F, O=O,
                         O_next=float(ev.O.iloc[j]) if j is not None else np.nan,
                         F_prev4=float(ev.F.iloc[k]) if k is not None else np.nan,
                         bars=stop - s if stop > 0 else -1, **t))
    T = pd.DataFrame(rows)
    for name in ("mid", "edge"):
        T[f"touch_{name}"] = T[f"t_{name}"] >= 0
        T[f"touch_{name}_p"] = T[f"t_{name}_p"] >= 0
    return T


def backtest(T: pd.DataFrame, variant: str, *, pen: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T: symbol · monday · F · O · O_next · F_prev4 · touch_mid · touch_edge (여러 코인).

    variant: "V2" | "V3m" | "V3e". ``pen`` 이면 관통 체결(touch_*_p). Returns (주별 포트폴리오, 코인·주 기록).
    """
    T = T.copy()
    if "wk" not in T:
        T["wk"] = et_week(T["monday"])
    cm, ce = ("touch_mid_p", "touch_edge_p") if pen else ("touch_mid", "touch_edge")
    recs = []
    for sym, g in T.sort_values(["symbol", "wk"]).groupby("symbol", sort=False):
        p = 0.0
        last = None
        for e in g.itertuples(index=False):
            if last is not None and (e.wk - last) != pd.Timedelta(days=7):
                p = 0.0                                              # 끊긴 주 뒤에는 새로 시작
            last = e.wk
            if not (np.isfinite(e.F_prev4) and np.isfinite(e.O_next) and e.O > 0):
                p = 0.0
                continue
            tgt = float(np.sign(e.F / e.F_prev4 - 1))
            base = e.O_next / e.O - 1
            act, r, fee, improve, missed = "hold", p * base, 0.0, np.nan, np.nan
            prev, fav = p, False
            if tgt != p:
                fav = (tgt > 0 and e.O > e.F) or (tgt < 0 and e.O < e.F)
                if variant == "V2" or not fav or tgt == 0:
                    act, r, fee, p_new = "market", tgt * base, abs(tgt - p) * TAKER, tgt
                else:
                    L = (e.O + e.F) / 2 if variant == "V3m" else e.F
                    hit = bool(getattr(e, cm) if variant == "V3m" else getattr(e, ce))
                    if hit:
                        act = "limit"
                        r = p * (L / e.O - 1) + tgt * (e.O_next / L - 1)
                        fee = abs(tgt - p) * MAKER
                        improve = tgt * (e.O - L) / e.O * 1e4
                        p_new = tgt
                    else:
                        act, p_new = "skip", p
                        missed = tgt * base                           # 신호대로 들었다면
                p = p_new
            recs.append(dict(symbol=sym, monday=e.monday, wk=e.wk, prev=prev, pos=p, fav=bool(fav), act=act,
                             ret=r - fee - p * FUND_WK,
                             gross=r, fee=fee, improve_bps=improve, missed=missed, tgt=tgt, base=base))
    R = pd.DataFrame(recs)
    wk = R.groupby("wk").agg(ret=("ret", "mean"), gross=("gross", "mean"), fee=("fee", "mean"),
                                 n=("ret", "size"))
    return wk, R
