"""비용 상태 스위치 — `claude/비용상태스위치-사전등록.md` 의 규칙 그대로.

거래마다 진입 시점에 알 수 있는 것만으로::

    ĝ   그 신호의 워크포워드 거래당 gross R (달마다 갱신, 결과가 난 거래만)
    p̂   워크포워드 익절 비율
    F   기대 왕복 비용 bps = 진입 bps + p̂·2 + (1−p̂)·6        (시간 청산은 고정)
    ρ   = 1R_bps · ĝ / F        (= 1R_bps / 손익분기 1R)

주 규칙: ``ĝ > 0 and ρ ≥ 2`` 이면 진입.

보조: 시장 변동성 상태 — 47심볼 30일 실현변동성의 횡단면 중앙값(전날까지)이
직전 365일 중앙값 이상이면 켬.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["walkforward_edge", "attach_rho", "market_state", "attach_state"]

MAKER, TAKER = 2.0, 6.0


def _month(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")


def walkforward_edge(t: pd.DataFrame, months: pd.PeriodIndex, *, lag_days: int = 7,
                     min_months: int = 12, window_months: int | None = None) -> pd.DataFrame:
    """달 m 마다 ĝ·p̂ 를 **그 달 시작 lag_days 일 전까지 진입한 거래**로 계산한다.

    ``window_months`` 가 None 이면 누적(첫 거래부터), 아니면 직전 N개월.
    이력이 ``min_months`` 개월 미만이면 ok=False.
    """
    time = t["time"]
    first = _month(time).min()
    rows = []
    ep = pd.Timestamp("1970-01-01", tz="UTC")
    sec = lambda x: (x - ep).total_seconds()                  # tz·단위 혼동 방지 — 초로 비교
    tt = (time - ep).dt.total_seconds().to_numpy()
    order = np.argsort(tt, kind="stable")
    tt = tt[order]
    g = t["gross"].to_numpy(float)[order]
    win = (t["kind"].to_numpy() == 1)[order]
    cg = np.concatenate(([0.0], np.cumsum(g)))
    cw = np.concatenate(([0], np.cumsum(win)))
    for m in months:
        start = pd.Timestamp(m.start_time, tz="UTC")
        hi = np.searchsorted(tt, sec(start - pd.Timedelta(days=lag_days)), side="left")
        if window_months is None:
            lo = 0
        else:
            lo = np.searchsorted(tt, sec(pd.Timestamp((m - window_months).start_time, tz="UTC")),
                                 side="left")
        n = hi - lo
        hist = (m - first).n if first is not pd.NaT else 0
        ok = n > 0 and hist >= min_months
        rows.append(dict(month=m, n=int(n), ok=bool(ok),
                         g=float((cg[hi] - cg[lo]) / n) if n > 0 else np.nan,
                         p=float((cw[hi] - cw[lo]) / n) if n > 0 else np.nan))
    return pd.DataFrame(rows).set_index("month")


def attach_rho(t: pd.DataFrame, wf: pd.DataFrame, *, entry_bps: float,
               fixed_fee_bps: float | None = None) -> pd.DataFrame:
    """거래마다 그 달의 ĝ·p̂ 로 F·ρ 를 붙인다. ok=False 인 달의 거래는 ρ = NaN."""
    m = _month(t["time"])
    w = wf.reindex(m.to_numpy())
    g = w["g"].to_numpy(float)
    p = w["p"].to_numpy(float)
    ok = w["ok"].fillna(False).to_numpy(bool)
    F = np.full(len(t), fixed_fee_bps) if fixed_fee_bps is not None else entry_bps + p * MAKER + (1 - p) * TAKER
    rho = t["risk_bps"].to_numpy(float) * g / F
    out = t.copy()
    out["g_hat"] = np.where(ok, g, np.nan)
    out["F"] = np.where(ok, F, np.nan)
    out["rho"] = np.where(ok & (g > 0), rho, np.nan)
    out["wf_ok"] = ok
    return out


def market_state(daily_close: pd.DataFrame, *, vol_days: int = 30, ref_days: int = 365,
                 min_ref: int = 180) -> pd.DataFrame:
    """일 단위 시장 변동성 상태. 값은 **전날까지**의 정보로 만든다(shift 1)."""
    C = daily_close.sort_index()
    r = np.log(C).diff()
    v = (r.rolling(vol_days, min_periods=vol_days // 2).std() * np.sqrt(365)).median(axis=1)
    ref = v.rolling(ref_days, min_periods=min_ref).median()
    out = pd.DataFrame(dict(V=v.shift(1), ref=ref.shift(1)))
    valid = out.V.notna() & out.ref.notna()
    out["high"] = np.where(valid, (out.V >= out.ref).astype(float), np.nan)   # 1.0 / 0.0 / NaN
    return out


def attach_state(t: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """진입 시각이 속한 UTC 일의 상태를 붙인다(그 날 값은 전날까지로 만들어져 있다)."""
    day = t["time"].dt.tz_convert("UTC").dt.floor("D")
    s = state.copy()
    s.index = pd.to_datetime(s.index).tz_convert("UTC") if s.index.tz is not None else \
        pd.to_datetime(s.index).tz_localize("UTC")
    s.index = s.index.floor("D")
    out = t.copy()
    out["V"] = s["V"].reindex(day.to_numpy()).to_numpy()
    out["mkt_high"] = s["high"].reindex(day.to_numpy()).to_numpy()
    return out
