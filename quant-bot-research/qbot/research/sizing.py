"""시계열 추세 — 코인별 사이징 · 교차 증거금 레버리지.

사전등록 `claude/시계열추세-사이징-레버리지-사전등록.md`.

- 비중은 **총노출 1배**(Σ|w| = 1)로 만든 모양 W 와 총노출 배수 G 로 나눈다. 노출 x = G · W (자산 대비 명목).
- 수수료는 **실제 거래량**: |새 목표 노출 − 지난주 노출이 가격 따라 흐른 값| × 6bps (자산 대비).
- 장중 경로: 주초 가격 P0 대비 r(t) = Σ w_i (P_i(t)/P0_i − 1), g(t) = Σ |w_i| P_i(t)/P0_i (총명목의 흐름).
- 교차 증거금 청산: 자산 ≤ 유지증거금 ⇔ 1 − G f + G r(t) ≤ G · mmr · g(t) ⇔ G ≥ 1 / (f + mmr g(t) − r(t)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["FEE", "FUND_WK", "unit_weights", "book", "book_cadence", "week_paths", "liq_threshold", "equity_path", "mdd"]

FEE = 6e-4
FUND_WK = 0.137 * 3 * 7 / 1e4


def unit_weights(sig: pd.DataFrame, sigma: pd.DataFrame | None, scheme: str) -> pd.DataFrame:
    """sig: 주 × 코인 (+1/−1/0, 없으면 NaN). 크기 a 는 쓸 수 있는 코인끼리 합 1, w = sig · a.

    ew = 1/N (현행과 같다), iv = ∝ 1/σ, vp = ∝ σ.
    """
    ok = sig.notna()
    if scheme == "ew":
        a = ok.astype(float)
    else:
        ok = ok & sigma.notna() & (sigma > 0)
        a = (1.0 / sigma) if scheme == "iv" else sigma.copy()
        a = a.where(ok, 0.0)
    a = a.where(ok, 0.0)
    a = a.div(a.sum(axis=1).replace(0, np.nan), axis=0)
    return (sig.fillna(0) * a).where(ok)


def book(W: pd.DataFrame, R: pd.DataFrame, gross: pd.Series | None = None, *, fee_mode: str = "actual") -> pd.DataFrame:
    """주별 장부. W: 단위 비중(주 × 코인), R: 그 주 코인 수익(주초 → 다음 주초), gross: 주별 G (기본 1).

    fee_mode "actual" = 흐른 노출까지 되맞춤 · "flip" = |x_t − x_{t−1}| 만(예전 백테스트와 같다).
    반환: ret(자산 대비 순수익) · gross_ret · fee · fund · turnover · G · fee_u(총노출 1배당 수수료).
    """
    G = pd.Series(1.0, index=W.index) if gross is None else gross.reindex(W.index)
    Wv = W.fillna(0).to_numpy(float)
    Rv = R.to_numpy(float)
    live = np.isfinite(Rv)
    Wv = np.where(live, Wv, 0.0)
    Rz = np.where(live, Rv, 0.0)
    Gv = G.to_numpy(float)
    n, m = Wv.shape
    out = np.zeros((n, 6))
    prev = np.zeros(m)                                        # 지난주 노출이 흐른 값 (새 자산 대비)
    for t in range(n):
        x = Gv[t] * Wv[t]
        trade = np.abs(x - prev).sum()
        fee = FEE * trade
        r = float(x @ Rz[t])
        fund = FUND_WK * x.sum()
        net = r - fee - fund
        out[t] = (net, r, fee, fund, trade, Gv[t])
        prev = x * (1 + Rz[t]) / (1 + net) if fee_mode == "actual" else x
    df = pd.DataFrame(out, index=W.index, columns=["ret", "gross_ret", "fee", "fund", "turnover", "G"])
    df["fee_u"] = df.fee / df.G.replace(0, np.nan)
    return df


def book_cadence(W: pd.DataFrame, R: pd.DataFrame, G: float, update, *, asym: bool = False,
                 rmin: np.ndarray | None = None, gliq: np.ndarray | None = None) -> tuple[pd.DataFrame, int]:
    """사이징 기준 자산 B 를 ``update`` 인 주에만 지금 자산 E 로 바꾸는 장부 (복리 주기).

    코인 노출 = G · (B / E) · W → 실효 G = G · B / E. ``asym`` 이면 갱신 안 하는 주에도 B = min(B, E) (줄이는 쪽은 매주).
    rmin · gliq (총노출 1배당 장중 최저 · 청산 임계)를 주면 장중 저점과 청산을 함께 본다. 청산 뒤 자산 0.
    반환: (주별 ret · G_eff · fee · fund · turnover · E_start · B · trough · E_end, 청산 주 위치 또는 −1).
    """
    Wv = W.fillna(0).to_numpy(float)
    Rv = R.to_numpy(float)
    live = np.isfinite(Rv)
    Wv = np.where(live, Wv, 0.0)
    Rz = np.where(live, Rv, 0.0)
    upd = np.asarray(update, bool)
    n, m = Wv.shape
    out = np.zeros((n, 9))
    E, B = 1.0, 1.0
    prev = np.zeros(m)
    liq = -1
    for t in range(n):
        if liq >= 0:
            continue                                          # 청산 뒤 전부 0
        if t == 0 or upd[t]:
            B = E
        elif asym:
            B = min(B, E)
        ge = G * B / E
        x = ge * Wv[t]
        trade = np.abs(x - prev).sum()
        fee = FEE * trade
        r = float(x @ Rz[t])
        fund = FUND_WK * x.sum()
        net = r - fee - fund
        trough = E * (1 - fee + ge * rmin[t]) if rmin is not None else np.nan
        if gliq is not None and ge >= gliq[t]:
            liq = t
            out[t] = (-1.0, ge, fee, fund, trade, E, B, 0.0, 0.0)
            continue
        out[t] = (net, ge, fee, fund, trade, E, B, trough, E * (1 + net))
        prev = x * (1 + Rz[t]) / (1 + net)
        E = E * (1 + net)
    cols = ["ret", "G_eff", "fee", "fund", "turnover", "E_start", "B", "trough", "E_end"]
    return pd.DataFrame(out, index=W.index, columns=cols), liq


def week_paths(close: np.ndarray, high: np.ndarray, low: np.ndarray, W: np.ndarray, bars: int = 2016):
    """주별 장중 경로 (단위 총노출). close/high/low: (1 + 주 수 × bars, 코인), 0행 = 첫 주초 직전 봉.

    반환 dict: r · g (종가 기준), rc · gc (보수적: 롱은 저가 · 숏은 고가), 모두 (주, bars).
    """
    nw = W.shape[0]
    r = np.empty((nw, bars))
    g = np.empty((nw, bars))
    rc = np.empty((nw, bars))
    gc = np.empty((nw, bars))
    for k in range(nw):
        s = 1 + k * bars
        w = np.nan_to_num(W[k])
        p0 = close[s - 1].astype(float)
        cl = close[s:s + bars].astype(float) / p0
        adv = np.where(w > 0, low[s:s + bars], high[s:s + bars]).astype(float) / p0
        aw = np.abs(w)
        r[k], g[k] = cl @ w - w.sum(), cl @ aw
        rc[k], gc[k] = adv @ w - w.sum(), adv @ aw
    return dict(r=r, g=g, rc=rc, gc=gc)


def liq_threshold(r: np.ndarray, g: np.ndarray, f: np.ndarray, mmr: float, stress: float = 1.0) -> np.ndarray:
    """주별로 청산이 나는 가장 작은 G (없으면 inf). r, g: (주, 봉), f: 주별 총노출 1배당 수수료."""
    den = f[:, None] + mmr * g - stress * r
    with np.errstate(divide="ignore"):
        th = np.where(den > 0, 1.0 / den, np.inf)
    return th.min(axis=1)


def equity_path(net: np.ndarray, rmin: np.ndarray, f: np.ndarray, G: np.ndarray, gliq: np.ndarray):
    """주별 자산 경로. net/rmin/f: 총노출 1배당, G: 주별 배수, gliq: 주별 청산 임계.

    반환: (주말 자산, 장중 저점, 청산 주 위치 또는 −1). 청산되면 그 주 저점 = 0, 이후 0.
    """
    n = len(net)
    end = np.empty(n)
    trough = np.empty(n)
    e = 1.0
    liq = -1
    for t in range(n):
        if liq >= 0:
            end[t] = trough[t] = 0.0
            continue
        if G[t] >= gliq[t]:
            liq = t
            end[t] = trough[t] = 0.0
            continue
        trough[t] = e * (1 - G[t] * f[t] + G[t] * rmin[t])
        e = e * (1 + G[t] * net[t])
        end[t] = e
    return end, trough, liq


def mdd(end: np.ndarray, trough: np.ndarray | None = None) -> float:
    """최대낙폭(양수). 고점 = 이전 주말 자산의 최고(처음 1 포함), 저점 = 주말 자산 · 장중 저점."""
    start = np.concatenate([[1.0], end[:-1]])
    peak = np.maximum.accumulate(np.maximum(start, 1.0))
    dd = 1 - end / np.maximum.accumulate(np.concatenate([[1.0], end]))[1:]
    if trough is not None:
        dd = np.maximum(dd, 1 - trough / peak)
    return float(np.max(dd))
