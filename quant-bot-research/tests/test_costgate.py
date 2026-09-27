"""비용 상태 스위치 — 워크포워드 인과성과 산술."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.research import costgate as G


def _trades(n=3000, seed=0, start="2021-04-01", days=900):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    time = t0 + pd.to_timedelta(np.sort(rng.uniform(0, days, n)), unit="D")
    kind = rng.choice([1, -1], n, p=[0.35, 0.65])
    gross = np.where(kind == 1, 2.0, -1.0)
    return pd.DataFrame(dict(time=time, gross=gross, kind=kind,
                             risk_bps=rng.uniform(50, 400, n)))


def test_walkforward_uses_only_resolved_past():
    t = _trades()
    months = pd.period_range("2021-04", "2023-09", freq="M")
    wf = G.walkforward_edge(t, months, min_months=0)
    m = pd.Period("2022-06", "M")
    cut = pd.Timestamp("2022-06-01", tz="UTC") - pd.Timedelta(days=7)
    past = t[t.time < cut]
    assert wf.loc[m, "n"] == len(past)
    assert np.isclose(wf.loc[m, "g"], past.gross.mean())
    assert np.isclose(wf.loc[m, "p"], (past.kind == 1).mean())


def test_walkforward_is_not_changed_by_future():
    t = _trades()
    months = pd.period_range("2021-04", "2023-09", freq="M")
    a = G.walkforward_edge(t, months)
    t2 = t.copy()
    fut = t2.time >= pd.Timestamp("2022-09-01", tz="UTC")
    t2.loc[fut, "gross"] = 5.0
    b = G.walkforward_edge(t2, months)
    k = months[months <= pd.Period("2022-09", "M")]
    assert np.allclose(a.loc[k, "g"], b.loc[k, "g"], equal_nan=True)


def test_min_months_and_window():
    t = _trades()
    months = pd.period_range("2021-04", "2023-09", freq="M")
    wf = G.walkforward_edge(t, months, min_months=12)
    assert not wf.loc[pd.Period("2022-02", "M"), "ok"]
    assert wf.loc[pd.Period("2022-05", "M"), "ok"]
    w24 = G.walkforward_edge(t, months, min_months=0, window_months=6)
    m = pd.Period("2023-01", "M")
    lo = pd.Timestamp("2022-07-01", tz="UTC")
    hi = pd.Timestamp("2023-01-01", tz="UTC") - pd.Timedelta(days=7)
    ref = t[(t.time >= lo) & (t.time < hi)].gross.mean()
    assert np.isclose(w24.loc[m, "g"], ref)


def test_rho_arithmetic():
    t = pd.DataFrame(dict(time=[pd.Timestamp("2023-02-10", tz="UTC")], gross=[0.0], kind=[2],
                          risk_bps=[200.0]))
    wf = pd.DataFrame(dict(n=[100], ok=[True], g=[0.05], p=[0.25]),
                      index=pd.PeriodIndex([pd.Period("2023-02", "M")], name="month"))
    out = G.attach_rho(t, wf, entry_bps=2.0)
    F = 2 + 0.25 * 2 + 0.75 * 6
    assert np.isclose(out.F.iloc[0], F)
    assert np.isclose(out.rho.iloc[0], 200 * 0.05 / F)
    fixed = G.attach_rho(t, wf, entry_bps=2.0, fixed_fee_bps=8.0)
    assert np.isclose(fixed.rho.iloc[0], 200 * 0.05 / 8)


def test_rho_is_nan_when_edge_not_positive():
    t = pd.DataFrame(dict(time=[pd.Timestamp("2023-02-10", tz="UTC")], gross=[0.0], kind=[2],
                          risk_bps=[200.0]))
    wf = pd.DataFrame(dict(n=[100], ok=[True], g=[-0.01], p=[0.3]),
                      index=pd.PeriodIndex([pd.Period("2023-02", "M")], name="month"))
    assert np.isnan(G.attach_rho(t, wf, entry_bps=6.0).rho.iloc[0])


def test_market_state_is_lagged():
    idx = pd.date_range("2021-01-01", periods=800, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    C = pd.DataFrame(dict(A=100 * np.exp(np.cumsum(rng.normal(0, 0.03, 800))),
                          B=100 * np.exp(np.cumsum(rng.normal(0, 0.03, 800)))), index=idx)
    s1 = G.market_state(C)
    C2 = C.copy()
    C2.iloc[500:] *= np.exp(np.cumsum(rng.normal(0, 0.2, 300)))[:, None]   # 미래만 요동
    s2 = G.market_state(C2)
    assert np.allclose(s1.V.iloc[:501], s2.V.iloc[:501], equal_nan=True)   # 500일 값은 499일까지로
    assert s1.high.dropna().isin([0.0, 1.0]).all()
