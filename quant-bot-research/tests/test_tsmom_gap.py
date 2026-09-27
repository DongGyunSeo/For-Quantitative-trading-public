"""금 신호 · 월 시가 · 주말갭 지정가 — 손계산 (보유 · 시장가 · 지정가 체결 · 패스 · 불리한 갭)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.research import tsmom_gap as G


def _T(rows):
    mons = pd.date_range("2024-01-08 14:30", periods=len(rows), freq="7D", tz="UTC")
    T = pd.DataFrame(rows)
    T["monday"] = mons
    T["symbol"] = "X"
    return T


def test_market_limit_skip_and_unfavorable():
    #   F     O     O'    F_prev4  mid 닿음 edge 닿음
    rows = [dict(F=100, O=100, O_next=110, F_prev4=90, touch_mid=False, touch_edge=False),    # 첫 주 롱, 갭 0 → 시가
            dict(F=108, O=110, O_next=104, F_prev4=100, touch_mid=True, touch_edge=True),    # 같은 방향 보유
            dict(F=95, O=100, O_next=90, F_prev4=100, touch_mid=True, touch_edge=False),     # 숏 신호, 위로 갭(불리) → 시가
            dict(F=85, O=80, O_next=86, F_prev4=90, touch_mid=True, touch_edge=True),        # 숏 유지
            dict(F=100, O=104, O_next=98, F_prev4=90, touch_mid=True, touch_edge=False)]     # 롱 신호, 위로 갭(유리)
    T = _T(rows)
    wk2, R2 = G.backtest(T, "V2")
    wkm, Rm = G.backtest(T, "V3m")
    wke, Re = G.backtest(T, "V3e")
    f = G.FUND_WK
    assert R2.act.tolist() == ["market", "hold", "market", "hold", "market"]
    assert R2.ret.iloc[0] == pytest.approx(0.10 - 6e-4 - f)
    assert R2.ret.iloc[2] == pytest.approx(-1 * (90 / 100 - 1) - 2 * 6e-4 + f)        # 롱→숏 뒤집기 2단위
    # V3m 마지막 주: 숏 보유 중 롱 신호, 위로 갭 → mid 102 에 매수 지정가 체결 (뒤집기 2단위, 메이커)
    assert Rm.act.tolist()[-1] == "limit"
    L = 102.0
    exp = -1 * (L / 104 - 1) + 1 * (98 / L - 1) - 2 * 2e-4 - f
    assert Rm.ret.iloc[-1] == pytest.approx(exp)
    assert Rm.improve_bps.iloc[-1] == pytest.approx((104 - 102) / 104 * 1e4)
    # V3e 마지막 주: edge(100) 안 닿음 → 패스, 숏 유지
    assert Re.act.tolist()[-1] == "skip" and Re.pos.iloc[-1] == -1
    assert Re.ret.iloc[-1] == pytest.approx(-1 * (98 / 104 - 1) + f)
    assert Re.missed.iloc[-1] == pytest.approx(98 / 104 - 1)


def test_skip_then_next_week_retries_with_current_position():
    rows = [dict(F=100, O=100, O_next=100, F_prev4=110, touch_mid=False, touch_edge=False),  # 숏 진입(시가)
            dict(F=100, O=103, O_next=106, F_prev4=95, touch_mid=False, touch_edge=False),   # 롱 신호·유리한 갭·안 닿음 → 패스
            dict(F=107, O=105, O_next=110, F_prev4=100, touch_mid=True, touch_edge=True)]    # 롱 신호, 아래로 갭(불리) → 시가
    _, R = G.backtest(_T(rows), "V3m")
    assert R.act.tolist() == ["market", "skip", "market"]
    assert R.pos.tolist() == [-1, -1, 1]


def test_week_table_touch_logic():
    idx = pd.date_range("2024-07-12 00:00", "2024-07-22 23:55", freq="5min", tz="UTC").as_unit("us")
    c = np.full(len(idx), 100.0)
    mon = pd.Timestamp("2024-07-15 13:30", tz="UTC")
    c[idx >= mon] = 104.0                                  # 위로 갭 4 (F 100 → O 104)
    df = pd.DataFrame(dict(open=c, high=c, low=c, close=c, volume=1.0), index=idx)
    i = int(idx.searchsorted(pd.Timestamp("2024-07-16 10:00", tz="UTC")))
    df.iloc[i, df.columns.get_loc("low")] = 101.5          # mid(102)는 닿고 edge(100)는 안 닿음
    T = G.week_table(df)
    r = T.iloc[0]
    assert r.F == 100 and r.O == 104 and bool(r.touch_mid) and not bool(r.touch_edge)


def test_week_table_dst_and_prev4_by_date():
    """2024-03-10 미국 DST 시작 — 월 09:30 ET 가 14:30 → 13:30 UTC 로 바뀌어도 주가 이어진다."""
    idx = pd.date_range("2024-02-02 00:00", "2024-03-19 23:55", freq="5min", tz="UTC").as_unit("us")
    t = np.arange(len(idx), dtype=float)
    c = 100 + t * 1e-3                                       # 천천히 오르는 가격 (갭 0 아님: 주말도 오른다)
    df = pd.DataFrame(dict(open=c, high=c, low=c, close=c, volume=1.0), index=idx)
    T = G.week_table(df)
    wk = T.set_index("wk")
    a, b = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-11")
    assert np.isfinite(wk.loc[a, "O_next"]) and wk.loc[a, "O_next"] == pytest.approx(wk.loc[b, "O"])
    assert wk.loc[b, "F_prev4"] == pytest.approx(wk.loc[pd.Timestamp("2024-02-12"), "F"])
    assert T.monday.dt.tz_convert("America/New_York").dt.strftime("%H:%M").eq("09:30").all()
    _, R = G.backtest(T.assign(symbol="X"), "V2")
    assert R.wk.diff().dropna().eq(pd.Timedelta(days=7)).all()     # DST 주에 끊겨서 다시 시작하지 않는다
    assert (R.act == "market").sum() == 1                          # 계속 오름 → 첫 주에 롱 한 번
