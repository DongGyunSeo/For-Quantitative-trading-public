"""독립 재계산 — tsmom_gap 모듈 없이.

1) 3개 심볼: 5m 원자료에서 pandas 로 F · O · O' · 4주 전 F · 주간 저가/고가 닿음을 다시 만들어 weeks.pkl 과 비교
2) V2: 코인×주 표에서 벡터로 · V3m/V3e: 배열 루프로 다시 계산해 backtest 주간 수익과 비교
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files  # noqa: E402
from qbot.research import tsmom_gap as G  # noqa: E402

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
W = pd.read_pickle("runs/tsmom_gap/weeks.pkl")

# ---- 1) 원자료 대조
files = symbol_files("cache_okx")
bad = 0
for sym in ("BTC", "DOGE", "UMA"):
    df, _ = load_ohlcv(files[sym])
    df = df[df.index < SEAL]
    et = df.index.tz_convert("America/New_York")
    fri = df[(et.dayofweek == 4) & (et.hour == 15) & (et.minute == 55)]
    mon = df[(et.dayofweek == 0) & (et.hour == 9) & (et.minute == 30)]
    Fd = pd.Series(fri.close.to_numpy(), index=(fri.index.tz_convert("America/New_York").tz_localize(None).normalize()
                                                 + pd.Timedelta(days=3)))      # 금 → 그 다음 월요일 날짜
    Od = pd.Series(mon.open.to_numpy(), index=mon.index.tz_convert("America/New_York").tz_localize(None).normalize())
    Ot = pd.Series(mon.index, index=Od.index)
    mine = W[W.symbol == sym].set_index("wk")
    for wk, row in mine.iterrows():
        F, O = Fd.get(wk), Od.get(wk)
        On = Od.get(wk + pd.Timedelta(days=7))
        Fp = Fd.get(wk - pd.Timedelta(days=28))
        ok = np.isclose(row.F, F) and np.isclose(row.O, O)
        ok &= (np.isnan(row.O_next) and On is None) or np.isclose(row.O_next, On)
        ok &= (np.isnan(row.F_prev4) and Fp is None) or np.isclose(row.F_prev4, Fp)
        if On is not None and O != F:
            seg = df[(df.index >= Ot[wk]) & (df.index < Ot[wk + pd.Timedelta(days=7)])]
            mid = (O + F) / 2
            if O > F:
                tm, te = seg.low.min() <= mid, seg.low.min() <= F
            else:
                tm, te = seg.high.max() >= mid, seg.high.max() >= F
            ok &= (bool(tm) == bool(row.touch_mid)) and (bool(te) == bool(row.touch_edge))
        bad += int(not ok)
    print(sym, "주", len(mine), "누적 불일치", bad)
assert bad == 0

# ---- 2) 포트폴리오 재계산
piv = {k: W.pivot(index="wk", columns="symbol", values=k) for k in ("F", "O", "O_next", "F_prev4", "touch_mid", "touch_edge")}
Fw, Ow, On, Fp = (piv[k].to_numpy(float) for k in ("F", "O", "O_next", "F_prev4"))
TM, TE = piv["touch_mid"].to_numpy(bool), piv["touch_edge"].to_numpy(bool)
nW, nC = Fw.shape
valid = np.isfinite(Fp) & np.isfinite(On)
sig = np.where(valid, np.sign(Fw / Fp - 1), np.nan)
fwd = On / Ow - 1
fund = 0.137 * 3 * 7 / 1e4

# V2 벡터
pos = sig
prev = np.vstack([np.zeros((1, nC)), np.nan_to_num(pos[:-1])])
r2 = np.where(valid, pos * fwd - np.abs(np.nan_to_num(pos) - prev) * 6e-4 - pos * fund, np.nan)
v2 = pd.Series(np.nanmean(r2, axis=1), index=piv["F"].index)

def v3(kind: str) -> pd.Series:
    out = np.full((nW, nC), np.nan)
    for j in range(nC):
        p = 0.0
        for i in range(nW):
            if not valid[i, j]:
                p = 0.0
                continue
            s, O, F, O2 = sig[i, j], Ow[i, j], Fw[i, j], On[i, j]
            if s == p:
                out[i, j] = p * (O2 / O - 1) - p * fund
                continue
            good = s != 0 and ((s > 0 and O > F) or (s < 0 and O < F))
            if not good:
                out[i, j] = s * (O2 / O - 1) - abs(s - p) * 6e-4 - s * fund
                p = s
                continue
            L = (O + F) / 2 if kind == "m" else F
            hit = TM[i, j] if kind == "m" else TE[i, j]
            if hit:
                out[i, j] = p * (L / O - 1) + s * (O2 / L - 1) - abs(s - p) * 2e-4 - s * fund
                p = s
            else:
                out[i, j] = p * (O2 / O - 1) - p * fund
    return pd.Series(np.nanmean(out, axis=1), index=piv["F"].index)

for name, mine in (("V2", v2), ("V3m", v3("m")), ("V3e", v3("e"))):
    b, _ = G.backtest(W, name)
    x = mine.dropna()
    y = b.ret.reindex(x.index)
    print(name, "주", len(x), "최대 차이", float(np.abs(x - y).max()), "샤프",
          round(float(x.mean() / x.std() * np.sqrt(52)), 4), round(float(y.mean() / y.std() * np.sqrt(52)), 4))
    assert np.allclose(x.to_numpy(), y.to_numpy(), atol=1e-12)
print("독립 재계산 일치")
