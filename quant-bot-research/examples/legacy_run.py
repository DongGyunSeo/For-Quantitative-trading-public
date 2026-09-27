"""예전 신호군 전부 — 47심볼 · 봉인 전(2026-07-05) · 신호별 거래 목록.

정본 설정은 각 문서의 사전등록 주 지표(없으면 결론 표의 대표 셀)를 쓰고,
비용은 현재 표준(지정가 2 / 시장가 6 bps), 측정해상도 규칙(1R/일간ATR ≥ 2)을 모두에 건다.
결과: runs/legacy/<신호>.pkl  +  runs/legacy/tsmom.pkl
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research import legacy as L
from qbot.research.fvg import FVGConfig, atr5m
from qbot.research.fvg_sweep import choch_on_tf, excursion, sweep_choch
from qbot.research.trader_pop import CANON, Archetype, population_state

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
OUT = Path("runs/legacy")
STRUCT = ("ma_cross_struct", "rsi_band", "pivot_break")


def run(job):
    sym, path = job
    t0 = time.time()
    df, _ = load_ohlcv(path)
    df = df[df.index <= SEAL]
    a = atr5m(df, 14, mode="daily")
    rk = L.tr_atr_rank_5m(df)
    out = {}
    rule = lambda t: L.apply_rule(t, a)
    out["htf_fib_base"] = rule(L.htf_fib_trades(df, tf="4h"))
    out["atr_top10"] = rule(L.htf_fib_trades(df, tf="4h", rank_min=0.90))
    out["lvl_break"] = rule(L.level_break_trades(df, rank_min=0.75, inside_only=True))
    out["bigbar_choch"] = rule(L.big_bar_choch_trades(df, top=0.20, rank=rk))
    ev = L.cascade_events(df, rank=rk)
    out["liq_rev"] = rule(L.liq_cascade_trades(df, direction="reversal", events=ev))
    out["liq_cont_top1"] = rule(L.liq_cascade_trades(df, direction="continuation", events=ev,
                                                     size_top=0.01, atr_daily=a))
    out["trend_fvg_15m"] = rule(L.trend_fvg_trades(df, mtf="15m", wait=96))
    out["trend_fvg_1h"] = rule(L.trend_fvg_trades(df, mtf="1h", wait=384))
    cache: dict = {}
    ch15 = choch_on_tf(df, "15m", 3, cache)
    ex = excursion(df, FVGConfig(htf="4h", atr_mode="daily"), horizon=576)
    sw = sweep_choch(df, ex, ch15, min_depth=0.0, max_wait=576, buffer_atr=0.0, atr=a)
    out["sweep_choch"] = L.limit_signal_trades(df, sw, a)
    out["choch_15m"] = L.limit_signal_trades(df, L.plain_choch_signals(df, ch15), a)
    out["weekend_gap"] = rule(L.weekend_gap_trades(df))
    archs = [Archetype(k, p, tf, stop=st, exit=ex_, label=lb)
             for lb, k, p, st, ex_, _ in CANON if lb in STRUCT for tf in ("5m", "15m", "1h", "4h", "1d")]
    R, _ = population_state(df, archs, a)
    out["pain_tail"] = L.pain_tail_trades(df, R.pain_imb_R3, a)
    for k in out:
        out[k] = out[k].assign(symbol=sym)
    closes = df.close.resample("1D").last()
    return sym, out, closes, round(time.time() - t0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = symbol_files("cache_okx")
    acc: dict[str, list] = {}
    closes = {}
    with ProcessPoolExecutor(2) as ex:
        for sym, out, cl, sec in ex.map(run, list(files.items())):
            for k, v in out.items():
                acc.setdefault(k, []).append(v)
            closes[sym] = cl
            print(f"  {sym:<6} {sec:>5}s  " + " ".join(f"{k}={len(v)}" for k, v in out.items()), flush=True)
    for k, v in acc.items():
        pd.concat(v, ignore_index=True).to_pickle(OUT / f"{k}.pkl")
    C = pd.DataFrame(closes)
    C.to_pickle(OUT / "daily_close.pkl")
    L.tsmom_weekly(C).to_pickle(OUT / "tsmom.pkl")
    # FVG 는 이미 계산된 47심볼 거래를 쓴다 (fvg_gate_stability.py 산출)
    u = pd.read_csv("runs/u47_trades.csv", parse_dates=["time"])
    u[u.tf == "4h"].to_pickle(OUT / "fvg_4h.pkl")
    u[(u.tf == "4h") & (u.risk_bps >= 150)].to_pickle(OUT / "fvg_4h_g150.pkl")
    u[u.tf == "1h"].to_pickle(OUT / "fvg_1h.pkl")
    print("완료 ->", OUT)


if __name__ == "__main__":
    main()
