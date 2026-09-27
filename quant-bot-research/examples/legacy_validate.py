"""재구축 검수 — 문서에 적힌 숫자가 다시 나오는가 (10심볼 · 문서와 같은 구간).

문서들은 IS 구간(심볼마다 시작일이 다른 두 번째 CSV 세트)만 읽고 돌렸다.
같은 조건을 만들려고 심볼별 IS 시작일로 **먼저 자른 뒤** 계산한다(워밍업도 IS 안에서).
규칙(1R/ATR ≥ 2)은 문서가 쓴 경우에만 건다.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research import legacy as L
from qbot.research.fvg import atr5m

IS_START = {"BTC": "2023-11-18", "ETH": "2023-11-18", "SOL": "2023-11-18",
            "ADA": "2023-12-18", "DOGE": "2023-12-18", "LINK": "2023-12-18",
            "TRX": "2023-12-18", "XRP": "2023-12-18", "AVAX": "2024-04-16", "BNB": "2024-12-12"}


def run(job):
    sym, path = job
    df, _ = load_ohlcv(path)
    df = df[df.index >= pd.Timestamp(IS_START[sym], tz="UTC")]
    out = {}
    out["atr_base"] = L.htf_fib_trades(df, tf="4h")
    out["atr_top25"] = L.htf_fib_trades(df, tf="4h", rank_min=0.75)
    out["atr_top10"] = L.htf_fib_trades(df, tf="4h", rank_min=0.90)
    out["lvl_all"] = L.level_break_trades(df, rank_min=None, inside_only=False, include_entry_bar=False)
    out["lvl_best"] = L.level_break_trades(df, rank_min=0.75, inside_only=True, include_entry_bar=False)
    out["bigbar"] = L.big_bar_choch_trades(df, top=0.20, wait=24, tp_r=1.0)
    ev = L.cascade_events(df)
    out["liq_vol"] = L.liq_cascade_trades(df, direction="reversal", tp_r=1.0, events=ev)
    tr = L.trend_fvg_trades(df, fib=0.7, tp_r=3.0)
    out["trend_inst"] = L.apply_rule(tr, atr5m(df, 14, mode="inst"))
    out["trend_daily"] = L.apply_rule(tr, atr5m(df, 14, mode="daily"))
    out["wgap"] = L.weekend_gap_trades(df)
    for k in out:
        out[k] = out[k].assign(symbol=sym)
    return sym, out


def main():
    files = symbol_files("cache_all")
    acc: dict[str, list] = {}
    with ProcessPoolExecutor(2) as ex:
        for sym, out in ex.map(run, list(files.items())):
            for k, v in out.items():
                acc.setdefault(k, []).append(v)
            print(f"  {sym} 완료", flush=True)
    T = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    Path("runs").mkdir(exist_ok=True)
    pd.to_pickle(T, "runs/legacy_validate.pkl")

    def win_resolved(t):
        r = t[t.kind.isin([1, -1])]
        return (r.kind == 1).mean(), len(r) / max(len(t), 1)

    print("\n항목 | 재구축 | 문서")
    for k, doc in (("atr_base", "n 16,196 · 승 40.2% · gross +0.020"),
                   ("atr_top25", "n 3,479 · 승 43.9% · gross +0.061"),
                   ("atr_top10", "n 1,293 · 승 45.0% · gross +0.057")):
        t = T[k]
        print(f"{k:<12} n {len(t):,} · 승 {(t.gross > 0).mean():.1%} · gross {t.gross.mean():+.3f}  | {doc}")
    for k, doc in (("lvl_all", "롱 0.9→1.7 n 12,252 · 승 52.82% (해결분)"),
                   ("lvl_best", "롱 안쪽돌파+ATR상위25% n 849 · 승 59.25% · 해결 19.8%")):
        t = T[k]
        lg = t[t.side > 0]
        w, res = win_resolved(lg)
        sh = t[t.side < 0]
        w2, res2 = win_resolved(sh)
        print(f"{k:<12} 롱 n {len(lg):,} 승 {w:.2%} 해결 {res:.1%} · 숏 n {len(sh):,} 승 {w2:.2%}  | {doc}")
    t = T["bigbar"]
    for sd, nm, doc in ((-1, "숏", "n 39,260 · 승 50.56% · 1R 73bps"), (1, "롱", "n 39,680 · 승 49.71% · 1R 78bps")):
        s = t[t.side == sd]
        w, _ = win_resolved(s)
        print(f"bigbar {nm}    n {len(s):,} · 승 {w:.2%} · 1R 중앙 {s.risk_bps.median():.0f}bps  | {doc}")
    t = T["liq_vol"]
    w, _ = win_resolved(t)
    print(f"liq_vol      n {len(t):,} · 승 {w:.2%} · 1R 중앙 {t.risk_bps.median():.1f}bps  | n 236,134 · 승 53.27% · 1R 29.5bps")
    for k in ("trend_inst", "trend_daily"):
        t = T[k]
        print(f"{k:<12} n {len(t):,} · 승 {(t.gross > 0).mean():.2%} · gross {t.gross.mean():+.4f} · 1R {t.risk_bps.median():.0f}bps"
              f"  | n 6,189 · 승 28.28% · gross +0.1318 · 1R 63bps")
    t = T["wgap"]
    print(f"wgap         n {len(t):,} · 메움 {(t.kind == 1).mean():.2%} · gross {t.gross.mean():+.3f} · 1R {t.risk_bps.median():.0f}bps"
          f"  | n 757 · 메움 60.73% · gross +0.160 · 1R 440bps")


if __name__ == "__main__":
    main()
