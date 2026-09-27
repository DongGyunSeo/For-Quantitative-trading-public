"""주말 갭 × 4H 본장정렬 × 15m CISD — 47심볼 · 봉인 전(2026-07-05).

사전등록: `claude/주말갭-4H정렬-CISD-사전등록.md`.
결과: runs/wgap/{events,B0,B0L,C,D1,D2,DC}.pkl
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
from qbot.research import wgap as W
from qbot.research.fvg import atr5m

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
OUT = Path("runs/wgap")
KEYS = ("B0", "B0L", "C", "D1", "D2", "DC")


def _utc(t: pd.DataFrame) -> pd.DataFrame:
    if "monday" in t:
        t["monday"] = pd.to_datetime(t["monday"], utc=True)
    return t


def run(job):
    sym, path = job
    t0 = time.time()
    df, _ = load_ohlcv(path)
    df = df[df.index <= SEAL]
    a = atr5m(df, 14, mode="daily")
    rule = lambda t: L.apply_rule(t, a)
    ev = W.weekend_events(df)
    mi = ev["mi"].to_numpy(np.int64)
    d = ev["d"].to_numpy(np.int64)
    ev["align4"] = W.htf_trend(df, weekday_only=True)[mi].astype(int) * d
    ev["align24"] = W.htf_trend(df, weekday_only=False)[mi].astype(int) * d
    M, pe = W.ltf_bars(df)
    out = {"B0": rule(W.base_trades(df, ev)), "B0L": rule(W.base_trades(df, ev, mtm=False)),
           "C": rule(W.cisd_entry_trades(df, ev, M, pe))}
    pf, d1, d2, dc = W.postfill(df, ev, M, pe)
    out.update(D1=rule(d1), D2=rule(d2), DC=rule(dc))
    ev = _utc(ev)
    pf = _utc(pf)
    ev = ev.merge(pf, on="monday", how="left")
    ev["et_date"] = ev["monday"].dt.tz_convert(W.ET).dt.date.astype(str)
    key = ev[["monday", "align4", "align24", "gap_bps", "G", "F", "O"]]
    for k in KEYS:
        out[k] = _utc(out[k]).merge(key, on="monday", how="left")
    out["events"] = ev
    for k in out:
        out[k] = out[k].assign(symbol=sym)
    return sym, out, round(time.time() - t0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = symbol_files("cache_okx")
    jobs = sorted(files.items())
    acc: dict[str, list] = {k: [] for k in ("events", *KEYS)}
    t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for i, (sym, out, sec) in enumerate(ex.map(run, jobs), 1):
            for k, v in out.items():
                acc[k].append(v)
            print(f"[{i:>2}/{len(jobs)}] {sym:<6} {sec:>5.1f}s  주말 {len(out['events']):>3}  "
                  f"B0 {len(out['B0']):>3}  C {len(out['C']):>3}  D1 {len(out['D1']):>3}  "
                  f"DC {len(out['DC']):>3}", flush=True)
    for k, v in acc.items():
        pd.concat(v, ignore_index=True).to_pickle(OUT / f"{k}.pkl")
    print(f"완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
