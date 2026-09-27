"""47심볼 주봉(UTC) · 일봉(뉴욕 · UTC) — 봉인 전. `claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`.

결과: runs/weekly/{weeks,days_ny,days_utc}.pkl
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research import weekly as K

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
OUT = Path("runs/weekly")


def run(job):
    sym, path = job
    t0 = time.time()
    df, _ = load_ohlcv(path)
    df = df[df.index < SEAL]                      # 봉인 시각의 봉은 빼서 마지막 주를 온전히 끝낸다
    W = K.weekly_utc(df).assign(symbol=sym).reset_index()
    Dn = K.daily(df, K.ET).assign(symbol=sym).reset_index()
    Du = K.daily(df, "UTC").assign(symbol=sym).reset_index()
    return sym, W, Dn, Du, round(time.time() - t0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sorted(symbol_files("cache_okx").items())
    acc = {"weeks": [], "days_ny": [], "days_utc": []}
    with ProcessPoolExecutor(2) as ex:
        for i, (sym, W, Dn, Du, sec) in enumerate(ex.map(run, jobs), 1):
            acc["weeks"].append(W)
            acc["days_ny"].append(Dn)
            acc["days_utc"].append(Du)
            print(f"[{i:>2}/{len(jobs)}] {sym:<6} {sec:>5.1f}s 주 {len(W)} 일 {len(Dn)}", flush=True)
    for k, v in acc.items():
        pd.concat(v, ignore_index=True).to_pickle(OUT / f"{k}.pkl")
    print("완료")


if __name__ == "__main__":
    main()
