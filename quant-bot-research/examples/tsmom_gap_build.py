"""금 신호 · 월 시가 · 주말갭 지정가 — 코인·주 표 만들기 (47심볼 · 봉인 전).

사전등록: `claude/시계열추세-금신호-월시가-주말갭지정가-사전등록.md`.
결과: runs/tsmom_gap/weeks.pkl
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files  # noqa: E402
from qbot.research import tsmom_gap as G  # noqa: E402

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
OUT = Path("runs/tsmom_gap")


def run(job):
    sym, path = job
    t0 = time.time()
    df, _ = load_ohlcv(path)
    df = df[df.index < SEAL]
    T = G.week_table(df)
    T["symbol"] = sym
    return sym, T, round(time.time() - t0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sorted(symbol_files("cache_okx").items())
    acc = []
    t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for i, (sym, T, sec) in enumerate(ex.map(run, jobs), 1):
            acc.append(T)
            ok = T.F_prev4.notna() & T.O_next.notna()
            print(f"[{i:>2}/{len(jobs)}] {sym:<6} {sec:>5.1f}s  주 {len(T):>3}  쓸 수 있는 주 {int(ok.sum()):>3}  "
                  f"mid 닿음 {T.touch_mid[ok].mean() * 100:5.1f}%  edge {T.touch_edge[ok].mean() * 100:5.1f}%", flush=True)
    W = pd.concat(acc, ignore_index=True)
    W.to_pickle(OUT / "weeks.pkl")
    print(f"완료 {time.time() - t0:.0f}s  행 {len(W)}")


if __name__ == "__main__":
    main()
