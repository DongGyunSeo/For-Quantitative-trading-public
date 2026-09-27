"""시간 종가 패널 (47심볼 · 봉인 전) — 체결 시각 전수 스윕용.

라벨 T 의 값 = T 직전에 끝난 5m 봉의 종가 = 시각 T 의 가격. 결과: runs/tsmom_anchor/hourly_close.pkl
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files  # noqa: E402

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")


def run(job):
    sym, path = job
    df, _ = load_ohlcv(path)
    c = df.loc[df.index < SEAL, "close"]
    return sym, c.resample("1h", label="right", closed="left").last()


def main() -> None:
    jobs = sorted(symbol_files("cache_okx").items())
    with ProcessPoolExecutor(2) as ex:
        H = pd.DataFrame(dict(ex.map(run, jobs)))
    H = H.asfreq("1h")
    H.to_pickle("runs/tsmom_anchor/hourly_close.pkl")
    print(H.shape, H.index[0], H.index[-1], "빈 칸", int(H.isna().sum().sum()))


if __name__ == "__main__":
    main()
