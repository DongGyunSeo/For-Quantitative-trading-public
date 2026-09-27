"""시기별 치환 대조군 — 거래마다 "같은 시기 안의 무작위 시점" 기대 R.

기존 대조군(`control_expectancy`)은 진입 시점을 **전 기간**에서 섞는다. 시기를 나눠
비교하려면 그걸로는 안 된다 — 2025 같은 하락장에서는 무작위 롱 자체가 지므로,
"FVG 롱이 졌다" 가 FVG 탓인지 시장 탓인지 가를 수 없다.

그래서 거래마다 **자기 시기(2024 전 / 2024 이후) 안에서만** 진입 시점을 K 번 새로 뽑고,
방향과 1R(bps)은 그대로 둔 채 같은 2R/1R 레이스를 돌린다. 거래별 값이라
어떤 부분집합(게이트·방향·반기·유동성)의 대조군이든 평균만 내면 된다.

출력: runs/u47_trades_era.csv = u47_trades.csv + ``era`` · ``ctrl_era`` 열
"""
from __future__ import annotations

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research.barrier import first_passage_r

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
SPLIT = pd.Timestamp("2024-01-01 00:00", tz="UTC")
K = 20
H = 2016


def run(job):
    sym, path, rows = job
    df, _ = load_ohlcv(path)
    df = df[df.index <= SEAL]
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    n = len(df)
    cut = int(df.index.searchsorted(SPLIT))
    rng = np.random.default_rng(zlib.crc32(("era|" + sym).encode()))
    bar = rows["bar"].to_numpy(np.int64)
    post = bar >= cut
    m = len(bar)
    lo = np.where(post, cut, 500)
    hi = np.where(post, n - H - 2, cut)          # 시기 안에서만 뽑는다
    b = (lo[:, None] + (rng.random((m, K)) * (hi - lo)[:, None]).astype(np.int64)).ravel()
    sd = np.repeat(rows["side"].to_numpy(np.int64), K)
    rb = np.repeat(rows["risk_bps"].to_numpy(np.float64), K)
    e = o[b]
    r = first_passage_r(h, l, b, e, e * rb / 1e4, sd, tp_r=2.0, sl_r=1.0,
                        horizon=H, skip_entry_bar=False)
    return sym, rows.index.to_numpy(), post, r.reshape(m, K).mean(1)


def main() -> None:
    t = pd.read_csv("runs/u47_trades.csv")
    files = symbol_files("cache_okx")
    jobs = [(s, str(files[s]), t[t.symbol == s][["bar", "side", "risk_bps"]]) for s in sorted(t.symbol.unique())]
    t["era"] = ""
    t["ctrl_era"] = np.nan
    with ProcessPoolExecutor(max_workers=2) as ex:
        for sym, idx, post, c in ex.map(run, jobs):
            t.loc[idx, "era"] = np.where(post, "2024+", "2021-23")
            t.loc[idx, "ctrl_era"] = c
            print(f"  {sym:<6} {len(idx):>5}건  대조 평균 {c.mean():+.4f}", flush=True)
    t.to_csv("runs/u47_trades_era.csv", index=False)
    print("-> runs/u47_trades_era.csv")


if __name__ == "__main__":
    main()
