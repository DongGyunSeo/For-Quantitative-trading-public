"""사이징 · 레버리지용 5분봉 패널 (47심볼 · 봉인 전).

분석 주: 월 00:00 UTC 2021-05-03 ~ 2026-06-22 시작 269주. 주 k 의 장중 경로 = 라벨 [T0, T0+7일) 5m 봉 2016개
(봉 τ 의 종가 = 시각 τ+5분 가격). 주초 가격 P0 = 라벨 T0−5분 봉 종가.
결과: runs/tsmom_size/{close,high,low}.npy (float32, 봉 × 47) · meta.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files  # noqa: E402

OUT = Path("runs/tsmom_size")
T_FIRST = pd.Timestamp("2021-05-03 00:00", tz="UTC")
NW = 269
T_END = T_FIRST + pd.Timedelta(days=7 * NW)                 # 2026-06-29 00:00 (배타)
IDX = pd.date_range(T_FIRST - pd.Timedelta(minutes=5), T_END - pd.Timedelta(minutes=5), freq="5min", tz="UTC")


def run(job):
    sym, path = job
    df, _ = load_ohlcv(path)
    df = df.reindex(IDX.as_unit(df.index.unit))
    assert df[["close", "high", "low"]].notna().all().all(), f"{sym} 빈 봉"
    return sym, df["close"].to_numpy(np.float32), df["high"].to_numpy(np.float32), df["low"].to_numpy(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sorted(symbol_files("cache_okx").items())
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run, jobs))
    syms = [r[0] for r in res]
    for k, name in ((1, "close"), (2, "high"), (3, "low")):
        np.save(OUT / f"{name}.npy", np.stack([r[k] for r in res], axis=1))
    meta = dict(symbols=syms, first_label=str(IDX[0]), n_bars=len(IDX), weeks=NW, t_first=str(T_FIRST))
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(syms), "심볼", len(IDX), "봉", IDX[0], "→", IDX[-1])


if __name__ == "__main__":
    main()
