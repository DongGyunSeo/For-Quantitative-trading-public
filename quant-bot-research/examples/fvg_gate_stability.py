"""1R 게이트 안정성 — 최적 컷오프를 찾지 않는다.

`FVG-1R확대-3레버-기각.md` 에서 관측된 것: 181칸에서 1R↔gross 상관이 −0.098
(사실상 0)이고 수수료만 떨어져서, **1R ≈ 150bps 부근이 손익분기**로 나왔다.

이 스크립트의 목적은 **그 관측이 흔들리는지 보는 것**이지 더 좋은 컷을 찾는 게 아니다.
그래서 게이트를 **100 / 125 / 150 / 175 / 200 bps 다섯 개로 미리 못 박고**,
각 구간을 심볼·연도·월·홀드아웃으로 쪼개 부호가 유지되는지만 본다.

**최적값 선택 금지.** 표에서 제일 좋은 칸을 골라 후보로 삼지 않는다.
판정은 오직 하나 — *연도별로 같은 부호인가.*

셋업 (전부 고정, 탐색 없음)
---------------------------
4h FVG · mid 진입(entry_level 0.5) · 손절 확장 0 · TP 2R / SL 1R ·
측정해상도 규칙 1R/일간중앙ATR ≥ 2 · 지평 2016봉 ·
비용 진입 maker 2 + 승 maker 2 / 패·모호·미해결 taker 6 bps.
1h FVG 는 같은 레버의 **두 번째 축**으로 같이 낸다(고르기 위한 것이 아니라 비교용).

봉인 (2026-09-25 부터)
----------------------
``--until`` 이후 데이터는 **읽기 전에 잘라낸다.** 기본값 2026-07-05 00:00 UTC —
기존 10심볼 캐시의 끝과 같다. 그 뒤 81일(→2026-09-24)은 가설·규칙을 확정한 다음
**딱 한 번** 여는 전방 검증 구간이다. 이 스크립트로 그 구간을 보지 않는다.

기존 결과 재현: ``--cache cache_all --until none``.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research.barrier import control_expectancy
from qbot.research.fvg import FVGConfig, run_fvg

TFS = ("4h", "1h")
GATES = (100.0, 125.0, 150.0, 175.0, 200.0)
ENTRY_BPS, TP_BPS, SL_BPS = 2.0, 2.0, 6.0
NDRAW = 12
SEAL = "2026-07-05 00:00"


def seed_of(sym: str) -> int:
    """심볼별 고정 시드. ``hash(str)`` 는 프로세스마다 바뀌므로 쓰지 않는다."""
    return zlib.crc32(sym.encode())


def run_symbol(job: tuple[str, str, str | None]) -> tuple[pd.DataFrame | None, list[dict], dict]:
    sym, path, until = job
    df, _ = load_ohlcv(path)
    if until:
        df = df[df.index <= pd.Timestamp(until, tz="UTC")]
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    rng = np.random.default_rng(seed_of(sym))
    hold_cut = int(df.index.searchsorted(df.index[-1] - pd.Timedelta(days=365)))
    year = df.index.year.to_numpy()
    month = df.index.year.to_numpy() * 12 + df.index.month.to_numpy()
    info = dict(symbol=sym, start=str(df.index[0]), end=str(df.index[-1]), rows=len(df),
                med_quote_vol=float(np.median(v * c)))

    trades, ctrl = [], []
    for tf in TFS:
        res = run_fvg(df, FVGConfig(htf=tf, entry_level=0.5, stop_ext=0.0,
                                    min_r_over_atr=2.0, atr_mode="daily",
                                    tp_r=2.0, sl_r=1.0, horizon=2016))
        if len(res) == 0:
            continue
        fee = (ENTRY_BPS + np.where(res.r > 0, TP_BPS, SL_BPS)) / res.risk_bps
        trades.append(pd.DataFrame(dict(
            symbol=sym, tf=tf, bar=res.entry_bar, time=df.index[res.entry_bar],
            side=res.side, year=year[res.entry_bar], month=month[res.entry_bar],
            holdout=res.entry_bar >= hold_cut,
            risk_bps=res.risk_bps, gross=res.r, fee=fee, net=res.r - fee)))
        for g in GATES:
            k = res.risk_bps >= g
            if k.sum() < 30:
                continue
            m, draws = control_expectancy(
                h, l, o, res.entry_bar[k], res.risk_bps[k], res.side[k],
                n_draw=NDRAW, rng=rng, tp_r=2.0, sl_r=1.0,
                horizon=2016, skip_entry_bar=False)
            ctrl.append(dict(symbol=sym, tf=tf, gate=g, n=int(k.sum()),
                             ctrl=m, ctrl_sd=float(draws.std())))
    t = pd.concat(trades, ignore_index=True) if trades else None
    return t, ctrl, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_okx")
    ap.add_argument("--until", default=SEAL, help="봉인 시각(포함). 'none' 이면 자르지 않음")
    ap.add_argument("--only", default="", help="쉼표 구분 심볼만")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results_fvg_gate_trades.csv")
    ap.add_argument("--ctrl", default="results_fvg_gate_ctrl.csv")
    ap.add_argument("--info", default="results_fvg_gate_symbols.csv")
    a = ap.parse_args()

    until = None if a.until.lower() == "none" else a.until
    files = symbol_files(a.cache)
    if a.only:
        want = {s.strip().upper() for s in a.only.split(",")}
        files = {k: v for k, v in files.items() if k in want}
    jobs = [(s, str(p), until) for s, p in files.items()]

    trades, ctrl, info = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for t, c, i in ex.map(run_symbol, jobs):
            if t is not None:
                trades.append(t)
            ctrl += c
            info.append(i)
            print(f"  {i['symbol']:<6} {i['start'][:10]}~{i['end'][:16]}  "
                  f"거래 {0 if t is None else len(t):>5}", flush=True)

    pd.concat(trades, ignore_index=True).to_csv(a.out, index=False)
    pd.DataFrame(ctrl).to_csv(a.ctrl, index=False)
    pd.DataFrame(info).to_csv(a.info, index=False)
    print(f"\n{len(info)}심볼 · 거래 {sum(len(t) for t in trades):,}건 -> {a.out}"
          f"  (봉인 {until or '없음'})")


if __name__ == "__main__":
    main()
