"""FVG 기준선 — 치환 대조군 대비, TF 6종, 측정해상도 규칙 on/off.

`규칙적용-전수감사-및-FVG기준선.md` 의 재현 스크립트. 컨테이너 유실로
코드가 사라져 2026-09-18 에 문서만 보고 재구축했다 — 숫자가 문서와 맞는지가
곧 재구축의 검수다.
"""
from __future__ import annotations
import zlib
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.data.loader import load_ohlcv
from qbot.research.barrier import control_expectancy
from qbot.research.fvg import FVGConfig, run_fvg

TFS = ("15m", "30m", "1h", "2h", "4h", "1d")
NDRAW = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_all")
    ap.add_argument("--out", default="results_fvg_baseline.csv")
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.cache).glob("*_5m_futures.csv")):
        sym = f.stem.split("_")[0]
        df, _ = load_ohlcv(f)
        h = df["high"].to_numpy(np.float64); l = df["low"].to_numpy(np.float64)
        o = df["open"].to_numpy(np.float64)
        for tf in TFS:
            for rule, mode in ((0.0, "daily"), (2.0, "daily"), (2.0, "inst")):
                res = run_fvg(df, FVGConfig(htf=tf, min_r_over_atr=rule, atr_mode=mode))
                if len(res) < 20:
                    continue
                rng = np.random.default_rng(zlib.crc32(repr((sym, tf, rule, mode)).encode()))
                ctrl, draws = control_expectancy(
                    h, l, o, res.entry_bar, res.risk_bps, res.side,
                    n_draw=NDRAW, rng=rng, tp_r=2.0, sl_r=1.0,
                    horizon=2016, skip_entry_bar=False)
                rows.append(dict(
                    symbol=sym, tf=tf, rule=rule, mode=mode, n=len(res),
                    risk_bps=float(np.median(res.risk_bps)),
                    real=float(res.r.mean()), ctrl=ctrl,
                    diff=float(res.r.mean()) - ctrl,
                    ctrl_sd=float(draws.std()),
                    unresolved=float((res.r == 0).mean()),
                    win=float((res.r > 0).sum() / max((res.r != 0).sum(), 1))))
        print(f"  {sym} 완료", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)

    print("\n" + "=" * 92)
    for rule, mode, title in ((0.0, "daily", "규칙 없음"),
                              (2.0, "daily", "규칙 적용 · 분모=일간중앙 ATR (표준)"),
                              (2.0, "inst", "규칙 적용 · 분모=터치봉 ATR14 (내생적)")):
        print(f"\n[{title}]")
        print(f"  {'TF':<5}{'거래수':>9}{'1R(bps)':>9}{'실측':>9}{'대조군':>9}"
              f"{'차이':>9}{'심볼+':>8}{'모호·미해결':>11}")
        for tf in TFS:
            g = d[(d.tf == tf) & (d.rule == rule) & (d["mode"] == mode)]
            if g.empty:
                continue
            w = g.n.to_numpy()
            print(f"  {tf:<5}{int(w.sum()):>9,}{np.average(g.risk_bps, weights=w):>9.1f}"
                  f"{np.average(g.real, weights=w):>+9.3f}"
                  f"{np.average(g.ctrl, weights=w):>+9.3f}"
                  f"{np.average(g['diff'], weights=w):>+9.3f}"
                  f"{int((g['diff'] > 0).sum())}/{len(g):<6}"
                  f"{np.average(g.unresolved, weights=w):>11.1%}")


if __name__ == "__main__":
    main()
