"""게이트 안정성 보고 — 부호가 유지되는가만 본다."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.yardstick import tstat

GATES = (100.0, 125.0, 150.0, 175.0, 200.0)
SYMS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "AVAX", "TRX"]


def block(t, c, tf):
    g = t[t.tf == tf]
    cc = c[c.tf == tf]
    print("\n" + "=" * 104)
    print(f"[{tf} FVG · mid 진입 · 손절확장 0]  게이트별 — 최적 컷 선택 금지, 부호 안정성만")
    print("=" * 104)
    print(f"  {'게이트':>7}{'거래수':>8}{'1R중앙':>8}{'승률':>7}{'gross':>9}{'치환대조':>9}"
          f"{'비용':>8}{'net':>9}{'net(월평균)':>12}{'t(월)':>7}{'월수':>6}{'양수월':>7}")
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        if len(s) < 30:
            continue
        mm = s.groupby("month").net.mean()
        _, tv, nm = tstat(mm)
        cs = cc[cc.gate == gt]
        cw = cs.n.to_numpy() if len(cs) else np.array([1])
        ct = np.average(cs.ctrl, weights=cw) if len(cs) else np.nan
        print(f"  {int(gt):>7}{len(s):>8,}{s.risk_bps.median():>8.0f}"
              f"{(s.gross > 0).mean():>7.1%}{s.gross.mean():>+9.4f}{ct:>+9.4f}"
              f"{s.fee.mean():>8.4f}{s.net.mean():>+9.4f}{mm.mean():>+12.4f}"
              f"{tv:>7.2f}{nm:>6}{(mm > 0).mean():>7.1%}")

    print(f"\n  연도별 net R  ★ 판정 축 — 부호가 유지되는가")
    yrs = sorted(g.year.unique())
    print(f"  {'게이트':>7}" + "".join(f"{y:>16}" for y in yrs))
    print(f"  {'':>7}" + "".join(f"{'n':>6}{'gross':>10}" for _ in yrs))
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        if len(s) < 30:
            continue
        line = f"  {int(gt):>7}"
        for y in yrs:
            ys = s[s.year == y]
            line += (f"{len(ys):>6}{ys.gross.mean():>+10.4f}" if len(ys) >= 15
                     else f"{len(ys):>6}{'—':>10}")
        print(line)
    print(f"  {'':>7}" + "".join(f"{'':>6}{'net':>10}" for _ in yrs))
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        if len(s) < 30:
            continue
        line = f"  {int(gt):>7}"
        for y in yrs:
            ys = s[s.year == y]
            line += (f"{'':>6}{ys.net.mean():>+10.4f}" if len(ys) >= 15
                     else f"{'':>6}{'—':>10}")
        print(line)

    print(f"\n  심볼별 net R")
    have = [x for x in SYMS if x in set(g.symbol)]
    print(f"  {'게이트':>7}" + "".join(f"{x:>9}" for x in have) + f"{'양수':>7}")
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        if len(s) < 30:
            continue
        line = f"  {int(gt):>7}"
        pos = tot = 0
        for x in have:
            xs = s[s.symbol == x]
            if len(xs) >= 10:
                line += f"{xs.net.mean():>+9.4f}"
                pos += xs.net.mean() > 0
                tot += 1
            else:
                line += f"{'—':>9}"
        print(line + f"{pos}/{tot:<5}")

    print(f"\n  홀드아웃")
    print(f"  {'게이트':>7}{'구간':<16}{'거래수':>8}{'gross':>9}{'비용':>8}{'net':>9}"
          f"{'t(월)':>7}{'양수월':>7}{'심볼+':>7}")
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        if len(s) < 30:
            continue
        for seg, m in (("홀드아웃 전", ~s.holdout), ("홀드아웃 12개월", s.holdout)):
            ss = s[m]
            if len(ss) < 20:
                continue
            mm = ss.groupby("month").net.mean()
            _, tv, _ = tstat(mm)
            per = ss.groupby("symbol").net.mean()
            print(f"  {int(gt):>7}{seg:<16}{len(ss):>8,}{ss.gross.mean():>+9.4f}"
                  f"{ss.fee.mean():>8.4f}{ss.net.mean():>+9.4f}{tv:>7.2f}"
                  f"{(mm > 0).mean():>7.1%}{(per > 0).sum()}/{len(per):<5}")


def main():
    t = pd.read_csv("results_fvg_gate_trades.csv")
    c = pd.read_csv("results_fvg_gate_ctrl.csv")
    for tf in ("4h", "1h"):
        block(t, c, tf)


if __name__ == "__main__":
    main()
