"""스윕+CHoCH 보고 — 주 지표 판정 + 대조군 B 비교."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.yardstick import tstat

CH_TFS = ("5m", "15m", "1h")


def cell(d, mn, seg="전체", **sel):
    g = d[d.seg == seg]
    for k, v in sel.items():
        g = g[g[k] == v]
    if g.empty:
        return None
    w = g.n.to_numpy()
    m = mn
    for k, v in sel.items():
        m = m[m[k] == v]
    _, t, nm = tstat(m.month_net) if len(m) > 3 else (np.nan, np.nan, 0)
    return dict(n=int(w.sum()), fill=np.average(g.fill, weights=w),
                risk=np.average(g.risk_bps, weights=w),
                gross=np.average(g.gross, weights=w),
                ctrl=np.average(g.ctrl, weights=w) if g.ctrl.notna().any() else np.nan,
                fee=np.average(g.fee, weights=w), net=np.average(g.net, weights=w),
                win=np.average(g.win, weights=w), t=t, n_month=nm,
                pos=int((g.net > 0).sum()), nsym=len(g))


def line(nm, c):
    if not c:
        return f"  {nm:<26}  표본 부족"
    return (f"  {nm:<26}{c['n']:>8,}{c['fill']:>7.0%}{c['risk']:>8.0f}{c['win']:>7.1%}"
            f"{c['gross']:>+9.4f}{c['ctrl']:>+8.4f}{c['fee']:>8.4f}{c['net']:>+9.4f}"
            f"{c['t']:>7.2f}{c['pos']}/{c['nsym']:<5}")


HDR = (f"  {'':<26}{'거래수':>8}{'체결':>7}{'1R':>8}{'승률':>7}{'gross':>9}"
       f"{'대조군':>8}{'수수료':>8}{'net':>9}{'t(월)':>7}{'심볼+':>7}")


def main():
    d = pd.read_csv("results_fvg_sweep_choch.csv")
    mn = d[d.seg == "월"].copy()
    d = d[d.seg != "월"]

    print("=" * 108)
    print("주 지표 — 4h FVG 관통 후 · CHoCH 15m 진입 · 스윕 최저 손절 · 버퍼 0 (사전등록)")
    print("=" * 108)
    print(HDR)
    for seg in ("전체", "홀드아웃 전", "홀드아웃 12개월"):
        print(line(seg, cell(d, mn, seg=seg, tag="스윕+CHoCH", fvg_tf="4h",
                             ch_tf="15m", buf=0.0)))
    print("\n  대조군 B (FVG 무관 순수 CHoCH 15m, 같은 잣대)")
    for seg in ("전체", "홀드아웃 전", "홀드아웃 12개월"):
        print(line(seg, cell(d, mn, seg=seg, tag="순수CHoCH", ch_tf="15m")))

    print("\n" + "=" * 108)
    print("전체 격자 — 전체 구간")
    print("=" * 108)
    print(HDR)
    for ctf in CH_TFS:
        print(line(f"순수CHoCH · {ctf}", cell(d, mn, tag="순수CHoCH", ch_tf=ctf)))
    print("  " + "-" * 104)
    for ftf in ("1h", "4h"):
        for ctf in CH_TFS:
            for buf in (0.0, 0.25):
                print(line(f"FVG {ftf} → CHoCH {ctf} · 버퍼{buf}",
                           cell(d, mn, tag="스윕+CHoCH", fvg_tf=ftf, ch_tf=ctf, buf=buf)))
        print("  " + "-" * 104)

    print("\n" + "=" * 108)
    print("홀드아웃 12개월 — 같은 격자")
    print("=" * 108)
    print(HDR)
    for ctf in CH_TFS:
        print(line(f"순수CHoCH · {ctf}",
                   cell(d, mn, seg="홀드아웃 12개월", tag="순수CHoCH", ch_tf=ctf)))
    print("  " + "-" * 104)
    for ftf in ("1h", "4h"):
        for ctf in CH_TFS:
            print(line(f"FVG {ftf} → CHoCH {ctf} · 버퍼0",
                       cell(d, mn, seg="홀드아웃 12개월", tag="스윕+CHoCH",
                            fvg_tf=ftf, ch_tf=ctf, buf=0.0)))


if __name__ == "__main__":
    main()
