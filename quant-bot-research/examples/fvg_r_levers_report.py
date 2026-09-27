"""1R 레버 보고 — 주 지표 판정 + 풍경 기술."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.yardstick import tstat

PRIMARY = dict(tf="4h", lvl=0.5, ext=0.0, gate=120.0)
GATES = (0.0, 40.0, 60.0, 80.0, 120.0, 160.0)


def agg(d, **sel):
    g = d
    for k, v in sel.items():
        g = g[g[k] == v]
    return g


def cell(d, mn, tf, lvl, ext, gate, seg="전체"):
    g = agg(d[d.seg == seg], tf=tf, lvl=lvl, ext=ext, gate=gate)
    if g.empty:
        return None
    w = g.n.to_numpy()
    mm = agg(mn, tf=tf, lvl=lvl, ext=ext, gate=gate)
    m, t, nm = tstat(mm.month_net) if len(mm) > 3 else (np.nan, np.nan, 0)
    return dict(n=int(w.sum()), risk=np.average(g.risk_bps, weights=w),
                gross=np.average(g.gross, weights=w),
                ctrl=np.average(g.ctrl, weights=w) if g.ctrl.notna().any() else np.nan,
                fee=np.average(g.fee, weights=w), net=np.average(g.net, weights=w),
                t=t, n_month=nm, pos=int((g.net > 0).sum()), nsym=len(g),
                win=np.average(g.win, weights=w), zero=np.average(g.zero, weights=w))


def main():
    d = pd.read_csv("results_fvg_levers.csv")
    mn = d[d.seg == "월"].copy()
    d = d[d.seg != "월"]

    print("=" * 100)
    print("주 지표 — 4h · mid 진입 · 손절 확장 없음 · 사전 1R 게이트 120bps (사전등록)")
    print("=" * 100)
    print(f"  {'구간':<16}{'거래수':>8}{'1R(bps)':>9}{'승률':>7}{'무승부':>7}"
          f"{'gross':>9}{'대조군':>8}{'수수료':>8}{'net':>9}{'t(월)':>8}{'심볼+':>8}")
    for seg in ("전체", "홀드아웃 전", "홀드아웃 12개월"):
        c = cell(d, mn, seg=seg, **PRIMARY)
        if not c:
            print(f"  {seg:<16}  표본 부족"); continue
        print(f"  {seg:<16}{c['n']:>8,}{c['risk']:>9.1f}{c['win']:>7.1%}{c['zero']:>7.1%}"
              f"{c['gross']:>+9.4f}{c['ctrl']:>+8.4f}{c['fee']:>8.4f}{c['net']:>+9.4f}"
              f"{c['t']:>8.2f}{c['pos']}/{c['nsym']:<6}")

    print("\n" + "=" * 100)
    print("레버 풍경 — 순R (전체 구간). **기술이지 판정이 아니다**")
    print("=" * 100)
    for tf in ("1h", "4h"):
        print(f"\n[{tf}]  행 = (진입깊이, 손절확장) · 열 = 사전 1R 게이트(bps)")
        print(f"  {'lvl':>5}{'ext':>6}" + "".join(f"{int(g):>12}" for g in GATES))
        for lvl in sorted(d.lvl.unique()):
            for ext in sorted(d.ext.unique()):
                line = f"  {lvl:>5}{ext:>6}"
                any_ = False
                for g in GATES:
                    c = cell(d, mn, tf=tf, lvl=lvl, ext=ext, gate=g)
                    if not c:
                        line += f"{'—':>12}"; continue
                    any_ = True
                    line += f"{c['net']:>+9.4f}{'*' if c['net'] > 0 and c['t'] > 2 else ' '}  "
                if any_:
                    print(line)

    print("\n" + "=" * 100)
    print("1R 을 키우면 엣지도 같이 줄어드는가 — 1H레그 기각의 핵심 질문")
    print("=" * 100)
    recs = []
    for tf in ("1h", "4h"):
        for lvl in sorted(d.lvl.unique()):
            for ext in sorted(d.ext.unique()):
                for g in GATES:
                    c = cell(d, mn, tf=tf, lvl=lvl, ext=ext, gate=g)
                    if c and c["n"] >= 500:
                        recs.append(dict(tf=tf, **c))
    R = pd.DataFrame(recs)
    print(f"  칸 {len(R)}개 (거래수 >= 500)")
    print(f"  1R 중앙 vs gross   상관 {R.risk.corr(R.gross):+.3f}")
    print(f"  1R 중앙 vs 수수료  상관 {R.risk.corr(R.fee):+.3f}")
    print(f"  1R 중앙 vs net     상관 {R.risk.corr(R.net):+.3f}")
    print(f"\n  {'1R 구간':<14}{'칸':>4}{'1R중앙':>8}{'gross':>9}{'대조군':>8}"
          f"{'수수료':>8}{'net':>9}{'최고net':>9}")
    bins = [(0, 70), (70, 100), (100, 140), (140, 200), (200, 1e9)]
    for lo, hi in bins:
        s = R[(R.risk >= lo) & (R.risk < hi)]
        if s.empty:
            continue
        nm = f"{lo}~{hi if hi < 1e9 else ''}bps"
        print(f"  {nm:<14}{len(s):>4}{s.risk.mean():>8.1f}{s.gross.mean():>+9.4f}"
              f"{s.ctrl.mean():>+8.4f}{s.fee.mean():>8.4f}{s.net.mean():>+9.4f}"
              f"{s.net.max():>+9.4f}")

    best = R.sort_values("net", ascending=False).head(8)
    print(f"\n  순R 상위 8칸 (사후 선택 — 참고만)")
    print(f"  {'TF':>4}{'lvl':>6}{'ext':>6}{'게이트':>7}{'n':>8}{'1R':>7}"
          f"{'gross':>9}{'수수료':>8}{'net':>9}{'t':>7}{'심볼+':>8}")
    for _, r in best.iterrows():
        print(f"  {r.tf:>4}{'':>6}{'':>6}{'':>7}{int(r.n):>8,}{r.risk:>7.0f}"
              f"{r.gross:>+9.4f}{r.fee:>8.4f}{r.net:>+9.4f}{r.t:>7.2f}"
              f"{int(r.pos)}/{int(r.nsym):<6}")
    R.to_csv("results_fvg_levers_cells.csv", index=False)


if __name__ == "__main__":
    main()
