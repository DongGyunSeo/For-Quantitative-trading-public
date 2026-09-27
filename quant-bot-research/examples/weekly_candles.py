"""A. 주봉 캔들(몸통·꼬리) → 다음 주 — 사전등록 그대로 (`claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`).

입력 runs/weekly/weeks.pkl → runs/weekly_candles.json · runs/weekly_candles_rows.pkl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import weekly as K  # noqa: E402
from qbot.research.yardstick import cluster_t  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
FEE, FUND = 6e-4, 0.137 * 3 * 7 / 1e4


def cluster_diff(y, d, g) -> tuple[float, float, int, int]:
    """y = a + b·d 의 b 와 주 군집 강건 t."""
    y, d, g = np.asarray(y, float), np.asarray(d, float), np.asarray(g)
    ok = np.isfinite(y) & np.isfinite(d)
    y, d, g = y[ok], d[ok], g[ok]
    X = np.column_stack([np.ones_like(d), d])
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    e = y - X @ beta
    S = pd.DataFrame({"g": g, "a": X[:, 0] * e, "b": X[:, 1] * e}).groupby("g")[["a", "b"]].sum().to_numpy()
    G = len(S)
    V = inv @ (S.T @ S) @ inv * G / (G - 1)
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1])), int(len(y)), G


def build_rows() -> pd.DataFrame:
    W = pd.read_pickle("runs/weekly/weeks.pkl")
    W = W[W.n >= 0.95 * 2016].sort_values(["symbol", "week"]).reset_index(drop=True)
    out = []
    for sym, g in W.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        sh = K.candle_shape(g.open, g.high, g.low, g.close)
        nxt_ok = (g.week.shift(-1) - g.week) == pd.Timedelta(days=7)
        r1 = (g.close.shift(-1) / g.close - 1).where(nxt_ok)
        f = pd.DataFrame(dict(symbol=sym, week=g.week, entry_week=g.week.shift(-1).where(nxt_ok),
                              dir=sh.dir, body=sh.body, upper=sh.upper, lower=sh.lower,
                              cell=K.shape_cell(sh), r1=r1,
                              up1=g.up_first_h.shift(-1).where(nxt_ok), dn1=g.dn_first_h.shift(-1).where(nxt_ok)))
        out.append(f)
    R = pd.concat(out, ignore_index=True)
    R = R[R.r1.notna() & (R.dir != 0) & R.body.notna()].reset_index(drop=True)
    R["against"] = np.where(R.dir > 0, R.upper, R.lower)
    R["adj"] = R.dir * R.r1                                     # 전주 색을 따랐을 때 수익
    R["half"] = np.where(R.entry_week < SPLIT, "발견", "확인")
    R["cl"] = R.entry_week.dt.strftime("%Y-%m-%d")              # 주 군집
    return R


def positions(R: pd.DataFrame) -> dict[str, np.ndarray]:
    d = R.dir.to_numpy()
    return dict(기준=d, F1=np.where(R.body >= 1 / 3, d, 0.0), F2=np.where(R.against < 1 / 3, d, 0.0),
                F3=np.where(R.against >= 1 / 3, -d, d))


def book(R: pd.DataFrame, pos: np.ndarray) -> pd.DataFrame:
    """주별 포트폴리오 (주마다 그 주에 데이터 있는 코인 1/N)."""
    T = R[["symbol", "entry_week", "r1"]].assign(pos=pos)
    T = T.sort_values(["symbol", "entry_week"])
    prev = T.groupby("symbol").pos.shift(1).fillna(0.0)
    gap = T.groupby("symbol").entry_week.diff() != pd.Timedelta(days=7)
    prev[gap] = 0.0
    T["net"] = T.pos * T.r1 - (T.pos - prev).abs() * FEE - T.pos * FUND
    wk = T.groupby("entry_week").agg(ret=("net", "mean"), gross_exp=("pos", lambda x: np.abs(x).mean()))
    return wk


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(weeks=int(len(r)), mean=float(r.mean()), sharpe=float(r.mean() / r.std() * np.sqrt(52)),
                t=float(r.mean() / r.std() * np.sqrt(len(r))), mdd=float((eq / eq.cummax() - 1).min()),
                cagr=float(eq.iloc[-1] ** (52 / len(r)) - 1))


def main() -> None:
    R = build_rows()
    R.to_pickle("runs/weekly_candles_rows.pkl")
    out: dict = dict(rows=int(len(R)), weeks=int(R.entry_week.nunique()),
                     first=str(R.entry_week.min().date()), last=str(R.entry_week.max().date()))

    # 1) 14칸 지도 — 발견 구간에서 선정
    cells = {}
    for c, g in R.groupby("cell"):
        row = {}
        for h in ("발견", "확인"):
            x = g[g.half == h]
            m, t, n, G = cluster_t(x.adj.to_numpy(), x.cl.to_numpy())
            row[h] = dict(mean=m, t=t, n=n, weeks=G, same=float((x.adj > 0).mean()) if n else np.nan)
        m, t, n, G = cluster_t(g.adj.to_numpy(), g.cl.to_numpy())
        row["전체"] = dict(mean=m, t=t, n=n, weeks=G, same=float((g.adj > 0).mean()))
        cells[c] = row
    picked = [c for c, v in cells.items() if np.isfinite(v["발견"]["t"]) and abs(v["발견"]["t"]) >= 2.5]
    verdict_cells = {}
    for c in picked:
        d, k = cells[c]["발견"], cells[c]["확인"]
        verdict_cells[c] = dict(same_sign=bool(np.sign(d["mean"]) == np.sign(k["mean"])),
                                t_ok=bool(abs(k["t"]) >= 2.0), weeks_ok=bool(k["weeks"] >= 50))
        verdict_cells[c]["passed"] = all(verdict_cells[c].values())
    out["cells"], out["picked"], out["verdict_cells"] = cells, picked, verdict_cells

    # 2) 규칙 거래 — 반기별 · 전체
    P = positions(R)
    strat = {}
    for k, p in P.items():
        wk = book(R, p)
        strat[k] = dict(전체=perf(wk.ret), 발견=perf(wk.ret[wk.index < SPLIT]), 확인=perf(wk.ret[wk.index >= SPLIT]),
                        gross=float(wk.gross_exp.mean()))
    out["strategies"] = strat

    # 3) 필터 효과 검정
    fx = {}
    for nm, kept in (("F1", R.body >= 1 / 3), ("F2", R.against < 1 / 3)):
        fx[nm] = {h: dict(zip(("diff", "t", "n", "G"), cluster_diff(R.adj[R.half == h], kept[R.half == h].astype(float),
                                                                    R.cl[R.half == h]))) for h in ("발견", "확인")}
        fx[nm]["share_kept"] = float(kept.mean())
    flip = R.against >= 1 / 3
    fx["F3"] = {h: dict(zip(("mean", "t", "n", "G"), cluster_t(R.adj[flip & (R.half == h)].to_numpy(),
                                                                 R.cl[flip & (R.half == h)].to_numpy()))) for h in ("발견", "확인")}
    fx["F3"]["share_flipped"] = float(flip.mean())
    verdict_f = {}
    for nm in ("F1", "F2", "F3"):
        s_ok = all(strat[nm][h]["sharpe"] > strat["기준"][h]["sharpe"] for h in ("발견", "확인"))
        key = "diff" if nm != "F3" else "mean"
        want = 1 if nm != "F3" else -1
        k_ok = (np.sign(fx[nm]["확인"][key]) == want) and abs(fx[nm]["확인"]["t"]) >= 2.0
        d_ok = np.sign(fx[nm]["발견"][key]) == want
        verdict_f[nm] = dict(sharpe_both=bool(s_ok), confirm_t=bool(k_ok), discover_sign=bool(d_ok))
        verdict_f[nm]["passed"] = all(verdict_f[nm].values())
    out["filter_tests"], out["verdict_filters"] = fx, verdict_f

    # 4) 서술 — 이번 주 전주 고저 돌파 (방향 × 꼬리 6묶음)
    R["grp"] = np.where(R.dir > 0, "양봉", "음봉") + " · " + np.where(
        (R.upper >= 1 / 3) & (R.upper > R.lower), "윗꼬리 김",
        np.where((R.lower >= 1 / 3) & (R.lower > R.upper), "아랫꼬리 김", "짧음"))
    brk = {}
    for gname, g in R.groupby("grp"):
        up, dn = g.up1.notna(), g.dn1.notna()
        both = up & dn
        brk[gname] = dict(n=int(len(g)), up=float(up.mean()), dn=float(dn.mean()), both=float(both.mean()),
                          neither=float((~up & ~dn).mean()),
                          up_first=float(((g.up1 < g.dn1) | (up & ~dn)).mean()),
                          dn_first=float(((g.dn1 < g.up1) | (dn & ~up)).mean()),
                          adj=float(g.adj.mean()), same=float((g.adj > 0).mean()))
    out["breaks"] = brk
    Path("runs/weekly_candles.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                                encoding="utf-8")

    print(f"코인·주 {out['rows']:,} · 주 {out['weeks']} · {out['first']} ~ {out['last']}")
    print(f"{'칸':<24}{'발견 평균':>10}{'t':>7}{'주':>5}{'확인 평균':>10}{'t':>7}{'전체 n':>8}{'같은방향':>9}")
    for c, v in sorted(cells.items(), key=lambda z: z[0]):
        d, k, a = v["발견"], v["확인"], v["전체"]
        print(f"{c:<24}{d['mean'] * 100:>9.2f}%{d['t']:>7.2f}{d['weeks']:>5}{k['mean'] * 100:>9.2f}%{k['t']:>7.2f}"
              f"{a['n']:>8}{a['same'] * 100:>8.1f}%")
    print("선정(발견 |t|≥2.5):", picked, verdict_cells)
    for k, v in strat.items():
        print(f"{k:<4} 전체 샤프 {v['전체']['sharpe']:+.2f} (t {v['전체']['t']:+.2f}, MDD {v['전체']['mdd'] * 100:.0f}%) · "
              f"발견 {v['발견']['sharpe']:+.2f} · 확인 {v['확인']['sharpe']:+.2f} · 총노출 {v['gross']:.2f}")
    for nm in ("F1", "F2", "F3"):
        print(nm, {h: {a: round(b, 4) if isinstance(b, float) else b for a, b in fx[nm][h].items()} for h in ("발견", "확인")},
              {k: round(v, 3) for k, v in fx[nm].items() if k.startswith("share")}, verdict_f[nm])
    for gname, v in brk.items():
        print(f"{gname:<14} n {v['n']:>5} 고점돌파 {v['up'] * 100:5.1f}% 저점이탈 {v['dn'] * 100:5.1f}% 둘다 {v['both'] * 100:5.1f}% "
              f"고점먼저 {v['up_first'] * 100:5.1f}% 저점먼저 {v['dn_first'] * 100:5.1f}% 같은방향 {v['same'] * 100:5.1f}% "
              f"방향맞춤 {v['adj'] * 100:+.2f}%")


if __name__ == "__main__":
    main()
