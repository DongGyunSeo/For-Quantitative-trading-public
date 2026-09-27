"""시계열 추세 + 뒤집기 보류(주봉 ATR) — 사전등록 그대로 (`claude/시계열추세-뒤집기보류-ATR-사전등록.md`).

입력 runs/legacy/daily_close.pkl · runs/weekly/weeks.pkl → runs/tsmom_hold.json · runs/tsmom_hold_weekly.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import legacy as L  # noqa: E402
from qbot.research import tsmom_rules as R  # noqa: E402
from qbot.research.yardstick import cluster_t  # noqa: E402

KS = (0.5, 1.0, 2.0)
HALF = pd.Timestamp("2023-11-01", tz="UTC")
RNG = np.random.default_rng(20260925)


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(weeks=int(len(r)), mean=float(r.mean()), sharpe=float(r.mean() / r.std() * np.sqrt(52)),
                t=float(r.mean() / r.std() * np.sqrt(len(r))), cagr=float(eq.iloc[-1] ** (52 / len(r)) - 1),
                total=float(eq.iloc[-1] - 1), mdd=float((eq / eq.cummax() - 1).min()), vol=float(r.std() * np.sqrt(52)))


def block_boot_t(d: np.ndarray, block: int = 4, reps: int = 5000) -> tuple[float, float, list[float]]:
    n = len(d)
    k = int(np.ceil(n / block))
    means = np.empty(reps)
    for b in range(reps):
        st = RNG.integers(0, n - block + 1, size=k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = d[idx].mean()
    se = means.std(ddof=1)
    return float(d.mean() / se), float(se), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def inputs(C: pd.DataFrame, lookback: int):
    wk = C[C.index.dayofweek == 6]
    sig = np.sign(C / C.shift(lookback) - 1).reindex(wk.index)
    chg = wk - wk.shift(1)
    fwd = wk.shift(-1) / wk - 1
    return sig, chg, fwd


def weekly_atr_anchor(cols) -> pd.DataFrame:
    W = pd.read_pickle("runs/weekly/weeks.pkl")
    W = W[W.n >= 0.95 * 2016]
    wide = {k: W.pivot(index="week", columns="symbol", values=k).reindex(columns=cols) for k in ("high", "low", "close")}
    atr = R.weekly_atr(wide)
    atr.index = atr.index + pd.Timedelta(days=6)            # 월 시작 주봉 → 그 주 일요일(종가) 라벨
    return atr


def run(C, atr, lookback: int, k):
    sig, chg, fwd = inputs(C, lookback)
    pos = R.hold_positions(sig, chg, atr, k)
    return pos, sig, fwd, R.book_from_positions(pos, fwd)


def main() -> None:
    C = pd.read_pickle("runs/legacy/daily_close.pkl")
    atr = weekly_atr_anchor(C.columns)
    pos0, sig, fwd, b0 = run(C, atr, 28, None)
    ref = L.tsmom_weekly(C)
    assert np.allclose(b0.ret.to_numpy(), ref.ret.to_numpy()), "기준이 legacy 와 다르다"
    base = b0.ret
    out = {"base": dict(all=perf(base), first=perf(base[base.index < HALF]), second=perf(base[base.index >= HALF]),
                        fee_pct_yr=float(b0.fee.mean() * 52 * 100), turnover=float(b0.turnover.mean()))}
    weekly = pd.DataFrame({"base": base})
    N = pos0.notna().sum(axis=1)
    flips_base = ((sig != pos0.shift(1)) & pos0.shift(1).notna() & sig.notna()).sum().sum()
    for k in KS:
        pos, _, _, b = run(C, atr, 28, k)
        r = b.ret
        weekly[f"k{k}"] = r
        d = (r - base).dropna().to_numpy()
        tb, se, ci = block_boot_t(d)
        p_all, p1, p2 = perf(r), perf(r[r.index < HALF]), perf(r[r.index >= HALF])
        v = dict(c1=bool(d.mean() > 0 and tb >= 2.4),
                 c2=bool(p1["sharpe"] > out["base"]["first"]["sharpe"] and p2["sharpe"] > out["base"]["second"]["sharpe"]),
                 c3=bool(p_all["mdd"] >= out["base"]["all"]["mdd"] - 0.05))
        v["passed"] = all(v.values())
        # 서술
        prev = pos.shift(1)
        atr_ok = atr.reindex(index=sig.index, columns=sig.columns).notna()
        flip_sig = (sig != prev) & prev.notna() & (prev != 0) & sig.notna() & (sig != 0) & atr_ok   # 규칙이 적용될 수 있는 뒤집기 신호
        held = flip_sig & (pos == prev)
        disagree = (pos != sig) & pos.notna() & sig.notna()
        diff = (pos != pos0) & pos.notna() & pos0.notna()
        gain = (pos * fwd)[diff].stack()
        wk_lab = gain.index.get_level_values(0).strftime("%Y-%m-%d")
        g_m, g_t, g_n, g_G = cluster_t(gain.to_numpy(), np.asarray(wk_lab))
        runs = []
        for c in pos.columns:
            s = pos[c].dropna()
            runs += s.groupby((s != s.shift()).cumsum()).size().tolist()
        out[f"k{k}"] = dict(all=p_all, first=p1, second=p2, d_mean_bps=float(d.mean() * 1e4), d_t_boot=tb,
                            d_ci_bps=[x * 1e4 for x in ci], verdict=v,
                            fee_pct_yr=float(b.fee.mean() * 52 * 100), turnover=float(b.turnover.mean()),
                            flip_signals=int(flip_sig.sum().sum()), held=int(held.sum().sum()),
                            held_share=float(held.sum().sum() / max(flip_sig.sum().sum(), 1)),
                            disagree_share=float(disagree.sum().sum() / pos.notna().sum().sum()),
                            diff_coinweeks=int(g_n), diff_gain_bps=float(g_m * 1e4), diff_gain_t=float(g_t),
                            hold_weeks_mean=float(np.mean(runs)))
    # lookback 민감도
    sens = {}
    for lb in (21, 35, 56):
        _, _, _, bb = run(C, atr, lb, None)
        row = {"base": perf(bb.ret)["sharpe"]}
        for k in KS:
            _, _, _, bk = run(C, atr, lb, k)
            row[f"k{k}"] = perf(bk.ret)["sharpe"]
        sens[lb] = row
    out["lookback_sens"] = sens
    # 연도별 (복리)
    yrs = {}
    for col in weekly.columns:
        yrs[col] = {int(y): float((1 + x).prod() - 1) for y, x in weekly[col].groupby(weekly.index.year)}
    out["years"] = yrs
    out["base_flip_signals"] = int(flips_base)
    Path("runs/tsmom_hold.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    weekly.to_csv("runs/tsmom_hold_weekly.csv")

    b = out["base"]
    print(f"기준  샤프 {b['all']['sharpe']:.3f} (t {b['all']['t']:.2f}) CAGR {b['all']['cagr'] * 100:.1f}% MDD {b['all']['mdd'] * 100:.1f}% "
          f"전반 {b['first']['sharpe']:.2f} 후반 {b['second']['sharpe']:.2f} 수수료 연 {b['fee_pct_yr']:.2f}% 뒤집기 신호 {out['base_flip_signals']}")
    for k in KS:
        o = out[f"k{k}"]
        print(f"k {k:<3} 샤프 {o['all']['sharpe']:.3f} (t {o['all']['t']:.2f}) CAGR {o['all']['cagr'] * 100:.1f}% MDD {o['all']['mdd'] * 100:.1f}% "
              f"전반 {o['first']['sharpe']:.2f} 후반 {o['second']['sharpe']:.2f} | 차이 {o['d_mean_bps']:+.1f}bps/주 t {o['d_t_boot']:+.2f} "
              f"CI [{o['d_ci_bps'][0]:+.1f}, {o['d_ci_bps'][1]:+.1f}] | 보류 {o['held']}/{o['flip_signals']} ({o['held_share'] * 100:.0f}%) "
              f"신호와 반대 {o['disagree_share'] * 100:.1f}% · 바뀐 코인주 {o['diff_coinweeks']} 평균 {o['diff_gain_bps']:+.1f}bps (t {o['diff_gain_t']:+.2f}) "
              f"· 보유 {o['hold_weeks_mean']:.1f}주 · 수수료 연 {o['fee_pct_yr']:.2f}% → {o['verdict']}")
    print("lookback", {lb: {a: round(v, 2) for a, v in r.items()} for lb, r in sens.items()})
    print("years", {c: {y: round(v * 100, 1) for y, v in r.items()} for c, r in yrs.items()})


if __name__ == "__main__":
    main()
