"""금 신호 · 월 시가 · 주말갭 지정가 — 판정 + 서술. 사전등록 그대로.

`claude/시계열추세-금신호-월시가-주말갭지정가-사전등록.md`
입력 runs/tsmom_gap/weeks.pkl · runs/legacy/daily_close.pkl → runs/tsmom_gap.json · runs/tsmom_gap_weekly.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import legacy as L  # noqa: E402
from qbot.research import tsmom_gap as G  # noqa: E402
from qbot.research.yardstick import cluster_t  # noqa: E402

HALF = pd.Timestamp("2023-11-01")
SEAL_DAY = pd.Timestamp("2026-07-05", tz="UTC")
RNG = np.random.default_rng(20260925)
VARS = ("V2", "V3m", "V3e")
COMPS = (("V2", "V0"), ("V3m", "V2"), ("V3e", "V2"))


def perf(r: pd.Series, per_year: float = 52) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(weeks=int(len(r)), mean=float(r.mean()), sharpe=float(r.mean() / r.std() * np.sqrt(per_year)),
                t=float(r.mean() / r.std() * np.sqrt(len(r))), cagr=float(eq.iloc[-1] ** (per_year / len(r)) - 1),
                total=float(eq.iloc[-1] - 1), mdd=float((eq / eq.cummax() - 1).min()),
                vol=float(r.std() * np.sqrt(per_year)))


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


def halves(r: pd.Series) -> dict:
    return dict(all=perf(r), first=perf(r[r.index < HALF]), second=perf(r[r.index >= HALF]))


def compare(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna().to_numpy()
    tb, se, ci = block_boot_t(d)
    pa, pb = halves(a), halves(b)
    v = dict(c1=bool(d.mean() > 0 and tb >= 2.4),
             c2=bool(pa["first"]["sharpe"] > pb["first"]["sharpe"] and pa["second"]["sharpe"] > pb["second"]["sharpe"]),
             c3=bool(pa["all"]["mdd"] >= pb["all"]["mdd"] - 0.05))
    v["passed"] = all(v.values())
    return dict(d_mean_bps=float(d.mean() * 1e4), d_t_boot=tb, d_ci_bps=[x * 1e4 for x in ci], verdict=v,
                weeks=int(len(d)), win_share=float((d > 0).mean()))


def rebal_freq(C: pd.DataFrame, k: int, phase: int, start: pd.Timestamp, lookback: int = 28) -> pd.Series:
    """일 종가 기준 k 일마다 리밸런스(28일 부호, 시장가 6bps, 펀딩) — 기간 수익(동일가중, 기간 안 보유)."""
    d = C.sort_index()
    days = np.asarray((d.index - d.index[0]).days)
    reb = d.index[((days - phase) % k == 0) & (d.index >= start)]
    P = d.loc[reb]
    sig = np.sign(d / d.shift(lookback) - 1).loc[reb]
    fwd = P.shift(-1) / P - 1
    pos = sig.where(fwd.notna())
    dpos = pos.fillna(0).diff().abs()
    dpos.iloc[0] = pos.iloc[0].abs()
    r = pos * fwd - dpos * 6e-4 - pos * (0.137 * 3 * k / 1e4)
    n = pos.notna().sum(axis=1)
    out = pd.DataFrame(dict(ret=r.mean(axis=1), gross=(pos * fwd).mean(axis=1), turn=dpos.where(pos.notna() | dpos.gt(0)).sum(axis=1) / n.clip(lower=1),
                            fee=(dpos * 6e-4).sum(axis=1) / n.clip(lower=1), n=n))
    return out[out.n > 0]


def main() -> None:
    W = pd.read_pickle("runs/tsmom_gap/weeks.pkl")
    C = pd.read_pickle("runs/legacy/daily_close.pkl")
    C = C[C.index < SEAL_DAY]                      # 07-05 행 = 분석 창 마지막 5m 봉(00:00) 하나로 만든 부분 일봉 → 온전한 날만 쓴다
    v0 = L.tsmom_weekly(C)
    v0.index = (v0.index + pd.Timedelta(days=1)).tz_localize(None).normalize()   # 일요일 종가 → 그 월요일
    books, R = {}, {}
    for v in VARS:
        books[v], R[v] = G.backtest(W, v)
    common = v0.index.intersection(books["V2"].index)
    ret = pd.DataFrame({"V0": v0.ret.reindex(common), **{v: books[v].ret.reindex(common) for v in VARS}})
    assert ret.notna().all().all(), "공통 주에 빈 값"
    out = {"weeks": [str(common[0].date()), str(common[-1].date()), int(len(common))],
           "perf": {k: halves(ret[k]) for k in ret}, "comp": {}}
    for a, b in COMPS:
        out["comp"][f"{a}_vs_{b}"] = compare(ret[a], ret[b])
    out["comp"]["V3m_vs_V0"] = compare(ret["V3m"], ret["V0"])
    out["comp"]["V3e_vs_V0"] = compare(ret["V3e"], ret["V0"])
    out["n_coins_mean"] = {"V0": float(v0.n.reindex(common).mean()), "V2": float(books["V2"].n.reindex(common).mean())}

    # ---- 비용 · 회전
    fees = {}
    for v in VARS:
        b = books[v].reindex(common)
        fees[v] = dict(fee_pct_yr=float(b.fee.mean() * 52 * 100), gross_pct_yr=float(b.gross.mean() * 52 * 100))
    out["fees"] = fees

    # ---- 진입 방식 · 체결률 · 가격 개선 · 체결까지 시간
    key = ["symbol", "wk"]
    Wk = W.set_index(key)
    mode = {}
    for v in ("V3m", "V3e"):
        r = R[v]
        r = r[r.wk.isin(common)]
        ch = r[r.act != "hold"]
        att = r[r.act.isin(["limit", "skip"])]
        lim = r[r.act == "limit"]
        tcol = "t_mid" if v == "V3m" else "t_edge"
        th = Wk.loc[pd.MultiIndex.from_frame(lim[key]), tcol].to_numpy() * 5 / 60
        mode[v] = dict(changes=int(len(ch)), share={a: float((ch.act == a).mean()) for a in ("limit", "market", "skip")},
                       attempts=int(len(att)), fill_rate=float((att.act == "limit").mean()),
                       improve_bps=float(lim.improve_bps.mean()),
                       fill_hours_median=float(np.median(th)), fill_within_24h=float((th < 24).mean()),
                       fill_within_1h=float((th < 1).mean()),
                       per_week=dict(changes=float(len(ch) / len(common)), skips=float((ch.act == "skip").sum() / len(common))))
    r2 = R["V2"]
    r2 = r2[r2.wk.isin(common)]
    mode["V2"] = dict(changes=int((r2.act != "hold").sum()), per_week=dict(changes=float((r2.act != "hold").sum() / len(common))))
    out["mode"] = mode

    # ---- 역선택 (V2 경로의 포지션 변화 · 유리한 갭) — 경로 무관 비교
    X = r2[(r2.act == "market") & (r2.tgt != 0)].merge(W[key + ["F", "O", "O_next", "touch_mid", "touch_edge"]], on=key)
    X["fav"] = ((X.tgt > 0) & (X.O > X.F)) | ((X.tgt < 0) & (X.O < X.F))
    X["nd"] = X.tgt * (X.O_next / X.O - 1)                     # 새 방향, 월 시가 → 다음 월 시가
    wlab = X.wk.dt.strftime("%Y-%m-%d").to_numpy()
    adv = dict(changes=int(len(X)), fav_share=float(X.fav.mean()))
    fav = X[X.fav]
    fl = fav.wk.dt.strftime("%Y-%m-%d").to_numpy()
    for lvl in ("mid", "edge"):
        hit = fav[f"touch_{lvl}"].to_numpy(bool)
        Lp = (fav.O + fav.F) / 2 if lvl == "mid" else fav.F
        fromL = (fav.tgt * (fav.O_next / Lp - 1)).to_numpy()
        m1 = cluster_t(fav.nd.to_numpy()[hit], fl[hit])
        m0 = cluster_t(fav.nd.to_numpy()[~hit], fl[~hit])
        mL = cluster_t(fromL[hit], fl[hit])
        adv[lvl] = dict(fill_rate=float(hit.mean()), filled_nd_bps=m1[0] * 1e4, filled_nd_t=m1[1], n_filled=m1[2],
                        unfilled_nd_bps=m0[0] * 1e4, unfilled_nd_t=m0[1], n_unfilled=m0[2],
                        filled_fromL_bps=mL[0] * 1e4, filled_fromL_t=mL[1])
    unf = X[~X.fav]
    mu = cluster_t(unf.nd.to_numpy(), unf.wk.dt.strftime("%Y-%m-%d").to_numpy())
    mf = cluster_t(fav.nd.to_numpy(), fl)
    adv["fav_nd_bps"], adv["fav_nd_t"] = mf[0] * 1e4, mf[1]
    adv["unfav_nd_bps"], adv["unfav_nd_t"] = mu[0] * 1e4, mu[1]
    ma = cluster_t(X.nd.to_numpy(), wlab)
    adv["all_nd_bps"], adv["all_nd_t"] = ma[0] * 1e4, ma[1]
    # 주말(F → O) 움직임: 새 방향 기준
    X["wkend"] = X.tgt * (X.O / X.F - 1)
    adv["weekend_nd_bps"] = float(X.wkend.mean() * 1e4)
    out["adverse"] = adv

    # ---- 주말 대기 비용 (V2 경로): 금 신호가 바뀐 코인 vs 안 바뀐 코인의 주말 움직임(신호 방향)
    Y = r2.merge(W[key + ["F", "O"]], on=key)
    Y["wkend"] = Y.tgt * (Y.O / Y.F - 1)
    ylab = Y.wk.dt.strftime("%Y-%m-%d").to_numpy()
    chg = (Y.act == "market").to_numpy()
    a1, a0 = cluster_t(Y.wkend.to_numpy()[chg], ylab[chg]), cluster_t(Y.wkend.to_numpy()[~chg], ylab[~chg])
    nY = Y.groupby("wk").symbol.transform("size")
    cost = ((Y.tgt - Y.prev) * (Y.O / Y.F - 1) / nY).groupby(Y.wk).sum()
    out["weekend"] = dict(changed_bps=a1[0] * 1e4, changed_t=a1[1], changed_n=a1[2],
                          unchanged_bps=a0[0] * 1e4, unchanged_t=a0[1], unchanged_n=a0[2],
                          flip_share=float((Y[chg].prev != 0).mean()), cost_bps_wk=float(cost.mean() * 1e4),
                          gap_abs_median_bps=float((Y[chg].O / Y[chg].F - 1).abs().median() * 1e4))

    # ---- V3 − V2 분해 (코인·주, V3 행동별 기여 bps/주)
    attr = {}
    for v in ("V3m", "V3e"):
        M = R[v][R[v].wk.isin(common)].merge(r2[key + ["ret", "act", "pos"]], on=key, suffixes=("", "_2"))
        nwk = M.groupby("wk").symbol.transform("size")
        M["c"] = (M.ret - M.ret_2) / nwk
        tot = float(M.c.sum() / len(common) * 1e4)
        rows = {}
        for a, g in M.groupby("act"):
            rows[a] = dict(n=int(len(g)), contrib_bps_wk=float(g.c.sum() / len(common) * 1e4),
                           mean_diff_bps=float((g.ret - g.ret_2).mean() * 1e4))
        sk = M[M.act == "skip"]
        rows["skip"]["missed_minus_held_bps"] = float(((sk.tgt - sk.prev) * sk.base).mean() * 1e4)
        attr[v] = dict(total_bps_wk=tot, by_act=rows,
                       pos_differs_share=float((M.pos != M.pos_2).mean()))
    out["attr"] = attr

    # ---- 관통 민감도 (1bps 넘어서야 체결)
    pen = {}
    for v in ("V3m", "V3e"):
        b, rr = G.backtest(W, v, pen=True)
        s = b.ret.reindex(common)
        c = compare(s, ret["V2"])
        att = rr[rr.act.isin(["limit", "skip"]) & rr.wk.isin(common)]
        pen[v] = dict(sharpe=halves(s)["all"]["sharpe"], d_mean_bps=c["d_mean_bps"], d_t_boot=c["d_t_boot"],
                      fill_rate=float((att.act == "limit").mean()))
    out["pen"] = pen

    # ---- 분해용 서술: V1 = 금 16:00 ET 신호 → 금 16:00 ET 체결 (주말도 새 포지션)
    F = W.pivot(index="wk", columns="symbol", values="F")
    Fp4 = W.pivot(index="wk", columns="symbol", values="F_prev4")
    sig = np.sign(F / Fp4 - 1)
    fwdF = F.shift(-1) / F - 1
    okw = (F.index.to_series().diff().shift(-1) == pd.Timedelta(days=7)).to_numpy()
    fwdF[~okw] = np.nan
    pos = sig.where(fwdF.notna())
    dpos = pos.fillna(0).diff().abs()
    dpos.iloc[0] = pos.iloc[0].abs()
    v1 = (pos * fwdF - dpos * 6e-4 - pos * G.FUND_WK).mean(axis=1)
    v1 = v1[pos.notna().sum(axis=1) > 0].reindex(common)
    v1fee = ((dpos * 6e-4).sum(axis=1) / pos.notna().sum(axis=1).clip(lower=1)).reindex(common)
    out["fees"]["V1"] = dict(fee_pct_yr=float(v1fee.mean() * 52 * 100),
                             gross_pct_yr=float((pos * fwdF).mean(axis=1).reindex(common).mean() * 52 * 100))
    out["V1"] = dict(perf=halves(v1.dropna()), vs_V0=compare(v1, ret["V0"]), V2_vs_V1=compare(ret["V2"], v1))
    ret["V1"] = v1

    # ---- 리밸런스 주기 (일 종가, 28일 부호, 시장가)
    start = pd.Timestamp("2021-05-02", tz="UTC")
    freq = {}
    for k in (1, 2, 3, 7, 14):
        ph = []
        for p in range(k):
            s = rebal_freq(C, k, p, start)
            pf = perf(s.ret, per_year=365 / k)
            ph.append(dict(phase=p, sharpe=pf["sharpe"], gross_sharpe=perf(s.gross, per_year=365 / k)["sharpe"],
                           cagr=pf["cagr"], mdd=pf["mdd"], periods=pf["weeks"],
                           turn_yr=float(s.turn.mean() * 365 / k), fee_pct_yr=float(s.fee.mean() * 365 / k * 100),
                           first=str(s.index[0].date()), dow=int(s.index[0].dayofweek)))
        sh = [x["sharpe"] for x in ph]
        freq[k] = dict(sharpe_mean=float(np.mean(sh)), sharpe_min=float(np.min(sh)), sharpe_max=float(np.max(sh)),
                       gross_sharpe_mean=float(np.mean([x["gross_sharpe"] for x in ph])),
                       cagr_mean=float(np.mean([x["cagr"] for x in ph])), mdd_mean=float(np.mean([x["mdd"] for x in ph])),
                       turn_yr=float(np.mean([x["turn_yr"] for x in ph])),
                       fee_pct_yr=float(np.mean([x["fee_pct_yr"] for x in ph])), phases=ph)
    sun = [x for x in freq[7]["phases"] if x["dow"] == 6][0]
    s_sun = rebal_freq(C, 7, sun["phase"], start).ret
    v0_full = L.tsmom_weekly(C).ret
    assert np.allclose(s_sun.to_numpy(), v0_full.reindex(s_sun.index).to_numpy()), "k=7 일요일 ≠ legacy"
    f_sun = rebal_freq(C, 7, sun["phase"], start)
    f_sun.index = (f_sun.index + pd.Timedelta(days=1)).tz_localize(None).normalize()
    out["fees"]["V0"] = dict(fee_pct_yr=float(f_sun.fee.reindex(common).mean() * 52 * 100),
                             gross_pct_yr=float(f_sun.gross.reindex(common).mean() * 52 * 100))
    out["freq"] = freq
    # 예전 V0 는 07-05 부분 일봉(분석 창 끝 00:00 봉)까지 써서 마지막 주가 06-29 ~ 07-05 00:05 의 6일 부분 주였다
    C_old = pd.read_pickle("runs/legacy/daily_close.pkl")
    old = L.tsmom_weekly(C_old).ret
    out["v0_old_vs_trunc"] = dict(old=perf(old)["sharpe"], old_weeks=int(len(old)), trunc=perf(v0_full)["sharpe"],
                                  trunc_weeks=int(len(v0_full)), last_week=str(old.index[-1].date()),
                                  last_week_ret=float(old.iloc[-1]), common_sharpe=halves(ret["V0"])["all"]["sharpe"])

    # ---- 요일 × 지연 격자 (일 종가, 서술) — 신호 요일의 UTC 종가에서 신호, k 일 뒤 종가에 체결
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tsmom_describe import book  # noqa: E402
    grid, gdiff = {}, {}
    for a in range(7):
        r0 = book(C, anchor=a, delay=0)["ret"]
        r0 = r0[r0.index >= start]                       # 리밸런스 주기 표와 같은 시작(2021-05-02 이후 신호)
        grid[a], gdiff[a] = [], []
        for dl in range(4):
            r = book(C, anchor=a, delay=dl)["ret"]
            r = r[r.index >= start]
            grid[a].append(float(r.mean() / r.std() * np.sqrt(52)))
            gdiff[a].append(float((r - r0).dropna().mean() * 1e4))
    out["delay_grid"] = dict(sharpe=grid, diff_bps=gdiff,
                             mean_sharpe=[float(np.mean([grid[a][d] for a in range(7)])) for d in range(4)],
                             mean_diff_bps=[float(np.mean([gdiff[a][d] for a in range(7)])) for d in range(4)])

    # ---- 연도별 (복리)
    out["years"] = {c: {int(y): float((1 + x).prod() - 1) for y, x in ret[c].dropna().groupby(ret[c].dropna().index.year)}
                    for c in ret}
    Path("runs/tsmom_gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ret.to_csv("runs/tsmom_gap_weekly.csv")
    for v in ("V3m", "V3e"):
        R[v].to_pickle(f"runs/tsmom_gap/R_{v}.pkl")
    R["V2"].to_pickle("runs/tsmom_gap/R_V2.pkl")

    # ---- 출력
    print("주", out["weeks"], "평균 코인 수", out["n_coins_mean"])
    for k in ("V0", "V1", "V2", "V3m", "V3e"):
        p = halves(ret[k].dropna())
        print(f"{k:<4} 샤프 {p['all']['sharpe']:.3f} (t {p['all']['t']:.2f}) CAGR {p['all']['cagr'] * 100:5.1f}% "
              f"MDD {p['all']['mdd'] * 100:5.1f}% 전반 {p['first']['sharpe']:.2f} 후반 {p['second']['sharpe']:.2f}")
    for k, c in out["comp"].items():
        print(f"{k:<11} 차이 {c['d_mean_bps']:+6.1f}bps/주 t {c['d_t_boot']:+.2f} CI [{c['d_ci_bps'][0]:+.1f}, {c['d_ci_bps'][1]:+.1f}] "
              f"이긴 주 {c['win_share'] * 100:.0f}% → {c['verdict']}")
    print("V1", {k: (round(v['d_mean_bps'], 1), round(v['d_t_boot'], 2)) for k, v in out["V1"].items() if k != "perf"})
    print("fees", json.dumps(out["fees"], ensure_ascii=False))
    print("mode", json.dumps(out["mode"], ensure_ascii=False, default=float))
    print("adverse", json.dumps(out["adverse"], ensure_ascii=False, default=float))
    print("weekend", json.dumps(out["weekend"], ensure_ascii=False, default=float))
    print("attr", json.dumps(out["attr"], ensure_ascii=False, default=float))
    print("pen", json.dumps(out["pen"], ensure_ascii=False))
    print("freq", {k: (round(v["sharpe_mean"], 2), round(v["sharpe_min"], 2), round(v["sharpe_max"], 2), round(v["gross_sharpe_mean"], 2),
                       round(v["turn_yr"], 1), round(v["fee_pct_yr"], 2)) for k, v in freq.items()})
    print("v0 old vs trunc", out["v0_old_vs_trunc"])
    print("delay grid mean", [round(x, 2) for x in out["delay_grid"]["mean_sharpe"]], [round(x, 1) for x in out["delay_grid"]["mean_diff_bps"]])
    print("years", {c: {y: round(v * 100, 1) for y, v in r.items()} for c, r in out["years"].items()})


if __name__ == "__main__":
    main()
