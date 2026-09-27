"""복리 주기 — 사이징 기준 자산 갱신 주기 비교 (사전등록 `claude/시계열추세-복리주기-사전등록.md` 그대로).

S1 역변동성 · G 0.75 · 교차 증거금. 입력 runs/tsmom_size/weights.pkl · {close,high,low}.npy
결과 runs/tsmom_compound.json · runs/tsmom_compound_weekly.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import sizing as S  # noqa: E402

D = Path("runs/tsmom_size")
G0 = 0.75
MMR = 0.02
P = 10_000
RNG = np.random.default_rng(20260926)
NAMES = {"C1": "매주", "C2": "2주", "C3": "매월", "C4": "분기", "C5": "매년", "C6": "갱신 안 함", "C7": "비대칭(줄임 매주 · 키움 매월)"}


def schedules(wk: pd.DatetimeIndex) -> dict:
    k = np.arange(len(wk))
    newm = np.r_[True, wk.month[1:] != wk.month[:-1]]
    return {"C1": np.ones(len(wk), bool), "C2": k % 2 == 0, "C3": newm,
            "C4": newm & np.isin(wk.month, [1, 4, 7, 10]) | (k == 0), "C5": newm & (wk.month == 1) | (k == 0),
            "C6": k == 0, "C7": newm}


def metrics(df: pd.DataFrame, liq: int, nw: int) -> dict:
    end, trough = df.E_end.to_numpy(), df.trough.to_numpy()
    r = df.ret[df.E_start > 0]
    return dict(mult=float(end[-1]), cagr=float(end[-1] ** (52 / nw) - 1) if liq < 0 else -1.0,
                sharpe=float(r.mean() / r.std() * np.sqrt(52)), vol=float(r.std() * np.sqrt(52)),
                arith=float(r.mean() * 52), mdd_close=S.mdd(end), mdd=S.mdd(end, trough), worst=float(r.min()),
                g_min=float(df.G_eff.min()), g_mean=float(df.G_eff.mean()), g_max=float(df.G_eff.max()),
                fee_pct_yr=float(df.fee.mean() * 52 * 100), liq=None if liq < 0 else int(liq))


def boot_run(net, f, rmin, gliq, idx, G, step, asym=False):
    Pn, n = idx.shape
    E, B, peak = np.ones(Pn), np.ones(Pn), np.ones(Pn)
    mdd = np.zeros(Pn)
    liq = np.zeros(Pn, bool)
    gsum, gmax = np.zeros(Pn), np.zeros(Pn)
    for k in range(n):
        j = idx[:, k]
        if k == 0 or (step and k % step == 0):
            B = E.copy()
        elif asym:
            B = np.minimum(B, E)
        ge = np.where(liq, 0.0, G * B / np.where(E > 0, E, 1.0))
        gsum += ge
        gmax = np.maximum(gmax, ge)
        liq |= ge >= gliq[j]
        trough = E * (1 - ge * f[j] + ge * rmin[j])
        mdd = np.maximum(mdd, 1 - trough / peak)
        E = np.where(liq, 0.0, E * (1 + ge * net[j]))
        peak = np.maximum(peak, E)
        mdd = np.maximum(mdd, 1 - E / peak)
    mdd = np.where(liq, 1.0, mdd)
    cagr = np.where(liq, -1.0, np.maximum(E, 0) ** (52 / n) - 1)
    return cagr, mdd, gsum / n, gmax, liq


def main() -> None:
    Z = pd.read_pickle(D / "weights.pkl")
    W, R = Z["W"]["S1"], Z["R"]
    wk = W.index
    nw = len(wk)
    close, high, low = (np.load(D / f"{k}.npy") for k in ("close", "high", "low"))
    pth = S.week_paths(close, high, low, W.to_numpy(float))
    ref = S.book(W, R, gross=pd.Series(G0, index=wk))
    assert np.allclose(pth["r"][:, -1] * G0, ref.gross_ret.to_numpy(), atol=2e-5)
    f_u = ref.fee_u.to_numpy()
    rmin = pth["r"].min(axis=1)
    gliq = S.liq_threshold(pth["r"], pth["g"], f_u, MMR)
    sch = schedules(wk)
    out = {"G": G0, "weeks": nw, "gliq_min": float(gliq.min()), "hist": {}, "g_sens": {}}
    weekly = {}
    for c, upd in sch.items():
        df, liq = S.book_cadence(W, R, G0, upd, asym=(c == "C7"), rmin=rmin, gliq=gliq)
        out["hist"][c] = metrics(df, liq, nw)
        out["hist"][c]["updates"] = int(upd.sum())
        out["hist"][c]["years"] = {int(y): float((1 + x).prod() - 1) for y, x in df.ret.groupby(wk.year)}
        weekly[c] = df
    assert np.allclose(weekly["C1"].ret.to_numpy(), ref.ret.to_numpy()), "매주 갱신 ≠ 기존 장부"
    pd.DataFrame({f"{c}_{k}": weekly[c][k] for c in weekly for k in ("ret", "G_eff", "E_end")}).to_csv("runs/tsmom_compound_weekly.csv")
    for g in (0.5, 1.0):
        out["g_sens"][str(g)] = {}
        for c in ("C1", "C3", "C6", "C7"):
            df, liq = S.book_cadence(W, R, g, sch[c], asym=(c == "C7"), rmin=rmin, gliq=gliq)
            out["g_sens"][str(g)][c] = metrics(df, liq, nw)
    # 부트스트랩 (같은 경로에서 짝 비교)
    net_u = (ref.ret / G0).to_numpy()
    k = int(np.ceil(nw / 4))
    st = RNG.integers(0, nw - 3, size=(P, k))
    idx = (st[:, :, None] + np.arange(4)[None, None, :]).reshape(P, -1)[:, :nw]
    steps = {"C1": 1, "C2": 2, "C3": 4, "C4": 13, "C5": 52, "C6": 0, "C7": 4}
    res = {c: boot_run(net_u, f_u, rmin, gliq, idx, G0, s, asym=(c == "C7")) for c, s in steps.items()}
    c1, m1 = res["C1"][0], res["C1"][1]
    out["boot"] = {}
    for c, (cg, md, gm, gx, lq) in res.items():
        out["boot"][c] = dict(cagr_p50=float(np.median(cg)), cagr_p05=float(np.percentile(cg, 5)), cagr_p95=float(np.percentile(cg, 95)),
                              mdd_p50=float(np.median(md)), mdd_p95=float(np.percentile(md, 95)),
                              p_beat_c1=float((cg > c1).mean()), d_cagr_p50=float(np.median(cg - c1)),
                              d_cagr_p05=float(np.percentile(cg - c1, 5)), d_cagr_p95=float(np.percentile(cg - c1, 95)),
                              d_mdd_p50=float(np.median(md - m1)), g_mean=float(np.median(gm)), g_max_p95=float(np.percentile(gx, 95)),
                              p_liq=float(lq.mean()), p_gmax_over1=float((gx > 1.0).mean()))
    # 판정: C3 매월 vs C1 매주
    h1, h3, b3 = out["hist"]["C1"], out["hist"]["C3"], out["boot"]["C3"]
    v = dict(c1=bool(h3["cagr"] >= h1["cagr"]), c2=bool(h3["mdd"] <= h1["mdd"] + 0.02),
             c3=bool(b3["p_beat_c1"] >= 0.60 and (out["boot"]["C3"]["mdd_p50"] <= out["boot"]["C1"]["mdd_p50"] + 0.01)))
    v["adopt_monthly"] = all(v.values())
    out["verdict"] = v
    # 서술: 전략 주간 수익의 자기상관 · 분산비 (섞은 순서와 비교)
    def vr(x, q):
        sm = pd.Series(x).rolling(q).sum().dropna().to_numpy()
        return float(sm.var(ddof=1) / (q * x.var(ddof=1)))
    g2 = np.random.default_rng(2)
    sims = {q: np.array([vr(g2.permutation(net_u), q) for _ in range(3000)]) for q in (4, 13, 26)}
    out["serial"] = dict(ac=[float(np.corrcoef(net_u[:-L], net_u[L:])[0, 1]) for L in (1, 2, 3, 4)],
                         vr={str(q): dict(obs=vr(net_u, q), null_p50=float(np.median(v)), p=float((v <= vr(net_u, q)).mean()))
                             for q, v in sims.items()})
    # 서술: 블록 길이 · 갱신 위치 민감도 (매월 = 4주, 분기 = 13주)
    sens = {}
    for block in (1, 4, 8, 13):
        g3 = np.random.default_rng(5)
        kb = int(np.ceil(nw / block))
        stb = g3.integers(0, nw - block + 1, size=(P, kb))
        ib = (stb[:, :, None] + np.arange(block)[None, None, :]).reshape(P, -1)[:, :nw]
        base = boot_run(net_u, f_u, rmin, gliq, ib, G0, 1)[0]
        row = {}
        for off in (0, 2):
            for name, stp in (("월", 4), ("분기", 13)):
                E, Bb = np.ones(P), np.ones(P)
                for t in range(nw):
                    j = ib[:, t]
                    if t == 0 or (t - off) % stp == 0:
                        Bb = E.copy()
                    E = E * (1 + G0 * Bb / E * net_u[j])
                cg = E ** (52 / nw) - 1
                row[f"{name}_{off}"] = dict(p_beat=float((cg > base).mean()), d_p50=float(np.median(cg - base)))
        sens[str(block)] = row
    out["align_sens"] = sens
    # 산술 vs 복리 (매주 갱신, G 격자)
    dec = []
    for g in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        b = S.book(W, R, gross=pd.Series(g, index=wk))
        eq = (1 + b.ret).prod()
        ar = float(b.ret.mean() * 52)
        cg = float(eq ** (52 / nw) - 1)
        dec.append(dict(G=g, arith=ar, cagr=cg, drag=ar - cg, half_var=float(b.ret.var() * 52 / 2), mult=float(eq)))
    out["decomp"] = dec
    Path("runs/tsmom_compound.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("청산 임계 최소", round(gliq.min(), 2))
    for c, h in out["hist"].items():
        b = out["boot"][c]
        print(f"{c} {NAMES[c]:<18} 갱신 {h['updates']:>3}회 | 역사 {h['mult']:.2f}배 연 {h['cagr'] * 100:5.1f}% MDD {h['mdd'] * 100:5.1f}% "
              f"(주말 {h['mdd_close'] * 100:5.1f}%) 샤프 {h['sharpe']:.3f} 실효G {h['g_min']:.2f}~{h['g_max']:.2f} 평균 {h['g_mean']:.2f} 수수료 {h['fee_pct_yr']:.2f}% "
              f"| 부트 연복리 중앙 {b['cagr_p50'] * 100:5.1f}% (5% {b['cagr_p05'] * 100:5.1f}%) MDD 중앙 {b['mdd_p50'] * 100:5.1f}% 95% {b['mdd_p95'] * 100:5.1f}% "
              f"매주보다 좋을 확률 {b['p_beat_c1'] * 100:4.1f}% 차이중앙 {b['d_cagr_p50'] * 100:+.2f}%p 낙폭차 {b['d_mdd_p50'] * 100:+.1f}%p "
              f"실효G 최대 95% {b['g_max_p95']:.2f} G>1 확률 {b['p_gmax_over1'] * 100:.0f}% 청산 {b['p_liq'] * 100:.1f}%")
    print("판정 매월 vs 매주:", v)
    for g, d in out["g_sens"].items():
        print("G", g, {c: (round(x["cagr"] * 100, 1), round(x["mdd"] * 100, 1), round(x["g_max"], 2)) for c, x in d.items()})
    print("산술 vs 복리", [(x["G"], round(x["arith"] * 100, 1), round(x["cagr"] * 100, 1), round(x["drag"] * 100, 1), round(x["half_var"] * 100, 1)) for x in dec])
    print("자기상관", [round(a, 3) for a in out["serial"]["ac"]], "VR", {q: (round(x["obs"], 3), round(x["p"], 3)) for q, x in out["serial"]["vr"].items()})
    print("블록 · 위치 민감도", {b: {k: (round(x["p_beat"] * 100), round(x["d_p50"] * 100, 2)) for k, x in r.items()} for b, r in out["align_sens"].items()})
    print("연도별", {c: {y: round(r * 100, 1) for y, r in out["hist"][c]["years"].items()} for c in ("C1", "C3", "C6", "C7")})


if __name__ == "__main__":
    main()
