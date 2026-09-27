"""2부 — 교차 증거금 레버리지 (사전등록 `claude/시계열추세-사이징-레버리지-사전등록.md` 그대로).

1부에서 채택된 사이징이 없어 **현행 S0(동일 명목) · 고정 G** 가 대상이다. VT(변동성 타깃)는 서술로만 함께 본다.
입력 runs/tsmom_size/weights.pkl · {close,high,low}.npy → runs/tsmom_size/part2.json
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
NW = 269
GRID = np.round(np.arange(0.05, 4.001, 0.05), 2)
MMR, MMR_STRESS = 0.02, 0.05
P = 10_000
RNG = np.random.default_rng(20260925)


def boot_index(n: int, p: int, block: int = 4) -> np.ndarray:
    k = int(np.ceil(n / block))
    st = RNG.integers(0, n - block + 1, size=(p, k))
    return (st[:, :, None] + np.arange(block)[None, None, :]).reshape(p, -1)[:, :n]


def paths_stats(net, rmin, f, Gw, gliq, idx=None):
    """(경로 수, 주) 로 자산 · 최대낙폭(장중 저점 포함) · 연복리 · 청산 여부. Gw: 주별 배수 (고정 G 면 상수 배열)."""
    if idx is None:
        idx = np.arange(len(net))[None, :]
    n_, r_, f_, g_, l_ = net[idx], rmin[idx], f[idx], Gw[idx], gliq[idx]
    liq = (g_ >= l_).any(axis=1)
    X = np.maximum(1 + g_ * n_, 1e-12)
    end = np.exp(np.cumsum(np.log(X), axis=1))
    start = np.concatenate([np.ones((end.shape[0], 1)), end[:, :-1]], axis=1)
    peak = np.maximum.accumulate(start, axis=1)
    trough = start * (1 - g_ * f_ + g_ * r_)
    dd = np.maximum(1 - trough / peak, 1 - end / np.maximum.accumulate(np.concatenate(
        [np.ones((end.shape[0], 1)), end], axis=1), axis=1)[:, 1:])
    mdd = np.where(liq, 1.0, dd.max(axis=1))
    cagr = np.where(liq, -1.0, end[:, -1] ** (52 / end.shape[1]) - 1)
    return mdd, cagr, liq


def full_path_mdd(r, f, net, G):
    """역사 경로의 5분 단위 자산으로 잰 최대낙폭 (주중 고점 → 저점까지)."""
    e = 1.0
    peak = 1.0
    worst = 0.0
    for k in range(r.shape[0]):
        path = e * (1 - G * f[k] + G * r[k])
        m = np.maximum.accumulate(np.concatenate([[peak], path]))[1:]
        worst = max(worst, float((1 - path / m).max()))
        peak = max(peak, float(path.max()))
        e = e * (1 + G * net[k])
        peak = max(peak, e)
    return worst


def main() -> None:
    Z = pd.read_pickle(D / "weights.pkl")
    meta = json.loads((D / "meta.json").read_text(encoding="utf-8"))
    W, R, books, pv = Z["W"], Z["R"], Z["books"], Z["pv"]
    assert list(W["S0"].columns) == meta["symbols"], "심볼 순서"
    close, high, low = (np.load(D / f"{k}.npy") for k in ("close", "high", "low"))
    b = books["S0"]
    pth = S.week_paths(close, high, low, W["S0"].to_numpy(float))
    assert np.allclose(pth["r"][:, -1], b.gross_ret.to_numpy(), atol=2e-5), "장중 경로 끝 ≠ 주간 수익"
    net, f = b.ret.to_numpy(), b.fee_u.to_numpy()
    rmin, rminc = pth["r"].min(axis=1), pth["rc"].min(axis=1)
    TH = {"close_2": S.liq_threshold(pth["r"], pth["g"], f, MMR),
          "close_2_x2": S.liq_threshold(pth["r"], pth["g"], f, MMR, 2.0),
          "cons_2": S.liq_threshold(pth["rc"], pth["gc"], f, MMR),
          "close_5": S.liq_threshold(pth["r"], pth["g"], f, MMR_STRESS),
          "cons_5_x2": S.liq_threshold(pth["rc"], pth["gc"], f, MMR_STRESS, 2.0)}
    wk = b.index
    out = {"thresholds": {k: dict(min=float(v.min()), week=str(wk[int(v.argmin())].date())) for k, v in TH.items()}}
    out["worst_week"] = dict(ret=float(net.min()), week=str(wk[int(net.argmin())].date()),
                             rmin=float(rmin.min()), rmin_week=str(wk[int(rmin.argmin())].date()),
                             rmin_cons=float(rminc.min()), rmin_cons_week=str(wk[int(rminc.argmin())].date()))
    # 고정 G 근사 확인: G = 2 를 장부로 다시 계산한 것과 G × (G=1 순수익) 비교
    b2 = S.book(W["S0"], R, gross=pd.Series(2.0, index=W["S0"].index))
    out["approx_check_G2"] = dict(exact_cagr=float((1 + b2.ret).prod() ** (52 / NW) - 1),
                                  scaled_cagr=float((1 + 2 * b.ret).prod() ** (52 / NW) - 1))
    # 역사 G 격자
    hist = []
    for G in GRID:                                             # 역사 경로는 G 마다 장부를 다시 계산(되맞춤 수수료 정확히)
        bg = S.book(W["S0"], R, gross=pd.Series(G, index=wk))
        net_g, f_g = (bg.ret / G).to_numpy(), bg.fee_u.to_numpy()
        th_g = S.liq_threshold(pth["r"], pth["g"], f_g, MMR)
        Gw = np.full(NW, G)
        mdd_i, cagr, liq = paths_stats(net_g, rmin, f_g, Gw, th_g)
        end, _, li = S.equity_path(net_g, rmin, f_g, Gw, th_g)
        h = dict(G=float(G), cagr=float(cagr[0]), mdd_intraweek=float(mdd_i[0]), mdd_close=S.mdd(end),
                 liq_week=str(wk[li].date()) if li >= 0 else None, worst_week=float(G * net_g.min()),
                 worst_intraweek=float(G * (rmin.min() - f_g[int(rmin.argmin())])), end_mult=float(end[-1]))
        if li < 0 and round(G * 20) % 5 == 0 or G in (0.2, 0.4):
            h["mdd_full5m"] = full_path_mdd(pth["r"], f_g, net_g, G) if li < 0 else 1.0
        hist.append(h)
    out["hist"] = hist
    L1 = float(GRID[GRID < TH["close_2_x2"].min()].max())
    out["L1"] = L1
    out["L1_variants"] = {k: float(GRID[GRID < v.min()].max()) if (GRID < v.min()).any() else 0.0 for k, v in TH.items()}
    # 부트스트랩
    idx = boot_index(NW, P)
    boot = []
    for G in GRID:
        m, c, l = paths_stats(net, rmin, f, np.full(NW, G), TH["close_2"], idx)
        boot.append(dict(G=float(G), mdd_p50=float(np.median(m)), mdd_p95=float(np.percentile(m, 95)),
                         cagr_p50=float(np.median(c)), cagr_p05=float(np.percentile(c, 5)), p_liq=float(l.mean()),
                         p_mdd30=float((m > 0.30).mean()), p_mdd50=float((m > 0.50).mean()), p_loss=float((c < 0).mean())))
    out["boot"] = boot
    # 역사 경로가 부트스트랩 분포의 몇 번째 백분위인가 (G = 1 · 0.4 · 0.2)
    pct = {}
    for G in (0.2, 0.4, 1.0):
        m, _, _ = paths_stats(net, rmin, f, np.full(NW, G), TH["close_2"], idx)
        hm = [h for h in hist if abs(h["G"] - G) < 1e-9][0]["mdd_intraweek"]
        pct[str(G)] = dict(hist_mdd=hm, pctile=float((m < hm).mean() * 100), p_mdd30=float((m > 0.3).mean()),
                           p_mdd50=float((m > 0.5).mean()))
    out["hist_pctile"] = pct
    # 블록 길이 민감도 (서술)
    blk = {}
    for bl in (12, 26):
        ib = boot_index(NW, P, bl)
        rows = []
        for G in GRID:
            m, c, l = paths_stats(net, rmin, f, np.full(NW, G), TH["close_2"], ib)
            rows.append((float(G), float(np.percentile(m, 95)), float(np.median(m))))
        blk[str(bl)] = dict(L2_30=max([g for g, p95, _ in rows if p95 <= 0.30] or [0.0]),
                           L2_50=max([g for g, p95, _ in rows if p95 <= 0.50] or [0.0]),
                           G1_p50=[x[2] for x in rows if abs(x[0] - 1.0) < 1e-9][0],
                           G1_p95=[x[1] for x in rows if abs(x[0] - 1.0) < 1e-9][0])
    out["block_sens"] = blk
    # 주말 종가 기준 최대낙폭 p95 (서술)
    close_p95 = {}
    for G in (0.2, 0.4, 0.5, 1.0):
        n_, g_ = net[idx], G
        end = np.cumprod(1 + g_ * n_, axis=1)
        dd = 1 - end / np.maximum.accumulate(np.concatenate([np.ones((P, 1)), end], axis=1), axis=1)[:, 1:]
        close_p95[str(G)] = float(np.percentile(dd.max(axis=1), 95))
    out["close_basis_p95"] = close_p95
    L2 = {lim: max([x["G"] for x in boot if x["mdd_p95"] <= lim] or [0.0]) for lim in (0.30, 0.50)}
    out["L2"] = {str(k): v for k, v in L2.items()}
    out["rec"] = {str(k): min(L1, v) for k, v in L2.items()}
    out["growth_opt"] = dict(hist=max(hist, key=lambda x: x["cagr"])["G"], boot_p50=max(boot, key=lambda x: x["cagr_p50"])["G"])
    # 샤프 0.3 시나리오
    mu, sd = net.mean(), net.std(ddof=1)
    delta = mu - 0.3 * sd / np.sqrt(52)
    net3, rmin3 = net - delta, rmin - delta
    th3 = S.liq_threshold(pth["r"] - delta, pth["g"], f, MMR)
    th3x2 = S.liq_threshold(pth["r"] - delta, pth["g"], f, MMR, 2.0)
    boot3 = []
    for G in GRID:
        m, c, l = paths_stats(net3, rmin3, f, np.full(NW, G), th3, idx)
        boot3.append(dict(G=float(G), mdd_p95=float(np.percentile(m, 95)), cagr_p50=float(np.median(c)),
                          p_loss=float((c < 0).mean()), p_liq=float(l.mean())))
    L1_3 = float(GRID[GRID < th3x2.min()].max())
    out["sharpe03"] = dict(delta_bps=float(delta * 1e4), boot=boot3, L1=L1_3,
                           rec={str(lim): min(L1_3, max([x["G"] for x in boot3 if x["mdd_p95"] <= lim] or [0.0])) for lim in (0.30, 0.50)},
                           growth_opt=max(boot3, key=lambda x: x["cagr_p50"])["G"])
    # VT (서술): 목표 변동성 격자, 주별 G = min(3, 목표 / 사전 변동성)
    vt = []
    pvv = pv["S0"].reindex(wk).to_numpy()
    for tgt in np.round(np.arange(0.05, 0.801, 0.025), 3):
        Gw = np.minimum(3.0, tgt / pvv)
        bv = S.book(W["S0"], R, gross=pd.Series(Gw, index=wk))
        netv = (bv.ret / bv.G).to_numpy()                      # 총노출 1배당
        fv = bv.fee_u.to_numpy()
        thv = S.liq_threshold(pth["r"], pth["g"], fv, MMR)
        thv2 = S.liq_threshold(pth["r"], pth["g"], fv, MMR, 2.0)
        mh, ch, lh = paths_stats(netv, rmin, fv, Gw, thv)
        mb, cb, lb = paths_stats(netv, rmin, fv, Gw, thv, idx)
        vt.append(dict(target=float(tgt), G_mean=float(Gw.mean()), G_max=float(Gw.max()), hist_cagr=float(ch[0]),
                       hist_mdd=float(mh[0]), hist_liq=bool(lh[0]), x2_safe=bool((Gw < thv2).all()),
                       mdd_p95=float(np.percentile(mb, 95)), cagr_p50=float(np.median(cb)), p_liq=float(lb.mean())))
    out["vt"] = vt
    out["vt_rec"] = {str(lim): max([x["target"] for x in vt if x["mdd_p95"] <= lim and x["x2_safe"]] or [0.0]) for lim in (0.30, 0.50)}
    Path(D / "part2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("청산 임계 G (주 최솟값):", {k: (round(v["min"], 2), v["week"]) for k, v in out["thresholds"].items()})
    print("최악의 주", out["worst_week"], "| G=2 근사 확인", out["approx_check_G2"])
    print("L1 (장중 최악 2배에도 청산 없음) =", L1, "| 변형", out["L1_variants"])
    print("L2 =", out["L2"], "→ 권장 =", out["rec"], "| 성장 최적", out["growth_opt"])
    print("역사 백분위", out["hist_pctile"], "| 블록 민감도", out["block_sens"], "| 주말 기준 p95", out["close_basis_p95"])
    for h, bb in zip(hist, boot):
        if h["G"] in (0.2, 0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            print(f"G {h['G']:.2f} 역사 CAGR {h['cagr'] * 100:6.1f}% MDD(주말) {h['mdd_close'] * 100:5.1f}% MDD(장중) {h['mdd_intraweek'] * 100:5.1f}% "
                  f"MDD(5분) {h.get('mdd_full5m', float('nan')) * 100:5.1f}% 최악주 {h['worst_week'] * 100:6.1f}% 청산 {h['liq_week']} | "
                  f"부트 MDD 중앙 {bb['mdd_p50'] * 100:5.1f}% p95 {bb['mdd_p95'] * 100:5.1f}% CAGR 중앙 {bb['cagr_p50'] * 100:6.1f}% "
                  f"p5 {bb['cagr_p05'] * 100:6.1f}% 청산 {bb['p_liq'] * 100:4.1f}% 5년손실 {bb['p_loss'] * 100:4.1f}%")
    s3 = out["sharpe03"]
    print("샤프 0.3 시나리오: 주 δ", round(s3["delta_bps"], 1), "bps  L1", s3["L1"], "권장", s3["rec"], "성장최적", s3["growth_opt"])
    for x in s3["boot"]:
        if x["G"] in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            print(f"   G {x['G']:.2f} MDD p95 {x['mdd_p95'] * 100:5.1f}% CAGR 중앙 {x['cagr_p50'] * 100:6.1f}% 5년손실 {x['p_loss'] * 100:4.1f}%")
    print("VT 권장 목표", out["vt_rec"])
    for x in vt:
        if abs(x["target"] * 40 - round(x["target"] * 40)) < 1e-9 and round(x["target"] * 100) % 10 == 0:
            print(f"   목표 {x['target'] * 100:.0f}% G평균 {x['G_mean']:.2f} 최대 {x['G_max']:.2f} 역사 CAGR {x['hist_cagr'] * 100:5.1f}% MDD {x['hist_mdd'] * 100:5.1f}% "
                  f"청산 {x['hist_liq']} 2배안전 {x['x2_safe']} | p95 MDD {x['mdd_p95'] * 100:5.1f}% CAGR 중앙 {x['cagr_p50'] * 100:5.1f}% 청산 {x['p_liq'] * 100:4.1f}%")


if __name__ == "__main__":
    main()
