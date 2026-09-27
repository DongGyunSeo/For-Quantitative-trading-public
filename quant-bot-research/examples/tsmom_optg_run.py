"""매월 갱신에서 복리가 가장 큰 G (사전등록 `claude/시계열추세-매월갱신-최적G-사전등록.md` 그대로).

S1 역변동성 · 매월 갱신 · 교차 증거금. 입력 runs/tsmom_size/weights.pkl · {close,high,low}.npy → runs/tsmom_optg.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbot.research import sizing as S  # noqa: E402
from tsmom_compound_run import boot_run, schedules  # noqa: E402

D = Path("runs/tsmom_size")
GRID = np.round(np.arange(0.10, 3.001, 0.05), 2)
MMR = 0.02
P = 10_000


def boot_idx(n: int, block: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    st = g.integers(0, n - block + 1, size=(P, k))
    return (st[:, :, None] + np.arange(block)[None, None, :]).reshape(P, -1)[:, :n]


def curve(net, f, rmin, gliq, idx, step):
    """G 격자마다 (경로별 연복리, 낙폭, 실효 G 최대, 청산)."""
    C, M, GX, L = [], [], [], []
    for G in GRID:
        cg, md, _, gx, lq = boot_run(net, f, rmin, gliq, idx, float(G), step)
        C.append(cg), M.append(md), GX.append(gx), L.append(lq)
    return np.array(C), np.array(M), np.array(GX), np.array(L)


def summarize(C, M, GX, L) -> dict:
    med = np.median(C, axis=1)
    j = int(med.argmax())
    per_path = GRID[C.argmax(axis=0)]
    rows = [dict(G=float(g), cagr_p50=float(med[i]), cagr_p05=float(np.percentile(C[i], 5)), cagr_p95=float(np.percentile(C[i], 95)),
                 mdd_p50=float(np.median(M[i])), mdd_p95=float(np.percentile(M[i], 95)), p_liq=float(L[i].mean()),
                 gmax_p95=float(np.percentile(GX[i], 95)), p_loss=float((C[i] < 0).mean())) for i, g in enumerate(GRID)]
    return dict(g_opt=float(GRID[j]), cagr_opt=float(med[j]), argmax_pct=[float(np.percentile(per_path, q)) for q in (5, 25, 50, 75, 95)],
                rows=rows)


def main() -> None:
    Z = pd.read_pickle(D / "weights.pkl")
    W, R = Z["W"]["S1"], Z["R"]
    wk = W.index
    nw = len(wk)
    close, high, low = (np.load(D / f"{k}.npy") for k in ("close", "high", "low"))
    pth = S.week_paths(close, high, low, W.to_numpy(float))
    ref = S.book(W, R)                                        # 매주 · G 1 — 총노출 1배당 값
    net, f = ref.ret.to_numpy(), ref.fee_u.to_numpy()
    rmin = pth["r"].min(axis=1)
    gliq = S.liq_threshold(pth["r"], pth["g"], f, MMR)
    gliq5 = S.liq_threshold(pth["r"], pth["g"], f, 0.05)
    sch = schedules(wk)
    out = {"grid": GRID.tolist()}
    # 1) 역사 경로
    hist = {}
    for name, upd in (("monthly", sch["C3"]), ("weekly", sch["C1"])):
        rows = []
        for G in GRID:
            df, liq = S.book_cadence(W, R, float(G), upd, rmin=rmin, gliq=gliq)
            end = df.E_end.to_numpy()
            rows.append(dict(G=float(G), cagr=float(end[-1] ** (52 / nw) - 1) if liq < 0 else -1.0, mult=float(end[-1]),
                             mdd=S.mdd(end, df.trough.to_numpy()) if liq < 0 else 1.0, gmax=float(df.G_eff.max()),
                             liq=str(wk[liq].date()) if liq >= 0 else None, arith=float(df.ret.mean() * 52)))
        j = int(np.argmax([x["cagr"] for x in rows]))
        hist[name] = dict(g_opt=rows[j]["G"], cagr_opt=rows[j]["cagr"], rows=rows,
                          first_liq=min([x["G"] for x in rows if x["liq"]] or [None]))
    out["hist"] = hist
    # 2) 부트스트랩 (주 추정: 4주 블록, 매월 = 4주마다 · 블록 경계)
    idx4 = boot_idx(nw, 4, 20260926)
    out["boot"] = {"monthly": summarize(*curve(net, f, rmin, gliq, idx4, 4)),
                   "weekly": summarize(*curve(net, f, rmin, gliq, idx4, 1))}
    # 3) 민감도
    sens = {}
    for bl in (1, 13):
        sens[f"block{bl}"] = summarize(*curve(net, f, rmin, gliq, boot_idx(nw, bl, 7 + bl), 4))
    sens["mmr5"] = summarize(*curve(net, f, rmin, gliq5, idx4, 4))
    sd = net.std(ddof=1)
    for target in (0.5, 0.3):
        delta = net.mean() - target * sd / np.sqrt(52)
        gl = S.liq_threshold(pth["r"] - delta, pth["g"], f, MMR)
        s = summarize(*curve(net - delta, f, rmin - delta, gl, idx4, 4))
        s["delta_bps"] = float(delta * 1e4)
        sens[f"sharpe{target}"] = s
    out["sens"] = sens
    out["sample_sharpe"] = float(net.mean() / sd * np.sqrt(52))
    # 4) 이론 대조: 켈리 근사 G* = μ / σ² (주 단위 · 4주 단위)
    m4 = pd.Series(net).rolling(4).sum().iloc[3::4].to_numpy()
    out["kelly_approx"] = dict(weekly=float(net.mean() / net.var(ddof=1)), monthly=float(m4.mean() / m4.var(ddof=1)))
    Path("runs/tsmom_optg.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    def row(rows, g):
        return [x for x in rows if abs(x["G"] - g) < 1e-9][0]
    for name in ("monthly", "weekly"):
        h, b = hist[name], out["boot"][name]
        print(f"[{name}] 역사 최적 G {h['g_opt']:.2f} 연 {h['cagr_opt'] * 100:.1f}% (첫 청산 G {h['first_liq']}) | 부트 중앙 최적 G {b['g_opt']:.2f} "
              f"연 {b['cagr_opt'] * 100:.1f}% | 경로별 최적 5·25·50·75·95%: {b['argmax_pct']}")
    for g in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
        h = row(hist["monthly"]["rows"], g)
        b = row(out["boot"]["monthly"]["rows"], g)
        print(f"  G {g:.2f} 역사 연 {h['cagr'] * 100:6.1f}% {h['mult']:.2f}배 MDD {h['mdd'] * 100:5.1f}% 실효G최대 {h['gmax']:.2f} 청산 {h['liq']} | "
              f"부트 연 중앙 {b['cagr_p50'] * 100:5.1f}% (5% {b['cagr_p05'] * 100:6.1f}%) MDD 중앙 {b['mdd_p50'] * 100:5.1f}% 95% {b['mdd_p95'] * 100:5.1f}% "
              f"청산 {b['p_liq'] * 100:4.1f}% 5년손실 {b['p_loss'] * 100:4.1f}% 실효G최대95% {b['gmax_p95']:.2f}")
    for k, s in sens.items():
        print("민감도", k, "최적 G", s["g_opt"], "연", round(s["cagr_opt"] * 100, 1), "경로별", s["argmax_pct"], "δ" if "delta_bps" in s else "", round(s.get("delta_bps", 0), 1))
    print("표본 샤프", round(out["sample_sharpe"], 3), "켈리 근사 μ/σ²", out["kelly_approx"])


if __name__ == "__main__":
    main()
