"""1부 — 사이징 판정 (사전등록 `claude/시계열추세-사이징-레버리지-사전등록.md` 그대로).

입력 runs/tsmom_anchor/hourly_close.pkl (매 정시 가격) · runs/tsmom_gap_weekly.csv (현행 V0 대조)
결과 runs/tsmom_size/part1.json · runs/tsmom_size/weekly_part1.csv · runs/tsmom_size/weights.pkl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import sizing as S  # noqa: E402

T0 = pd.Timestamp("2021-05-03 00:00", tz="UTC")
NW = 269
HALF = pd.Timestamp("2023-11-01", tz="UTC")
TARGET, CAP = 0.30, 3.0
RNG = np.random.default_rng(20260925)


def inputs(win: int = 60):
    H = pd.read_pickle("runs/tsmom_anchor/hourly_close.pkl")
    D = H[H.index.hour == 0]
    T = pd.DatetimeIndex([T0 + pd.Timedelta(days=7 * k) for k in range(NW + 1)])
    sig = np.sign(D.loc[T[:-1]] / D.shift(28).loc[T[:-1]].to_numpy() - 1)
    R = pd.DataFrame(D.loc[T[1:]].to_numpy() / D.loc[T[:-1]].to_numpy() - 1, index=T[:-1], columns=D.columns)
    ld = np.log(D).diff()
    sig_v = {w: (ld.rolling(w, min_periods=20).std() * np.sqrt(365)).loc[T[:-1]] for w in (20, 60, 90)}
    return D, T, sig, R, sig_v


def port_vol(D: pd.DataFrame, W: pd.DataFrame, win: int = 60) -> pd.Series:
    rd = D.pct_change()
    out = {}
    for t, w in W.iterrows():
        rows = rd.loc[t - pd.Timedelta(days=win - 1): t]
        rp = rows.fillna(0).to_numpy() @ w.fillna(0).to_numpy()
        rp = rp[np.isfinite(rp)]
        out[t] = rp.std(ddof=1) * np.sqrt(365) if len(rp) >= 20 else np.nan
    return pd.Series(out)


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(52)), vol=float(r.std() * np.sqrt(52)),
                cagr=float(eq.iloc[-1] ** (52 / len(r)) - 1), mdd=float((eq / eq.cummax() - 1).min()),
                first=float(r[r.index < HALF].mean() / r[r.index < HALF].std() * np.sqrt(52)),
                second=float(r[r.index >= HALF].mean() / r[r.index >= HALF].std() * np.sqrt(52)))


def at_vol(r: pd.Series, vol: float = 0.30) -> dict:
    """같은 변동성(연 vol)으로 사후 스케일한 연복리 · 최대낙폭 — 비교용."""
    c = vol / (r.std() * np.sqrt(52))
    x = r * c
    eq = (1 + x).cumprod()
    return dict(scale=float(c), cagr=float(eq.iloc[-1] ** (52 / len(x)) - 1), mdd=float((eq / eq.cummax() - 1).min()))


def sharpe_boot(a: np.ndarray, b: np.ndarray, block: int = 4, reps: int = 5000):
    n = len(a)
    k = int(np.ceil(n / block))
    sh = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(52)
    d = np.empty(reps)
    for i in range(reps):
        st = RNG.integers(0, n - block + 1, size=k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        d[i] = sh(a[idx]) - sh(b[idx])
    obs = sh(a) - sh(b)
    se = d.std(ddof=1)
    return dict(diff=float(obs), t=float(obs / se), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def main() -> None:
    D, T, sig, R, sig_v = inputs()
    W = {"S0": S.unit_weights(sig, None, "ew"), "S1": S.unit_weights(sig, sig_v[60], "iv"),
         "S2": S.unit_weights(sig, sig_v[60], "vp")}
    # 현행 대조: 동일 명목 + 부호 바뀐 몫만 수수료 = 예전 V0
    v0 = pd.read_csv("runs/tsmom_gap_weekly.csv", index_col=0, parse_dates=True)["V0"]
    b_flip = S.book(W["S0"], R, fee_mode="flip")
    assert np.allclose(b_flip.ret.to_numpy(), v0.reindex(b_flip.index.tz_localize(None)).to_numpy(), atol=1e-12), "V0 재현 실패"
    books = {k: S.book(w, R) for k, w in W.items()}
    pv = {k: port_vol(D, W[k]) for k in ("S0", "S1", "S2")}
    for k in ("S0", "S1", "S2"):
        G = (TARGET / pv[k]).clip(upper=CAP)
        books[k + "+VT"] = S.book(W[k], R, gross=G)
    ret = pd.DataFrame({k: b.ret for k, b in books.items()})
    out = {"weeks": NW, "flip_S0": perf(b_flip.ret), "perf": {k: perf(ret[k]) for k in ret},
           "at30": {k: at_vol(ret[k]) for k in ret},
           "fee_pct_yr": {k: float(b.fee.mean() * 52 * 100) for k, b in books.items()},
           "turn_yr": {k: float(b.turnover.mean() * 52) for k, b in books.items()},
           "G_mean": {k: float(b.G.mean()) for k, b in books.items()},
           "G_max": {k: float(b.G.max()) for k, b in books.items()},
           "exante_vol": {k: float(pv[k].median()) for k in pv}}

    def verdict(a, b, strict: bool):
        pa, pb = out["perf"][a], out["perf"][b]
        bt = sharpe_boot(ret[a].to_numpy(), ret[b].to_numpy())
        c_mdd = out["at30"][a]["mdd"] >= out["at30"][b]["mdd"] - 0.05
        if strict:
            ok = bt["t"] >= 2 and pa["first"] > pb["first"] and pa["second"] > pb["second"] and c_mdd
        else:
            ok = pa["sharpe"] >= pb["sharpe"] and pa["first"] >= pb["first"] and pa["second"] >= pb["second"] and c_mdd
        return dict(boot=bt, mdd_ok=bool(c_mdd), adopt=bool(ok))

    v = {"S1_vs_S0": verdict("S1", "S0", False), "S2_vs_S0": verdict("S2", "S0", True)}
    X = "S1" if v["S1_vs_S0"]["adopt"] else ("S2" if v["S2_vs_S0"]["adopt"] else "S0")
    v[f"{X}+VT_vs_{X}"] = verdict(X + "+VT", X, False)
    out["verdict"], out["chosen_base"] = v, X
    # 서술: 창 민감도 (역변동성 20 · 90일), 비중 쏠림
    sens = {}
    for win in (20, 90):
        Wi = S.unit_weights(sig, sig_v[win], "iv")
        sens[f"S1_{win}d"] = perf(S.book(Wi, R).ret)
        Wv = S.unit_weights(sig, sig_v[win], "vp")
        sens[f"S2_{win}d"] = perf(S.book(Wv, R).ret)
    out["window_sens"] = sens
    aw = W["S1"].abs()
    out["s1_weight_mean"] = {c: float(x) for c, x in aw.mean().sort_values(ascending=False).items()}
    out["s1_top5_share"] = float(aw.apply(lambda r: r.nlargest(5).sum(), axis=1).mean())
    out["years"] = {k: {int(y): float((1 + x).prod() - 1) for y, x in ret[k].groupby(ret.index.year)} for k in ret}
    Path("runs/tsmom_size/part1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ret.to_csv("runs/tsmom_size/weekly_part1.csv")
    pd.to_pickle(dict(W=W, R=R, books=books, pv=pv, T=T), "runs/tsmom_size/weights.pkl")

    print("대조: 동일 명목·부호 수수료 = V0 재현, 샤프", round(out["flip_S0"]["sharpe"], 3))
    for k in ret:
        p, a = out["perf"][k], out["at30"][k]
        print(f"{k:<7} 샤프 {p['sharpe']:.3f} 전반 {p['first']:.2f} 후반 {p['second']:.2f} 변동성 {p['vol'] * 100:4.0f}% "
              f"CAGR {p['cagr'] * 100:5.1f}% MDD {p['mdd'] * 100:5.1f}% | 30%로 맞춤 CAGR {a['cagr'] * 100:5.1f}% MDD {a['mdd'] * 100:5.1f}% | "
              f"수수료 연 {out['fee_pct_yr'][k]:.2f}% 회전 {out['turn_yr'][k]:.1f} G평균 {out['G_mean'][k]:.2f} 최대 {out['G_max'][k]:.2f}")
    for k, x in v.items():
        print(k, f"샤프차 {x['boot']['diff']:+.3f} t {x['boot']['t']:+.2f} CI [{x['boot']['ci'][0]:+.2f}, {x['boot']['ci'][1]:+.2f}] MDD조건 {x['mdd_ok']} → 채택 {x['adopt']}")
    print("선택된 비중:", X, "| 사전 변동성 중앙", {k: round(x, 3) for k, x in out["exante_vol"].items()})
    print("창 민감도", {k: round(x["sharpe"], 3) for k, x in sens.items()})
    print("S1 평균 비중 상위", {c: round(x, 3) for c, x in list(out["s1_weight_mean"].items())[:6]}, "하위",
          {c: round(x, 3) for c, x in list(out["s1_weight_mean"].items())[-4:]}, "상위5 합", round(out["s1_top5_share"], 3))
    print("years", {k: {y: round(x * 100, 1) for y, x in r.items()} for k, r in out["years"].items()})


if __name__ == "__main__":
    main()
