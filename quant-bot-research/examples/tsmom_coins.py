"""시계열 추세 — 코인별 수익과 변동성 (서술, 판정 아님).

사용자 가설: "코인들의 차이는 변동성이다." 즉 코인마다 같은 베팅(시장 추세)을 크기만 다르게 하고 있어서
원수익은 변동성에 비례하고, 위험 1단위당 수익(샤프)은 비슷할 것이다.

코인별 단독 수익(명목 1배): ret_i = s_i · r_i − 수수료_i − 펀딩_i  (포트폴리오 = 47개 평균, 검산)
확인하는 것:
  1. 코인별 성과표 — 주평균 · 연(×52) · 복리 · 샤프 · MDD · 매수보유
  2. 변동성 · 베타(주간 수익을 자기 뺀 47개 평균에 회귀)
  3. 코인 수익 = 시장 몫(s_i · β_i · m) + 고유 몫(s_i · (r_i − β_i · m))
  4. 횡단면: 연수익 ~ 변동성 기울기·상관, 샤프 ~ 변동성 — 주 4주 블록 부트스트랩(코인끼리 상관을 그대로 둔다)
  5. 귀무 검사 "모든 코인의 샤프가 같다" · "모든 코인의 평균수익이 같다" — 관측 분산이 잡음 수준인가
  6. 변동성 3분위 묶음(전 구간 변동성 · 직전 90일 변동성 두 가지) · 위험 기여

입력 runs/legacy/daily_close.pkl → runs/tsmom_coins.json · runs/tsmom_coins.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_describe import book, perf  # noqa: E402

RNG = np.random.default_rng(20260925)


def block_idx(n: int, block: int = 4) -> np.ndarray:
    k = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=k)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def sharpe_cols(X: np.ndarray) -> np.ndarray:
    return X.mean(0) / X.std(0, ddof=1) * np.sqrt(52)


def main() -> None:
    C = pd.read_pickle("runs/legacy/daily_close.pkl")
    B = book(C)
    pos, fwd = B["pos"], B["fwd"]
    ret = pos * fwd - B["fee"] - B["fund"]                        # 코인별 주간 순수익 (명목 1배)
    assert np.allclose(ret.mean(axis=1), B["ret"]), "코인별 평균 ≠ 포트폴리오"
    gross = pos * fwd
    N = pos.shape[1]
    m = fwd.mean(axis=1)

    dvol = np.log(C).diff().std() * np.sqrt(365)                   # 전 구간 일간 로그수익 변동성(연)
    rows = []
    for c in C.columns:
        r = ret[c]
        loo = (m * N - fwd[c]) / (N - 1)
        beta = float(np.cov(fwd[c], loo)[0, 1] / loo.var())
        mk = pos[c] * beta * m
        own = gross[c] - mk
        eq = (1 + r).cumprod()
        bh = float((1 + fwd[c]).prod() - 1)
        sd_w = float(r.std())
        rows.append(dict(coin=c, vol=float(dvol[c]), beta=beta, corr=float(fwd[c].corr(loo)),
                         mean_bps=float(r.mean() * 1e4), ann=float(r.mean() * 52),
                         sharpe=float(r.mean() / sd_w * np.sqrt(52)), t=float(r.mean() / sd_w * np.sqrt(len(r))),
                         se_ann=float(sd_w * 52 / np.sqrt(len(r))), worst=float(r.min()), worst_date=str(r.idxmin().date()),
                         busted=bool((eq <= 0).any()), hit=float((r > 0).mean()),
                         bh_total=bh, bh_ann=float(fwd[c].mean() * 52),
                         mkt_bps=float(mk.mean() * 1e4), own_bps=float(own.mean() * 1e4),
                         cost_bps=float((B["fee"][c] + B["fund"][c]).mean() * 1e4),
                         long_share=float((pos[c] > 0).mean()), flips=int((pos[c].diff().abs() == 2).sum())))
    T = pd.DataFrame(rows).sort_values("vol").reset_index(drop=True)

    # 위험 기여 (동일 명목 포트폴리오)
    Rp = ret.mean(axis=1)
    rc = {c: float((ret[c] / N).cov(Rp) / Rp.var()) for c in C.columns}
    T["risk_share"] = T.coin.map(rc)

    # 횡단면 관계
    def xs(ann: np.ndarray, sh: np.ndarray, vol: np.ndarray, beta: np.ndarray) -> dict:
        b1, a1 = np.polyfit(vol, ann, 1)
        b2, a2 = np.polyfit(vol, sh, 1)
        return dict(slope_ann_vol=float(b1), icpt_ann_vol=float(a1), corr_ann_vol=float(np.corrcoef(vol, ann)[0, 1]),
                    slope_sh_vol=float(b2), icpt_sh_vol=float(a2), corr_sh_vol=float(np.corrcoef(vol, sh)[0, 1]),
                    corr_ann_beta=float(np.corrcoef(beta, ann)[0, 1]))
    cols = list(T.coin)
    R = ret[cols].to_numpy()
    vol = T.vol.to_numpy()
    beta = T.beta.to_numpy()
    obs = xs(T.ann.to_numpy(), T.sharpe.to_numpy(), vol, beta)

    # 부트스트랩 (주 4주 블록) — 기울기 신뢰구간 · 귀무 두 가지
    n = R.shape[0]
    mu, sd = R.mean(0), R.std(0, ddof=1)
    common_sh = float(np.mean(mu / sd))                            # 주 단위 공통 샤프
    R_eqsh = R - mu + common_sh * sd                               # 귀무 A: 모든 코인 샤프 같음(변동성만 다름)
    R_eqmu = R - mu + mu.mean()                                    # 귀무 B: 모든 코인 평균수익 같음(변동성 무관)
    bs = {"slope_ann_vol": [], "corr_sh_vol": [], "sd_sh_nullA": [], "sd_mu_nullB": [], "slope_nullB": []}
    for _ in range(2000):
        ix = block_idx(n)
        X = R[ix]
        a = xs(X.mean(0) * 52, sharpe_cols(X), vol, beta)
        bs["slope_ann_vol"].append(a["slope_ann_vol"])
        bs["corr_sh_vol"].append(a["corr_sh_vol"])
        bs["sd_sh_nullA"].append(sharpe_cols(R_eqsh[ix]).std(ddof=1))
        XB = R_eqmu[ix]
        bs["sd_mu_nullB"].append((XB.mean(0) * 52).std(ddof=1))
        bs["slope_nullB"].append(np.polyfit(vol, XB.mean(0) * 52, 1)[0])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    sd_sh_obs = float(T.sharpe.std(ddof=1))
    sd_mu_obs = float(T.ann.std(ddof=1))
    tests = dict(
        slope_ann_vol_ci=q(bs["slope_ann_vol"]), corr_sh_vol_ci=q(bs["corr_sh_vol"]),
        sd_sharpe_obs=sd_sh_obs, sd_sharpe_nullA_mean=float(np.mean(bs["sd_sh_nullA"])),
        p_sd_sharpe_nullA=float(np.mean(np.array(bs["sd_sh_nullA"]) >= sd_sh_obs)),
        sd_ann_obs=sd_mu_obs, sd_ann_nullB_mean=float(np.mean(bs["sd_mu_nullB"])),
        p_sd_ann_nullB=float(np.mean(np.array(bs["sd_mu_nullB"]) >= sd_mu_obs)),
        p_slope_nullB=float(np.mean(np.abs(np.array(bs["slope_nullB"])) >= abs(obs["slope_ann_vol"]))),
        common_sharpe_ann=common_sh * np.sqrt(52))

    # 변동성 3분위 — (a) 전 구간 변동성 (b) 매주 직전 90일 변동성(인과)
    terc = {}
    grp = np.array_split(np.arange(N), 3)
    names = ("낮음", "중간", "높음")
    for nm, g in zip(names, grp):
        cs = [cols[i] for i in g]
        rp = ret[cs].mean(axis=1)
        p = perf(rp)
        terc[f"전구간 {nm}"] = dict(coins=len(cs), vol=float(T.vol.iloc[g].mean()), ann=float(rp.mean() * 52),
                                   sharpe=p["sharpe"], mdd=p["mdd"], cagr=p["cagr"], vol_strat=p["vol"],
                                   members=cs)
    tv = (np.log(C).diff().rolling(90, min_periods=60).std() * np.sqrt(365)).reindex(ret.index)
    rk = tv.rank(axis=1, pct=True)
    for nm, lo, hi in (("낮음", 0, 1 / 3), ("중간", 1 / 3, 2 / 3), ("높음", 2 / 3, 1.0001)):
        msk = (rk > lo) & (rk <= hi) if lo > 0 else (rk <= hi)
        rp = ret.where(msk).mean(axis=1).dropna()
        p = perf(rp)
        terc[f"직전90일 {nm}"] = dict(coins=float(msk.sum(axis=1).mean()), vol=float(tv.where(msk).stack().mean()),
                                     ann=float(rp.mean() * 52), sharpe=p["sharpe"], mdd=p["mdd"], cagr=p["cagr"],
                                     vol_strat=p["vol"], weeks=int(len(rp)))

    # Fama-MacBeth: 매주 코인 수익을 직전 90일 변동성에 횡단면 회귀 → 기울기의 시계열 평균과 t
    fm = {}
    V = tv[cols]
    for nm, Y in (("raw", ret[cols]), ("per_vol", ret[cols] / V)):
        sl = []
        for t_ in Y.index:
            y, x = Y.loc[t_].to_numpy(float), V.loc[t_].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() >= 20:
                sl.append(np.polyfit(x[ok], y[ok], 1)[0])
        sl = np.array(sl)
        fm[nm] = dict(slope_mean=float(sl.mean()), t=float(sl.mean() / sl.std(ddof=1) * np.sqrt(len(sl))), weeks=int(len(sl)))
    # 직전 90일 3분위 — 높음 − 낮음 샤프 차이의 부트스트랩
    hi_r = ret.where(rk > 2 / 3).mean(axis=1)
    lo_r = ret.where(rk <= 1 / 3).mean(axis=1)
    both = pd.concat([hi_r, lo_r], axis=1).dropna().to_numpy()
    diffs = []
    for _ in range(2000):
        X = both[block_idx(len(both))]
        s_ = sharpe_cols(X)
        diffs.append(s_[0] - s_[1])
    obs_d = sharpe_cols(both)
    tests["terc_sharpe_diff"] = float(obs_d[0] - obs_d[1])
    tests["terc_sharpe_diff_ci"] = q(diffs)
    tests["fm"] = fm

    # 변동성으로 나눈(위험 1단위) 코인별 수익 — 흩어짐 비교
    disp = dict(cv_ann=float(T.ann.std() / abs(T.ann.mean())),
                cv_ann_per_vol=float((T.ann / T.vol).std() / abs((T.ann / T.vol).mean())),
                top10_vol_risk_share=float(T.risk_share.iloc[-10:].sum()),
                bottom10_vol_risk_share=float(T.risk_share.iloc[:10].sum()))

    out = dict(obs=obs, tests=tests, terciles=terc, dispersion=disp,
               summary=dict(pos_ann=int((T.ann > 0).sum()), pos_sharpe=int((T.sharpe > 0).sum()),
                            beat_bh_ann=int((T.ann > T.bh_ann).sum()), busted=[c for c in T.coin[T.busted]],
                            se_ann_med=float(T.se_ann.median()), coins=N,
                            vol_min=float(T.vol.min()), vol_max=float(T.vol.max()), vol_med=float(T.vol.median()),
                            mkt_share=float(T.mkt_bps.sum() / (T.mkt_bps.sum() + T.own_bps.sum()))))
    Path("runs/tsmom_coins.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    T.to_csv("runs/tsmom_coins.csv", index=False)

    pd.set_option("display.width", 220)
    show = T[["coin", "vol", "beta", "ann", "se_ann", "sharpe", "worst", "busted", "bh_ann", "mkt_bps", "own_bps", "risk_share"]].copy()
    for k in ("vol", "ann", "se_ann", "worst", "bh_ann", "risk_share"):
        show[k] = (show[k] * 100).round(1)
    print(show.round(2).to_string(index=False))
    print("obs", {k: round(v, 3) for k, v in obs.items()})
    print("tests", {k: (np.round(v, 3).tolist() if isinstance(v, list) else v if isinstance(v, dict) else round(v, 3))
                    for k, v in tests.items()})
    print("disp", {k: round(v, 3) for k, v in disp.items()})
    print("summary", out["summary"])
    for k, v in terc.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "members"})


if __name__ == "__main__":
    main()
