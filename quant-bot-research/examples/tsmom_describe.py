"""시계열 추세 — 운용 설명서용 기술통계 (47심볼 · 봉인 전). 판정이 아니라 서술이다.

기준 규칙은 `legacy.tsmom_weekly` 와 같다(28일 부호 · 일요일 UTC 종가 · 동일가중 · 변화분 6bps ·
펀딩 0.137bps/8h 롱 지급). 여기서는 그 위에 운용에 필요한 숫자(낙폭, 연도별, 다리 분해, 노출,
회전율, 비중 되맞춤 비용)와 민감도(lookback · 요일 · 실행 지연 · 변동성 타깃 · 비용 · 심볼 묶음)를 낸다.

입력 runs/legacy/daily_close.pkl → runs/tsmom_describe.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import legacy as L  # noqa: E402

LARGE4 = ["BTC", "ETH", "SOL", "XRP"]                 # 47심볼엔 BNB 가 없다
SMALL5 = ["DOGE", "ADA", "LINK", "AVAX", "TRX"]


def book(C: pd.DataFrame, *, lookback: int = 28, anchor: int = 6, delay: int = 0, fee_bps: float = 6.0,
         fund_bps_8h: float = 0.137, vol_target: float | None = None, vol_days: int = 20,
         cap: float = 3.0) -> dict[str, pd.DataFrame]:
    """주별 포지션·수익 표. anchor = 신호 요일(0 월 … 6 일, UTC 일봉 라벨), delay = 실행까지 일수."""
    d = C.sort_index()
    wk = d.index[d.index.dayofweek == anchor]
    sig = np.sign(d / d.shift(lookback) - 1).reindex(wk)
    if vol_target is not None:
        vol = np.log(d).diff().rolling(vol_days, min_periods=vol_days).std() * np.sqrt(365)
        scale = (vol_target / vol).clip(upper=cap).reindex(wk)
        sig = sig * scale
    px = d.shift(-delay).reindex(wk)
    fwd = px.shift(-1) / px - 1
    pos = sig.where(fwd.notna())
    dpos = pos.fillna(0).diff().abs()
    dpos.iloc[0] = pos.iloc[0].abs()
    fund = fund_bps_8h * 3 * 7 / 1e4
    gross = pos * fwd
    fee = dpos * fee_bps / 1e4
    fnd = pos * fund
    n = pos.notna().sum(axis=1)
    keep = n > 0
    return dict(pos=pos[keep], fwd=fwd[keep], gross=gross[keep], fee=fee[keep], fund=fnd[keep],
                ret=(gross - fee - fnd)[keep].mean(axis=1), bh=fwd[keep].mean(axis=1))


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    trough = dd.idxmin()
    pk = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[pk]]
    under = (dd < 0).astype(int)
    runs = under.groupby((under == 0).cumsum()).sum()
    n = len(r)
    return dict(weeks=n, mean=float(r.mean()), sd=float(r.std()), sharpe=float(r.mean() / r.std() * np.sqrt(52)),
                t=float(r.mean() / r.std() * np.sqrt(n)), cagr=float(eq.iloc[-1] ** (52 / n) - 1),
                total=float(eq.iloc[-1] - 1), vol=float(r.std() * np.sqrt(52)), mdd=float(dd.min()),
                mdd_peak=str(pk.date()), mdd_trough=str(trough.date()),
                mdd_recover=str(rec.index[0].date()) if len(rec) else None,
                longest_dd_weeks=int(runs.max()), worst=float(r.min()), worst_date=str(r.idxmin().date()),
                best=float(r.max()), best_date=str(r.idxmax().date()), pos_share=float((r > 0).mean()),
                skew=float(r.skew()))


def by_year(r: pd.Series) -> dict:
    g = r.groupby(r.index.year)
    return {int(y): dict(ret=float((1 + x).prod() - 1), sharpe=float(x.mean() / x.std() * np.sqrt(52)),
                         weeks=int(len(x))) for y, x in g}


def main() -> None:
    C = pd.read_pickle("runs/legacy/daily_close.pkl")
    B = book(C)
    ref = L.tsmom_weekly(C)
    assert np.allclose(B["ret"].to_numpy(), ref["ret"].to_numpy()), "기준 규칙과 불일치"
    r, bh = B["ret"], B["bh"]
    pos, fwd = B["pos"], B["fwd"]
    N = pos.notna().sum(axis=1)
    out: dict = {}
    out["base"] = perf(r)
    out["base_gross"] = perf(B["gross"].mean(axis=1))
    out["bh"] = perf(bh)
    out["years"] = by_year(r)
    out["years_bh"] = by_year(bh)

    # 다리 분해 — 주별 포트폴리오 기여(1/N 가중) · 포지션 평균
    long_leg = B["gross"].where(pos > 0).sum(axis=1) / N
    short_leg = B["gross"].where(pos < 0).sum(axis=1) / N
    out["legs"] = dict(
        long_contrib_bps_wk=float(long_leg.mean() * 1e4), short_contrib_bps_wk=float(short_leg.mean() * 1e4),
        long_pos_avg_bps=float(fwd.where(pos > 0).stack().mean() * 1e4),
        short_pos_avg_bps=float((-fwd).where(pos < 0).stack().mean() * 1e4),
        long_pos_n=int((pos > 0).sum().sum()), short_pos_n=int((pos < 0).sum().sum()))

    # 노출 · 회전율 · 보유 기간
    net = pos.mean(axis=1)
    flips = (np.sign(pos).diff().abs() == 2).sum(axis=1) / N
    runs = []
    for c in pos.columns:
        s = np.sign(pos[c].dropna())
        grp = (s != s.shift()).cumsum()
        runs += s.groupby(grp).size().tolist()
    out["exposure"] = dict(net_mean=float(net.mean()), long_share_mean=float(((pos > 0).sum(axis=1) / N).mean()),
                           abs_net_gt_half=float((net.abs() > 0.5).mean()), all_long=int((net >= 0.999).sum()),
                           all_short=int((net <= -0.999).sum()), flip_share_wk=float(flips.iloc[1:].mean()),
                           hold_weeks_mean=float(np.mean(runs)), hold_weeks_median=float(np.median(runs)),
                           corr_bh=float(r.corr(bh)), beta_bh=float(np.cov(r, bh)[0, 1] / bh.var()),
                           up_weeks_ret=float(r[bh > 0].mean()), down_weeks_ret=float(r[bh < 0].mean()),
                           up_weeks=int((bh > 0).sum()), down_weeks=int((bh < 0).sum()))

    # 비용 — 부과된 것 + 비중 되맞춤(백테스트에 없는 것)
    fee_wk = B["fee"].sum(axis=1) / N
    fund_wk = B["fund"].sum(axis=1) / N
    Rp = B["gross"].sum(axis=1) / N
    drift = pos.shift(1) * (1 + fwd.shift(1))                  # 지난주 포지션이 가격 따라 불어난 명목
    tgt = pos.mul(1 + Rp.shift(1), axis=0)                     # 이번 주 목표 = 부호 × 새 자산 / N
    full = (tgt - drift).abs().div(1 + Rp.shift(1), axis=0).sum(axis=1) / N
    charged = pos.fillna(0).diff().abs().sum(axis=1) / N
    out["costs"] = dict(fee_pct_yr=float(fee_wk.mean() * 52 * 100), fund_pct_yr=float(fund_wk.mean() * 52 * 100),
                        turnover_charged_wk=float(charged.iloc[1:].mean()), turnover_full_wk=float(full.iloc[1:].mean()),
                        drift_extra_cost_pct_yr=float((full - charged).iloc[1:].mean() * 6e-4 * 52 * 100))

    # 민감도 (서술)
    sens = {}
    for lb in (7, 14, 21, 28, 35, 42, 56, 84):
        sens[f"lookback {lb}"] = perf(book(C, lookback=lb)["ret"])
    for a, nm in enumerate("월화수목금토일"):
        sens[f"요일 {nm}"] = perf(book(C, anchor=a)["ret"])
    for dl in (1, 2):
        sens[f"지연 {dl}일"] = perf(book(C, delay=dl)["ret"])
    vt = book(C, vol_target=0.40)
    sens["변동성 타깃 40%"] = perf(vt["ret"]) | dict(avg_gross=float(vt["pos"].abs().mean(axis=1).mean()))
    for fb in (10, 20):
        sens[f"수수료 {fb}bps"] = perf(book(C, fee_bps=fb)["ret"])
    for fu in (1.0, 3.0):
        sens[f"펀딩 {fu}bps/8h"] = perf(book(C, fund_bps_8h=fu)["ret"])
    others = [c for c in C.columns if c not in LARGE4 + SMALL5]
    for nm, cols in (("대형4", LARGE4), ("소형5", SMALL5), ("기존9", LARGE4 + SMALL5), ("새38", others)):
        sens[f"묶음 {nm}"] = perf(book(C[cols])["ret"])
    half = pd.Timestamp("2023-11-01", tz="UTC")
    sens["전반 2021-05~2023-10"] = perf(r[r.index < half])
    sens["후반 2023-11~2026-06"] = perf(r[r.index >= half])
    sens["신호 반전 (대조)"] = perf((-B["gross"] - B["fee"] + B["fund"]).mean(axis=1))
    out["sens"] = sens
    Path("runs/tsmom_describe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    f = lambda x: f"{x:+.3f}"
    print("base", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["base"].items()})
    print("gross", round(out["base_gross"]["sharpe"], 3), round(out["base_gross"]["total"], 3))
    print("bh", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["bh"].items()})
    print("years", {y: (round(v["ret"], 3), round(v["sharpe"], 2)) for y, v in out["years"].items()})
    print("years_bh", {y: round(v["ret"], 3) for y, v in out["years_bh"].items()})
    print("legs", {k: round(v, 2) if isinstance(v, float) else v for k, v in out["legs"].items()})
    print("exposure", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["exposure"].items()})
    print("costs", {k: round(v, 3) for k, v in out["costs"].items()})
    print(f"{'민감도':<22}{'샤프':>7}{'t':>7}{'CAGR':>8}{'MDD':>8}{'연변동':>8}")
    for k, s in sens.items():
        extra = f"  평균 총노출 {s['avg_gross']:.2f}" if "avg_gross" in s else ""
        print(f"{k:<22}{s['sharpe']:>7.2f}{s['t']:>7.2f}{s['cagr'] * 100:>7.1f}%{s['mdd'] * 100:>7.1f}%"
              f"{s['vol'] * 100:>7.1f}%{extra}")


if __name__ == "__main__":
    main()
