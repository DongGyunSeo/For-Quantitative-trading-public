"""시계열 추세 수익 분해 — '다 같이 움직이는 부분' 과 '코인별 부분' (서술, 판정 아님).

매주 t, 코인 i 의 포지션 s_i(±1), 다음 주 수익 r_i, 코인 수 N 에 대해

    m      = (1/N) Σ r_i                 시장(동일가중) 수익 = 매수보유
    s̄      = (1/N) Σ s_i                 순노출
    전략 gross = (1/N) Σ s_i r_i
               = s̄ · m                                   ① 시장 방향: 순노출 × 시장
               + (1/N) Σ (s_i − s̄)(r_i − m)              ② 코인 선별: 평균보다 더 롱인 코인이 평균보다 더 올랐나

두 항의 합은 gross 와 **정확히** 같다. 비용(수수료 · 펀딩)은 따로 뺀다.
보조: ①을 코인별 베타(직전 90일 일간 수익, 인과)로 보정한 판, 정적(평균 기울기)·동적(시기) 분할.

입력 runs/legacy/daily_close.pkl → runs/tsmom_decomp.json · runs/tsmom_decomp_weekly.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_describe import book  # noqa: E402


def comp_stats(x: pd.Series, total_mean: float | None = None) -> dict:
    x = x.dropna()
    sd = x.std()
    out = dict(mean_bps=float(x.mean() * 1e4), ann_pct=float(x.mean() * 52 * 100), sd_bps=float(sd * 1e4),
               sharpe=float(x.mean() / sd * np.sqrt(52)) if sd > 0 else np.nan,
               t=float(x.mean() / sd * np.sqrt(len(x))) if sd > 0 else np.nan, sum_pct=float(x.sum() * 100))
    if total_mean:
        out["share"] = float(x.mean() / total_mean)
    return out


def trailing_beta(C: pd.DataFrame, days: int = 90, min_days: int = 30) -> pd.DataFrame:
    """코인별 베타 — 직전 ``days`` 일 일간 수익을 동일가중 시장(자기 제외)에 회귀. 그날 종가까지(인과)."""
    r = C.pct_change()
    N = r.notna().sum(axis=1)
    tot = r.sum(axis=1)
    out = {}
    for c in r.columns:
        mkt = (tot - r[c].fillna(0)) / (N - r[c].notna().astype(int)).clip(lower=1)
        cov = r[c].rolling(days, min_periods=min_days).cov(mkt)
        var = mkt.rolling(days, min_periods=min_days).var()
        out[c] = cov / var
    return pd.DataFrame(out)


def main() -> None:
    C = pd.read_pickle("runs/legacy/daily_close.pkl")
    B = book(C)
    pos, fwd = B["pos"], B["fwd"]
    N = pos.notna().sum(axis=1)
    m = fwd.mean(axis=1)
    sbar = pos.mean(axis=1)
    gross = (pos * fwd).mean(axis=1)
    timing = sbar * m
    select = gross - timing
    alt = ((pos.sub(sbar, axis=0)) * (fwd.sub(m, axis=0))).mean(axis=1)
    assert np.allclose(select, alt), "분해 항등식 불일치"
    fee = B["fee"].mean(axis=1)
    fund = B["fund"].mean(axis=1)
    net = gross - fee - fund
    assert np.allclose(net, B["ret"])

    # 베타 보정: 시장 항 = (1/N) Σ s_i β_i · m
    beta = trailing_beta(C).reindex(pos.index)
    beta = beta.where(beta.notna(), 1.0)
    timing_b = (pos * beta).mean(axis=1) * m
    idio_b = gross - timing_b

    # 정적 / 동적
    t_static = float(sbar.mean() * m.mean())
    dev_pos = pos.sub(sbar, axis=0)
    dev_ret = fwd.sub(m, axis=0)
    s_static = float((dev_pos.mean() * dev_ret.mean()).sum() / N.mean())

    tm = float(gross.mean())
    out = dict(
        weeks=int(len(gross)), coins=int(N.iloc[0]),
        components=dict(gross=comp_stats(gross, tm), timing=comp_stats(timing, tm), select=comp_stats(select, tm),
                        fee=comp_stats(-fee, tm), fund=comp_stats(-fund, tm), net=comp_stats(net, tm),
                        timing_beta=comp_stats(timing_b, tm), idio_beta=comp_stats(idio_b, tm), market=comp_stats(m)),
        static_dynamic=dict(timing_static_bps=t_static * 1e4, timing_dynamic_bps=(timing.mean() - t_static) * 1e4,
                            select_static_bps=s_static * 1e4, select_dynamic_bps=(select.mean() - s_static) * 1e4,
                            sbar_mean=float(sbar.mean()), m_mean_bps=float(m.mean() * 1e4)),
        variance=dict(var_gross=float(gross.var()), var_timing=float(timing.var()), var_select=float(select.var()),
                      cov2=float(2 * np.cov(timing, select)[0, 1]),
                      corr_gross_timing=float(gross.corr(timing)), corr_gross_select=float(gross.corr(select))),
    )

    # 연도별
    yr = {}
    for y, idx in gross.groupby(gross.index.year).groups.items():
        yr[int(y)] = dict(weeks=int(len(idx)), timing_pct=float(timing[idx].sum() * 100),
                          select_pct=float(select[idx].sum() * 100), cost_pct=float(-(fee + fund)[idx].sum() * 100),
                          net_pct=float(net[idx].sum() * 100), market_pct=float(m[idx].sum() * 100),
                          sbar=float(sbar[idx].mean()))
    out["years"] = yr

    # 순노출 구간별 분해
    reg = {}
    for nm, msk in (("순숏 (≤ −0.5)", sbar <= -0.5), ("중간", (sbar > -0.5) & (sbar < 0.5)), ("순롱 (≥ +0.5)", sbar >= 0.5)):
        reg[nm] = dict(weeks=int(msk.sum()), timing_bps=float(timing[msk].mean() * 1e4),
                       select_bps=float(select[msk].mean() * 1e4), gross_bps=float(gross[msk].mean() * 1e4))
    out["regimes"] = reg

    # 동조성 — 코인이 얼마나 같이 움직이나
    Rw = fwd
    corr = Rw.corr().to_numpy()
    iu = np.triu_indices_from(corr, 1)
    loo = {c: (m * N - Rw[c]) / (N - 1) for c in Rw.columns}
    r2 = {c: float(Rw[c].corr(loo[c]) ** 2) for c in Rw.columns}
    Z = (Rw - Rw.mean()) / Rw.std()
    ev = np.linalg.eigvalsh(np.cov(Z.dropna().to_numpy().T))[::-1]
    breadth = (Rw > 0).mean(axis=1)
    dr = C.pct_change().iloc[1:]
    dcorr = dr.corr().to_numpy()
    yearly_corr = {}
    for y, idx in Rw.groupby(Rw.index.year).groups.items():
        cc = Rw.loc[idx].corr().to_numpy()
        yearly_corr[int(y)] = float(np.nanmean(cc[np.triu_indices_from(cc, 1)]))
    idx_level = (1 + dr.mean(axis=1)).cumprod()
    isig = np.sign(idx_level / idx_level.shift(28) - 1).reindex(sbar.index)
    out["comove"] = dict(
        pair_corr_weekly=float(np.nanmean(corr[iu])), pair_corr_daily=float(np.nanmean(dcorr[iu])),
        r2_mean=float(np.mean(list(r2.values()))), r2_min=min(r2.values()), r2_max=max(r2.values()),
        r2_min_coin=min(r2, key=r2.get), r2_max_coin=max(r2, key=r2.get),
        pc1_share=float(ev[0] / ev.sum()),
        breadth_ge80=float((breadth >= 0.8).mean()), breadth_le20=float((breadth <= 0.2).mean()),
        breadth_hist=[int(x) for x in np.histogram(breadth, bins=np.linspace(0, 1, 11))[0]],
        yearly_pair_corr=yearly_corr,
        signal_agree_index=float((np.sign(sbar) == isig)[sbar != 0].mean()),
        abs_sbar_gt_half=float((sbar.abs() > 0.5).mean()), abs_sbar_ge_08=float((sbar.abs() >= 0.8).mean()),
        abs_sbar_ge_06=float((sbar.abs() >= 0.6 - 1e-9).mean()),          # 신호의 80% 이상이 한쪽
    )
    Path("runs/tsmom_decomp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(dict(gross=gross, timing=timing, select=select, fee=fee, fund=fund, net=net, market=m, sbar=sbar,
                      timing_beta=timing_b, idio_beta=idio_b, breadth=breadth)).to_csv("runs/tsmom_decomp_weekly.csv")

    print("주", out["weeks"], "코인", out["coins"])
    for k, v in out["components"].items():
        print(f"{k:<12} 주평균 {v['mean_bps']:+7.1f}bps  연 {v['ann_pct']:+6.1f}%  샤프 {v['sharpe']:+5.2f}  t {v['t']:+5.2f}  "
              f"합 {v['sum_pct']:+7.1f}%  몫 {v.get('share', float('nan')):+.2f}")
    print("static/dynamic", {k: round(v, 3) for k, v in out["static_dynamic"].items()})
    print("variance", {k: round(v, 5) for k, v in out["variance"].items()})
    print("years", {y: {k: round(v, 1) if isinstance(v, float) else v for k, v in d.items()} for y, d in yr.items()})
    print("regimes", {k: {a: round(b, 1) if isinstance(b, float) else b for a, b in d.items()} for k, d in reg.items()})
    print("comove", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["comove"].items()})


if __name__ == "__main__":
    main()
