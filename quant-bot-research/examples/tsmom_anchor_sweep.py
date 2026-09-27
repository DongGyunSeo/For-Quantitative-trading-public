"""시계열 추세 — 체결 시각 전수(주 168시간) 서술. 사전등록 아님: '금 장마감 체결이 더 낫나' 에 답하기 위한 잡음 지도.

규칙은 현행 그대로(28일 부호 · 동일가중 롱·숏 · 1주 보유 · 6bps · 펀딩), **신호 = 체결 시각**만 주 168개 시각으로 돌린다.
각 시각마다 2021-04-30 00:00 UTC 이후 첫 269주. ET 격자(금 16:00 ET = V1)와 UTC 격자(월 00:00 UTC = 현행 V0).
입력 runs/tsmom_anchor/hourly_close.pkl · runs/tsmom_gap_weekly.csv → runs/tsmom_anchor.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
START = pd.Timestamp("2021-04-30 00:00", tz="UTC")
NW = 269
FEE, FUND_WK = 6e-4, 0.137 * 3 * 7 / 1e4
HALF = pd.Timestamp("2023-11-01", tz="UTC")
DAYS = "월화수목금토일"
RNG = np.random.default_rng(20260925)


def anchor_times(H: pd.DataFrame, a: int, tz: str) -> pd.DatetimeIndex:
    loc = H.index.tz_convert(tz)
    how = loc.dayofweek * 24 + loc.hour
    sel = H.index[how == a]
    d = sel.tz_convert(tz).tz_localize(None).normalize()
    return sel[~pd.Index(d).duplicated(keep="first")]          # 가을 DST 로 같은 시각이 두 번이면 첫 번째


def book(H: pd.DataFrame, times: pd.DatetimeIndex) -> pd.Series:
    P = H.loc[times]
    t = pd.Series(times, index=times)
    lo, hi = pd.Timedelta(hours=1), pd.Timedelta(days=7)
    ok4 = ((t - t.shift(4)) - 4 * hi).abs() <= lo              # 4주 전(DST 로 ±1시간)
    okn = ((t.shift(-1) - t) - hi).abs() <= lo
    sig = np.sign(P / P.shift(4) - 1)
    sig.loc[~ok4.to_numpy()] = np.nan
    fwd = P.shift(-1) / P - 1
    fwd.loc[~okn.to_numpy()] = np.nan
    keep = times[times >= START][:NW]
    sig, fwd = sig.loc[keep], fwd.loc[keep]
    pos = sig.where(fwd.notna())
    dpos = pos.fillna(0).diff().abs()
    dpos.iloc[0] = pos.iloc[0].abs()
    r = (pos * fwd - dpos * FEE - pos * FUND_WK).mean(axis=1)
    return r[pos.notna().sum(axis=1) > 0]


def perf(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(52)), weeks=int(len(r)),
                cagr=float(eq.iloc[-1] ** (52 / len(r)) - 1), mdd=float((eq / eq.cummax() - 1).min()),
                first=float(r[r.index < HALF].mean() / r[r.index < HALF].std() * np.sqrt(52)),
                second=float(r[r.index >= HALF].mean() / r[r.index >= HALF].std() * np.sqrt(52)))


def block_boot_t(d: np.ndarray, block: int = 4, reps: int = 5000):
    n = len(d)
    k = int(np.ceil(n / block))
    means = np.empty(reps)
    for b in range(reps):
        st = RNG.integers(0, n - block + 1, size=k)
        means[b] = d[(st[:, None] + np.arange(block)[None, :]).ravel()[:n]].mean()
    se = means.std(ddof=1)
    return float(d.mean() / se), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def tranches(H: pd.DataFrame) -> pd.DataFrame:
    """요일마다 00:00 UTC 에 자본의 1/7 씩 리밸런스하는 7개 묶음 — 묶음별 일 손익(리밸런스 시 자본 대비, 주중 보유)."""
    D = H[H.index.hour == 0]
    assert (D.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    sig = np.sign(D / D.shift(28) - 1)
    dP = D.shift(-1) - D
    out = {}
    for k in range(7):
        m = pd.Series((D.index.dayofweek == k) & (D.index >= START), index=D.index)
        s_k = sig.where(m, axis=0)
        pos = s_k.ffill()
        P0 = D.where(m, axis=0).ffill()
        prev = s_k[m].shift(1).fillna(0)
        fee = ((s_k[m] - prev).abs() * FEE).mean(axis=1).reindex(D.index).fillna(0)
        out[k] = (pos * dP / P0).mean(axis=1) - fee - (pos * FUND_WK / 7).mean(axis=1)
    return pd.DataFrame(out)


def lab(a: int) -> str:
    return f"{DAYS[a // 24]} {a % 24:02d}:00"


def main() -> None:
    H = pd.read_pickle("runs/tsmom_anchor/hourly_close.pkl")
    wk = pd.read_csv("runs/tsmom_gap_weekly.csv", index_col=0, parse_dates=True)
    out = {}
    for tz, name in ((ET, "et"), ("UTC", "utc")):
        rows, rets = [], {}
        for a in range(168):
            r = book(H, anchor_times(H, a, tz))
            rets[a] = r
            rows.append(dict(a=a, label=lab(a), **perf(r)))
        out[name] = rows
        if name == "et":
            r1 = rets[4 * 24 + 16]                                  # 금 16:00 ET = V1
            m = r1.index.tz_convert(ET).tz_localize(None).normalize() + pd.Timedelta(days=3)
            assert len(r1) == NW and np.allclose(r1.to_numpy(), wk["V1"].reindex(m).to_numpy(), atol=1e-12), "V1 재현 실패"
        else:
            r0 = rets[0]                                            # 월 00:00 UTC = V0
            m = r0.index.tz_localize(None)
            assert len(r0) == NW and np.allclose(r0.to_numpy(), wk["V0"].reindex(m).to_numpy(), atol=1e-12), "V0 재현 실패"
    print("V0 · V1 주간 수익 정확히 재현")

    # ---- 지형 요약
    def summary(rows):
        s = np.array([x["sharpe"] for x in rows])
        shift = {h: float(np.mean(np.abs(s - np.roll(s, -h)))) for h in (1, 3, 4, 6, 12, 24)}
        i, j = int(s.argmin()), int(s.argmax())
        return dict(mean=float(s.mean()), sd=float(s.std(ddof=1)), min=float(s[i]), min_at=rows[i]["label"],
                    max=float(s[j]), max_at=rows[j]["label"], q10=float(np.percentile(s, 10)), q90=float(np.percentile(s, 90)),
                    shift_abs=shift, by_day={DAYS[d]: float(s[d * 24:(d + 1) * 24].mean()) for d in range(7)})
    out["et_summary"], out["utc_summary"] = summary(out["et"]), summary(out["utc"])
    s_et = np.array([x["sharpe"] for x in out["et"]])
    v1 = out["et"][4 * 24 + 16]["sharpe"]
    out["v1_rank"] = dict(sharpe=v1, higher=int((s_et > v1).sum()), pct=float((s_et < v1).mean()))
    v0 = out["utc"][0]["sharpe"]
    out["v0"] = dict(sharpe=v0, et_share_above=float((s_et > v0).mean()))
    out["fri_zoom"] = [dict(label=x["label"], sharpe=x["sharpe"]) for x in out["et"][4 * 24 + 10: 5 * 24 + 7]]

    # ---- V1 vs V0 (공통 269주, 같은 주 월요일끼리 짝)
    d = (wk["V1"] - wk["V0"]).dropna()
    tb, ci = block_boot_t(d.to_numpy())
    yrs = {c: {int(y): float((1 + x).prod() - 1) for y, x in wk[c].groupby(wk.index.year)} for c in ("V0", "V1")}
    ex21 = d[d.index.year >= 2022]
    tb21, ci21 = block_boot_t(ex21.to_numpy())
    comp = {c: float((1 + wk[c][wk.index.year >= 2022]).prod()) for c in ("V0", "V1")}
    out["v1_vs_v0"] = dict(d_bps=float(d.mean() * 1e4), t=tb, ci_bps=[x * 1e4 for x in ci], weeks=int(len(d)),
                           weeks_for_t2=float(len(d) * (2 / tb) ** 2), years_for_t2=float(len(d) * (2 / tb) ** 2 / 52.18),
                           ex2021_d_bps=float(ex21.mean() * 1e4), ex2021_t=tb21, ex2021_ci_bps=[x * 1e4 for x in ci21],
                           mult_2022_on=comp, years=yrs,
                           halves={c: [float(wk[c][wk.index < HALF.tz_localize(None)].pipe(lambda x: x.mean() / x.std() * np.sqrt(52))),
                                       float(wk[c][wk.index >= HALF.tz_localize(None)].pipe(lambda x: x.mean() / x.std() * np.sqrt(52)))]
                                   for c in ("V0", "V1")},
                           win_share=float((d > 0).mean()))
    # ---- 나눠 리밸런스 (서술): 7개 요일 묶음, 달력 주(월 00:00 UTC ~) 로 합산
    T = tranches(H)
    T = T[(T.index >= pd.Timestamp("2021-05-10", tz="UTC")) & (T.index < pd.Timestamp("2026-06-29", tz="UTC"))]
    wkT = T.groupby(T.index.tz_localize(None).to_period("W-SUN")).sum()
    wkT.index = wkT.index.start_time
    assert np.allclose(wkT[0].to_numpy(), wk["V0"].reindex(wkT.index).to_numpy(), atol=1e-12), "월 묶음 ≠ V0"
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(52))
    mdd = lambda x: float(((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())
    comb = wkT.mean(axis=1)
    cc = wkT.corr().to_numpy()
    out["tranche"] = dict(weeks=int(len(wkT)), single={DAYS[k]: dict(sharpe=sh(wkT[k]), mdd=mdd(wkT[k])) for k in range(7)},
                          combined=dict(sharpe=sh(comb), mdd=mdd(comb), cagr=float((1 + comb).prod() ** (52 / len(comb)) - 1)),
                          single_mean=dict(sharpe=float(np.mean([sh(wkT[k]) for k in range(7)])),
                                           mdd=float(np.mean([mdd(wkT[k]) for k in range(7)]))),
                          corr_mean=float(cc[np.triu_indices(7, 1)].mean()))

    # ---- 요일 수준 대비 (사후): 수~금 시각들 vs 토~월 시각들, ET 벽시계 주 번호로 맞춘 주간 수익
    start_wall = START.tz_convert(ET).tz_localize(None)
    cols = {}
    for a in range(168):
        if a == 6 * 24 + 2:                                        # 일 02:00 ET — 봄 DST 로 해마다 한 번 없는 시각
            continue
        r = book(H, anchor_times(H, a, ET))
        wall = r.index.tz_convert(ET).tz_localize(None)
        cols[a] = pd.Series(r.to_numpy(), index=((wall - start_wall) // pd.Timedelta(days=7)).astype(int))
    A = pd.DataFrame(cols).dropna()
    day = np.array([a // 24 for a in A.columns])
    cst = (A.loc[:, (day >= 2) & (day <= 4)].mean(axis=1) - A.loc[:, (day >= 5) | (day == 0)].mean(axis=1)).to_numpy()
    tc, cic = block_boot_t(cst)
    h2 = len(cst) // 2
    f1 = np.array([x["first"] for x in out["et"]])
    f2 = np.array([x["second"] for x in out["et"]])
    out["midweek"] = dict(d_bps=float(cst.mean() * 1e4), t=tc, first_bps=float(cst[:h2].mean() * 1e4),
                          second_bps=float(cst[h2:].mean() * 1e4), halves_corr=float(np.corrcoef(f1, f2)[0, 1]),
                          by_day_first={DAYS[d]: float(f1[d * 24:(d + 1) * 24].mean()) for d in range(7)},
                          by_day_second={DAYS[d]: float(f2[d * 24:(d + 1) * 24].mean()) for d in range(7)})
    Path("runs/tsmom_anchor.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for k in ("et_summary", "utc_summary"):
        s = out[k]
        print(k, f"평균 {s['mean']:.2f} SD {s['sd']:.2f} 10~90% {s['q10']:.2f}~{s['q90']:.2f} 최저 {s['min']:.2f}({s['min_at']}) "
                 f"최고 {s['max']:.2f}({s['max_at']})", {h: round(v, 3) for h, v in s["shift_abs"].items()},
              {d: round(v, 2) for d, v in s["by_day"].items()})
    print("V1 금 16:00 ET", round(v1, 3), "더 높은 시각", out["v1_rank"]["higher"], "/168", "V0", round(v0, 3),
          "ET 격자 중 V0 보다 높은 비율", round(out["v0"]["et_share_above"], 2))
    print("금 10:00 ~ 토 06:00 ET", [(x["label"], round(x["sharpe"], 2)) for x in out["fri_zoom"]])
    v = out["v1_vs_v0"]
    print(f"V1−V0 {v['d_bps']:+.1f}bps/주 t {v['t']:+.2f} CI [{v['ci_bps'][0]:+.0f},{v['ci_bps'][1]:+.0f}] 이긴 주 {v['win_share'] * 100:.0f}% "
          f"| t2 까지 {v['weeks_for_t2']:.0f}주 = {v['years_for_t2']:.0f}년 | 2021 제외 {v['ex2021_d_bps']:+.1f} t {v['ex2021_t']:+.2f} "
          f"CI [{v['ex2021_ci_bps'][0]:+.0f},{v['ex2021_ci_bps'][1]:+.0f}] | 2022~ 배수 {v['mult_2022_on']} | 반기 {v['halves']}")
    print("years", {c: {y: round(x * 100, 1) for y, x in r.items()} for c, r in v["years"].items()})
    t = out["tranche"]
    print("나눠 리밸런스", t["weeks"], "주 | 단독", {d: (round(x["sharpe"], 2), round(x["mdd"] * 100)) for d, x in t["single"].items()},
          "| 7개 나눔 샤프", round(t["combined"]["sharpe"], 2), "MDD", round(t["combined"]["mdd"] * 100), "CAGR", round(t["combined"]["cagr"] * 100, 1),
          "| 단독 평균", round(t["single_mean"]["sharpe"], 2), round(t["single_mean"]["mdd"] * 100), "| 상관", round(t["corr_mean"], 2))
    m = out["midweek"]
    print(f"수~금 − 토~월 {m['d_bps']:+.1f}bps/주 t {m['t']:+.2f} (전반 {m['first_bps']:+.1f} 후반 {m['second_bps']:+.1f}) 반기 지형 상관 {m['halves_corr']:.2f}")


if __name__ == "__main__":
    main()
