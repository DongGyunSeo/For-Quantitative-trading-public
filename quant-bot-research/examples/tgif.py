"""B. TGIF 요일 패턴 실존 검정 — 사전등록 그대로 (`claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`).

하락형 D: 월 혼조 · 화 약상승 · 수 약상승 · 목 큰 하락 · 금 소폭 상승 (U 는 역방향)
H1  준비(X−3 혼조 · X−2 · X−1 약상승) → X 수익(σ). 본 = X 목요일, 위약 = X 다른 요일 6개
H1b 준비 → X 가 직전 3일 저점(U 는 고점)을 깨는 비율
H2  X 큰 하락 + 직전 3일 저점 이탈 → X+1 수익(σ). 본 = X 목요일(→ 금), 위약 = 다른 요일
H3  전체 패턴 빈도 ÷ 요일별 조건 확률의 곱

입력 runs/weekly/days_{ny,utc}.pkl → runs/tgif.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbot.research import weekly as K  # noqa: E402
from qbot.research.yardstick import cluster_t  # noqa: E402
from weekly_candles import cluster_diff  # noqa: E402

SPLIT_DAY = "2024-01-01"
RNG = np.random.default_rng(7)
DOW = "월화수목금토일"


def frame(path: str, tz: str) -> pd.DataFrame:
    """코인별로 달력 연속 일봉 → X 기준 앞뒤 날짜 열."""
    D = pd.read_pickle(path)
    out = []
    for sym, g in D.groupby("symbol", sort=False):
        g = g.set_index("day").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D", tz=tz)
        g = g.reindex(full)
        g.loc[g.n < 250, ["open", "high", "low", "close"]] = np.nan      # 빠진 봉이 많은 날은 버린다
        r = g.close / g.close.shift(1) - 1
        s = K.trailing_sigma(r)
        f = pd.DataFrame(index=full)
        f["symbol"] = sym
        f["dow"] = full.dayofweek
        for k in (-3, -2, -1, 0, 1):
            f[f"r{k}"] = r.shift(-k)
            f[f"lo{k}"] = g.low.shift(-k)
            f[f"hi{k}"] = g.high.shift(-k)
        f["o0"], f["c0"] = g.open, g.close
        f["o1"], f["c1"] = g.open.shift(-1), g.close.shift(-1)
        f["s"] = s.shift(3)                          # σ(X−3): X−3 전날까지 30일 (그 주 월요일 전 30일)
        out.append(f)
    F = pd.concat(out)
    F["day"] = F.index
    F["week"] = (F.index.normalize() - pd.to_timedelta(F.index.dayofweek, unit="D")).strftime("%Y-%m-%d")
    F["half"] = np.where(F.index < pd.Timestamp(SPLIT_DAY, tz=tz), "발견", "확인")
    for k in (-3, -2, -1, 0, 1):
        F[f"z{k}"] = F[f"r{k}"] / F["s"]
    F["low3"] = F[["lo-3", "lo-2", "lo-1"]].min(axis=1, skipna=False)
    F["high3"] = F[["hi-3", "hi-2", "hi-1"]].max(axis=1, skipna=False)
    return F.dropna(subset=["s", "z-3", "z-2", "z-1", "z0"]).reset_index(drop=True)


def cls(z: pd.Series) -> dict[str, pd.Series]:
    return dict(mixed=z.abs() < 0.5, up=(z > 0) & (z <= 1), dn=(z < 0) & (z >= -1), bigdn=z <= -1, bigup=z >= 1)


def test_block(F: pd.DataFrame, cond: pd.Series, y: pd.Series, sign: int) -> dict:
    """본(목요일 X) vs 위약(다른 요일) — 평균·t·차이·반기."""
    main = F.dow == 3
    out = {}
    for nm, m in (("main", cond & main), ("placebo", cond & ~main)):
        a = cluster_t(y[m].to_numpy(), F.week[m].to_numpy())
        out[nm] = dict(mean=a[0], t=a[1], n=a[2], weeks=a[3])
    sel = cond & y.notna()
    d = cluster_diff(y[sel], main[sel].astype(float), F.week[sel])
    out["diff"] = dict(zip(("diff", "t", "n", "G"), d))
    out["halves"] = {}
    for h in ("발견", "확인"):
        m = cond & main & (F.half == h)
        a = cluster_t(y[m].to_numpy(), F.week[m].to_numpy())
        out["halves"][h] = dict(mean=a[0], t=a[1], n=a[2])
    out["uncond_thu"] = float(y[main & y.notna()].mean()) if y.name != "brk" else None
    ok = (np.sign(out["main"]["mean"]) == sign and abs(out["main"]["t"]) >= 2
          and np.sign(out["diff"]["diff"]) == sign and abs(out["diff"]["t"]) >= 2
          and all(np.sign(out["halves"][h]["mean"]) == sign for h in ("발견", "확인")))
    out["passed"] = bool(ok)
    return out


def analyse(F: pd.DataFrame) -> dict:
    c3, c2, c1, c0 = cls(F["z-3"]), cls(F["z-2"]), cls(F["z-1"]), cls(F["z0"])
    res = {}
    for form, up_pre, sign_h1, big, brk_col, brk_cmp, sign_h2 in (
            ("D", "up", -1, "bigdn", "low3", "lt", +1), ("U", "dn", +1, "bigup", "high3", "gt", -1)):
        setup = c3["mixed"] & c2[up_pre] & c1[up_pre]
        y0 = F["z0"].rename("z0")
        h1 = test_block(F, setup, y0, sign_h1)
        brk = (F.lo0 < F.low3) if brk_cmp == "lt" else (F.hi0 > F.high3)
        h1b = {}
        for nm, m in (("main", setup & (F.dow == 3)), ("placebo", setup & (F.dow != 3))):
            a = cluster_t(brk[m].astype(float).to_numpy(), F.week[m].to_numpy())
            h1b[nm] = dict(rate=a[0], n=a[2])
        d = cluster_diff(brk[setup].astype(float), (F.dow[setup] == 3).astype(float), F.week[setup])
        h1b["diff"] = dict(zip(("diff", "t", "n", "G"), d))
        h1b["uncond_thu"] = float(brk[F.dow == 3].mean())
        # 목 크기 서술 (준비 후 목요일)
        m = setup & (F.dow == 3)
        z = F.z0[m]
        h1b["thu_z_q"] = [float(x) for x in z.quantile([0.1, 0.25, 0.5, 0.75, 0.9])]
        h1b["thu_pct_q"] = [float(x) for x in (F.r0[m] * 100).quantile([0.1, 0.25, 0.5, 0.75, 0.9])]
        h1b["thu_big_share"] = float(c0[big][m].mean())
        h1b["thu_big_share_uncond"] = float(c0[big][F.dow == 3].mean())
        # H2
        trig = c0[big] & brk
        y1 = F["z1"].rename("z1")
        h2 = test_block(F, trig & F.z1.notna(), y1, sign_h2)
        # 거래 확인 (보조): 준비 → X 시가 진입·종가 청산 / 트리거 → X+1 시가·종가, 왕복 12bps
        tr = {}
        r_x = F.c0 / F.o0 - 1
        r_x1 = F.c1 / F.o1 - 1
        for nm, m_, rr, sd in (("목 거래", setup & (F.dow == 3), r_x, sign_h1), ("금 거래", trig & (F.dow == 3), r_x1, sign_h2)):
            x = (sd * rr[m_] - 12e-4).dropna()
            a = cluster_t(x.to_numpy(), F.week[x.index].to_numpy())
            tr[nm] = dict(mean_pct=a[0] * 100, t=a[1], n=a[2], weeks=a[3], win=float((x > 0).mean()))
        res[form] = dict(h1=h1, h1b=h1b, h2=h2, trades=tr)
    return res


def full_pattern(F: pd.DataFrame) -> dict:
    """주(월~금) 단위 전체 패턴 빈도 vs 독립 기대 — 주 블록 부트스트랩."""
    thu = F[F.dow == 3].copy()                       # X = 목: z-3 월, z-2 화, z-1 수, z0 목, z1 금
    thu = thu.dropna(subset=["z1"])
    c3, c2, c1, c0, cf = cls(thu["z-3"]), cls(thu["z-2"]), cls(thu["z-1"]), cls(thu["z0"]), cls(thu["z1"])
    out = {}
    for form, pre, big, post in (("D", "up", "bigdn", "up"), ("U", "dn", "bigup", "dn")):
        parts = [c3["mixed"], c2[pre], c1[pre], c0[big], cf[post]]
        full = parts[0] & parts[1] & parts[2] & parts[3] & parts[4]
        weeks = thu.week.to_numpy()
        uw = np.unique(weeks)
        idx = {w: np.flatnonzero(weeks == w) for w in uw}
        P = np.column_stack([p.to_numpy(float) for p in parts])
        fv = full.to_numpy(float)

        def lift(rows):
            obs = fv[rows].mean()
            exp = np.prod(P[rows].mean(axis=0))
            return obs, exp
        obs, exp = lift(np.arange(len(thu)))
        bs = []
        for _ in range(2000):
            pick = RNG.choice(uw, size=len(uw), replace=True)
            rows = np.concatenate([idx[w] for w in pick])
            o, e = lift(rows)
            bs.append(o / e if e > 0 else np.nan)
        out[form] = dict(n_weeks_coins=int(len(thu)), obs=float(obs), exp=float(exp), count=int(full.sum()),
                         lift=float(obs / exp), lift_ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                         distinct_weeks=int(thu.week[full].nunique()))
    return out


def main() -> None:
    res = {}
    for tz, path in (("NY", "runs/weekly/days_ny.pkl"), ("UTC", "runs/weekly/days_utc.pkl")):
        F = frame(path, K.ET if tz == "NY" else "UTC")
        res[tz] = dict(analysis=analyse(F), full=full_pattern(F), rows=int(len(F)))
        # 무조건 요일별 평균(σ 단위) — 서술
        res[tz]["dow_mean_z"] = {DOW[d]: float(F.z0[F.dow == d].mean()) for d in range(7)}
    Path("runs/tgif.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for tz in ("NY", "UTC"):
        print(f"===== {tz} 일봉 · 행 {res[tz]['rows']:,}")
        print("요일별 평균 z:", {k: round(v, 3) for k, v in res[tz]["dow_mean_z"].items()})
        for form, a in res[tz]["analysis"].items():
            h1, h1b, h2, tr = a["h1"], a["h1b"], a["h2"], a["trades"]
            print(f"[{form}] H1 목 z 평균 {h1['main']['mean']:+.3f} (t {h1['main']['t']:+.2f}, n {h1['main']['n']}, 주 {h1['main']['weeks']}) "
                  f"| 위약 {h1['placebo']['mean']:+.3f} (n {h1['placebo']['n']}) | 차이 {h1['diff']['diff']:+.3f} t {h1['diff']['t']:+.2f} "
                  f"| 반기 {h1['halves']['발견']['mean']:+.3f}/{h1['halves']['확인']['mean']:+.3f} | 무조건 목 {h1['uncond_thu']:+.3f} → {'통과' if h1['passed'] else '기각'}")
            print(f"     H1b 목 이탈률 {h1b['main']['rate'] * 100:.1f}% vs 위약 {h1b['placebo']['rate'] * 100:.1f}% (차 t {h1b['diff']['t']:+.2f}) "
                  f"무조건 목 {h1b['uncond_thu'] * 100:.1f}% · 목 z 분위 {np.round(h1b['thu_z_q'], 2).tolist()} · % 분위 {np.round(h1b['thu_pct_q'], 2).tolist()} "
                  f"· 큰 움직임 비율 {h1b['thu_big_share'] * 100:.1f}% (무조건 {h1b['thu_big_share_uncond'] * 100:.1f}%)")
            print(f"     H2 금 z 평균 {h2['main']['mean']:+.3f} (t {h2['main']['t']:+.2f}, n {h2['main']['n']}, 주 {h2['main']['weeks']}) "
                  f"| 위약 {h2['placebo']['mean']:+.3f} (n {h2['placebo']['n']}) | 차이 {h2['diff']['diff']:+.3f} t {h2['diff']['t']:+.2f} "
                  f"| 반기 {h2['halves']['발견']['mean']:+.3f}/{h2['halves']['확인']['mean']:+.3f} → {'통과' if h2['passed'] else '기각'}")
            print("     거래:", "; ".join("%s %+.3f%% t %+.2f n %d 승 %.0f%%" % (k, v['mean_pct'], v['t'], v['n'], v['win'] * 100) for k, v in tr.items()))
        for form, v in res[tz]["full"].items():
            print(f"[{form}] 전체 패턴: {v['count']}건 ({v['distinct_weeks']}주) 관측 {v['obs'] * 100:.3f}% · 독립 기대 {v['exp'] * 100:.3f}% · "
                  f"배수 {v['lift']:.2f} [{v['lift_ci'][0]:.2f}, {v['lift_ci'][1]:.2f}]")


if __name__ == "__main__":
    main()
