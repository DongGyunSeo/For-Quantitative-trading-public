"""47심볼 게이트 안정성 보고 — 10심볼 결과가 표본 탓이었는지 가른다.

사전 등록 (47심볼 결과를 보기 전에 적음, 2026-09-25)
-----------------------------------------------------
데이터: OKX 47심볼 5분봉 2021-04-02 ~ 2026-07-05 00:00. 이후 81일은 봉인.
셋업·게이트·비용은 10심볼 검사와 **글자 하나 다르지 않다.**

판정 질문
  Q1 새 심볼 재현   기존 10심볼에 없던 38심볼(독립 횡단면)에서 4h 게이트150 순R > 0, 월 t ≥ 2
  Q2 연도별 부호    47심볼 4h 게이트150 순R 이 2021~2026H1 모든 해에서 같은 부호
  Q3 TF 일관        4h↔1h 연도별 순R 상관 (10심볼: −0.93). 양수로 돌아서면 표본 잡음이었다는 뜻
  Q4 홀드아웃       마지막 12개월(2025-07-05~2026-07-05) 순R > 0

"안정적" 판정은 Q1·Q2·Q4 가 **모두** 예일 때만. 하나라도 아니면 150bps 관측은 기각 유지.
보조 절단(판정에 쓰지 않음): 유동성 3분위, 롱/숏 × 연도.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.yardstick import tstat

GATES = (100.0, 125.0, 150.0, 175.0, 200.0)
ORIG = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "TRX"]  # 기존 10 − BNB
MIN_YEAR_N = 15
MD: list[str] = []


def md(s: str = "") -> None:
    MD.append(s)


def f4(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{x:+.3f}"


def summary_rows(g: pd.DataFrame, cc: pd.DataFrame) -> list[dict]:
    out = []
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        if len(s) < 30:
            continue
        mm = s.groupby("month").net.mean()
        _, tv, nm = tstat(mm)
        cs = cc[cc.gate == gt]
        ct = np.average(cs.ctrl, weights=cs.n) if len(cs) else np.nan
        out.append(dict(gate=int(gt), n=len(s), r_med=float(s.risk_bps.median()),
                        win=float((s.gross > 0).mean()), gross=float(s.gross.mean()),
                        ctrl=float(ct), fee=float(s.fee.mean()), net=float(s.net.mean()),
                        net_m=float(mm.mean()), t=float(tv), months=int(nm),
                        pos_m=float((mm > 0).mean())))
    return out


def year_table(g: pd.DataFrame, col: str = "net") -> tuple[pd.DataFrame, pd.DataFrame]:
    yrs = sorted(g.year.unique())
    val = pd.DataFrame(index=[int(x) for x in GATES], columns=yrs, dtype=float)
    cnt = pd.DataFrame(index=[int(x) for x in GATES], columns=yrs, dtype=float)
    for gt in GATES:
        s = g[g.risk_bps >= gt]
        for y in yrs:
            ys = s[s.year == y]
            cnt.loc[int(gt), y] = len(ys)
            val.loc[int(gt), y] = ys[col].mean() if len(ys) >= MIN_YEAR_N else np.nan
    return val, cnt


def md_summary(title: str, rows: list[dict]) -> None:
    md(f"#### {title}")
    md()
    md("| 게이트 | 거래수 | 1R중앙 | 승률 | gross | 치환대조 | 비용 | 순R | 순R(월평균) | t(월) | 양수월 |")
    md("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md(f"| {r['gate']} | {r['n']:,} | {r['r_med']:.0f} | {r['win']:.1%} | {r['gross']:+.4f} | "
           f"{r['ctrl']:+.4f} | {r['fee']:.4f} | **{r['net']:+.4f}** | {r['net_m']:+.4f} | "
           f"{r['t']:.2f} | {r['pos_m']:.1%} |")
    md()


def md_years(title: str, val: pd.DataFrame, cnt: pd.DataFrame) -> None:
    yrs = list(val.columns)
    md(f"#### {title}")
    md()
    md("| 게이트 | " + " | ".join(str(y) + ("H1" if y == 2026 else "") for y in yrs) + " | 음수 해 |")
    md("|---|" + "---:|" * (len(yrs) + 1))
    for gt in val.index:
        cells = []
        neg = 0
        for y in yrs:
            x, n = val.loc[gt, y], int(cnt.loc[gt, y])
            if np.isfinite(x):
                neg += x < 0
                cells.append((f"**{x:+.3f}**" if x < 0 else f"{x:+.3f}") + f" ({n})")
            else:
                cells.append(f"— ({n})")
        md(f"| {gt} | " + " | ".join(cells) + f" | {neg} |")
    md()


def corr_years(a: pd.Series, b: pd.Series) -> float:
    k = a.notna() & b.notna()
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() >= 3 else np.nan


def main() -> None:
    t = pd.read_csv("runs/u47_trades.csv")
    c = pd.read_csv("runs/u47_ctrl.csv")
    info = pd.read_csv("runs/u47_info.csv")
    syms = sorted(t.symbol.unique())
    new = [s for s in syms if s not in ORIG]
    q = info.set_index("symbol").med_quote_vol.rank(method="first")
    terc = pd.qcut(q, 3, labels=["얇음", "중간", "두꺼움"])
    t["liq"] = t.symbol.map(terc)
    t["cohort"] = np.where(t.symbol.isin(ORIG), "기존9", "신규38")
    key: dict = dict(symbols=len(syms), new=len(new), orig=len(ORIG),
                     trades=len(t), start=str(info.start.min()), end=str(info.end.max()))

    cohorts = {"47심볼 전체": t, "신규 38심볼 (독립 재현)": t[t.cohort == "신규38"],
               "기존 9심볼": t[t.cohort == "기존9"]}

    md("## 1. 게이트별 요약")
    md()
    for tf in ("4h", "1h"):
        for name, sub in cohorts.items():
            g = sub[sub.tf == tf]
            cc = c[(c.tf == tf) & c.symbol.isin(g.symbol.unique())]
            rows = summary_rows(g, cc)
            key[f"summary|{tf}|{name}"] = rows
            md_summary(f"{tf} · {name}", rows)

    md("## 2. 연도별 순R — 판정 축")
    md()
    ycache = {}
    for tf in ("4h", "1h"):
        for name, sub in cohorts.items():
            val, cnt = year_table(sub[sub.tf == tf])
            ycache[(tf, name)] = val
            key[f"years|{tf}|{name}"] = {str(k): {str(y): v for y, v in r.items()}
                                         for k, r in val.to_dict("index").items()}
            md_years(f"{tf} · {name}  (괄호 = 거래 수, 굵게 = 음수)", val, cnt)

    md("## 3. TF 간 일관성 — 4h↔1h 연도별 순R 상관")
    md()
    md("| 게이트 | 47심볼 | 신규38 | 기존9 |")
    md("|---|---:|---:|---:|")
    for gt in (int(x) for x in GATES):
        cs = [corr_years(ycache[("4h", n)].loc[gt], ycache[("1h", n)].loc[gt]) for n in cohorts]
        key[f"tfcorr|{gt}"] = cs
        md(f"| {gt} | " + " | ".join(f4(x) for x in cs) + " |")
    md()

    md("## 4. 홀드아웃 — 마지막 12개월 (2025-07-05 ~ 2026-07-05)")
    md()
    for tf in ("4h", "1h"):
        md(f"#### {tf}")
        md()
        md("| 게이트 | 코호트 | 구간 | 거래수 | gross | 비용 | 순R | t(월) | 양수월 | 심볼+ |")
        md("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for gt in GATES:
            for name, sub in (("47", t), ("신규38", t[t.cohort == "신규38"])):
                s = sub[(sub.tf == tf) & (sub.risk_bps >= gt)]
                for seg, m in (("이전", ~s.holdout), ("홀드아웃", s.holdout)):
                    ss = s[m]
                    if len(ss) < 20:
                        continue
                    mm = ss.groupby("month").net.mean()
                    _, tv, _ = tstat(mm)
                    per = ss.groupby("symbol").net.mean()
                    key[f"hold|{tf}|{int(gt)}|{name}|{seg}"] = dict(n=len(ss), net=float(ss.net.mean()), t=float(tv))
                    net = ss.net.mean()
                    md(f"| {int(gt)} | {name} | {seg} | {len(ss):,} | {ss.gross.mean():+.4f} | "
                       f"{ss.fee.mean():.4f} | " + (f"**{net:+.4f}**" if net < 0 else f"{net:+.4f}")
                       + f" | {tv:.2f} | {(mm > 0).mean():.0%} | {(per > 0).sum()}/{len(per)} |")
        md()

    md("## 5. 심볼별 — 게이트150")
    md()
    per_rows = []
    for tf in ("4h", "1h"):
        s = t[(t.tf == tf) & (t.risk_bps >= 150)]
        p = s.groupby("symbol").agg(n=("net", "size"), net=("net", "mean"), gross=("gross", "mean"))
        p = p[p.n >= 10]
        p["tf"] = tf
        per_rows.append(p.reset_index())
        key[f"persym|{tf}"] = dict(pos=int((p.net > 0).sum()), tot=len(p),
                                   med=float(p.net.median()),
                                   pos_new=int((p[~p.index.isin(ORIG)].net > 0).sum()),
                                   tot_new=int((~p.index.isin(ORIG)).sum()))
        md(f"- **{tf}**: 순R 양수 {int((p.net > 0).sum())}/{len(p)}심볼 "
           f"(신규 {key[f'persym|{tf}']['pos_new']}/{key[f'persym|{tf}']['tot_new']}) · "
           f"심볼별 순R 중앙값 {p.net.median():+.3f}")
    pd.concat(per_rows).to_csv("runs/u47_persym150.csv", index=False)
    both = pd.concat(per_rows).pivot(index="symbol", columns="tf", values="net").dropna()
    key["persym_tfcorr"] = float(both.corr().iloc[0, 1])
    md(f"- 심볼별 순R 4h↔1h 상관 {key['persym_tfcorr']:+.2f} (10심볼: +0.48)")
    md()

    md("## 6. 보조 절단 (판정에 쓰지 않음)")
    md()
    md("#### 유동성 3분위 — 봉인 전 5분 중앙 거래대금 기준, 게이트150")
    md()
    md("| TF | 분위 | 심볼 | 거래수 | gross | 비용 | 순R | t(월) |")
    md("|---|---|---:|---:|---:|---:|---:|---:|")
    for tf in ("4h", "1h"):
        for lv in ("얇음", "중간", "두꺼움"):
            s = t[(t.tf == tf) & (t.risk_bps >= 150) & (t.liq == lv)]
            mm = s.groupby("month").net.mean()
            _, tv, _ = tstat(mm)
            key[f"liq|{tf}|{lv}"] = dict(n=len(s), net=float(s.net.mean()), t=float(tv))
            md(f"| {tf} | {lv} | {s.symbol.nunique()} | {len(s):,} | {s.gross.mean():+.4f} | "
               f"{s.fee.mean():.4f} | {s.net.mean():+.4f} | {tv:.2f} |")
    md()
    md("#### 롱/숏 × 연도 — 47심볼 게이트150 순R (괄호 = 거래 수)")
    md()
    yrs = sorted(t.year.unique())
    md("| TF | 방향 | " + " | ".join(str(y) for y in yrs) + " |")
    md("|---|---|" + "---:|" * len(yrs))
    for tf in ("4h", "1h"):
        for sd, nm in ((1, "롱"), (-1, "숏")):
            s = t[(t.tf == tf) & (t.risk_bps >= 150) & (t.side == sd)]
            cells = []
            for y in yrs:
                ys = s[s.year == y]
                cells.append(f"{ys.net.mean():+.3f} ({len(ys)})" if len(ys) >= MIN_YEAR_N else f"— ({len(ys)})")
            md(f"| {tf} | {nm} | " + " | ".join(cells) + " |")
    md()

    Path("out").mkdir(exist_ok=True)
    Path("out/fvg_universe_tables.md").write_text("\n".join(MD), encoding="utf-8")
    Path("runs/u47_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1, default=float),
                                         encoding="utf-8")
    print("\n".join(MD))


if __name__ == "__main__":
    main()
