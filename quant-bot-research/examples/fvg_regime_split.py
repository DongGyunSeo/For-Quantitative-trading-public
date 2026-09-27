"""2024 전/후 분할 — 무엇이 뒤집혔고, 뒤집힌 채로 안정적인가.

입력: runs/u47_trades_era.csv (fvg_era_control.py 산출) · runs/u47_dailyqv.csv
표는 out/fvg_regime_tables.md 로 쓴다.

용어
  gross    수수료 전 R
  대조     같은 시기 안에서 진입 시점만 무작위로 바꾼 기대 R (방향·1R 동일)
  엣지     gross − 대조.  "FVG 자리라서" 생긴 몫. 시장 방향 효과는 대조가 흡수한다
  순R      gross − 수수료

역방향 거래
  FVG 롱 자리에서 숏(반대도 같음). 손익은 −gross 로 정확히 거울이지만 **체결이 다르다** —
  원래는 되돌림을 지정가로 기다리는(maker) 진입인데, 반대로 치려면 가격이 내려와 닿는 순간
  팔아야 하므로 **스톱 주문(taker 6)** 이다. 익절(원래 손절 자리)은 지정가 maker 2,
  손절(원래 익절 자리)은 taker 6.

워크포워드 ("최근 1~2년으로 고르면 되는가")
  메뉴 20개 = TF{4h,1h} × 게이트{100..200} × 방향{정,역}.
  매 분기 시작일에 직전 L개월(12·24) 성과로 순R 최고 변형을 고르고(거래 100건 이상,
  순R ≤ 0 이면 쉼), 그 분기에만 적용한다. 결과가 아직 안 난 거래를 보지 않도록
  분기 시작 7일 전까지 진입한 거래만 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.yardstick import cluster_t, tstat

GATES = (100, 125, 150, 175, 200)
ERAS = ("2021-23", "2024+")
MD: list[str] = []
KEY: dict = {}


def md(s: str = "") -> None:
    MD.append(s)


def neg(x: float, fmt: str = "{:+.3f}") -> str:
    s = fmt.format(x)
    return f"**{s}**" if np.isfinite(x) and x < 0 else s


def mt(s: pd.DataFrame, col: str) -> float:
    """거래 가중 평균의 t (월 군집 강건). 월평균의 t 가 아니다 — yardstick.cluster_t 참조."""
    return cluster_t(s[col].to_numpy(), s["month"].to_numpy())[1]


def load() -> pd.DataFrame:
    t = pd.read_csv("runs/u47_trades_era.csv", parse_dates=["time"])
    t["edge"] = t.gross - t.ctrl_era
    t["half"] = t.year.astype(str) + np.where(t.time.dt.month <= 6, "H1", "H2")
    # 역방향: 손익 거울, 진입 taker 6 · 승(=원래 패) maker 2 · 그 외 taker 6
    t["inv_gross"] = -t.gross
    t["inv_fee"] = (6.0 + np.where(t.gross < 0, 2.0, 6.0)) / t.risk_bps
    t["inv_net"] = t.inv_gross - t.inv_fee
    t["inv_edge"] = -t.edge
    # 시점 유동성 3분위 (진입 전날까지 30일 중앙 거래대금의 횡단면 순위)
    p = pd.read_csv("runs/u47_dailyqv.csv", index_col=0, parse_dates=True)
    r = p.rolling(30, min_periods=20).median().shift(1).rank(axis=1, pct=True)
    lab = r.stack().rename("pct").reset_index()
    lab.columns = ["day", "symbol", "pct"]
    lab["day"] = pd.to_datetime(lab.day).dt.tz_localize(None)
    t["day"] = t.time.dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    t = t.merge(lab, on=["day", "symbol"], how="left")
    t["liq"] = pd.cut(t.pct, [0, 1 / 3, 2 / 3, 1.0001], labels=["얇음", "중간", "두꺼움"]).astype(str)
    return t


def era_table(t: pd.DataFrame) -> None:
    md("## 1. 시기별 요약 — 대조군도 같은 시기 안에서")
    md()
    for tf in ("4h", "1h"):
        md(f"#### {tf}")
        md()
        md("| 게이트 | 시기 | 거래수 | 승률 | gross | 대조 | **엣지** | t(엣지) | 비용 | 순R | t(순R) |")
        md("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for g in GATES:
            for e in ERAS:
                s = t[(t.tf == tf) & (t.risk_bps >= g) & (t.era == e)]
                te, tn = mt(s, "edge"), mt(s, "net")
                KEY[f"era|{tf}|{g}|{e}"] = dict(n=len(s), gross=s.gross.mean(), ctrl=s.ctrl_era.mean(),
                                                edge=s.edge.mean(), t_edge=te, net=s.net.mean(), t_net=tn,
                                                win=(s.gross > 0).mean())
                md(f"| {g} | {e} | {len(s):,} | {(s.gross > 0).mean():.1%} | {s.gross.mean():+.3f} | "
                   f"{s.ctrl_era.mean():+.3f} | {neg(s.edge.mean())} | {te:.2f} | {s.fee.mean():.3f} | "
                   f"{neg(s.net.mean())} | {tn:.2f} |")
        md()


def half_table(t: pd.DataFrame) -> None:
    md("## 2. 반기별 — 2024 이후가 '뒤집힌 채로' 안정적인가")
    md()
    halves = sorted(t.half.unique())
    for g in (100, 150, 200):
        md(f"#### 게이트 {g} — 엣지(gross − 대조) / 순R  (괄호 = 거래 수)")
        md()
        md("| 반기 | 4h 엣지 | 4h 순R | 1h 엣지 | 1h 순R |")
        md("|---|---:|---:|---:|---:|")
        for hf in halves:
            cells = []
            for tf in ("4h", "1h"):
                s = t[(t.tf == tf) & (t.risk_bps >= g) & (t.half == hf)]
                if len(s) < 15:
                    cells += [f"— ({len(s)})", "—"]
                    continue
                cells += [neg(s.edge.mean()) + f" ({len(s)})", neg(s.net.mean())]
                KEY[f"half|{tf}|{g}|{hf}"] = dict(n=len(s), edge=s.edge.mean(), net=s.net.mean())
            md(f"| {hf} | " + " | ".join(cells) + " |")
        md()
    # 부호 일관성 요약
    md("#### 2024 이후 반기 5개 중 엣지가 음수인 반기 수")
    md()
    md("| TF | " + " | ".join(str(g) for g in GATES) + " |")
    md("|---|" + "---:|" * len(GATES))
    post_halves = [h for h in halves if h >= "2024"]
    for tf in ("4h", "1h"):
        row = []
        for g in GATES:
            k = 0
            tot = 0
            for hf in post_halves:
                s = t[(t.tf == tf) & (t.risk_bps >= g) & (t.half == hf)]
                if len(s) >= 15:
                    tot += 1
                    k += s.edge.mean() < 0
            row.append(f"{k}/{tot}")
            KEY[f"negpost|{tf}|{g}"] = (k, tot)
        md(f"| {tf} | " + " | ".join(row) + " |")
    md()


def side_liq(t: pd.DataFrame) -> None:
    md("## 3. 방향 · 유동성 — 게이트150")
    md()
    md("| TF | 절단 | 시기 | 거래수 | 엣지 | t(엣지) | 순R |")
    md("|---|---|---|---:|---:|---:|---:|")
    for tf in ("4h", "1h"):
        for lab, m in (("롱", t.side > 0), ("숏", t.side < 0), ("얇음", t.liq == "얇음"),
                       ("중간", t.liq == "중간"), ("두꺼움", t.liq == "두꺼움")):
            for e in ERAS:
                s = t[(t.tf == tf) & (t.risk_bps >= 150) & m & (t.era == e)]
                te = mt(s, "edge")
                KEY[f"cut|{tf}|{lab}|{e}"] = dict(n=len(s), edge=s.edge.mean(), t=te, net=s.net.mean())
                md(f"| {tf} | {lab} | {e} | {len(s):,} | {neg(s.edge.mean())} | {te:.2f} | {neg(s.net.mean())} |")
    md()


def inverse_table(t: pd.DataFrame) -> None:
    md("## 4. 뒤집어서 치면? — 역방향 실비용 (진입 taker 6 · 승 maker 2 · 패 taker 6)")
    md()
    md("| TF | 게이트 | 시기 | 거래수 | 역방향 gross | 역방향 비용 | 역방향 순R | t | 정방향 순R |")
    md("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for tf in ("4h", "1h"):
        for g in GATES:
            for e in ERAS:
                s = t[(t.tf == tf) & (t.risk_bps >= g) & (t.era == e)]
                ti = mt(s, "inv_net")
                KEY[f"inv|{tf}|{g}|{e}"] = dict(n=len(s), gross=s.inv_gross.mean(), fee=s.inv_fee.mean(),
                                                net=s.inv_net.mean(), t=ti)
                md(f"| {tf} | {g} | {e} | {len(s):,} | {s.inv_gross.mean():+.3f} | {s.inv_fee.mean():.3f} | "
                   f"{neg(s.inv_net.mean())} | {ti:.2f} | {neg(s.net.mean())} |")
    md()


# ------------------------------------------------------------------ 워크포워드


def variants(t: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for tf in ("4h", "1h"):
        for g in GATES:
            s = t[(t.tf == tf) & (t.risk_bps >= g)][["time", "month", "net", "inv_net"]]
            out[f"{tf}·{g}·정"] = s.rename(columns={"net": "r"})[["time", "month", "r"]]
            out[f"{tf}·{g}·역"] = s.rename(columns={"inv_net": "r"})[["time", "month", "r"]]
    return out


def walk_forward(t: pd.DataFrame) -> None:
    V = variants(t)
    q0 = pd.Timestamp("2021-04-01", tz="UTC")
    seal = pd.Timestamp("2026-07-05", tz="UTC")
    qs = pd.date_range("2021-04-01", "2026-04-01", freq="QS", tz="UTC")   # 마지막 = 2026Q2 (→ 봉인)
    rows, oos = [], []
    for L in (12, 24):
        for i, qa in enumerate(qs):
            qb = min(qs[i + 1] if i + 1 < len(qs) else seal, seal)
            lo = qa - pd.DateOffset(months=L)
            if lo < q0:
                continue
            hi = qa - pd.Timedelta(days=7)           # 결과가 난 거래만
            best, bv, bn = None, 0.0, 0
            for k, s in V.items():
                w = s[(s.time >= lo) & (s.time < hi)]
                if len(w) >= 100 and w.r.mean() > bv:
                    best, bv, bn = k, float(w.r.mean()), len(w)
            if best is None:
                rows.append(dict(L=L, q=f"{qa.year}Q{(qa.month - 1) // 3 + 1}", pick="쉼", trail=np.nan,
                                 n=0, oos=0.0))
                continue
            s = V[best]
            o = s[(s.time >= qa) & (s.time < qb)]
            oos.append(o.assign(L=L, q=f"{qa.year}Q{(qa.month - 1) // 3 + 1}"))
            rows.append(dict(L=L, q=f"{qa.year}Q{(qa.month - 1) // 3 + 1}", pick=best, trail=bv,
                             n=len(o), oos=float(o.r.mean()) if len(o) else 0.0, oos_sum=float(o.r.sum())))
    W = pd.DataFrame(rows)
    W.to_csv("runs/u47_walkforward.csv", index=False)
    O = pd.concat(oos, ignore_index=True)

    # 고정 기준선: 같은 분기들에 항상 같은 변형
    def fixed(k):
        s = V[k]
        out = {}
        for i, qa in enumerate(qs):
            qb = min(qs[i + 1] if i + 1 < len(qs) else seal, seal)
            o = s[(s.time >= qa) & (s.time < qb)]
            out[f"{qa.year}Q{(qa.month - 1) // 3 + 1}"] = (len(o), float(o.r.sum()))
        return out

    base = {k: fixed(k) for k in ("4h·150·정", "4h·150·역", "1h·150·정", "1h·150·역")}

    md("## 5. \"최근 1~2년으로 고르기\" 워크포워드")
    md()
    md("매 분기 직전 L개월 성과로 20개 변형 중 순R 최고를 골라 **다음 분기에만** 적용. 순R ≤ 0 이면 쉰다.")
    md()
    md("#### 분기별 선택과 결과 (표본 밖)")
    md()
    md("| 분기 | L=12 선택 | 직전 순R | 분기 순R (n) | L=24 선택 | 직전 순R | 분기 순R (n) |")
    md("|---|---|---:|---:|---|---:|---:|")
    for q in W[W.L == 12].q:
        cells = []
        for L in (12, 24):
            r = W[(W.L == L) & (W.q == q)]
            if r.empty:
                cells += ["—", "", ""]
                continue
            r = r.iloc[0]
            if r.pick == "쉼":
                cells += ["쉼", "", "0"]
            else:
                cells += [r.pick, f"{r.trail:+.3f}", neg(r.oos) + f" ({int(r.n)})"]
        md(f"| {q} | " + " | ".join(cells) + " |")
    md()

    def qlab(ts):
        return ts.dt.year.astype(str) + "Q" + ((ts.dt.month - 1) // 3 + 1).astype(str)

    def agg(qsel):
        out = {}
        for L in (12, 24):
            o = O[(O.L == L) & O.q.isin(qsel)]
            w = W[(W.L == L) & W.q.isin(qsel) & (W.n > 0)]
            m, tv, n, _ = cluster_t(o.r.to_numpy(), o.month.to_numpy())
            out[f"L={L}"] = (n, m, tv, int((w.oos > 0).sum()), len(w))
        for k in base:
            s = V[k]
            s = s[qlab(s.time).isin(qsel)]
            m, tv, n, _ = cluster_t(s.r.to_numpy(), s.month.to_numpy())
            qm = s.groupby(qlab(s.time)).r.mean()
            out[f"고정 {k}"] = (n, m, tv, int((qm > 0).sum()), len(qm))
        return out

    both = sorted(set(W[W.L == 24].q))
    post = [q for q in both if q >= "2024"]
    md("#### 합계 — 같은 분기 집합에서 비교")
    md()
    md("| 방식 | 구간 | 거래수 | 거래당 순R | t(월 군집) | 양수 분기 |")
    md("|---|---|---:|---:|---:|---:|")
    for nm, qsel in ((f"{both[0]}~{both[-1]}", both), (f"{post[0]}~{post[-1]}", post)):
        for k, (n, m, tq, pos, nq) in agg(qsel).items():
            KEY[f"wf|{nm}|{k}"] = dict(n=n, mean=m, t=tq, pos=pos, nq=nq)
            md(f"| {k} | {nm} | {n:,} | {neg(m)} | {tq:.2f} | {pos}/{nq} |")
    md()

    # 선택이 얼마나 자주 바뀌나
    for L in (12, 24):
        p = W[W.L == L].pick.tolist()
        ch = sum(a != b for a, b in zip(p, p[1:]))
        KEY[f"wf_changes|{L}"] = (ch, len(p) - 1)
    md(f"- 선택이 바뀐 횟수: L=12 {KEY['wf_changes|12'][0]}/{KEY['wf_changes|12'][1]}분기 · "
       f"L=24 {KEY['wf_changes|24'][0]}/{KEY['wf_changes|24'][1]}분기")

    # 부호 지속성: 직전 12개월 순R 부호가 다음 분기 순R 부호와 같은가 (정방향 변형만)
    hits = tot = 0
    for k, s in V.items():
        if not k.endswith("정"):
            continue
        for i, qa in enumerate(qs):
            qb = min(qs[i + 1] if i + 1 < len(qs) else seal, seal)
            lo = qa - pd.DateOffset(months=12)
            if lo < q0:
                continue
            w = s[(s.time >= lo) & (s.time < qa - pd.Timedelta(days=7))]
            o = s[(s.time >= qa) & (s.time < qb)]
            if len(w) >= 100 and len(o) >= 20:
                tot += 1
                hits += np.sign(w.r.mean()) == np.sign(o.r.mean())
    KEY["persist12"] = (int(hits), tot)
    md(f"- 부호 지속성(정방향 10개 변형 × 분기): 직전 12개월 순R 부호 = 다음 분기 순R 부호 {hits}/{tot} "
       f"({hits / tot:.0%}). 동전 던지기는 50%.")
    md()


def busy_table(t: pd.DataFrame) -> None:
    """진입 직전 7일 동안 유니버스 전체에서 같은 TF FVG 진입(게이트100 이상)이 몇 건이었나.
    진입 시점에 알 수 있는 값이다. 시기 안에서 3분위로 나눈다(서술용 — 판정에 안 씀)."""
    md("## 6. 신호가 몰릴 때 — 월평균과 거래평균이 갈리는 이유")
    md()
    md("| TF | 시기 | 직전 7일 유니버스 신호 | 거래수 | 엣지 | 순R | t |")
    md("|---|---|---|---:|---:|---:|---:|")
    wk = 7 * 86400.0
    ep = pd.Timestamp("1970-01-01", tz="UTC")
    sec = lambda c: (c - ep).dt.total_seconds().to_numpy()   # 단위(ns/us) 혼동 방지 — 초로 통일
    for tf in ("4h", "1h"):
        base = t[(t.tf == tf) & (t.risk_bps >= 100)].sort_values("time")
        ts = sec(base.time)
        s = t[(t.tf == tf) & (t.risk_bps >= 150)].copy()
        x = sec(s.time)
        s["busy"] = np.searchsorted(ts, x, "left") - np.searchsorted(ts, x - wk, "left")
        for e in ERAS:
            se = s[s.era == e].copy()
            se["b3"] = pd.qcut(se.busy.rank(method="first"), 3, labels=["한산", "보통", "몰림"])
            for lv in ("한산", "보통", "몰림"):
                q = se[se.b3 == lv]
                tv = mt(q, "net")
                KEY[f"busy|{tf}|{e}|{lv}"] = dict(n=len(q), edge=q.edge.mean(), net=q.net.mean(), t=tv,
                                                   lo=int(q.busy.min()), hi=int(q.busy.max()))
                md(f"| {tf} | {e} | {lv} ({int(q.busy.min())}~{int(q.busy.max())}건) | {len(q):,} | "
                   f"{neg(q.edge.mean())} | {neg(q.net.mean())} | {tv:.2f} |")
    md()
    md("#### 달마다 거래 수 vs 그 달 거래당 순R — 게이트150")
    md()
    md("| TF | 시기 | 상관(거래 수, 월 순R) | 거래 가중 순R | 월 평균 순R |")
    md("|---|---|---:|---:|---:|")
    for tf in ("4h", "1h"):
        for e in ERAS:
            s = t[(t.tf == tf) & (t.risk_bps >= 150) & (t.era == e)]
            g = s.groupby("month").net.agg(["size", "mean"])
            c = float(np.corrcoef(g["size"], g["mean"])[0, 1])
            KEY[f"cntcorr|{tf}|{e}"] = c
            md(f"| {tf} | {e} | {c:+.2f} | {neg(s.net.mean())} | {neg(g['mean'].mean())} |")
    md()


def main() -> None:
    t = load()
    era_table(t)
    half_table(t)
    side_liq(t)
    inverse_table(t)
    walk_forward(t)
    busy_table(t)
    Path("out").mkdir(exist_ok=True)
    Path("out/fvg_regime_tables.md").write_text("\n".join(MD), encoding="utf-8")
    Path("runs/u47_regime_key.json").write_text(json.dumps(KEY, ensure_ascii=False, indent=1, default=float),
                                                encoding="utf-8")
    print("\n".join(MD))


if __name__ == "__main__":
    main()
