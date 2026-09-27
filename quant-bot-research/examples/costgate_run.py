"""비용 상태 스위치 — 사전등록(`claude/비용상태스위치-사전등록.md`) 그대로 실행.

입력: runs/legacy/*.pkl (예전 신호 15종 거래) · runs/legacy/daily_close.pkl
출력: runs/costgate_*.csv · runs/costgate_key.json · out/costgate_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import costgate as G
from qbot.research.yardstick import cluster_t

D = Path("runs/legacy")
MONTHS = pd.period_range("2021-04", "2026-06", freq="M")
OOS0, OOS1 = pd.Period("2022-04", "M"), pd.Period("2026-06", "M")
PERIODS = (("2022-04~12", "2022-04", "2022-12"), ("2023", "2023-01", "2023-12"),
           ("2024", "2024-01", "2024-12"), ("2025", "2025-01", "2025-12"), ("2026H1", "2026-01", "2026-06"))
K = 2.0

# 키 · 이름 · 진입 bps · 고정 왕복 비용(시간 청산)
SIGNALS = [
    ("lvl_break", "피보 0.9/0.1 레벨 돌파", 6.0, None),
    ("htf_fib_base", "HTF 피보 연속 (4H)", 6.0, None),
    ("atr_top10", "ATR 봉선별 상위10%", 6.0, None),
    ("bigbar_choch", "장대봉 → CHoCH", 6.0, None),
    ("liq_rev", "청산 캐스케이드 반전", 6.0, None),
    ("liq_cont_top1", "청산 재테스트 상위1% (대리)", 6.0, None),
    ("fvg_1h", "FVG mid 1h", 2.0, None),
    ("fvg_4h", "FVG mid 4h", 2.0, None),
    ("fvg_4h_g150", "FVG mid 4h · 150bps", 2.0, None),
    ("trend_fvg_15m", "추세추종 FVG 15m", 2.0, None),
    ("trend_fvg_1h", "추세추종 1H 레그", 2.0, None),
    ("sweep_choch", "FVG 관통 → CHoCH", 2.0, None),
    ("choch_15m", "순수 CHoCH 15m", 2.0, None),
    ("weekend_gap", "주말 갭 메움", 6.0, None),
    ("pain_tail", "painR3 꼬리 (시간청산)", 2.0, 8.0),
]


def load(key: str) -> pd.DataFrame:
    t = pd.read_pickle(D / f"{key}.pkl")
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t = t[t.time < pd.Timestamp("2026-07-01", tz="UTC")].sort_values("time").reset_index(drop=True)
    if "kind" not in t.columns:                     # FVG (u47) 거래: 결과 R 로 판정
        t["kind"] = np.where(t.gross > 0, 1, np.where(t.gross < 0, -1, 0)).astype(np.int8)
    t["mkey"] = t.time.dt.year * 12 + t.time.dt.month
    t["per"] = t.time.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    return t


def stats(s: pd.DataFrame, n_all: int) -> dict:
    m, tv, n, _ = cluster_t(s.net.to_numpy(), s.mkey.to_numpy()) if len(s) >= 3 else (np.nan, np.nan, len(s), 0)
    out = dict(n=int(len(s)), keep=len(s) / max(n_all, 1), net=float(s.net.mean()) if len(s) else np.nan,
               t=float(tv), gross=float(s.gross.mean()) if len(s) else np.nan,
               fee=float(s.fee.mean()) if len(s) else np.nan, risk=float(s.risk_bps.median()) if len(s) else np.nan)
    pos = 0
    for nm, a, b in PERIODS:
        q = s[(s.per >= pd.Period(a, "M")) & (s.per <= pd.Period(b, "M"))]
        out[f"n_{nm}"] = int(len(q))
        out[f"net_{nm}"] = float(q.net.mean()) if len(q) else np.nan
        pos += (len(q) >= 30) and (q.net.mean() > 0)
    out["pos"] = int(pos)
    return out


def main() -> None:
    C = pd.read_pickle(D / "daily_close.pkl")
    C.index = pd.to_datetime(C.index)
    state = G.market_state(C)
    rows, diag, mdiag = [], [], []
    for key, name, ebps, fixed in SIGNALS:
        t = load(key)
        wf = G.walkforward_edge(t, MONTHS)
        wf24 = G.walkforward_edge(t, MONTHS, window_months=24)
        a = G.attach_rho(t, wf, entry_bps=ebps, fixed_fee_bps=fixed)
        a24 = G.attach_rho(t, wf24, entry_bps=ebps, fixed_fee_bps=fixed)
        a = G.attach_state(a, state)
        oos = (a.per >= OOS0) & (a.per <= OOS1)
        A = a[oos]
        A24 = a24[oos]
        n_all = len(A)
        rules = {
            "항상 켬": A,
            "ρ≥2 (주)": A[A.rho >= K],
            "ρ≥1": A[A.rho >= 1.0],
            "ρ≥3": A[A.rho >= 3.0],
            "ρ≥2 · ĝ 24개월": A24[A24.rho >= K],
            "시장 고변동": A[A.mkt_high == 1.0],
        }
        g_on = wf.loc[OOS0:OOS1]
        for rn, s in rules.items():
            st = stats(s, n_all)
            rows.append(dict(key=key, name=name, rule=rn, **st,
                             g_pos_months=int((g_on.g > 0).sum()), months=len(g_on)))
        # 진단: ρ 3분위별 gross (ĝ > 0 인 달의 거래만)
        d = A[A.rho.notna()].copy()
        if len(d) >= 90:
            d["q"] = pd.qcut(d.rho.rank(method="first"), 3, labels=["낮음", "중간", "높음"])
            for q, s in d.groupby("q", observed=True):
                diag.append(dict(key=key, name=name, q=str(q), n=len(s), rho_med=float(s.rho.median()),
                                 risk=float(s.risk_bps.median()), gross=float(s.gross.mean()),
                                 fee=float(s.fee.mean()), net=float(s.net.mean())))
        for hv, s in A[A.mkt_high.notna()].groupby("mkt_high"):
            mdiag.append(dict(key=key, name=name, high=bool(hv), n=len(s), gross=float(s.gross.mean()),
                              fee=float(s.fee.mean()), net=float(s.net.mean()), risk=float(s.risk_bps.median())))
        p = [r for r in rows if r["key"] == key]
        print(f"  {name:<24} 항상 {p[0]['net']:+.4f} (n {p[0]['n']:,})  ρ≥2 {p[1]['net']:+.4f} "
              f"(n {p[1]['n']:,}, t {p[1]['t']:+.2f}, 양수구간 {p[1]['pos']})  시장고변동 {p[5]['net']:+.4f}", flush=True)

    R = pd.DataFrame(rows)
    Dg = pd.DataFrame(diag)
    Mg = pd.DataFrame(mdiag)
    R.to_csv("runs/costgate_rules.csv", index=False)
    Dg.to_csv("runs/costgate_rho_terciles.csv", index=False)
    Mg.to_csv("runs/costgate_market.csv", index=False)

    # 판정 (사전등록 네 조건)
    prim = R[R.rule == "ρ≥2 (주)"].set_index("key")
    alw = R[R.rule == "항상 켬"].set_index("key")
    verdict = {}
    for key, name, *_ in SIGNALS:
        p, a_ = prim.loc[key], alw.loc[key]
        c1 = bool(p.net > 0 and p.t >= 2.0)
        c2 = bool(p.pos >= 4)
        c3 = bool(p.net > a_.net) if np.isfinite(p.net) else False
        c4 = bool(p.n >= 300)
        verdict[key] = dict(c1=c1, c2=c2, c3=c3, c4=c4, passed=c1 and c2 and c3 and c4)
    Path("runs/costgate_key.json").write_text(json.dumps(dict(verdict=verdict), ensure_ascii=False, indent=1),
                                              encoding="utf-8")

    md = ["## 주 규칙 판정 (ρ ≥ 2, 표본 밖 2022-04 ~ 2026-06)", "",
          "| 신호 | 항상 켬 순R | ρ≥2 거래수 (비율) | ρ≥2 순R | t | 양수 구간 | ① t≥2 | ② 구간 4+ | ③ 항상보다 ↑ | ④ 300건+ | 판정 |",
          "|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|"]
    ok = lambda b: "○" if b else "·"
    for key, name, *_ in SIGNALS:
        p, a_, v = prim.loc[key], alw.loc[key], verdict[key]
        net = f"{p.net:+.4f}" if np.isfinite(p.net) else "—"
        tt = f"{p.t:.2f}" if np.isfinite(p.t) else "—"
        md.append(f"| {name} | {a_.net:+.4f} | {int(p.n):,} ({p.keep:.1%}) | {net} | {tt} | {int(p.pos)}/5 | "
                  f"{ok(v['c1'])} | {ok(v['c2'])} | {ok(v['c3'])} | {ok(v['c4'])} | "
                  f"{'**통과**' if v['passed'] else '기각'} |")
    md += ["", "## 규칙별 거래당 순R (표본 밖) — 민감도·보조는 판정에 안 씀", "",
           "| 신호 | 항상 켬 | ρ≥1 | **ρ≥2** | ρ≥3 | ρ≥2·ĝ24개월 | 시장 고변동 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for key, name, *_ in SIGNALS:
        cells = []
        for rn in ("항상 켬", "ρ≥1", "ρ≥2 (주)", "ρ≥3", "ρ≥2 · ĝ 24개월", "시장 고변동"):
            r = R[(R.key == key) & (R.rule == rn)].iloc[0]
            cells.append("—" if not np.isfinite(r.net) else f"{r.net:+.3f} ({int(r.n):,})")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", "## 구간별 순R — 주 규칙 (괄호 = 거래 수)", "",
           "| 신호 | " + " | ".join(p[0] for p in PERIODS) + " |", "|---|" + "---:|" * len(PERIODS)]
    for key, name, *_ in SIGNALS:
        p = prim.loc[key]
        cells = []
        for nm, *_ in PERIODS:
            n = int(p[f"n_{nm}"])
            v = p[f"net_{nm}"]
            cells.append("—" if n == 0 else f"{v:+.3f} ({n})")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", "## 진단 — ρ 3분위별 거래당 gross (가설의 전제: 1R 이 커도 gross 가 같은가)", "",
           "| 신호 | 낮음 gross (1R) | 중간 gross (1R) | 높음 gross (1R) | 높음 순R |", "|---|---:|---:|---:|---:|"]
    for key, name, *_ in SIGNALS:
        s = Dg[Dg.key == key].set_index("q") if len(Dg) else pd.DataFrame()
        if s.empty:
            md.append(f"| {name} | — | — | — | — |")
            continue
        c = [f"{s.loc[q, 'gross']:+.3f} ({s.loc[q, 'risk']:.0f}bps)" for q in ("낮음", "중간", "높음")]
        md.append(f"| {name} | " + " | ".join(c) + f" | {s.loc['높음', 'net']:+.3f} |")
    md += ["", "## 진단 — 시장 변동성 상태별 거래당 gross", "",
           "| 신호 | 낮음 gross | 높음 gross | 낮음 1R | 높음 1R | 낮음 순R | 높음 순R |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for key, name, *_ in SIGNALS:
        s = Mg[Mg.key == key].set_index("high")
        if len(s) < 2:
            continue
        md.append(f"| {name} | {s.loc[False, 'gross']:+.3f} | {s.loc[True, 'gross']:+.3f} | "
                  f"{s.loc[False, 'risk']:.0f} | {s.loc[True, 'risk']:.0f} | {s.loc[False, 'net']:+.3f} | "
                  f"{s.loc[True, 'net']:+.3f} |")
    Path("out").mkdir(exist_ok=True)
    Path("out/costgate_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
