"""주말 갭 × 4H 본장정렬 × 15m CISD — 판정과 진단 집계 (사전등록 기준 그대로).

입력 runs/wgap/*.pkl → runs/wgap_cells.csv · runs/wgap_key.json · out/wgap_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.wgap import PATH_HOURS  # noqa: E402
from qbot.research.yardstick import cluster_t  # noqa: E402

D = Path("runs/wgap")
PER = ("2021", "2022", "2023", "2024", "2025", "2026H1")
KEY = ["symbol", "monday"]


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for k in ("events", "B0", "B0L", "C", "D1", "D2", "DC"):
        t = pd.read_pickle(D / f"{k}.pkl")
        t["monday"] = pd.to_datetime(t["monday"], utc=True)
        out[k] = t
    return out


def per_label(monday: pd.Series) -> pd.Series:
    y = monday.dt.year
    return y.astype(str).where(y < 2026, "2026H1")


def month_key(monday: pd.Series) -> np.ndarray:
    return monday.dt.strftime("%Y-%m").to_numpy()


def stats(t: pd.DataFrame, col: str = "net") -> dict:
    if len(t) == 0:
        return dict(n=0, weeks=0, net=np.nan, t=np.nan, gross=np.nan, fee=np.nan, win=np.nan,
                    risk=np.nan, tp_r=np.nan, ex25=np.nan, ex25_n=0,
                    per={p: (np.nan, 0) for p in PER})
    m, tc, n, _ = cluster_t(t[col].to_numpy(float), month_key(t["monday"]))
    per = per_label(t["monday"])
    g = t[col].groupby(per).agg(["mean", "size"])
    ex = t[t["monday"].dt.year != 2025]
    return dict(n=int(n), weeks=int(t["monday"].nunique()), net=float(m), t=float(tc),
                gross=float(t["gross"].mean()), fee=float(t["fee"].mean()),
                win=float((t["kind"] == 1).mean()), risk=float(t["risk_bps"].median()),
                tp_r=float(t["tp_r"].median()), ex25=float(ex[col].mean()) if len(ex) else np.nan,
                ex25_n=int(len(ex)),
                per={p: (float(g.loc[p, "mean"]), int(g.loc[p, "size"])) if p in g.index else (np.nan, 0)
                     for p in PER})


def n_pos(s: dict) -> int:
    return sum(1 for p in PER if s["per"][p][1] >= 30 and s["per"][p][0] > 0)


def verdict(s: dict, better: list[float]) -> dict:
    c = dict(c1=bool(s["n"] > 0 and s["net"] > 0 and s["t"] >= 2.0), c2=bool(n_pos(s) >= 5),
             c3=bool(np.isfinite(s["ex25"]) and s["ex25"] > 0),
             c4=bool(s["n"] > 0 and all(s["net"] > b for b in better)), c5=bool(s["n"] >= 300))
    c["passed"] = all(c.values())
    return c


def keyset(t: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(t[KEY])


def cells(W: dict) -> dict[str, pd.DataFrame]:
    B0, C, D1, D2, DC = W["B0"], W["C"], W["D1"], W["D2"], W["DC"]
    has_c = keyset(B0).isin(keyset(C))
    return {
        "B0": B0, "B0L": W["B0L"],
        "A+": B0[B0.align4 == 1], "A−": B0[B0.align4 == -1], "A0": B0[B0.align4 == 0],
        "A24+": B0[B0.align24 == 1], "A24−": B0[B0.align24 == -1],
        "C": C, "A+∧C": C[C.align4 == 1], "A−∧C": C[C.align4 == -1],
        "B0|C있음": B0[has_c], "B0|C없음": B0[~has_c],
        "D1": D1, "D2": D2,
        "D-T": pd.concat([D2[D2.align4 == 1], D1[D1.align4 == -1]], ignore_index=True),
        "D-CISD": DC, "D-CISD|A+": DC[DC.align4 == 1], "D-CISD|A−": DC[DC.align4 == -1],
        "D1|A+": D1[D1.align4 == 1], "D1|A−": D1[D1.align4 == -1],
        "D2|A+": D2[D2.align4 == 1], "D2|A−": D2[D2.align4 == -1],
    }


JUDGED = {"A+": ["B0", "A−"], "C": ["B0"], "A+∧C": ["A+", "C"], "D-CISD": ["D1"], "D-T": ["D1", "D2"]}

NAMES = {
    "B0": "기본형 (필터 없음)", "B0L": "기본형 · 예전 규약(미해결 0R)",
    "A+": "4H 정렬 (메움 = 추세 방향)", "A−": "4H 역정렬 (메움 = 추세 반대)", "A0": "4H 추세 없음",
    "A24+": "24/7 4H 정렬 (진단)", "A24−": "24/7 4H 역정렬 (진단)",
    "C": "CISD 진입 확인", "A+∧C": "4H 정렬 + CISD", "A−∧C": "4H 역정렬 + CISD (진단)",
    "B0|C있음": "기본형 · CISD 가 난 주", "B0|C없음": "기본형 · CISD 가 먼저 안 난 주",
    "D1": "메운 뒤 반전 (지정가, CISD 없음)", "D2": "메운 뒤 연속 (역지정가, CISD 없음)",
    "D-T": "메운 뒤 4H 추세 방향", "D-CISD": "메운 뒤 반전 + CISD 확인",
    "D-CISD|A+": "메운 뒤 반전 + CISD · 정렬 주", "D-CISD|A−": "메운 뒤 반전 + CISD · 역정렬 주",
    "D1|A+": "메운 뒤 반전 · 정렬 주", "D1|A−": "메운 뒤 반전 · 역정렬 주",
    "D2|A+": "메운 뒤 연속 · 정렬 주", "D2|A−": "메운 뒤 연속 · 역정렬 주",
}


def d0_groups(ev: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    f = ev[ev.filled == True].copy()  # noqa: E712
    days = ["월", "화", "수", "목", "금"]
    out = [("전체", f), ("정렬 A+", f[f.align4 == 1]), ("역정렬 A−", f[f.align4 == -1]),
           ("위로 갭 (숏 메움)", f[f.d == -1]), ("아래로 갭 (롱 메움)", f[f.d == 1])]
    out += [(f"{days[k]}요일에 메움", f[f.fill_dow == k]) for k in range(5)]
    return out


def d0_row(name: str, f: pd.DataFrame) -> dict:
    if len(f) == 0:
        return dict(group=name, n=0)
    m, tc, n, _ = cluster_t(f.d0_g.to_numpy(float), month_key(f["monday"]))
    return dict(group=name, n=int(n), cont=float((f.d0_kind == 1).mean()),
                rev=float((f.d0_kind == -1).mean()), none=float((f.d0_kind == 2).mean()),
                amb=float((f.d0_kind == 0).mean()), mean=float(m), t=float(tc),
                fri=float(f.p_fri.mean()))


def paths(ev: pd.DataFrame) -> dict:
    f = ev[ev.filled == True]  # noqa: E712
    cols = [f"p{h}" for h in PATH_HOURS] + ["p_fri"]
    out = {}
    for name, g in (("all", f), ("A+", f[f.align4 == 1]), ("A−", f[f.align4 == -1])):
        out[name] = dict(mean=[float(g[c].mean()) for c in cols], n=[int(g[c].notna().sum()) for c in cols],
                         q25=[float(g[c].quantile(0.25)) for c in cols],
                         q75=[float(g[c].quantile(0.75)) for c in cols])
    out["x"] = [str(h) for h in PATH_HOURS] + ["금 종가"]
    return out


def fmt(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main() -> None:
    W = load()
    C = cells(W)
    S = {k: stats(v) for k, v in C.items()}
    V = {k: verdict(S[k], [S[b]["net"] for b in cmp]) for k, cmp in JUDGED.items()}
    ev = W["events"]
    D0 = [d0_row(n, g) for n, g in d0_groups(ev)]
    P = paths(ev)

    rows = []
    for k, s in S.items():
        r = dict(cell=k, name=NAMES[k], **{x: s[x] for x in ("n", "weeks", "net", "t", "gross", "fee", "win",
                                                              "risk", "tp_r", "ex25")}, pos=n_pos(s))
        for p in PER:
            r[f"net_{p}"], r[f"n_{p}"] = s["per"][p]
        rows.append(r)
    pd.DataFrame(rows).to_csv("runs/wgap_cells.csv", index=False)

    ev_all = ev
    info = dict(
        weekends=int(ev_all.monday.nunique()), events=int(len(ev_all)), symbols=int(ev_all.symbol.nunique()),
        first=str(ev_all.monday.min()), last=str(ev_all.monday.max()),
        filled_week=float(ev_all.filled.mean()),
        filled_week_by_align={str(k): float(v) for k, v in ev_all.groupby("align4").filled.mean().items()},
        align4_share={str(k): float(v) for k, v in ev_all.align4.value_counts(normalize=True).items()},
        align24_share={str(k): float(v) for k, v in ev_all.align24.value_counts(normalize=True).items()},
        fill_h_median=float(ev_all.loc[ev_all.filled == True, "fill_h"].median()),  # noqa: E712
        fill_dow_share={str(k): float(v) for k, v in
                        ev_all.loc[ev_all.filled == True, "fill_dow"].value_counts(normalize=True).sort_index().items()},  # noqa: E712
        c_delay_h_median=float(((C["C"].time - C["C"].monday).dt.total_seconds() / 3600).median()),
        c_share_of_b0=float(len(C["B0|C있음"]) / max(len(C["B0"]), 1)),
        dc_share_of_filled=float((ev_all.dc_bar >= 0).sum() / max(ev_all.filled.sum(), 1)),
    )
    key = dict(verdict=V, stats={k: {x: (v if not isinstance(v, float) or np.isfinite(v) else None)
                                     for x, v in s.items() if x != "per"} | {"per": s["per"]}
                                 for k, s in S.items()},
               d0=D0, paths=P, info=info)
    Path("runs/wgap_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")

    L = ["## 판정 (사전등록 5개)", "",
         "| 판정 대상 | 거래 | 주 | 순R | t | 양수 구간 | 2025 제외 | 비교 대상 | ① | ② | ③ | ④ | ⑤ | 판정 |",
         "|---|---:|---:|---:|---:|---:|---:|---|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for k, cmp in JUDGED.items():
        s, v = S[k], V[k]
        cmpt = " · ".join(f"{c} {fmt(S[c]['net'])}" for c in cmp)
        ox = lambda b: "○" if b else "·"
        L.append(f"| {NAMES[k]} | {s['n']:,} | {s['weeks']} | {fmt(s['net'])} | {fmt(s['t'], 2)} | {n_pos(s)}/6 | "
                 f"{fmt(s['ex25'])} | {cmpt} | {ox(v['c1'])} | {ox(v['c2'])} | {ox(v['c3'])} | {ox(v['c4'])} | "
                 f"{ox(v['c5'])} | {'**통과**' if v['passed'] else '기각'} |")
    L += ["", "① 순R > 0 · 월 군집 t ≥ 2 　② 6구간 중 5개 이상 양수 　③ 2025 제외 양수 　④ 비교 대상보다 높음 　"
          "⑤ 300건 이상. B0 = 기본형, A± = 4H 본장 정렬/역정렬, C = CISD 진입 확인, "
          "D1 = 메운 뒤 반전(지정가), D2 = 메운 뒤 연속(역지정가)."]
    L += ["", "## 셀 전체 (거래당, R)", "",
          "| 셀 | 거래 | 주 | 순R | t | gross | 비용 | 익절률 | 1R 중앙 bps | 익절 배수 | 2025 제외 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, s in S.items():
        L.append(f"| {NAMES[k]} | {s['n']:,} | {s['weeks']} | {fmt(s['net'])} | {fmt(s['t'], 2)} | "
                 f"{fmt(s['gross'])} | {fmt(s['fee'], 3, False)} | {fmt(s['win'] * 100 if s['n'] else np.nan, 1, False)}% | "
                 f"{fmt(s['risk'], 0, False)} | {fmt(s['tp_r'], 2, False)} | {fmt(s['ex25'])} |")
    L += ["", "## 구간별 순R (괄호 = 거래 수)", "", "| 셀 | " + " | ".join(PER) + " |",
          "|---|" + "---:|" * len(PER)]
    for k in ("B0", "A+", "A−", "C", "A+∧C", "D1", "D2", "D-T", "D-CISD"):
        s = S[k]
        L.append(f"| {NAMES[k]} | " + " | ".join(
            "—" if s["per"][p][1] == 0 else f"{fmt(s['per'][p][0])} ({s['per'][p][1]})" for p in PER) + " |")
    L += ["", "## 메운 뒤 1:1 레이스 (D0, 서술) — F 에서 한 갭 더(연속) vs 월 개장가(반전), 금 종가까지", "",
          "| 묶음 | 주·심볼 | 연속 먼저 | 반전 먼저 | 둘 다 못 닿음 | 평균 (G 단위, 연속 +) | t | 금 종가 평균 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in D0:
        if r["n"] == 0:
            continue
        L.append(f"| {r['group']} | {r['n']:,} | {r['cont'] * 100:.1f}% | {r['rev'] * 100:.1f}% | "
                 f"{r['none'] * 100:.1f}% | {fmt(r['mean'])} | {fmt(r['t'], 2)} | {fmt(r['fri'])} |")
    Path("out/wgap_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print("paths", {k: [round(x, 3) for x in v["mean"]] for k, v in P.items() if k != "x"})


if __name__ == "__main__":
    main()
