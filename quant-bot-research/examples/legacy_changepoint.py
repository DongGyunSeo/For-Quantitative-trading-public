"""예전 신호 전부 — 수익 곡선과 변곡점.

입력: runs/legacy/*.pkl (legacy_run.py)
출력: runs/legacy_cp.json · out/legacy_curves.html · out/legacy_cp_tables.md

변곡점 = 월별 순손익(Σ 순R) 수열의 평균 변화점 (`qbot.research.changepoint`).
보조로 **거래당 gross**(수수료 전, 신호의 질) 수열의 변화점도 같이 낸다 —
수수료가 지배하는 고빈도 신호는 순손익 곡선이 거의 직선이라 기울기 변화가
거래 수 변화에 끌려간다. 질의 변화는 gross 쪽에서 본다.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import changepoint as CP
from qbot.research.yardstick import cluster_t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams["axes.unicode_minus"] = False

D = Path("runs/legacy")
M0, M1 = pd.Period("2021-04", "M"), pd.Period("2026-06", "M")
MONTHS = pd.period_range(M0, M1, freq="M")

# 이름 · 출처 문서 · 진입 · 재구축 상태
META = {
    "lvl_break": ("피보 0.9/0.1 레벨 돌파 (4H·ATR 상위25%)", "피보레벨-진입승률-분석", "스톱", "재현 ✓ 승률 59.29% (문서 59.25%)"),
    "htf_fib_base": ("HTF 피보 연속 진입 (4H 기준선)", "실데이터-4H-검증결과", "시장가", "정의 복원 불완전 — 문서 숫자와 불일치"),
    "atr_top10": ("ATR 봉선별 상위10% (4H 연속)", "ATR-봉선별-검증결과", "시장가", "정의 복원 불완전 — 문서 숫자와 불일치"),
    "bigbar_choch": ("장대봉 → CHoCH (5m·상위20%)", "장대봉-CHoCH-검증결과", "시장가", "재현 ✓ 거래수 −3% · 승률 ±0.3%p"),
    "liq_rev": ("청산 캐스케이드 반전 (거래량 복귀)", "청산이벤트-거래량복귀-검증결과", "시장가", "근사 — 거래수 −23% · 승률 +1.9%p"),
    "liq_cont_top1": ("청산 극단 재테스트 상위1% (크기 대리)", "청산-좁힌컷-트리거3종-전면비교", "시장가", "대리 변수 — 원본은 히트맵 소진량 순위"),
    "fvg_1h": ("FVG mid 1h", "규칙적용-전수감사-및-FVG기준선", "지정가", "47심볼 실측"),
    "fvg_4h": ("FVG mid 4h", "규칙적용-전수감사-및-FVG기준선", "지정가", "47심볼 실측"),
    "fvg_4h_g150": ("FVG mid 4h · 1R 150bps 이상", "FVG-47심볼-게이트안정성-재검사", "지정가", "47심볼 실측"),
    "trend_fvg_15m": ("추세추종 FVG 15m (4H 정렬·fib0.7·3R)", "추세추종-FVG-IS결과-기각", "지정가", "재현 ✓ 거래수 6,172 (문서 6,189)"),
    "trend_fvg_1h": ("추세추종 1H 레그 (4H 정렬·fib0.7·3R)", "1H레그-1단계결과-기각", "지정가", "문서 정의대로 (대형5 결과만 있어 직접 대조 불가)"),
    "sweep_choch": ("FVG 관통 → 15m CHoCH", "FVG관통-스윕CHoCH-기각-및-엣지부재", "지정가", "기존 코드 그대로"),
    "choch_15m": ("순수 CHoCH 15m", "FVG관통-스윕CHoCH-기각-및-엣지부재", "지정가", "기존 코드 그대로 (대조군 B)"),
    "weekend_gap": ("주말 갭 메움 (200bps+)", "주말갭-OOS결과-기각", "시장가", "재현 ✓ 거래수 770 (문서 757)"),
    "pain_tail": ("painR3 꼬리 (48봉 시간청산)", "체결모델확정-및-painR3-3종검정", "지정가", "문서 정의대로 · 1R = 2×일간ATR 로 환산"),
    "tsmom": ("시계열 추세 28일 (주간 리밸런스)", "시계열추세-시뮬레이션결과", "시장가", "재현 ✓ 대형5 샤프 0.99 (문서 0.99)"),
}
ORDER = list(META)


def monthly(t: pd.DataFrame, col: str) -> pd.Series:
    p = t.time.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    return t.groupby(p)[col].sum().reindex(MONTHS, fill_value=0.0)


def monthly_mean(t: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    p = t.time.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    g = t.groupby(p)[col].agg(["mean", "size"]).reindex(MONTHS)
    return g["mean"], g["size"].fillna(0)


def load(name: str) -> pd.DataFrame:
    t = pd.read_pickle(D / f"{name}.pkl")
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t = t[t.time < pd.Timestamp("2026-07-01", tz="UTC")]
    if "month" not in t.columns:
        t["month"] = t.time.dt.year * 12 + t.time.dt.month
    return t


def fmt_month(k: int) -> str:
    return str(MONTHS[k]) if 0 <= k < len(MONTHS) else "—"


def analyse() -> dict:
    res = {}
    for name in ORDER:
        if name == "tsmom":
            w = pd.read_pickle(D / "tsmom.pkl")
            w.index = pd.to_datetime(w.index)
            w = w[w.index < pd.Timestamp("2026-07-01", tz="UTC")]
            per = w.index.tz_convert("UTC").tz_localize(None).to_period("M")
            net_m = w.ret.groupby(per).sum().reindex(MONTHS, fill_value=0.0)
            bh_m = w.bh.groupby(per).sum().reindex(MONTHS, fill_value=0.0)
            cps = CP.detect(net_m.to_numpy(), seed=7)
            res[name] = dict(n=int(len(w)), unit="%", net_m=net_m.tolist(), gross_m=bh_m.tolist(),
                             daily=dict(t=[str(x) for x in w.index], net=(np.log1p(w.ret).cumsum()).tolist(),
                                        gross=(np.log1p(w.gross).cumsum()).tolist(),
                                        bh=(np.log1p(w.bh).cumsum()).tolist()),
                             total=float(np.expm1(np.log1p(w.ret).sum())),
                             mean=float(w.ret.mean()), t=float(w.ret.mean() / w.ret.std() * np.sqrt(len(w))),
                             cps=[c.__dict__ for c in cps], q_cps=[],
                             pre=float(w.ret[w.index < pd.Timestamp("2024-01-01", tz="UTC")].mean()),
                             post=float(w.ret[w.index >= pd.Timestamp("2024-01-01", tz="UTC")].mean()),
                             sharpe=float(w.ret.mean() / w.ret.std() * np.sqrt(52)))
            continue
        t = load(name)
        net_m = monthly(t, "net")
        gross_m = monthly(t, "gross")
        q, cnt = monthly_mean(t, "gross")
        cps = CP.detect(net_m.to_numpy(), seed=7)
        qq = q.fillna(0.0).to_numpy()
        q_cps = CP.detect(qq, seed=11, second=False) if (cnt > 0).sum() >= 24 else []
        m, tc, n, _ = cluster_t(t.net.to_numpy(), t.month.to_numpy())
        pre = t[t.time < pd.Timestamp("2024-01-01", tz="UTC")]
        post = t[t.time >= pd.Timestamp("2024-01-01", tz="UTC")]
        daily = t.set_index("time").sort_index()[["net", "gross"]].resample("1D").sum().cumsum()
        res[name] = dict(n=int(n), unit="R", net_m=net_m.tolist(), gross_m=gross_m.tolist(),
                         count_m=cnt.tolist(), q_m=q.tolist(),
                         daily=dict(t=[str(x) for x in daily.index], net=daily.net.tolist(),
                                    gross=daily.gross.tolist()),
                         total=float(t.net.sum()), mean=float(m), t=float(tc),
                         gross_mean=float(t.gross.mean()), fee_mean=float(t.fee.mean()),
                         cps=[c.__dict__ for c in cps], q_cps=[c.__dict__ for c in q_cps],
                         pre=float(pre.net.mean()), post=float(post.net.mean()),
                         pre_g=float(pre.gross.mean()), post_g=float(post.gross.mean()),
                         pre_n=int(len(pre)), post_n=int(len(post)))
        print(f"  {name:<14} n {n:>9,}  순R {m:+.4f} (t {tc:+.2f})  변곡 {fmt_month(cps[0]['k'] if False else cps[0].k)}"
              f" p {cps[0].p:.3f}  {cps[0].before:+.2f} → {cps[0].after:+.2f}", flush=True)
    return res


def panel(ax, name, r):
    d = pd.to_datetime(pd.Series(r["daily"]["t"]), utc=True)
    net = np.asarray(r["daily"]["net"])
    gross = np.asarray(r["daily"]["gross"])
    if r["unit"] == "%":
        # 다른 패널과 같은 규약: 회색 = 수수료 전, 검정 = 수수료 후. 매수보유는 따로 점선.
        # (처음엔 회색 자리에 매수보유를 그리고 범례를 빠뜨려 "검정이 회색 위" 로 보였다)
        bh = np.asarray(r["daily"]["bh"])
        ax.plot(d, np.expm1(bh) * 100, color="#9ca3af", lw=1.0, ls=(0, (2, 2)), label="매수보유(동일가중)")
        ax.plot(d, np.expm1(gross) * 100, color="#9ca3af", lw=1.0, label="전략 수수료 전")
        ax.plot(d, np.expm1(net) * 100, color="#1f2937", lw=1.4, label="전략 수수료 후")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.set_ylabel("누적 수익 %", fontsize=8)
    else:
        ax.plot(d, gross, color="#9ca3af", lw=1.0, label="수수료 전")
        ax.plot(d, net, color="#1f2937", lw=1.4, label="수수료 후")
        ax.set_ylabel("누적 R", fontsize=8)
    ax.axhline(0, color="#d1d5db", lw=0.8)
    ax.axvline(pd.Timestamp("2024-01-01", tz="UTC"), color="#60a5fa", lw=0.8, ls=":")
    for i, c in enumerate(r["cps"]):
        col = "#dc2626" if c["p"] < 0.05 else "#f59e0b"
        x = MONTHS[c["k"]].to_timestamp().tz_localize("UTC")
        ax.axvline(x, color=col, lw=1.3, ls="--")
    for c in r.get("q_cps", [])[:1]:
        if c["p"] < 0.05:
            x = MONTHS[c["k"]].to_timestamp().tz_localize("UTC")
            ax.axvline(x, color="#7c3aed", lw=1.0, ls="-.")
    c0 = r["cps"][0]
    sig = "유의" if c0["p"] < 0.05 else "약함" if c0["p"] < 0.2 else "없음"
    ax.set_title(f"{META[name][0]}\n변곡 {fmt_month(c0['k'])} · p {c0['p']:.3f} ({sig})", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def timeline(res) -> str:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    names = [n for n in ORDER if n in res]
    for i, n in enumerate(names[::-1]):
        r = res[n]
        for j, c in enumerate(r["cps"]):
            x = MONTHS[c["k"]].to_timestamp()
            up = c["after"] > c["before"]
            col = ("#16a34a" if up else "#dc2626") if c["p"] < 0.05 else "#9ca3af"
            ax.scatter([x], [i], s=70 if c["p"] < 0.05 else 35, color=col,
                       marker="^" if up else "v", zorder=3)
        for c in r.get("q_cps", [])[:1]:
            if c["p"] < 0.05:
                ax.scatter([MONTHS[c["k"]].to_timestamp()], [i], s=40, facecolors="none",
                           edgecolors="#7c3aed", lw=1.2, zorder=2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([META[n][0] for n in names[::-1]], fontsize=8)
    ax.axvline(pd.Timestamp("2024-01-01"), color="#60a5fa", ls=":", lw=1)
    ax.set_xlim(MONTHS[0].to_timestamp(), MONTHS[-1].to_timestamp())
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("신호별 변곡점 — ▲ 좋아짐 / ▼ 나빠짐 · 색 = 유의(p<0.05) · 회색 = 유의하지 않음 · ○ = 거래당 질의 변화점",
                 fontsize=9, loc="left")
    return fig_b64(fig)


def main() -> None:
    res = analyse()
    Path("runs/legacy_cp.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    names = [n for n in ORDER if n in res]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(13, 3.1 * rows))
    for ax, n in zip(axs.ravel(), names):
        panel(ax, n, res[n])
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    grid = fig_b64(fig)
    tl = timeline(res)
    Path("runs/legacy_figs.json").write_text(json.dumps(dict(grid=grid, timeline=tl)), encoding="utf-8")
    print("그림 완료")


if __name__ == "__main__":
    main()
