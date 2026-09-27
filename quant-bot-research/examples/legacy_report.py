"""예전 신호 16종 보고서 — 변곡점 원인 분해 + HTML + 문서용 표.

입력: runs/legacy_cp.json · runs/legacy_figs.json · runs/legacy/*.pkl
출력: out/legacy_curves.html · out/legacy_tables.md · runs/legacy_summary.csv
"""
from __future__ import annotations

import base64
import io
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import changepoint as CP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_changepoint import META, MONTHS, ORDER  # noqa: E402

D = Path("runs/legacy")


def vol_regime():
    """시장 변동성 = 심볼별 **월 실현변동성**(그 달 일간 로그수익 표준편차 × √365)의 횡단면 중앙.
    30일 롤링을 쓰면 달끼리 겹쳐 자기상관이 커지고, 첫 달을 메우는 방식에 변화점이 끌려간다
    (처음에 그렇게 해서 2021-10 이 나왔다). 겹치지 않는 월 단위로 잰다."""
    C = pd.read_pickle(D / "daily_close.pkl")
    C.index = pd.to_datetime(C.index)
    r = np.log(C).diff()
    per = r.index.tz_convert("UTC").tz_localize(None).to_period("M")
    mm = r.groupby(per).std().mul(np.sqrt(365)).median(axis=1).reindex(MONTHS)
    cp = CP.detect(mm.to_numpy(), seed=3, n_perm=4000)[0]
    return mm, cp


def decompose(name, r):
    """변곡 전후로 거래당 gross·수수료가 얼마나 변했나 → 순손익 변화의 원인."""
    if r["unit"] != "R":
        return None
    t = pd.read_pickle(D / f"{name}.pkl")
    t["time"] = pd.to_datetime(t.time, utc=True)
    t = t[t.time < pd.Timestamp("2026-07-01", tz="UTC")]
    c = MONTHS[r["cps"][0]["k"]].to_timestamp().tz_localize("UTC")
    a, b = t[t.time < c], t[t.time >= c]
    dg = b.gross.mean() - a.gross.mean()
    df_ = b.fee.mean() - a.fee.mean()
    dn = dg - df_
    if abs(dn) < 1e-9:
        cause = "—"
    elif abs(df_) >= 0.6 * (abs(dg) + abs(df_)):
        cause = "비용 ↑ (1R 축소)" if df_ > 0 else "비용 ↓"
    elif abs(dg) >= 0.6 * (abs(dg) + abs(df_)):
        cause = "엣지 ↓" if dg < 0 else "엣지 ↑"
    else:
        cause = "엣지·비용 둘 다"
    return dict(risk_a=float(a.risk_bps.median()), risk_b=float(b.risk_bps.median()),
                g_a=float(a.gross.mean()), g_b=float(b.gross.mean()),
                f_a=float(a.fee.mean()), f_b=float(b.fee.mean()),
                n_a=float(a.net.mean()), n_b=float(b.net.mean()), cause=cause)


def timeline(res, vcp) -> str:
    names = [n for n in ORDER if n in res]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    rows = names[::-1]
    for i, n in enumerate(rows):
        for c in res[n]["cps"]:
            x = MONTHS[c["k"]].to_timestamp()
            up = c["after"] > c["before"]
            col = ("#16a34a" if up else "#dc2626") if c["p"] < 0.05 else "#9ca3af"
            ax.scatter([x], [i], s=80 if c["p"] < 0.05 else 36, color=col, marker="^" if up else "v", zorder=3)
    i = len(rows)
    x = MONTHS[vcp.k].to_timestamp()
    ax.scatter([x], [i], s=90, color="#2563eb", marker="v", zorder=3)
    ax.axvspan(pd.Timestamp("2022-06-01"), pd.Timestamp("2022-09-30"), color="#dbeafe", zorder=0)
    ax.set_yticks(range(len(rows) + 1))
    ax.set_yticklabels([META[n][0] for n in rows] + ["[참고] 시장 변동성 (47심볼 월 실현변동성 중앙)"], fontsize=8)
    ax.axvline(pd.Timestamp("2024-01-01"), color="#60a5fa", ls=":", lw=1)
    ax.set_xlim(MONTHS[0].to_timestamp(), MONTHS[-1].to_timestamp())
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("변곡점 — ▲ 월손익 좋아짐 / ▼ 나빠짐 · 빨강·초록 = 유의(p<0.05) · 회색 = 유의하지 않음",
                 fontsize=9, loc="left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    res = json.loads(Path("runs/legacy_cp.json").read_text(encoding="utf-8"))
    figs = json.loads(Path("runs/legacy_figs.json").read_text(encoding="utf-8"))
    mm, vcp = vol_regime()
    tl = timeline(res, vcp)

    rows = []
    for n in ORDER:
        r = res[n]
        c = r["cps"][0]
        dec = decompose(n, r)
        rows.append(dict(key=n, 이름=META[n][0], 문서=META[n][1], 진입=META[n][2], 재구축=META[n][3],
                         거래수=r["n"], 순R=r["mean"], t=r["t"], gross=r.get("gross_mean", np.nan),
                         비용=r.get("fee_mean", np.nan), 변곡=str(MONTHS[c["k"]]), p=c["p"],
                         전=c["before"], 후=c["after"],
                         변곡2=(str(MONTHS[r["cps"][1]["k"]]) if len(r["cps"]) > 1 else ""),
                         질변곡=(str(MONTHS[r["q_cps"][0]["k"]]) if r.get("q_cps") else ""),
                         질p=(r["q_cps"][0]["p"] if r.get("q_cps") else np.nan),
                         전2024=r["pre"], 후2024=r["post"],
                         원인=(dec["cause"] if dec else "—"),
                         risk_a=(dec["risk_a"] if dec else np.nan), risk_b=(dec["risk_b"] if dec else np.nan),
                         g_a=(dec["g_a"] if dec else np.nan), g_b=(dec["g_b"] if dec else np.nan),
                         f_a=(dec["f_a"] if dec else np.nan), f_b=(dec["f_b"] if dec else np.nan),
                         sharpe=r.get("sharpe", np.nan)))
    S = pd.DataFrame(rows)
    S.to_csv("runs/legacy_summary.csv", index=False)

    def sig_str(p):
        return "**유의**" if p < 0.05 else ("약함" if p < 0.2 else "없음")

    md = []
    md.append("| 신호 | 진입 | 거래수 | 거래당 순R (t) | gross | 비용 | 변곡 | p | 월손익 전→후 | 원인 | 2024 전→후 순R |")
    md.append("|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|")
    for _, r in S.iterrows():
        if r.key == "tsmom":
            md.append(f"| {r.이름} | {r.진입} | {r.거래수}주 | 주 {r.순R * 100:+.2f}% (t {r.t:.2f}) · 샤프 {r.sharpe:.2f} | — | — | "
                      f"{r.변곡} | {r.p:.3f} | {r.전 * 100:+.2f}% → {r.후 * 100:+.2f}% | — | "
                      f"주 {r.전2024 * 100:+.2f}% → {r.후2024 * 100:+.2f}% |")
            continue
        md.append(f"| {r.이름} | {r.진입} | {r.거래수:,} | {r.순R:+.4f} ({r.t:.2f}) | {r.gross:+.4f} | {r.비용:.3f} | "
                  f"{r.변곡} | {r.p:.3f} {sig_str(r.p)} | {r.전:+.1f} → {r.후:+.1f} | {r.원인} | "
                  f"{r.전2024:+.4f} → {r.후2024:+.4f} |")
    table_main = "\n".join(md)

    md2 = ["| 신호 | 변곡 | 1R 중앙 전→후 (bps) | 거래당 gross 전→후 | 거래당 비용 전→후 | 거래당 순R 전→후 |",
           "|---|---|---:|---:|---:|---:|"]
    for _, r in S[S.key != "tsmom"].iterrows():
        md2.append(f"| {r.이름} | {r.변곡} | {r.risk_a:.0f} → {r.risk_b:.0f} | {r.g_a:+.4f} → {r.g_b:+.4f} | "
                   f"{r.f_a:.3f} → {r.f_b:.3f} | {r.g_a - r.f_a:+.4f} → {r.g_b - r.f_b:+.4f} |")
    table_dec = "\n".join(md2)

    md3 = ["| 신호 | 출처 문서 | 재구축 상태 |", "|---|---|---|"]
    for _, r in S.iterrows():
        md3.append(f"| {r.이름} | `{r.문서}.md` | {r.재구축} |")
    table_src = "\n".join(md3)

    Path("out").mkdir(exist_ok=True)
    Path("out/legacy_tables.md").write_text(
        "## 요약표\n\n" + table_main + "\n\n## 변곡 원인 분해\n\n" + table_dec + "\n\n## 출처·재구축\n\n" + table_src
        + f"\n\n시장 변동성 변곡: {MONTHS[vcp.k]} p {vcp.p:.3f} · 연율 {vcp.before:.0%} → {vcp.after:.0%}\n",
        encoding="utf-8")

    # ---------------------------------------------------------------- HTML
    def tr_rows():
        out = []
        for _, r in S.iterrows():
            pcol = "#dc2626" if (r.p < 0.05 and r.후 < r.전) else "#16a34a" if r.p < 0.05 else "var(--mut)"
            if r.key == "tsmom":
                cells = [f"주 {r.순R * 100:+.2f}%<small>샤프 {r.sharpe:.2f}</small>", "—", "—"]
                d24 = f"{r.전2024 * 100:+.2f}% → {r.후2024 * 100:+.2f}%"
                n_ = f"{r.거래수}주"
            else:
                cells = [f"{r.순R:+.4f}<small>t {r.t:.2f}</small>", f"{r.gross:+.4f}", f"{r.비용:.3f}"]
                d24 = f"{r.전2024:+.3f} → {r.후2024:+.3f}"
                n_ = f"{r.거래수:,}"
            out.append(f"<tr><th>{escape(r.이름)}<small>{escape(r.재구축)}</small></th><td>{r.진입}</td><td>{n_}</td>"
                       + "".join(f"<td>{c}</td>" for c in cells)
                       + f"<td style='color:{pcol};font-weight:600'>{r.변곡}<small>p {r.p:.3f}</small></td>"
                       + f"<td>{escape(r.원인)}</td><td>{d24}</td></tr>")
        return "".join(out)

    n_sig = int((S.p < 0.05).sum())
    n_q = int((S.질p < 0.05).sum())
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>예전 신호 변곡점</title>
<style>
:root{{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#1d4ed8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#171717;--fg:#e7e5e4;--mut:#a8a29e;--card:#1f1f1f;--line:#333;--acc:#93c5fd}}}}
:root[data-theme="dark"]{{--bg:#171717;--fg:#e7e5e4;--mut:#a8a29e;--card:#1f1f1f;--line:#333;--acc:#93c5fd}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 56px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:30px 0 10px}}
.sub{{color:var(--mut);margin-bottom:16px}}
.box{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:14px 18px}}
.box li{{margin:4px 0}}
.fig{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px;overflow-x:auto}}
.fig img{{width:100%;min-width:640px;height:auto;display:block}}
.tbl{{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}}
th:first-child{{text-align:left;white-space:normal;min-width:220px}}
td small,th small{{display:block;font-size:11px;color:var(--mut);font-weight:400}}
thead th{{color:var(--mut);font-weight:600;text-align:right}}
.note{{color:var(--mut);font-size:13px}}
</style></head><body><main>
<h1>예전 신호 16종 — 수익 곡선과 변곡점</h1>
<div class="sub">OKX 47심볼 5분봉 · 2021-04 ~ 2026-06 (봉인 81일 제외) · 비용 지정가 2 / 시장가 6 bps · 측정해상도 규칙 1R/일간ATR ≥ 2 · 거래당 1R 위험</div>

<div class="box"><ul>
<li><b>변곡점이 유의한 신호는 16개 중 {n_sig}개.</b> 셋(장대봉 CHoCH · 청산 반전 · 추세추종 15m)은 <b>2022년 7~8월</b>에 월손익이 나빠졌고, FVG 4h 는 <b>2025년 9월</b>에 꺾였다.</li>
<li><b>개별 p 는 16개 동시 검정을 보정하면 모두 문턱 밖이다</b>(가장 작은 0.009 × 16 = 0.15). 다만 유의 4개는 우연 기대치 0.8개보다 많고(이항 p ≈ 0.007), 그중 셋은 사실상 같은 사건이다 — 아래.</li>
<li><b>2022년 여름 변곡은 신호가 아니라 시장이 바꾼 것이다.</b> 47심볼 월 변동성이 같은 시기({MONTHS[vcp.k]}, p {vcp.p:.3f}) 연율 {vcp.before:.0%} → {vcp.after:.0%} 로 내려앉으면서 1R(bps)이 대부분 33~38% 줄었고, 고정 bps 수수료의 R 부담이 60~70% 늘었다. 청산 반전은 거래당 gross 가 +0.047 → +0.045 로 그대로인데 순손익만 꺾였다.</li>
<li><b>거래당 질(수수료 전 gross)만 떼어 보면 16개 중 유의한 변화점이 {n_q}개다.</b> 신호 자체가 어느 날 뒤집힌 흔적은 없다.</li>
<li><b>2024년에 공통 변곡은 없다.</b> 2024 전후로 순R 이 나빠진 신호와 좋아진 신호가 섞여 있다. 4h FVG 만 2024 부터 엣지가 줄고 2025-09 에 꺾였다.</li>
<li>비용 후에도 꾸준히 오른 곡선은 <b>시계열 추세 하나(샤프 0.64, 변곡 없음)</b>. 롱·숏 양쪽이 다 벌었다(롱 +83 / 숏 +63 bps/주, 수수료 전). 주말 갭은 2025 이후에만 올랐다.</li>
</ul></div>

<h2>변곡점 한눈에</h2>
<div class="fig"><img alt="변곡점 타임라인" src="data:image/png;base64,{tl}"></div>

<h2>수익 곡선 — 검정 = 수수료 후 · 회색 = 수수료 전 · 세로 점선 = 변곡(빨강 유의 / 주황 비유의) · 파란 점선 = 2024-01 · 시계열 추세의 회색 점선 = 매수보유</h2>
<div class="fig"><img alt="신호별 누적 R" src="data:image/png;base64,{figs['grid']}"></div>

<h2>신호별 요약</h2>
<div class="tbl"><table><thead><tr><th>신호 <small>재구축 상태</small></th><th>진입</th><th>거래수</th><th>거래당 순R</th><th>gross</th><th>비용 R</th><th>변곡</th><th>원인</th><th>2024 전→후 순R</th></tr></thead>
<tbody>{tr_rows()}</tbody></table></div>

<h2>읽는 법</h2>
<ul class="note">
<li>변곡 = 월별 순손익(Σ 순R)의 평균이 바뀐 달. 모든 분할점을 훑어 차이가 가장 큰 곳을 고르고, 3개월 블록을 섞은 귀무분포로 p 를 낸다 — "가장 큰 곳을 고른" 효과까지 보정한 값이다.</li>
<li>원인 = 변곡 전후 거래당 gross 변화와 비용 변화 중 어느 쪽이 순손익 변화의 60% 이상을 설명하나.</li>
<li>곡선의 R 합계는 모든 신호를 동시에 1R 씩 걸었다는 가정이라 크기는 비현실적이다. 기울기와 꺾임만 본다.</li>
<li>16개를 동시에 검정했으므로 우연히 유의가 나오는 기대 개수는 약 0.8개다.</li>
</ul>
</main></body></html>"""
    Path("out/legacy_curves.html").write_text(html, encoding="utf-8")
    print(S[["key", "변곡", "p", "원인", "질변곡", "질p"]].to_string(index=False))
    print(f"\n변동성 변곡 {MONTHS[vcp.k]} p {vcp.p:.3f} {vcp.before:.2f}→{vcp.after:.2f}")


if __name__ == "__main__":
    main()
