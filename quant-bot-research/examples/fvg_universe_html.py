"""47심볼 재검사 한 장 요약 (HTML). 숫자는 전부 runs/u47_*.csv 에서 다시 계산한다."""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research.yardstick import tstat

GATES = (100, 125, 150, 175, 200)
ORIG = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "TRX"}


def color(x: float, lim: float = 0.25) -> str:
    if not np.isfinite(x):
        return "var(--na)"
    a = min(abs(x) / lim, 1.0)
    return f"rgba(22,163,74,{0.12 + 0.6 * a:.2f})" if x > 0 else f"rgba(220,38,38,{0.12 + 0.6 * a:.2f})"


def year_grid(t: pd.DataFrame, title: str) -> str:
    yrs = sorted(t.year.unique())
    h = [f"<div class='grid'><h4>{escape(title)}</h4><table><tr><th>게이트</th>"
         + "".join(f"<th>{y}{'H1' if y == 2026 else ''}</th>" for y in yrs) + "<th>음수 해</th></tr>"]
    for g in GATES:
        s = t[t.risk_bps >= g]
        row, neg = [], 0
        for y in yrs:
            ys = s[s.year == y]
            if len(ys) >= 15:
                v = ys.net.mean()
                neg += v < 0
                row.append(f"<td style='background:{color(v)}'>{v:+.3f}<small>{len(ys)}</small></td>")
            else:
                row.append(f"<td class='na'>—<small>{len(ys)}</small></td>")
        h.append(f"<tr><th>{g}</th>{''.join(row)}<td class='neg'>{neg}</td></tr>")
    h.append("</table></div>")
    return "".join(h)


def main() -> None:
    t = pd.read_csv("runs/u47_trades.csv")
    info = pd.read_csv("runs/u47_info.csv").set_index("symbol")
    terc = pd.qcut(info.med_quote_vol.rank(method="first"), 3, labels=["얇음", "중간", "두꺼움"])
    t["liq"] = t.symbol.map(terc).astype(str)
    new = t[~t.symbol.isin(ORIG)]
    old = t[t.symbol.isin(ORIG)]

    def yr(sub, tf, g):
        """연도별 순R — 보고서와 같은 규칙: 거래 15건 미만인 해는 빈칸."""
        s = sub[(sub.tf == tf) & (sub.risk_bps >= g)]
        m = s.groupby("year").net.agg(["mean", "size"])
        return m["mean"].where(m["size"] >= 15)

    corr = {}
    for nm, sub in (("47심볼", t), ("신규38", new), ("기존9", old)):
        vals = []
        for g in GATES:
            a, b = yr(sub, "4h", g), yr(sub, "1h", g)
            k = a.index.intersection(b.index)
            a, b = a.loc[k], b.loc[k]
            ok = a.notna() & b.notna()
            vals.append(float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 else np.nan)
        corr[nm] = vals

    s150 = t[(t.tf == "4h") & (t.risk_bps >= 150)]
    n150 = new[(new.tf == "4h") & (new.risk_bps >= 150)]
    _, t_new, _ = tstat(n150.groupby("month").net.mean())

    hold_rows = []
    for tf in ("4h", "1h"):
        for g in GATES:
            cells = []
            for sub in (t, new):
                h = sub[(sub.tf == tf) & (sub.risk_bps >= g) & sub.holdout]
                v = h.net.mean()
                cells.append(f"<td style='background:{color(v, 0.12)}'>{v:+.3f}<small>{len(h)}</small></td>")
            hold_rows.append(f"<tr><th>{tf}</th><th>{g}</th>{''.join(cells)}</tr>")

    liq_rows = []
    yrs = sorted(t.year.unique())
    for tf in ("4h", "1h"):
        for lv in ("얇음", "중간", "두꺼움"):
            s = t[(t.tf == tf) & (t.risk_bps >= 150) & (t.liq == lv)]
            _, tv, _ = tstat(s.groupby("month").net.mean())
            neg = sum(1 for y in yrs if len(s[s.year == y]) >= 15 and s[s.year == y].net.mean() < 0)
            h = s[s.holdout].net.mean()
            liq_rows.append(f"<tr><th>{tf}</th><th>{lv}</th><td>{s.gross.mean():+.3f}</td>"
                            f"<td style='background:{color(s.net.mean(), 0.1)}'>{s.net.mean():+.3f}</td>"
                            f"<td>{tv:.2f}</td><td class='neg'>{neg}</td>"
                            f"<td style='background:{color(h, 0.12)}'>{h:+.3f}</td></tr>")

    ycells = "".join(f"<span class='chip' style='background:{color(v)}'>{int(y)}{'H1' if y == 2026 else ''} {v:+.3f}</span>"
                     for y, v in s150.groupby("year").net.mean().items())

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FVG 47심볼 재검사</title>
<style>
:root{{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--na:#f5f5f4;--acc:#b91c1c}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#171717;--fg:#e7e5e4;--mut:#a8a29e;--card:#1f1f1f;--line:#333;--na:#262626;--acc:#f87171}}}}
:root[data-theme="dark"]{{--bg:#171717;--fg:#e7e5e4;--mut:#a8a29e;--card:#1f1f1f;--line:#333;--na:#262626;--acc:#f87171}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 8px}}h4{{margin:0 0 6px;font-size:13px;color:var(--mut)}}
.sub{{color:var(--mut);margin-bottom:16px}}
.verdict{{border:1px solid var(--line);border-left:4px solid var(--acc);background:var(--card);padding:14px 16px;border-radius:8px}}
.verdict b{{color:var(--acc)}}
.q{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin-top:10px}}
.q div{{background:var(--na);border-radius:6px;padding:8px 10px}}
.grids{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px}}
.grid,.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:4px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{font-weight:600;color:var(--mut)}}td small{{display:block;font-size:10px;color:var(--mut)}}
td.na{{background:var(--na);color:var(--mut)}}td.neg{{font-weight:700}}
.chip{{display:inline-block;padding:3px 8px;border-radius:999px;margin:2px;font-variant-numeric:tabular-nums}}
ul{{padding-left:18px}}li{{margin:3px 0}}
</style></head><body><main>
<h1>FVG 1R 게이트 — 47심볼 재검사</h1>
<div class="sub">OKX 5분봉 47심볼 · 2021-04-02 ~ 2026-07-05 (이후 81일 봉인) · 셋업·게이트·비용은 10심볼 검사와 동일 · 순R = 비용 차감 후 R</div>

<div class="verdict"><b>기각 유지.</b> 사전 등록한 필수 세 항목(신규 심볼 재현 · 연도별 부호 · 홀드아웃)이 모두 미달.
10심볼의 4h↔1h 연도 상관 −0.93 은 잡음이었지만, 걷어내자 보이는 건 <b>2024 년 이후 두 TF 공통의 음수 구간</b>이다.
<div class="q">
<div><b>Q1 신규 38 재현</b><br>4h 게이트150 순R {n150.net.mean():+.3f} · 월 t {t_new:.2f} (기준 t ≥ 2) → 아니오</div>
<div><b>Q2 연도별 부호</b><br>{ycells}</div>
<div><b>Q3 −0.93 의 정체</b><br>게이트150 연도 상관: 47심볼 {corr['47심볼'][2]:+.2f} · 신규38 {corr['신규38'][2]:+.2f} · 기존9 {corr['기존9'][2]:+.2f}</div>
<div><b>Q4 홀드아웃</b><br>4h 다섯 게이트 전부 음수 (아래 표)</div>
</div></div>

<h2>연도별 순R — 4h (칸 아래 숫자 = 거래 수)</h2>
<div class="grids">{year_grid(t[t.tf == '4h'], '47심볼')}{year_grid(new[new.tf == '4h'], '신규 38심볼 (독립 재현)')}{year_grid(old[old.tf == '4h'], '기존 9심볼')}</div>

<h2>연도별 순R — 1h</h2>
<div class="grids">{year_grid(t[t.tf == '1h'], '47심볼')}{year_grid(new[new.tf == '1h'], '신규 38심볼 (독립 재현)')}{year_grid(old[old.tf == '1h'], '기존 9심볼')}</div>

<h2>4h↔1h 연도별 순R 상관</h2>
<div class="card"><table><tr><th>게이트</th>{''.join(f'<th>{g}</th>' for g in GATES)}</tr>
{''.join(f"<tr><th>{nm}</th>" + ''.join(f"<td style='background:{color(v, 1.0)}'>{v:+.2f}</td>" for v in vs) + "</tr>" for nm, vs in corr.items())}
</table></div>

<div class="grids" style="margin-top:12px">
<div class="card"><h4>홀드아웃 순R — 마지막 12개월</h4><table><tr><th>TF</th><th>게이트</th><th>47심볼</th><th>신규38</th></tr>{''.join(hold_rows)}</table></div>
<div class="card"><h4>보조 절단: 유동성 3분위 · 게이트150 (판정에 안 씀)</h4>
<table><tr><th>TF</th><th>분위</th><th>gross</th><th>순R</th><th>t(월)</th><th>음수 해</th><th>홀드아웃</th></tr>{''.join(liq_rows)}</table>
<ul><li>얇은 16심볼은 비용 전 gross 부터 음수.</li><li>나머지 31심볼도 음수 해 2~3개, 4h 홀드아웃은 게이트 다섯 개 전부 음수.</li><li>"유동 심볼만" 은 표를 보고 고르는 사후 선택이라 규칙으로 만들지 않는다.</li></ul></div>
</div>

<h2>검수</h2>
<ul>
<li>개정 스크립트로 기존 10심볼 캐시를 다시 돌려 9,264건 전부 동일(차이 0).</li>
<li>새 데이터의 겹치는 9심볼: 기존 거래 8,664건 중 8,657건 동일. 7건은 기존 캐시에 빠져 있던 봉 주변.</li>
<li>OKX 47심볼 파일은 manifest 와 47/47 일치, 공백·중복·OHLC 위반 0. 기존 캐시와 겹치는 480만 봉에서 가격·볼륨 완전 일치.</li>
<li>봉인 81일은 약 270건 → 표준오차 약 0.09R. +0.05R 짜리 효과는 확인할 수 없고, 큰 실패만 거를 수 있다.</li>
</ul>
</main></body></html>"""
    Path("out").mkdir(exist_ok=True)
    Path("out/fvg_universe47.html").write_text(html, encoding="utf-8")
    print("out/fvg_universe47.html", len(html))


if __name__ == "__main__":
    main()
