"""비용 상태 스위치 결과 보고서 (HTML) — 판정표 + 'bps 엣지 vs 비용' 점그림.

입력: runs/costgate_rules.csv · runs/costgate_key.json · out/costgate_tables.md · runs/legacy/*.pkl
출력: out/costgate.html · runs/costgate_bps.csv
"""
from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from costgate_run import OOS0, OOS1, SIGNALS, load  # noqa: E402


def bps_table() -> pd.DataFrame:
    rows = []
    for key, name, _e, _f in SIGNALS:
        t = load(key)
        t = t[(t.per >= OOS0) & (t.per <= OOS1)]
        gb = t.gross * t.risk_bps
        fb = t.fee * t.risk_bps
        q = pd.qcut(t.risk_bps.rank(method="first"), 3, labels=["small", "mid", "large"])
        g3 = gb.groupby(q, observed=True).mean()
        r3 = t.risk_bps.groupby(q, observed=True).median()
        rows.append(dict(key=key, name=name, n=len(t), gross=gb.mean(), cost=fb.mean(), net=(gb - fb).mean(),
                         g_small=g3["small"], g_mid=g3["mid"], g_large=g3["large"],
                         r_small=r3["small"], r_mid=r3["mid"], r_large=r3["large"]))
    return pd.DataFrame(rows).sort_values("gross", ascending=False).reset_index(drop=True)


def svg_dots(B: pd.DataFrame) -> str:
    W, left, right, top, rowh = 780, 210, 24, 58, 28
    H = top + rowh * len(B) + 40
    lo, hi = -60.0, 55.0
    x = lambda v: left + (min(max(v, lo), hi) - lo) / (hi - lo) * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="신호별 거래당 gross 엣지와 비용 (bps)" '
           f'style="width:100%;min-width:640px;height:auto;display:block;font-family:inherit">']
    # 범례
    lx = left
    for i, (lab, cls) in enumerate((("1R 작음", "s1"), ("중간", "s2"), ("1R 큼", "s3"))):
        out.append(f'<circle cx="{lx + 6}" cy="18" r="5" class="{cls}"/>'
                   f'<text x="{lx + 16}" y="22" class="lg">{lab}</text>')
        lx += 70 if i else 84
    out.append(f'<line x1="{lx + 6}" x2="{lx + 6}" y1="11" y2="25" class="cost"/>'
               f'<text x="{lx + 14}" y="22" class="lg">거래당 비용</text>')
    # 격자
    for v in range(-60, 60, 10):
        cls = "zero" if v == 0 else "grid"
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 8}" y2="{H - 34}" class="{cls}"/>')
        out.append(f'<text x="{x(v):.1f}" y="{H - 18}" class="tick" text-anchor="middle">{v:+d}</text>')
    out.append(f'<text x="{(left + W - right) / 2:.0f}" y="{H - 2}" class="tick" text-anchor="middle">'
               f'거래당 bps (표본 밖 2022-04~2026-06, 항상 켬)</text>')
    for i, r in B.iterrows():
        y = top + i * rowh + rowh / 2
        out.append(f'<text x="{left - 10}" y="{y + 4:.1f}" class="lab" text-anchor="end">{escape(r["name"])}</text>')
        vals = [r.g_small, r.g_mid, r.g_large]
        out.append(f'<line x1="{x(min(vals)):.1f}" x2="{x(max(vals)):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="span"/>')
        out.append(f'<g><title>{escape(r["name"])} · 거래당 비용 {r.cost:.1f}bps</title>'
                   f'<rect x="{x(r.cost) - 6:.1f}" y="{y - 10:.1f}" width="12" height="20" class="hit"/>'
                   f'<line x1="{x(r.cost):.1f}" x2="{x(r.cost):.1f}" y1="{y - 8:.1f}" y2="{y + 8:.1f}" class="cost"/></g>')
        for cls, v, rb, lab in (("s1", r.g_small, r.r_small, "1R 작음"), ("s2", r.g_mid, r.r_mid, "중간"),
                                ("s3", r.g_large, r.r_large, "1R 큼")):
            clip = " (축 밖)" if v < lo or v > hi else ""
            out.append(f'<g><title>{escape(r["name"])} · {lab} (1R 중앙 {rb:.0f}bps): gross {v:+.1f}bps{clip}</title>'
                       f'<circle cx="{x(v):.1f}" cy="{y:.1f}" r="10" class="hit"/>'
                       f'<circle cx="{x(v):.1f}" cy="{y:.1f}" r="5" class="{cls}"/></g>')
    out.append("</svg>")
    return "".join(out)


def md_tables_to_html(md: str) -> str:
    html, rows, in_tbl = [], [], False
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            if re.match(r"^\|[-:| ]+\|$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
            in_tbl = True
            continue
        if in_tbl:
            head, *body = rows
            h = "".join(f"<th>{c}</th>" for c in head)
            b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
            html.append(f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
            rows, in_tbl = [], False
        if line.startswith("## "):
            html.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.strip():
            html.append(f"<p>{line}</p>")
    s = "\n".join(html)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


def main() -> None:
    B = bps_table()
    B.to_csv("runs/costgate_bps.csv", index=False)
    v = json.loads(Path("runs/costgate_key.json").read_text(encoding="utf-8"))["verdict"]
    n_pass = sum(x["passed"] for x in v.values())
    tables = md_tables_to_html(Path("out/costgate_tables.md").read_text(encoding="utf-8"))
    wg = B[B.key == "weekend_gap"].iloc[0]
    others_max = B[B.key != "weekend_gap"].gross.max()
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>비용 상태 스위치 결과</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#8a8984;--line:#e4e3df;--grid:#ecebe7;--s1:#86b6ef;--s2:#2a78d6;--s3:#104281;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#8f8e86;--line:#34332f;
--grid:#2a2a27;--s1:#184f95;--s2:#3987e5;--s3:#9ec5f4;--acc:#3987e5}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#8f8e86;--line:#34332f;--grid:#2a2a27;--s1:#184f95;--s2:#3987e5;--s3:#9ec5f4;--acc:#3987e5}}
*{{box-sizing:border-box}}body{{margin:0}}
.viz-root{{background:var(--bg);color:var(--text-primary);font:14px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;min-height:100vh}}
main{{max-width:1060px;margin:0 auto;padding:24px 16px 56px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 10px}}
.sub{{color:var(--text-secondary);margin-bottom:16px}}
.box{{background:var(--surface-1);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 18px}}
.box li{{margin:4px 0}}
.card{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}}
.tbl{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;overflow-x:auto;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--text-secondary);font-weight:600}}
p{{color:var(--text-secondary)}}
svg .lab{{fill:var(--text-primary);font-size:12px}}svg .lg,svg .tick{{fill:var(--text-secondary);font-size:11px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .zero{{stroke:var(--muted);stroke-width:1}}
svg .span{{stroke:var(--line);stroke-width:2}}svg .cost{{stroke:var(--text-primary);stroke-width:2;stroke-linecap:round}}
svg .s1{{fill:var(--s1);stroke:var(--surface-1);stroke-width:2}}svg .s2{{fill:var(--s2);stroke:var(--surface-1);stroke-width:2}}
svg .s3{{fill:var(--s3);stroke:var(--surface-1);stroke-width:2}}svg .hit{{fill:transparent}}
svg g:hover .s1,svg g:hover .s2,svg g:hover .s3{{stroke:var(--text-primary)}}
</style></head><body><div class="viz-root"><main>
<h1>비용 상태 스위치 — 사전등록 결과</h1>
<div class="sub">예전 신호 15종 · OKX 47심볼 · 표본 밖 2022-04 ~ 2026-06 · 주 규칙 ρ = 1R(bps) × ĝ / 비용 ≥ 2 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>판정: 15개 중 통과 {n_pass}개.</b> 스위치를 켠 거래가 양수로 간 신호가 없다. 봉인 창은 그대로 닫아 둔다.</li>
<li><b>원인은 사전등록에 적어 둔 실패 모드 1번 그대로다.</b> 1R 이 큰 거래일수록 거래당 gross(R)가 작다 —
    15개 중 14개에서 1R 3분위 '큼'의 gross 가 '작음'보다 낮다(예외 FVG 관통 → CHoCH 는 둘 다 음수).
    스위치가 비용을 아끼는 만큼 엣지도 같이 빠진다.</li>
<li><b>bps 로 보면 이유가 보인다.</b> 엣지는 1R 을 키워도 커지지 않는다 — 대부분 거래당 −8 ~ +5 bps 인데
    비용은 6.6 ~ 11.5 bps 로 고정이다. 1R 이나 시장 변동성으로 골라도 부호가 바뀌지 않는다.</li>
<li>bps 로 비용을 넘는 건 주말 갭 하나다: gross {wg.gross:+.1f}bps &gt; 비용 {wg.cost:.1f}bps (나머지 최대 {others_max:+.1f}bps).
    하지만 항상 켬 순R +0.040 의 월 군집 t 는 1.04 이고, 2025 년을 빼면 +0.002R (t 0.05) 이다 — 상위 10개 주말이
    순이익의 142% 를 낸다. 이미 기각된 신호(주말갭-OOS결과-기각 문서)의 '2025 한 해' 패턴이 47심볼에서도 그대로다.</li>
<li>보조로 본 시장 고변동 스위치도 같다 — 고변동기엔 1R 이 커져 비용은 가벼워지지만 gross 가 더 줄어,
    gross 엣지가 뚜렷했던 신호(FVG 4h · 주말 갭 · ATR)는 오히려 나빠졌다.</li>
</ul></div>

<h2>거래당 gross 엣지(bps) — 1R 크기 3분위별, 세로 눈금 = 거래당 비용</h2>
<div class="card">{svg_dots(B)}</div>
<p>점이 비용 눈금보다 오른쪽에 있어야 수수료를 넘는다. 15개 중 14개에서 '1R 큼' 점이 세 점 중 가장 왼쪽이다
(예외 FVG mid 4h) — 1R 이 큰 거래를 골라도 bps 엣지가 따라 커지지 않는다. 점에 마우스를 올리면
정확한 값이 나온다.</p>

{tables}
</main></div></body></html>"""
    Path("out/costgate.html").write_text(html, encoding="utf-8")
    print(B[["name", "gross", "cost", "net", "g_small", "g_mid", "g_large"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
