"""주말 갭 × 4H 본장정렬 × 15m CISD — HTML 보고서.

입력 runs/wgap_key.json · out/wgap_tables.md (examples/wgap_stats.py 산출) → out/wgap.html
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from costgate_report import md_tables_to_html  # noqa: E402


def _hbar(x0: float, x1: float, y: float, h: float, cls: str, r: float = 3.0) -> str:
    """가로 막대 — 바깥쪽 끝만 둥글게(기준선 쪽은 각지게)."""
    if abs(x1 - x0) < 0.5:
        return ""
    right = x1 > x0
    r = min(r, abs(x1 - x0) / 2, h / 2)
    if right:
        d = (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
             f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")
    else:
        d = (f"M{x0:.1f},{y:.1f} H{x1 + r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
             f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 + r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return f'<path d="{d}" class="{cls}"/>'


def race_chart(d0: list[dict]) -> str:
    groups = [["전체"], ["정렬 A+", "역정렬 A−"], ["위로 갭 (숏 메움)", "아래로 갭 (롱 메움)"],
              [f"{d}요일에 메움" for d in "월화수목금"]]
    by = {r["group"]: r for r in d0 if r.get("n")}
    W, left, right, top, rowh, gap = 760, 170, 118, 46, 28, 12
    rows = sum(len(g) for g in groups)
    H = top + rows * rowh + gap * (len(groups) - 1) + 44
    cx = left + (W - left - right) / 2
    span = 0.6                                              # ±60%
    sx = lambda v: (W - left - right) / 2 / span * v
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="메운 뒤 1:1 레이스 결과 — 반전 먼저 대 연속 먼저" '
           f'style="width:100%;min-width:620px;height:auto;display:block;font-family:inherit">']
    out.append(f'<rect x="{left}" y="14" width="12" height="12" rx="2" class="rev"/>'
               f'<text x="{left + 18}" y="24" class="lg">반전 먼저 (월 개장가로 돌아감)</text>'
               f'<rect x="{cx + 40}" y="14" width="12" height="12" rx="2" class="cont"/>'
               f'<text x="{cx + 58}" y="24" class="lg">연속 먼저 (한 갭 더 감)</text>')
    for v in (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6):
        x = cx + sx(v)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{H - 38}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{H - 22}" class="tick" text-anchor="middle">{abs(v) * 100:.0f}%</text>')
    out.append(f'<text x="{W - right + 10}" y="{top - 12}" class="tick">못 닿음</text>')
    y = top
    for gi, g in enumerate(groups):
        for name in g:
            r = by.get(name)
            if r is None:
                continue
            yb = y + (rowh - 16) / 2
            out.append(f'<text x="{left - 10}" y="{y + rowh / 2 + 4:.1f}" class="lab" text-anchor="end">'
                       f'{escape(name)}</text>')
            tip = (f"{name} · 연속 먼저 {r['cont'] * 100:.1f}% · 반전 먼저 {r['rev'] * 100:.1f}% · "
                   f"못 닿음 {r['none'] * 100:.1f}% · {r['n']:,}건")
            out.append(f'<g><title>{escape(tip)}</title>'
                       f'<rect x="{left}" y="{y}" width="{W - left - right}" height="{rowh}" class="hit"/>'
                       + _hbar(cx - 1, cx - 1 - sx(r["rev"]), yb, 16, "rev")
                       + _hbar(cx + 1, cx + 1 + sx(r["cont"]), yb, 16, "cont")
                       + f'<text x="{cx - 5 - sx(r["rev"]):.1f}" y="{yb + 12:.1f}" class="val" text-anchor="end">'
                         f'{r["rev"] * 100:.0f}</text>'
                       + f'<text x="{cx + 5 + sx(r["cont"]):.1f}" y="{yb + 12:.1f}" class="val">'
                         f'{r["cont"] * 100:.0f}</text>'
                       + f'<text x="{W - right + 10}" y="{yb + 12:.1f}" class="val2">{r["none"] * 100:.0f}% · '
                         f'{r["n"]:,}</text></g>')
            y += rowh
        y += gap
    out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{top - 6}" y2="{H - 38}" class="zero"/>')
    out.append(f'<text x="{cx:.1f}" y="{H - 6}" class="tick" text-anchor="middle">'
               f'메운 뒤 금 16:00 ET 까지 — 어느 쪽에 먼저 닿았나 (주·심볼 비율)</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/wgap_key.json").read_text(encoding="utf-8"))
    S, V, I = K["stats"], K["verdict"], K["info"]
    d0 = {r["group"]: r for r in K["d0"]}
    n_pass = sum(v["passed"] for v in V.values())
    f = lambda k, x="net": S[k][x]
    tables = md_tables_to_html(Path("out/wgap_tables.md").read_text(encoding="utf-8"))
    a4 = I["align4_share"]
    a24 = I["align24_share"]
    box = f"""
<li><b>판정: 사전등록 5개 중 통과 {n_pass}개.</b> 봉인 창은 그대로 닫아 둔다.
    기본형(필터 없음)은 {S['B0']['n']:,}건 {f('B0'):+.3f}R (t {f('B0', 't'):.2f}), 2025 를 빼면 {f('B0', 'ex25'):+.3f}R.</li>
<li><b>4H 정렬 — 주말 봉을 빼야 필터가 된다.</b> 주말 포함 4H 로 보면 {a24['-1'] * 100:.0f}% 가 '역정렬' 이다
    (주말 움직임이 이미 4H 구조를 갭 방향으로 돌려놓는다). 주말 봉을 빼면 정렬 {a4['1'] * 100:.0f}% · 역정렬 {a4['-1'] * 100:.0f}% 로 갈린다.
    그런데 <b>정렬(메움 = 추세 방향)이 {f('A+'):+.3f}R (t {f('A+', 't'):.2f})로 기본형보다 나쁘다.</b>
    역정렬이 {f('A−'):+.3f}R 로 오히려 낫지만 둘의 차이는 유의하지 않고(월별 차이 t 0.56), 2025 를 빼면 둘 다 0 근처.</li>
<li><b>CISD 진입 확인은 거르지 못하고 비싸게만 만든다.</b> 기본형 주의 {I['c_share_of_b0'] * 100:.0f}% 에서 개장 후 중앙
    {I['c_delay_h_median']:.0f}시간 만에 CISD 가 나온다. 손절이 가까워져 1R 이 {f('B0', 'risk'):.0f} → {f('C', 'risk'):.0f}bps,
    비용이 {f('B0', 'fee'):.3f} → {f('C', 'fee'):.3f}R 로 커지고 gross 는 {f('C', 'gross'):+.3f} 로 0 이라 순 {f('C'):+.3f}R.
    4H 정렬과 합치면 {f('A+∧C'):+.3f}R.</li>
<li><b>메운 뒤: 되돌아가기보다 한 갭 더 가는 쪽이 조금 많다.</b> 연속 먼저 {d0['전체']['cont'] * 100:.1f}% · 반전 먼저
    {d0['전체']['rev'] * 100:.1f}% · 금요일까지 못 닿음 {d0['전체']['none'] * 100:.1f}% (6구간 중 5개에서 연속 &gt; 반전, 2023 은 같음).
    그래서 <b>금 종가 레벨은 메운 뒤 지지·저항이 아니다</b> — 메우는 순간 반대로 잡으면 {f('D1'):+.3f}R (t {f('D1', 't'):.2f}),
    반전 CISD 를 기다려 잡으면 {f('D-CISD'):+.3f}R (t {f('D-CISD', 't'):.2f}, 6구간 모두 음수)로 더 나쁘다.</li>
<li>연속 쪽은 gross {f('D2', 'gross'):+.3f} 이지만 역지정가 수수료를 내면 {f('D2'):+.3f}R (t {f('D2', 't'):.2f}).
    4H 추세 방향만 타면(D-T) {f('D-T'):+.3f}R — 6구간 중 5개 양수지만 t {f('D-T', 't'):.2f} 로 크기가 없다.</li>
"""
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>주말 갭 · 4H 정렬 · CISD</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--cont:#2a78d6;--rev:#eb6834;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--cont:#3987e5;--rev:#d95926;--acc:#3987e5}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--cont:#3987e5;--rev:#d95926;--acc:#3987e5}}
*{{box-sizing:border-box}}body{{margin:0}}
.viz-root{{background:var(--bg);color:var(--text-primary);font:14px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;min-height:100vh}}
main{{max-width:1060px;margin:0 auto;padding:24px 16px 56px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 10px}}
.sub{{color:var(--text-secondary);margin-bottom:16px}}
.box{{background:var(--surface-1);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 18px}}
.box ul{{padding-left:18px;margin:0}}.box li{{margin:6px 0}}
.card{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}}
.tbl{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;overflow-x:auto;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--text-secondary);font-weight:600}}
p{{color:var(--text-secondary)}}
svg .lab{{fill:var(--text-primary);font-size:12px}}svg .lg,svg .tick{{fill:var(--text-secondary);font-size:11px}}
svg .val{{fill:var(--text-primary);font-size:11px;font-variant-numeric:tabular-nums}}
svg .val2{{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .zero{{stroke:var(--base);stroke-width:1.5}}
svg .cont{{fill:var(--cont)}}svg .rev{{fill:var(--rev)}}svg .hit{{fill:transparent}}
svg g:hover .hit{{fill:var(--grid);fill-opacity:.45}}
</style></head><body><div class="viz-root"><main>
<h1>주말 갭 × 4H 본장정렬 × 15m CISD</h1>
<div class="sub">OKX 47심볼 · {I['first'][:10]} ~ {I['last'][:10]} ({I['weekends']}주) · |갭| ≥ 200bps · 사전등록 판정 ·
봉인 창은 열지 않음</div>
<div class="box"><ul>{box}</ul></div>

<h2>메운 뒤 어디로 가나 — 금 종가(F)에서 한 갭 더 vs 월 개장가(O)</h2>
<div class="card">{race_chart(K['d0'])}</div>
<p>같은 주 금요일 16:00 ET 전에 메운 주만 센다({d0['전체']['n']:,}건, 메움률 {I['filled_week'] * 100:.1f}%, 메움까지 중앙
{I['fill_h_median']:.0f}시간). 두 목표는 F 에서 같은 거리(G)라 무작위 보행이면 반반이다. 막대 끝 숫자는 %, 오른쪽은
못 닿은 비율과 건수. 막대에 마우스를 올리면 정확한 값이 나온다. 요일별로 두 칸이 t 2 를 넘지만 10개 묶음 중이라 우연 범위이고,
금요일 메움은 시간이 모자라 {d0['금요일에 메움']['none'] * 100:.0f}% 가 미해결이다.</p>

{tables}
<p>정의 · 판정 기준은 프로젝트 문서 <b>주말갭-4H정렬-CISD-사전등록</b> 그대로다. 모든 셀은 만료를 시장가(MTM)로 평가하고
측정해상도 규칙(1R ≥ 2 × 일간중앙 5m ATR14)을 건다. '기본형 · CISD 가 먼저 안 난 주' 는 CISD 진입이 성립하지 않은 주
(메움 전에 CISD 가 없었거나 손절이 너무 가까워 규칙에 걸린 주)라 사후에야 알 수 있는 구분이다 — 필터로 쓸 수 없다.</p>
</main></div></body></html>"""
    Path("out/wgap.html").write_text(html, encoding="utf-8")
    print("out/wgap.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
