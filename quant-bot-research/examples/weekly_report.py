"""주봉 캔들 + TGIF — HTML 보고서.

입력 runs/weekly_candles.json · runs/tgif.json · runs/weekly_extra.json → out/weekly_tgif.html
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np


def bar(x, y0, y1, w, cls, r=3.0):
    h = abs(y1 - y0)
    if h < 0.5:
        return ""
    r = min(r, w / 2, h / 2)
    if y1 < y0:
        d = (f"M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} "
             f"Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z")
    else:
        d = (f"M{x:.1f},{y0:.1f} V{y1 - r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} "
             f"Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 - r:.1f} V{y0:.1f} Z")
    return f'<path d="{d}" class="{cls}"/>'


def svg(W, H, aria, minw=560):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;height:auto;display:block;font-family:inherit">')


def rules_chart(S: dict) -> str:
    names = ["기준", "F1", "F2", "F3"]
    labels = {"기준": "기준 (전주 색)", "F1": "F1 결단 캔들", "F2": "F2 역방향 꼬리 없음", "F3": "F3 거부 꼬리 반전"}
    W, H, l, r, t, b = 720, 300, 56, 16, 40, 44
    lo, hi = -0.6, 0.8
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    gw = (W - l - r) / 4
    bw = 34
    out = [svg(W, H, "규칙별 순 샤프 — 발견 구간 대 확인 구간")]
    out.append(f'<rect x="{l}" y="10" width="12" height="12" rx="2" class="s1"/><text x="{l + 18}" y="20" class="lg">발견 2021-04 ~ 2023-12</text>'
               f'<rect x="{l + 190}" y="10" width="12" height="12" rx="2" class="s2"/><text x="{l + 208}" y="20" class="lg">확인 2024-01 ~ 2026-06</text>')
    for v in np.arange(-0.6, 0.81, 0.2):
        cls = "base" if abs(v) < 1e-9 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:+.1f}</text>')
    for i, k in enumerate(names):
        cx = l + i * gw + gw / 2
        d, c = S[k]["발견"]["sharpe"], S[k]["확인"]["sharpe"]
        tip = f"{labels[k]} · 발견 샤프 {d:+.2f} · 확인 샤프 {c:+.2f} · 전체 {S[k]['전체']['sharpe']:+.2f} · 평균 총노출 {S[k]['gross']:.2f}"
        out.append(f'<g><title>{escape(tip)}</title><rect x="{l + i * gw:.1f}" y="{t}" width="{gw:.1f}" height="{H - t - b}" class="hit"/>')
        for j, (v, cls) in enumerate(((d, "s1"), (c, "s2"))):
            x = cx - bw - 1 + j * (bw + 2)
            out.append(bar(x, sy(0), sy(v), bw, cls))
            ty = sy(v) - 5 if v >= 0 else sy(v) + 13
            out.append(f'<text x="{x + bw / 2:.1f}" y="{ty:.1f}" class="val" text-anchor="middle">{v:+.2f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - b + 18}" class="tick" text-anchor="middle">{escape(labels[k])}</text></g>')
    out.append("</svg>")
    return "".join(out)


def tgif_chart(A: dict) -> str:
    rows = [("D", "h1", "하락형 · 목 (월 혼조 → 화·수 약상승 뒤)", -1), ("U", "h1", "상승형 · 목 (월 혼조 → 화·수 약하락 뒤)", +1),
            ("D", "h2", "하락형 · 금 (목 큰 하락 + 저점 이탈 뒤)", +1), ("U", "h2", "상승형 · 금 (목 큰 상승 + 고점 돌파 뒤)", -1)]
    W, l, r, t, rowh = 760, 290, 30, 44, 44
    H = t + rowh * len(rows) + 44
    lo, hi = -0.6, 0.6
    sx = lambda v: l + (min(max(v, lo), hi) - lo) / (hi - lo) * (W - l - r)
    out = [svg(W, H, "TGIF — 목요일·금요일 조건부 평균(σ) 대 다른 요일 위약", 640)]
    out.append(f'<circle cx="{l + 6}" cy="16" r="5" class="pt"/><text x="{l + 16}" y="20" class="lg">본 (목·금) ± 95%</text>'
               f'<circle cx="{l + 146}" cy="16" r="5" class="pl"/><text x="{l + 156}" y="20" class="lg">위약 (같은 조건, 다른 요일)</text>'
               f'<text x="{l + 350}" y="20" class="lg">▲ 예측 방향</text>')
    for v in np.arange(-0.6, 0.61, 0.2):
        cls = "base" if abs(v) < 1e-9 else "grid"
        out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t - 8}" y2="{H - 40}" class="{cls}"/>'
                   f'<text x="{sx(v):.1f}" y="{H - 24}" class="tick" text-anchor="middle">{v:+.1f}σ</text>')
    for i, (form, h, name, sign) in enumerate(rows):
        y = t + i * rowh + rowh / 2
        a = A[form][h]
        m, tt = a["main"]["mean"], a["main"]["t"]
        se = abs(m / tt) if tt else np.nan
        pl = a["placebo"]["mean"]
        tip = (f"{name} · 본 {m:+.3f}σ (t {tt:+.2f}, {a['main']['n']}건 · {a['main']['weeks']}주) · 위약 {pl:+.3f}σ "
               f"({a['placebo']['n']}건) · 차이 t {a['diff']['t']:+.2f} · 반기 {a['halves']['발견']['mean']:+.2f} / {a['halves']['확인']['mean']:+.2f} · "
               f"예측 {'+' if sign > 0 else '−'} → {'통과' if a['passed'] else '기각'}")
        out.append(f'<g><title>{escape(tip)}</title><rect x="0" y="{y - rowh / 2:.1f}" width="{W}" height="{rowh}" class="hit"/>'
                   f'<text x="{l - 12}" y="{y + 4:.1f}" class="lab" text-anchor="end">{escape(name)}</text>'
                   f'<line x1="{sx(m - 1.96 * se):.1f}" x2="{sx(m + 1.96 * se):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci"/>'
                   f'<circle cx="{sx(pl):.1f}" cy="{y:.1f}" r="5" class="pl"/>'
                   f'<circle cx="{sx(m):.1f}" cy="{y:.1f}" r="5.5" class="pt"/>'
                   f'<text x="{sx(0.55 if sign > 0 else -0.55):.1f}" y="{y + 4:.1f}" class="arrow" text-anchor="middle">▲</text></g>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 6}" class="tick" text-anchor="middle">그날 수익 ÷ 직전 30일 일간 변동성 (σ 단위) · 뉴욕 일봉</text>')
    out.append("</svg>")
    return "".join(out)


def dow_chart(D: dict) -> str:
    days = list("월화수목금토일")
    W, H, l, r, t, b = 640, 260, 56, 16, 24, 50
    lo, hi = -0.15, 0.15
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    gw = (W - l - r) / 7
    bw = 36
    out = [svg(W, H, "요일별 평균 수익 (σ 단위, 조건 없음)", 520)]
    for v in (-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:+.2f}</text>')
    for i, d in enumerate(days):
        v = D[d]
        x = l + i * gw + (gw - bw) / 2
        tip = f"{d}요일 · 평균 {v['z']:+.3f}σ ({v['pct']:+.2f}%) · t {v['t']:+.2f} · 오른 날 {v['up'] * 100:.1f}% · {v['n']:,}건"
        cls = "s1" if d in "목금" else "s1m"
        out.append(f'<g><title>{escape(tip)}</title><rect x="{l + i * gw:.1f}" y="{t}" width="{gw:.1f}" height="{H - t - b}" class="hit"/>'
                   + bar(x, sy(0), sy(v["z"]), bw, cls)
                   + f'<text x="{x + bw / 2:.1f}" y="{(sy(v["z"]) - 5) if v["z"] >= 0 else (sy(v["z"]) + 13):.1f}" class="val" text-anchor="middle">{v["z"]:+.2f}</text>'
                   + f'<text x="{x + bw / 2:.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{d} (t {v["t"]:+.1f})</text></g>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 8}" class="tick" text-anchor="middle">뉴욕 일봉 · 47심볼 · 2021-04 ~ 2026-06 · 목·금 강조</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    A = json.loads(Path("runs/weekly_candles.json").read_text(encoding="utf-8"))
    T = json.loads(Path("runs/tgif.json").read_text(encoding="utf-8"))
    X = json.loads(Path("runs/weekly_extra.json").read_text(encoding="utf-8"))
    S, fx, vf = A["strategies"], A["filter_tests"], A["verdict_filters"]
    ny, utc = T["NY"], T["UTC"]
    an = ny["analysis"]
    fp = ny["full"]
    maxt = max(abs(v["발견"]["t"]) for v in A["cells"].values() if v["발견"]["t"] == v["발견"]["t"])
    cell_rows = "".join(
        f"<tr><td>{escape(c)}</td><td>{v['발견']['mean'] * 100:+.2f}%</td><td>{v['발견']['t']:+.2f}</td><td>{v['발견']['weeks']}</td>"
        f"<td>{v['확인']['mean'] * 100:+.2f}%</td><td>{v['확인']['t']:+.2f}</td><td>{v['전체']['n']:,}</td><td>{v['전체']['same'] * 100:.1f}%</td></tr>"
        for c, v in sorted(A["cells"].items()) if v["전체"]["n"] >= 30)
    br = A["breaks"]
    brk_rows = "".join(
        f"<tr><td>{escape(g)}</td><td>{v['n']:,}</td><td>{v['up'] * 100:.1f}%</td><td>{v['dn'] * 100:.1f}%</td>"
        f"<td>{X['clv'][g]['clv']:.2f}</td><td>{X['clv'][g]['high_first_given_touch'] * 100:.1f}%</td>"
        f"<td>{v['same'] * 100:.1f}%</td><td>{v['adj'] * 100:+.2f}%</td></tr>" for g, v in br.items())
    strat_rows = "".join(
        f"<tr><td>{k}</td><td>{v['전체']['sharpe']:+.2f}</td><td>{v['전체']['t']:+.2f}</td><td>{v['발견']['sharpe']:+.2f}</td>"
        f"<td>{v['확인']['sharpe']:+.2f}</td><td>{v['전체']['mdd'] * 100:.0f}%</td><td>{v['gross']:.2f}</td>"
        f"<td>{'—' if k == '기준' else ('통과' if vf[k]['passed'] else '기각')}</td></tr>" for k, v in S.items())

    def trow(form, h, nm):
        a = an[form][h]
        return (f"<tr><td>{escape(nm)}</td><td>{a['main']['mean']:+.3f}</td><td>{a['main']['t']:+.2f}</td><td>{a['main']['n']:,} · {a['main']['weeks']}주</td>"
                f"<td>{a['placebo']['mean']:+.3f}</td><td>{a['diff']['t']:+.2f}</td>"
                f"<td>{a['halves']['발견']['mean']:+.2f} / {a['halves']['확인']['mean']:+.2f}</td><td>{'통과' if a['passed'] else '기각'}</td></tr>")
    tg_rows = (trow("D", "h1", "H1 하락형 — 목 (예측 −)") + trow("U", "h1", "H1 상승형 — 목 (예측 +)")
               + trow("D", "h2", "H2 하락형 — 금 (예측 +)") + trow("U", "h2", "H2 상승형 — 금 (예측 −)"))
    b1d, b1u = an["D"]["h1b"], an["U"]["h1b"]
    trd, tru = an["D"]["trades"], an["U"]["trades"]
    ua = utc["analysis"]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>주봉 캔들 · TGIF 검정</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s1m:#86b6ef;--ci:#86b6ef;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--s1m:#184f95;--ci:#184f95;--acc:#3987e5}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--s1m:#184f95;--ci:#184f95;--acc:#3987e5}}
*{{box-sizing:border-box}}body{{margin:0}}
.viz-root{{background:var(--bg);color:var(--text-primary);font:14px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;min-height:100vh}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 56px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:34px 0 10px}}h3{{font-size:15px;margin:22px 0 8px}}
.sub{{color:var(--text-secondary);margin-bottom:16px}}
.box{{background:var(--surface-1);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 18px}}
.box ul{{padding-left:18px;margin:0}}.box li{{margin:6px 0}}
.card{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}}
.tbl{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;overflow-x:auto;margin:10px 0}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:var(--text-secondary);font-weight:600}}
p{{color:var(--text-secondary)}}
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .val{{fill:var(--text-primary);font-size:11px;font-variant-numeric:tabular-nums}}
svg .lab{{fill:var(--text-primary);font-size:12px}}svg .arrow{{fill:var(--muted);font-size:11px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .s1{{fill:var(--s1)}}svg .s1m{{fill:var(--s1m)}}svg .s2{{fill:var(--s2)}}
svg .pt{{fill:var(--s1);stroke:var(--surface-1);stroke-width:2}}svg .pl{{fill:var(--s2);stroke:var(--surface-1);stroke-width:2}}
svg .ci{{stroke:var(--ci);stroke-width:3;stroke-linecap:round}}
svg .hit{{fill:transparent}}svg g:hover .hit{{fill:var(--grid);fill-opacity:.45}}
</style></head><body><div class="viz-root"><main>
<h1>주봉 캔들(몸통·꼬리) 패턴 · TGIF 요일 패턴 — 사전등록 검정</h1>
<div class="sub">OKX 47심볼 · 2021-04 ~ 2026-06 · 봉인 전 · 주봉 = UTC 월 00:00 · 요일 = 뉴욕 자정 · t 는 주 군집</div>
<div class="box"><ul>
<li><b>판정: 둘 다 없다.</b> 주봉 몸통·꼬리 필터 3개 모두 기각, 14칸 중 후보로 뽑힌 칸 0개, TGIF 4개 가설 모두 기각. 봉인 창은 열지 않는다.</li>
<li><b>전주 색 따라가기(기준)</b>는 샤프 {S['기준']['전체']['sharpe']:.2f} (t {S['기준']['전체']['t']:.2f}) — 28일 추세(0.64)보다 약하다.
    몸통·꼬리 필터는 <b>2021~23 에서는 전부 좋아 보였다가 2024~26 에서 전부 나빠졌다</b>(F3 거부 꼬리 반전은 확인 구간 샤프 {S['F3']['확인']['sharpe']:+.2f}).</li>
<li><b>"다음 주가 전주 고점을 먼저 치나 저점을 먼저 치나" 는 꼬리 모양이 아니라 전주 종가 위치로 전부 설명된다.</b>
    예: 꼬리 짧은 양봉 뒤 고점 먼저 {X['clv']['양봉 · 짧음']['high_first_given_touch'] * 100:.0f}% — 종가가 범위의 {X['clv']['양봉 · 짧음']['clv'] * 100:.0f}% 위치(고점 가까이)라 생기는 기하학이다.</li>
<li><b>TGIF 는 이 데이터에 없다.</b> 월 혼조 · 화·수 약상승 뒤 목요일은 평균 {an['D']['h1']['main']['mean']:+.2f}σ (t {an['D']['h1']['main']['t']:+.2f}) 로
    다른 요일 같은 조건({an['D']['h1']['placebo']['mean']:+.2f}σ)과 다르지 않고, 월~수 저점을 깨는 비율도 {b1d['main']['rate'] * 100:.0f}% (위약 {b1d['placebo']['rate'] * 100:.0f}%) 로 같다.
    목 큰 하락 뒤 금요일 반등도 {an['D']['h2']['main']['mean']:+.2f}σ (t {an['D']['h2']['main']['t']:+.2f}) 로 유의하지 않다.
    역방향은 금요일이 되돌리지 않고 오히려 이어갔다({an['U']['h2']['main']['mean']:+.2f}σ).</li>
<li>5일 전체 순서는 <b>우연보다 덜 나온다</b>: 하락형 {fp['D']['count']}건({fp['D']['distinct_weeks']}주) — 요일별 조건이 독립일 때 기대의 {fp['D']['lift']:.2f}배
    [{fp['D']['lift_ci'][0]:.2f}, {fp['D']['lift_ci'][1]:.2f}], 상승형 {fp['U']['lift']:.2f}배.</li>
</ul></div>

<h2>A. 주봉 캔들 — 전주 캔들로 이번 주 맞히기</h2>
<div class="card">{rules_chart(S)}</div>
<p>막대 = 규칙별 순 샤프(수수료 6bps · 펀딩 포함). 네 규칙 모두 발견 구간(파랑)보다 확인 구간(주황)이 낮고, 필터를 건 셋은
기준보다 더 크게 떨어졌다 — 발견 구간에 맞춰진 모양이다.</p>
<div class="tbl"><table><thead><tr><th>규칙</th><th>전체 샤프</th><th>t</th><th>발견</th><th>확인</th><th>최대낙폭</th><th>평균 총노출</th><th>판정</th></tr></thead>
<tbody>{strat_rows}</tbody></table></div>
<p>필터 효과(확인 구간, 주 군집): F1 남긴 − 버린 거래 {fx['F1']['확인']['diff'] * 100:+.2f}%p/주 (t {fx['F1']['확인']['t']:+.2f}) ·
F2 {fx['F2']['확인']['diff'] * 100:+.2f}%p (t {fx['F2']['확인']['t']:+.2f}) · F3 뒤집은 거래의 원래 방향 수익 {fx['F3']['확인']['mean'] * 100:+.2f}% (t {fx['F3']['확인']['t']:+.2f}, 뒤집어서 손해).</p>

<h3>14칸 지도 — 전주 캔들 모양별 다음 주 '같은 방향' 수익</h3>
<p>사전등록대로 발견 구간에서 |t| ≥ 2.5 인 칸만 후보로 뽑는다. 가장 큰 |t| 가 {maxt:.2f} 라 <b>후보가 없다</b>. 확인 구간 값은 참고로만 적는다.
(거의 비는 두 칸은 뺐다 — 몸통이 1/3 미만이면서 두 꼬리가 다 짧을 수는 없다.)</p>
<div class="tbl"><table><thead><tr><th>전주 캔들</th><th>발견 평균</th><th>t</th><th>주</th><th>확인 평균</th><th>t</th><th>전체 건수</th><th>같은 방향 비율</th></tr></thead>
<tbody>{cell_rows}</tbody></table></div>

<h3>이번 주가 전주 고점·저점을 넘나 — 꼬리 모양 vs 종가 위치</h3>
<div class="tbl"><table><thead><tr><th>전주 캔들</th><th>건수</th><th>고점 돌파</th><th>저점 이탈</th><th>전주 종가 위치</th><th>고점 먼저 (닿은 것 중)</th>
<th>같은 방향 마감</th><th>방향 맞춤 수익</th></tr></thead><tbody>{brk_rows}</tbody></table></div>
<p>종가 위치 = (종가 − 저가) ÷ (고가 − 저가). 같은 종가 위치의 모든 캔들과 비교하면 '고점 먼저' 비율 차이가 1~2%p 이내다 —
꼬리 모양이 주는 정보는 종가가 어디 붙었는지 이상이 아니다. 방향(같은 방향 마감)은 46~59% 로 반반에 가깝다.</p>

<h2>B. TGIF — 월 혼조 · 화·수 약상승 · 목 큰 하락 · 금 소폭 상승 (그리고 역방향)</h2>
<div class="card">{tgif_chart(an)}</div>
<p>파란 점과 막대 = 목·금의 조건부 평균과 95% 구간, 주황 점 = 같은 조건을 다른 요일에 둔 위약. 요일이 특별하다면 파란 점이 ▲ 쪽으로,
주황 점보다 멀리 가야 한다. 넷 다 0 을 걸치거나 반대쪽이다.</p>
<div class="tbl"><table><thead><tr><th>가설</th><th>본 평균 σ</th><th>t</th><th>건수</th><th>위약 σ</th><th>차이 t</th><th>반기 (발견 / 확인)</th><th>판정</th></tr></thead>
<tbody>{tg_rows}</tbody></table></div>

<h3>목요일은 얼마나 떨어지나 (하락형 준비 뒤)</h3>
<p>월 혼조 · 화·수 약상승 뒤 목요일 {b1d['main']['n']:,}건: 중앙 {b1d['thu_pct_q'][2]:+.2f}% ({b1d['thu_z_q'][2]:+.2f}σ), 하위 25% {b1d['thu_pct_q'][1]:+.2f}%,
하위 10% {b1d['thu_pct_q'][0]:+.2f}% ({b1d['thu_z_q'][0]:+.2f}σ), 상위 10% {b1d['thu_pct_q'][4]:+.2f}%.
큰 하락(−1σ 이하)은 {b1d['thu_big_share'] * 100:.1f}% 로 목요일 전체({b1d['thu_big_share_uncond'] * 100:.1f}%)보다 오히려 적다.
월~수 저점을 깨는 비율 {b1d['main']['rate'] * 100:.1f}% — 목요일 전체({b1d['uncond_thu'] * 100:.1f}%)보다 낮은 건 이틀 올라서 저점이 멀어졌기 때문이다.
상승형도 같다: 월~수 고점 돌파 {b1u['main']['rate'] * 100:.1f}% (위약 {b1u['placebo']['rate'] * 100:.1f}%).</p>
<p>거래로 보면(왕복 12bps, 보조): 하락형 목요일 숏 {trd['목 거래']['mean_pct']:+.2f}% (t {trd['목 거래']['t']:+.2f}) · 금요일 롱 {trd['금 거래']['mean_pct']:+.2f}%
(t {trd['금 거래']['t']:+.2f}) / 상승형 목요일 롱 {tru['목 거래']['mean_pct']:+.2f}% · 금요일 숏 {tru['금 거래']['mean_pct']:+.2f}% — 유의한 것 없음.</p>

<h3>요일 효과 자체 (조건 없음)</h3>
<div class="card">{dow_chart(X['dow'])}</div>
<p>조건 없이 봐도 목요일 평균 {X['dow']['목']['z']:+.3f}σ (t {X['dow']['목']['t']:+.2f}, 오른 날 {X['dow']['목']['up'] * 100:.0f}%), 금요일 {X['dow']['금']['z']:+.3f}σ
(t {X['dow']['금']['t']:+.2f}) — 방향은 TGIF 와 비슷하게 기울지만 어느 요일도 유의하지 않다(|t| ≤ 1.3).</p>

<h3>UTC 일봉 민감도 (판정 외)</h3>
<p>요일을 UTC 자정으로 끊어도 넷 다 통과하지 않는다: 하락형 목 {ua['D']['h1']['main']['mean']:+.2f}σ · 금 {ua['D']['h2']['main']['mean']:+.2f}σ /
상승형 목 {ua['U']['h1']['main']['mean']:+.2f}σ · 금 {ua['U']['h2']['main']['mean']:+.2f}σ (t {ua['U']['h2']['main']['t']:+.2f}) — 상승형 금요일은 예측(되돌림)의 반대로
이어갔고, 뉴욕 일봉에서도 같은 쪽({an['U']['h2']['main']['mean']:+.2f}σ)이다. 사전등록 방향이 아니고 여러 검정 중 하나라 우연 범위로 본다.</p>
</main></div></body></html>"""
    Path("out/weekly_tgif.html").write_text(html, encoding="utf-8")
    print("out/weekly_tgif.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
