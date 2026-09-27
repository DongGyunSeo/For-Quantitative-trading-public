"""매월 갱신 최적 G — HTML. 입력 runs/tsmom_optg.json → out/tsmom_optg.html"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np


def svg(W, H, aria, minw=600, maxw=None):
    mx = f"max-width:{maxw}px;margin:0 auto;" if maxw else ""
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;{mx}height:auto;display:block;font-family:inherit">')


def rows_of(K, key):
    return {round(x["G"], 2): x for x in key}


def g_axis(out, sx, W, l, r, H, b, t, gmax=3.0):
    for g in np.arange(0, gmax + 1e-9, 0.5):
        out.append(f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="{t}" y2="{H - b}" class="grid"/>'
                   f'<text x="{sx(g):.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{g:g}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 4}" class="tick" text-anchor="middle">G (월초 기준 총명목 ÷ 자산)</text>')


def cagr_chart(K: dict) -> str:
    hm = K["hist"]["monthly"]["rows"]
    bm = K["boot"]["monthly"]["rows"]
    bw = K["boot"]["weekly"]["rows"]
    s3 = K["sens"]["sharpe0.3"]["rows"]
    W, H, l, r, t, b = 880, 400, 52, 20, 62, 40
    lo, hi = -0.10, 0.45
    sx = lambda g: l + (W - l - r) * g / 3.0
    sy = lambda v: t + (H - t - b) * (hi - min(max(v, lo), hi)) / (hi - lo)
    out = [svg(W, H, "G 별 연복리 — 매월 갱신 역사 · 부트스트랩 중앙, 매주 갱신, 샤프 0.3 이라면")]
    for v in (-0.1, 0, 0.1, 0.2, 0.3, 0.4):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"base" if v == 0 else "grid"}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v * 100:+.0f}%</text>')
    g_axis(out, sx, W, l, r, H, b, t)
    ser = ((hm, "cagr", "ink", "역사 경로 (매월)"), (bm, "cagr_p50", "c1", "부트스트랩 중앙 (매월)"),
           (bw, "cagr_p50", "c3", "부트스트랩 중앙 (매주)"), (s3, "cagr_p50", "c2", "샤프 0.3 이라면 (매월)"))
    lx = l
    for rows, k, cls, name in ser:
        out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x[k]):.1f}" for x in rows) + f'" class="{cls}" fill="none"/>')
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="14" y2="14" class="{cls}"/><text x="{lx + 24}" y="18" class="lg">{escape(name)}</text>')
        lx += 44 + 10.5 * len(name)
    marks = ((K["hist"]["monthly"]["g_opt"], hm, "cagr", "mk0", "역사 최적", -12),
             (K["boot"]["monthly"]["g_opt"], bm, "cagr_p50", "mk1", "최적", -12),
             (K["boot"]["weekly"]["g_opt"], bw, "cagr_p50", "mk3", "매주 최적", 22),
             (K["sens"]["sharpe0.3"]["g_opt"], s3, "cagr_p50", "mk2", "샤프 0.3 최적", -12))
    for g, rows, k, cls, lab, dy in marks:
        y = [x for x in rows if abs(x["G"] - g) < 1e-9][0][k]
        out.append(f'<circle cx="{sx(g):.1f}" cy="{sy(y):.1f}" r="5" class="{cls}"/>'
                   f'<text x="{sx(g):.1f}" y="{sy(y) + dy:.1f}" class="dl" text-anchor="middle">{escape(lab)} {g:.2f}</text>')
    y75 = [x for x in bm if abs(x["G"] - 0.75) < 1e-9][0]["cagr_p50"]
    out.append(f'<line x1="{sx(0.75):.1f}" x2="{sx(0.75):.1f}" y1="{t}" y2="{H - b}" class="refl"/>'
               f'<text x="{sx(0.75) + 4:.1f}" y="{t + 10}" class="tick">현재 0.75 (절반 켈리 근처)</text>')
    out.append(f'<text x="{l}" y="40" class="tick">청산된 경로는 −100% 로 셌다 · 세로축은 −10% 아래를 잘랐다</text>')
    hmd = rows_of(K, hm)
    bwd = rows_of(K, bw)
    bw_ = (W - l - r) / len(bm)
    for x in bm:
        g = round(x["G"], 2)
        hh = hmd[g]
        hc = "청산" if hh["liq"] else f"{hh['cagr'] * 100:+.1f}%"
        tip = (f"G {g:.2f} · 역사(매월) {hc} · 부트스트랩 중앙(매월) {x['cagr_p50'] * 100:+.1f}% (5% {x['cagr_p05'] * 100:+.0f}%) · "
               f"매주 {bwd[g]['cagr_p50'] * 100:+.1f}% · 낙폭 중앙 −{x['mdd_p50'] * 100:.0f}% · 청산 {x['p_liq'] * 100:.1f}%")
        out.append(f'<rect x="{sx(g) - bw_ / 2:.1f}" y="{t}" width="{bw_:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def mdd_chart(K: dict) -> str:
    hm = K["hist"]["monthly"]["rows"]
    bm = K["boot"]["monthly"]["rows"]
    W, H, l, r, t, b = 880, 320, 52, 20, 44, 40
    sx = lambda g: l + (W - l - r) * g / 3.0
    sy = lambda v: t + (H - t - b) * v
    out = [svg(W, H, "G 별 최대낙폭 — 매월 갱신 역사 · 부트스트랩 중앙과 95번째 백분위")]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{"0" if v == 0 else f"−{v * 100:.0f}%"}</text>')
    g_axis(out, sx, W, l, r, H, b, t)
    band = " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p50']):.1f}" for x in bm) + " " + \
        " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p95']):.1f}" for x in reversed(bm))
    out.append(f'<polygon points="{band}" class="band"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p95']):.1f}" for x in bm) + '" class="c1" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p50']):.1f}" for x in bm) + '" class="c1d" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['mdd']):.1f}" for x in hm) + '" class="ink" fill="none"/>')
    for g in (0.75, K["boot"]["monthly"]["g_opt"]):
        out.append(f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="{t}" y2="{H - b}" class="refl"/>'
                   f'<text x="{sx(g) + 4:.1f}" y="{H - b - 6}" class="tick">{"현재 0.75" if g == 0.75 else f"최적 {g:.2f}"}</text>')
    out.append(f'<line x1="{l}" x2="{l + 18}" y1="14" y2="14" class="ink"/><text x="{l + 24}" y="18" class="lg">역사 경로</text>'
               f'<line x1="{l + 110}" x2="{l + 128}" y1="14" y2="14" class="c1d"/><text x="{l + 134}" y="18" class="lg">부트스트랩 중앙</text>'
               f'<line x1="{l + 260}" x2="{l + 278}" y1="14" y2="14" class="c1"/><text x="{l + 284}" y="18" class="lg">부트스트랩 95번째 백분위</text>')
    hmd = rows_of(K, hm)
    bw_ = (W - l - r) / len(bm)
    for x in bm:
        g = round(x["G"], 2)
        tip = (f"G {g:.2f} · 역사 낙폭 −{hmd[g]['mdd'] * 100:.0f}% (실효 G 최대 {hmd[g]['gmax']:.2f}) · 부트스트랩 중앙 −{x['mdd_p50'] * 100:.0f}% · "
               f"95% −{x['mdd_p95'] * 100:.0f}% · 실효 G 최대 95% {x['gmax_p95']:.2f} · 5년 손실 {x['p_loss'] * 100:.0f}%")
        out.append(f'<rect x="{sx(g) - bw_ / 2:.1f}" y="{t}" width="{bw_:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def argmax_chart(K: dict) -> str:
    rows = (("주 추정 (4주 블록 · 매월)", K["boot"]["monthly"]), ("13주 블록", K["sens"]["block13"]), ("주 순서 무작위 (IID)", K["sens"]["block1"]),
            ("매주 갱신", K["boot"]["weekly"]), ("샤프가 0.5 라면", K["sens"]["sharpe0.5"]), ("샤프가 0.3 이라면", K["sens"]["sharpe0.3"]))
    W, l, r, t, rh = 760, 190, 30, 30, 32
    H = t + rh * len(rows) + 34
    sx = lambda g: l + (W - l - r) * g / 3.0
    out = [svg(W, H, "경로마다 연복리가 가장 큰 G 의 분포 — 5 · 25 · 50 · 75 · 95%", 560, 880)]
    for g in np.arange(0, 3.001, 0.5):
        out.append(f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="{t - 6}" y2="{H - 34}" class="grid"/>'
                   f'<text x="{sx(g):.1f}" y="{t - 12}" class="tick" text-anchor="middle">{g:g}</text>')
    out.append(f'<line x1="{sx(0.75):.1f}" x2="{sx(0.75):.1f}" y1="{t - 6}" y2="{H - 34}" class="refl"/>')
    y = t
    for name, s in rows:
        p5, p25, p50, p75, p95 = s["argmax_pct"]
        cy = y + rh / 2
        tip = f"{name}: 중앙 곡선의 최적 G {s['g_opt']:.2f} (연 {s['cagr_opt'] * 100:.1f}%) · 경로별 최적 G 5% {p5:.2f} · 25% {p25:.2f} · 50% {p50:.2f} · 75% {p75:.2f} · 95% {p95:.2f}"
        out.append(f'<g><title>{escape(tip)}</title><rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                   f'<text x="8" y="{cy + 4:.1f}" class="lg">{escape(name)}</text>'
                   f'<line x1="{sx(p5):.1f}" x2="{sx(p95):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="whg"/>'
                   f'<rect x="{sx(p25):.1f}" y="{cy - 7:.1f}" width="{sx(p75) - sx(p25):.1f}" height="14" rx="3" class="box"/>'
                   f'<circle cx="{sx(s["g_opt"]):.1f}" cy="{cy:.1f}" r="5.5" class="mk1"/></g>')
        y += rh
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 10}" class="tick" text-anchor="middle">G · 막대 = 경로별 최적 G 의 25 ~ 75%, 선 = 5 ~ 95%, 점 = 중앙 곡선의 최적 · 점선 = 현재 0.75</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_optg.json").read_text(encoding="utf-8"))
    hm = rows_of(K, K["hist"]["monthly"]["rows"])
    bm = rows_of(K, K["boot"]["monthly"]["rows"])
    bwk = rows_of(K, K["boot"]["weekly"]["rows"])
    go = K["boot"]["monthly"]["g_opt"]
    gh = K["hist"]["monthly"]["g_opt"]
    gw = K["boot"]["weekly"]["g_opt"]
    sens = K["sens"]
    ap = K["boot"]["monthly"]["argmax_pct"]
    ratio = bm[0.75]["cagr_p50"] / bm[go]["cagr_p50"]
    trs = ""
    for g in (0.5, 0.75, 1.0, 1.25, go, 1.75, gh, 2.25):
        g = round(g, 2)
        h, b = hm[g], bm[g]
        cls = ' class="hl"' if g in (0.75, go) else ""
        hc = "청산" if h["liq"] else f"{h['cagr'] * 100:+.1f}%"
        trs += (f"<tr{cls}><td>{g:g}</td><td>{hc}</td><td>{h['mult']:.2f}배</td><td>−{h['mdd'] * 100:.0f}%</td><td>{h['gmax']:.2f}</td>"
                f"<td>{b['cagr_p50'] * 100:+.1f}%</td><td>{b['cagr_p05'] * 100:+.0f}%</td><td>−{b['mdd_p50'] * 100:.0f}% · −{b['mdd_p95'] * 100:.0f}%</td>"
                f"<td>{b['p_loss'] * 100:.0f}%</td><td>{b['p_liq'] * 100:.1f}%</td><td>{b['gmax_p95']:.2f}</td></tr>")
    srows = "".join(f"<tr><td>{escape(n)}</td><td>{s['g_opt']:.2f}</td><td>{s['cagr_opt'] * 100:+.1f}%</td><td>{s['argmax_pct'][1]:.2f} ~ {s['argmax_pct'][3]:.2f}</td></tr>"
                    for n, s in (("주 추정 (4주 블록 · 매월)", K["boot"]["monthly"]), ("매주 갱신 (비교)", K["boot"]["weekly"]),
                                 ("주 순서 무작위 (IID)", sens["block1"]), ("13주 블록", sens["block13"]), ("MMR 5%", sens["mmr5"]),
                                 (f"샤프 0.5 (주 {sens['sharpe0.5']['delta_bps']:.0f}bps 차감)", sens["sharpe0.5"]),
                                 (f"샤프 0.3 (주 {sens['sharpe0.3']['delta_bps']:.0f}bps 차감)", sens["sharpe0.3"])))
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>매월 갱신 최적 G</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--acc:#2a78d6;--hl:#eef4fc}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--acc:#3987e5;--hl:#1c2533}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--acc:#3987e5;--hl:#1c2533}}
*{{box-sizing:border-box}}body{{margin:0}}
.viz-root{{background:var(--bg);color:var(--text-primary);font:14px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;min-height:100vh}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 56px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:30px 0 10px}}
.sub{{color:var(--text-secondary);margin-bottom:16px}}
.box{{background:var(--surface-1);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 18px}}
.box ul{{padding-left:18px;margin:0}}.box li{{margin:6px 0}}
.card{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}}
.tbl{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;overflow-x:auto;margin:10px 0}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:var(--text-secondary);font-weight:600}}
tr.hl td{{background:var(--hl);font-weight:600}}
p{{color:var(--text-secondary)}}.note{{font-size:13px}}
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .dl{{fill:var(--text-primary);font-size:12px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .ink{{stroke:var(--text-primary);stroke-width:2}}svg .c1{{stroke:var(--c1);stroke-width:2}}svg .c1d{{stroke:var(--c1);stroke-width:2;stroke-dasharray:6 4}}
svg .c2{{stroke:var(--c2);stroke-width:2}}svg .c3{{stroke:var(--c3);stroke-width:2}}
svg .band{{fill:var(--c1);fill-opacity:.12}}svg .refl{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
svg .mk0{{fill:var(--text-primary);stroke:var(--surface-1);stroke-width:2}}svg .mk1{{fill:var(--c1);stroke:var(--surface-1);stroke-width:2}}
svg .mk2{{fill:var(--c2);stroke:var(--surface-1);stroke-width:2}}svg .mk3{{fill:var(--c3);stroke:var(--surface-1);stroke-width:2}}
svg .whg{{stroke:var(--muted);stroke-width:2}}svg .box{{fill:var(--c1);fill-opacity:.35;stroke:var(--c1)}}
svg .hit{{fill:transparent}}svg g:hover .hit{{fill:var(--grid);fill-opacity:.45}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.6}}
</style></head><body><div class="viz-root"><main>
<h1>매월 갱신에서 복리가 가장 큰 G</h1>
<div class="sub">역변동성(S1) · 매월 갱신 · 교차 증거금(장중 청산 = 파산) · 47심볼 · 269주 · G 0.10 ~ 3.00 · 사전등록한 추정 방법 그대로 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>답: 꼭대기가 있다.</b> 과거 주를 섞은 5년 경로 10,000개의 연복리 중앙값이 가장 큰 G = <b>{go:.2f}</b> (연 {K['boot']['monthly']['cagr_opt'] * 100:.1f}%).
    실제 역사 경로 한 줄로는 <b>{gh:.2f}</b> (연 {K['hist']['monthly']['cagr_opt'] * 100:.1f}%). 매주 갱신이면 {gw:.2f} (연 {K['boot']['weekly']['cagr_opt'] * 100:.1f}%) —
    매월 갱신은 한 달 단위 변동이 작아서 꼭대기가 오른쪽으로 간다.</li>
<li><b>꼭대기는 평평하고 위험은 가파르다.</b> G {go:.2f} 에서 섞은 경로의 최대낙폭 중앙 −{bm[go]['mdd_p50'] * 100:.0f}% · 95% −{bm[go]['mdd_p95'] * 100:.0f}%,
    5년 뒤 손실일 확률 {bm[go]['p_loss'] * 100:.0f}%. 현재 0.75 는 연복리 중앙 {bm[0.75]['cagr_p50'] * 100:.1f}% — 최대의 {ratio * 100:.0f}% 를 얻으면서 낙폭 중앙은 −{bm[0.75]['mdd_p50'] * 100:.0f}%.</li>
<li><b>0.75 는 사실상 '절반 켈리' 다.</b> 이론대로라면 최적의 절반에서 최대 복리의 약 3/4 를 변동성 절반으로 얻는다 — 데이터도 {ratio * 100:.0f}%.
    최적을 넘으면 복리가 급히 줄고(G 2.2 부터 22% 경로가 청산, 2.6 이면 절반 넘게), 최적보다 모자라면 천천히 준다 — 모를 땐 작게 거는 쪽이 싸다.</li>
<li><b>최적 G 는 아주 불확실하다.</b> 경로마다 최적 G 의 가운데 50% 가 {ap[1]:.2f} ~ {ap[3]:.2f}. 엣지가 작으면 확 줄어든다: 샤프 0.5 라면 {sens['sharpe0.5']['g_opt']:.2f},
    0.3 이라면 {sens['sharpe0.3']['g_opt']:.2f}. 주 순서를 무작위로 섞으면(되돌림 없음) {sens['block1']['g_opt']:.2f}. 표본 샤프 0.68 은 유의하지 않다(t 1.5).</li>
<li><b>매월 갱신 + 큰 G 는 한 달 안에서 실효 G 가 치솟는다.</b> G {go:.2f} 에서 실효 G 가 역사 최대 {hm[go]['gmax']:.2f}, 섞은 경로 95% {bm[go]['gmax_p95']:.2f}
    (G 2 면 역사 {hm[2.0]['gmax']:.2f}). 청산 한계(약 2.6)에 가깝다 — 큰 G 를 쓰려면 달 중간 실효 G 상한(넘으면 B 를 앞당겨 갱신)이 필요하다(검증 안 함).</li>
</ul></div>

<h2>1. G 별 연복리</h2>
<div class="card">{cagr_chart(K)}</div>
<p>세로 띠에 마우스를 올리면 그 G 의 숫자가 나온다. 산술 수익은 G 에 비례해 늘지만 변동성 끌림이 G² 로 늘어 꼭대기가 생긴다. 역사 경로는 1.5 ~ 2.25 가 거의 평평하다.</p>

<h2>2. G 별 최대낙폭 (장중 포함)</h2>
<div class="card">{mdd_chart(K)}</div>

<h2>3. 최적 G 는 얼마나 확실한가</h2>
<div class="card">{argmax_chart(K)}</div>
<p>경로마다(섞은 5년 한 줄마다) 연복리가 가장 큰 G 를 찾으면 0.1 부터 3 까지 흩어진다. 한 줄의 역사로 고른 '최적' 은 이 분포에서 한 점을 뽑은 것뿐이다.</p>

<h2>4. G 표 (매월 갱신)</h2>
<div class="tbl"><table><thead><tr><th>G</th><th>역사 연복리</th><th>5년</th><th>역사 낙폭</th><th>역사 G 최대</th><th>섞은 연복리</th><th>5%</th>
<th>섞은 낙폭 중앙·95%</th><th>5년 손실</th><th>청산</th><th>섞은 G 최대</th></tr></thead><tbody>{trs}</tbody></table></div>
<p class="note">'섞은' = 부트스트랩 5년 경로 10,000개. 섞은 G 최대 = 경로마다 실효 G 최댓값의 95번째 백분위. 청산 = 청산된 경로 비율.</p>

<h2>5. 민감도</h2>
<div class="tbl"><table><thead><tr><th>설정</th><th>최적 G (중앙 곡선)</th><th>그때 연복리 중앙</th><th>경로별 최적 25 ~ 75%</th></tr></thead><tbody>{srows}</tbody></table></div>
<p class="note">켈리 근사(평균 ÷ 분산, 주 단위): {K['kelly_approx']['weekly']:.2f} — 매주 갱신 최적 {gw:.2f} 와 맞는다. 샤프 할인은 주 수익에서 상수를 빼 평균만 낮췄다(흔들림은 그대로).</p>

<h2>6. 어떻게 쟀나</h2>
<p class="note">매월 갱신: 달력 월 첫 월요일에 B = 계좌 자산, 매주 코인 명목 = G × B × 역변동성 비중, 실효 G = G × B ÷ 지금 자산. 장중 5분마다 교차 증거금 유지증거금(MMR 2%)을 확인해
청산이면 자산 0. 부트스트랩: 과거 269주를 4주 블록으로 다시 뽑은 5년 경로 10,000개, 매월 = 4주마다(블록 경계). 검증: 역사 연복리를 모듈 없이 반복문으로 다시 계산해 일치
(G 0.75 · 1.45 · 1.9 → 24.09% · 37.46% · 39.88%, 실효 G 최대 0.93 · 2.29 · 3.67), 부트스트랩 중앙을 단순 반복문 3,000 경로로 다시 뽑아 21.2% · 29.5% (본 계산 20.9% · 29.2%).</p>
</main></div></body></html>"""
    Path("out/tsmom_optg.html").write_text(html, encoding="utf-8")
    print("out/tsmom_optg.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
