"""복리 주기 — HTML. 입력 runs/tsmom_compound.json · runs/tsmom_compound_weekly.csv → out/tsmom_compound.html"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {"C1": "매주", "C2": "2주", "C3": "매월", "C4": "분기", "C5": "매년", "C6": "갱신 안 함 (단리)", "C7": "비대칭 (줄임 매주 · 키움 매월)"}
SER = (("C1", "매주", "ink"), ("C3", "매월 (채택)", "c1"), ("C7", "비대칭", "c3"), ("C6", "갱신 안 함", "c2"))


def svg(W, H, aria, minw=600, maxw=None):
    mx = f"max-width:{maxw}px;margin:0 auto;" if maxw else ""
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;{mx}height:auto;display:block;font-family:inherit">')


def legend(out, l, items, y=14):
    lx = l
    for _, name, cls in items:
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{y}" y2="{y}" class="{cls}"/><text x="{lx + 24}" y="{y + 4}" class="lg">{escape(name)}</text>')
        lx += 44 + 11 * len(name)


def years_axis(out, idx, xs, t, H, b):
    for yv in sorted(set(idx.year)):
        k = int(np.searchsorted(idx.year, yv))
        out.append(f'<line x1="{xs[k]:.1f}" x2="{xs[k]:.1f}" y1="{t}" y2="{H - b}" class="grid"/><text x="{xs[k] + 4:.1f}" y="{H - b + 16}" class="tick">{yv}</text>')


def equity_chart(w: pd.DataFrame) -> str:
    W, H, l, r, t, b = 860, 360, 52, 120, 40, 30
    eq = pd.concat([pd.DataFrame({k: [1.0] for k, _, _ in SER}, index=[w.index[0]]),
                    pd.DataFrame({k: w[f"{k}_E_end"].to_numpy() for k, _, _ in SER}, index=w.index + pd.Timedelta(days=7))])
    lo, hi = np.log(0.7), np.log(4.0)
    xs = np.linspace(l, W - r, len(eq))
    sy = lambda v: t + (H - t - b) * (hi - np.log(max(v, 0.7))) / (hi - lo)
    out = [svg(W, H, "갱신 주기별 누적 자산 (로그 축) — 역변동성 · G 0.75")]
    for v in (0.75, 1, 1.5, 2, 3):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"base" if v == 1 else "grid"}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}배</text>')
    years_axis(out, eq.index, xs, t, H, b)
    legend(out, l, SER)
    for k, _, cls in SER:
        out.append('<polyline points="' + " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, eq[k])) + f'" class="{cls}" fill="none"/>')
    last = -1e9
    for yv, k, n in sorted((sy(eq[k].iloc[-1]), k, n) for k, n, _ in SER):
        yv = max(yv, last + 15)
        last = yv
        out.append(f'<text x="{W - r + 8}" y="{yv + 4:.1f}" class="dl">{escape(n.split(" (")[0])} {eq[k].iloc[-1]:.2f}배</text>')
    bw = (W - l - r) / len(eq)
    for i, ts in enumerate(eq.index):
        tip = f"{ts.date()} · " + " · ".join(f"{n.split(' (')[0]} {eq[k].iloc[i]:.2f}배" for k, n, _ in SER)
        out.append(f'<rect x="{xs[i] - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def geff_chart(w: pd.DataFrame) -> str:
    items = (("C3", "매월", "c1"), ("C7", "비대칭", "c3"), ("C6", "갱신 안 함", "c2"))
    W, H, l, r, t, b = 860, 300, 52, 20, 40, 30
    lo, hi = 0.25, 1.05
    xs = np.linspace(l, W - r, len(w))
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    out = [svg(W, H, "갱신 주기별 실효 G (명목 ÷ 지금 자산) — 매주는 늘 0.75")]
    for v in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"refl" if v == 0.75 else "grid"}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    years_axis(out, w.index, xs, t, H, b)
    legend(out, l, items + (("", "매주 = 0.75 (점선)", "refl"),))
    for k, _, cls in items:
        out.append('<polyline points="' + " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, w[f"{k}_G_eff"])) + f'" class="{cls}" fill="none"/>')
    bw = (W - l - r) / len(w)
    for i, ts in enumerate(w.index):
        tip = f"{ts.date()} 주 · " + " · ".join(f"{n} {w[f'{k}_G_eff'].iloc[i]:.2f}" for k, n, _ in items)
        out.append(f'<rect x="{xs[i] - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def diff_chart(K: dict) -> str:
    rows = [c for c in ("C2", "C3", "C4", "C5", "C6", "C7")]
    W, l, r, t, rh = 760, 250, 150, 34, 34
    H = t + rh * len(rows) + 30
    lo, hi = -0.10, 0.10
    sx = lambda v: l + (W - l - r) * (min(max(v, lo), hi) - lo) / (hi - lo)
    out = [svg(W, H, "매주 갱신 대비 연복리 차이 — 부트스트랩 중앙값과 5 ~ 95%", 560, 860)]
    for v in (-0.10, -0.05, 0, 0.05, 0.10):
        lab = "" if v == 0.10 else ("0" if v == 0 else f"{v * 100:+.0f}%p")
        out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t - 8}" y2="{H - 30}" class="{"base" if v == 0 else "grid"}"/>'
                   f'<text x="{sx(v):.1f}" y="{t - 14}" class="tick" text-anchor="middle">{lab}</text>')
    y = t
    for c in rows:
        bb = K["boot"][c]
        cy = y + rh / 2
        cls = "mk1" if c == "C3" else "mkg"
        tip = (f"{NAMES[c]} − 매주: 중앙 {bb['d_cagr_p50'] * 100:+.2f}%p · 5 ~ 95% {bb['d_cagr_p05'] * 100:+.1f} ~ {bb['d_cagr_p95'] * 100:+.1f}%p · "
               f"매주보다 좋을 확률 {bb['p_beat_c1'] * 100:.0f}% · 낙폭 중앙 차이 {bb['d_mdd_p50'] * 100:+.1f}%p")
        out.append(f'<g><title>{escape(tip)}</title><rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                   f'<text x="8" y="{cy + 4:.1f}" class="lg">{escape(NAMES[c])}</text>'
                   f'<line x1="{sx(bb["d_cagr_p05"]):.1f}" x2="{sx(bb["d_cagr_p95"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="wh{"1" if c == "C3" else "g"}"/>'
                   f'<circle cx="{sx(bb["d_cagr_p50"]):.1f}" cy="{cy:.1f}" r="5.5" class="{cls}"/>'
                   f'<text x="{W - r + 10}" y="{cy + 4:.1f}" class="val">{bb["d_cagr_p50"] * 100:+.2f}%p · {bb["p_beat_c1"] * 100:.0f}%</text></g>')
        y += rh
    out.append(f'<text x="{W - r + 16}" y="{t - 14}" class="tick">중앙 · 매주 이김</text>')
    out.append(f'<text x="{sx(0):.1f}" y="{H - 8}" class="tick" text-anchor="middle">같은 5년 경로에서 (그 주기 − 매주) 연복리 차이</text>')
    out.append("</svg>")
    return "".join(out)


def drag_chart(K: dict) -> str:
    dec = K["decomp"]
    W, H, l, r, t, b = 640, 300, 52, 130, 40, 40
    gmax, hi = 1.5, 0.6
    sx = lambda g: l + (W - l - r) * g / gmax
    sy = lambda v: t + (H - t - b) * (hi - v) / hi
    out = [svg(W, H, "G 별 산술 연수익과 연복리 — 차이가 변동성 끌림", 520, 760)]
    for v in (0, 0.2, 0.4, 0.6):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"base" if v == 0 else "grid"}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v * 100:.0f}%</text>')
    for g in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        out.append(f'<text x="{sx(g):.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{g:g}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 6}" class="tick" text-anchor="middle">G (매주 갱신 · 역변동성)</text>')
    pa = [(0, 0)] + [(x["G"], x["arith"]) for x in dec]
    pc = [(0, 0)] + [(x["G"], x["cagr"]) for x in dec]
    band = " ".join(f"{sx(g):.1f},{sy(v):.1f}" for g, v in pa) + " " + " ".join(f"{sx(g):.1f},{sy(v):.1f}" for g, v in reversed(pc))
    out.append(f'<polygon points="{band}" class="band2"/>')
    out.append('<polyline points="' + " ".join(f"{sx(g):.1f},{sy(v):.1f}" for g, v in pa) + '" class="c2" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(g):.1f},{sy(v):.1f}" for g, v in pc) + '" class="c1" fill="none"/>')
    for g, v in pa[1:]:
        out.append(f'<circle cx="{sx(g):.1f}" cy="{sy(v):.1f}" r="3.5" class="mk2"/>')
    for g, v in pc[1:]:
        out.append(f'<circle cx="{sx(g):.1f}" cy="{sy(v):.1f}" r="3.5" class="mk1"/>')
    a15, c15 = pa[-1][1], pc[-1][1]
    out.append(f'<text x="{sx(1.5) + 8:.1f}" y="{sy(a15) + 4:.1f}" class="dl">산술 {a15 * 100:.0f}%</text>'
               f'<text x="{sx(1.5) + 8:.1f}" y="{sy(c15) + 4:.1f}" class="dl">연복리 {c15 * 100:.0f}%</text>')
    x75 = [x for x in dec if x["G"] == 0.75][0]
    out.append(f'<line x1="{sx(0.75):.1f}" x2="{sx(0.75):.1f}" y1="{sy(x75["arith"]):.1f}" y2="{sy(x75["cagr"]):.1f}" class="ink"/>'
               f'<text x="{sx(0.75) - 6:.1f}" y="{sy((x75["arith"] + x75["cagr"]) / 2) + 4:.1f}" class="dl" text-anchor="end">끌림 {x75["drag"] * 100:.1f}%p</text>')
    out.append(f'<line x1="{l}" x2="{l + 18}" y1="14" y2="14" class="c2"/><text x="{l + 24}" y="18" class="lg">산술 연수익 (주 평균 × 52)</text>'
               f'<line x1="{l + 200}" x2="{l + 218}" y1="14" y2="14" class="c1"/><text x="{l + 224}" y="18" class="lg">연복리 (실제로 불어나는 속도)</text>')
    for x in dec:
        tip = f"G {x['G']:g} · 산술 {x['arith'] * 100:.1f}% · 연복리 {x['cagr'] * 100:.1f}% · 끌림 {x['drag'] * 100:.1f}%p · 5년 {x['mult']:.2f}배"
        out.append(f'<rect x="{sx(x["G"]) - 20:.1f}" y="{t}" width="40" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_compound.json").read_text(encoding="utf-8"))
    w = pd.read_csv("runs/tsmom_compound_weekly.csv", index_col=0, parse_dates=True)
    h, bt, v = K["hist"], K["boot"], K["verdict"]
    h1, h3, h6, h7 = h["C1"], h["C3"], h["C6"], h["C7"]
    b3 = bt["C3"]
    HL = ' class="hl"'
    rows = "".join(
        f"<tr{HL if c == 'C3' else ''}><td>{escape(NAMES[c])}</td><td>{h[c]['updates']}</td><td>{h[c]['mult']:.2f}배</td><td>{h[c]['cagr'] * 100:+.1f}%</td>"
        f"<td>{h[c]['mdd'] * 100:.0f}%</td><td>{h[c]['sharpe']:.2f}</td><td>{h[c]['g_min']:.2f} ~ {h[c]['g_max']:.2f}</td>"
        f"<td>{bt[c]['cagr_p50'] * 100:+.1f}%</td><td>{bt[c]['mdd_p50'] * 100:.0f}% · {bt[c]['mdd_p95'] * 100:.0f}%</td>"
        f"<td>{bt[c]['p_beat_c1'] * 100:.0f}%</td><td>{bt[c]['g_max_p95']:.2f}</td></tr>" for c in NAMES)
    gs = K["g_sens"]
    grow = "".join(f"<tr><td>G {g}</td>" + "".join(f"<td>{gs[g][c]['cagr'] * 100:+.1f}% · {gs[g][c]['mdd'] * 100:.0f}%</td>" for c in ("C1", "C3", "C7", "C6")) + "</tr>"
                   for g in ("0.5", "1.0"))
    grow = (f"<tr><td>G 0.75</td>" + "".join(f"<td>{h[c]['cagr'] * 100:+.1f}% · {h[c]['mdd'] * 100:.0f}%</td>" for c in ("C1", "C3", "C7", "C6")) + "</tr>") + grow
    se = K["serial"]
    al = K["align_sens"]
    arow = "".join(f"<tr><td>{bl}주{' (IID)' if bl == '1' else ''}</td>"
                   + "".join(f"<td>{al[bl][k]['p_beat'] * 100:.0f}% · {al[bl][k]['d_p50'] * 100:+.2f}%p</td>" for k in ("월_0", "월_2", "분기_0", "분기_2")) + "</tr>"
                   for bl in ("1", "4", "8", "13"))
    x75 = [x for x in K["decomp"] if x["G"] == 0.75][0]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>복리 주기</title>
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
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .dl{{fill:var(--text-primary);font-size:12px}}svg .val{{fill:var(--text-primary);font-size:11px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .ink{{stroke:var(--text-primary);stroke-width:2}}svg .c1{{stroke:var(--c1);stroke-width:2}}svg .c2{{stroke:var(--c2);stroke-width:2}}svg .c3{{stroke:var(--c3);stroke-width:2}}
svg .refl{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}svg .band2{{fill:var(--c2);fill-opacity:.12}}
svg .mk1{{fill:var(--c1);stroke:var(--surface-1);stroke-width:2}}svg .mk2{{fill:var(--c2);stroke:var(--surface-1);stroke-width:2}}
svg .mkg{{fill:var(--muted);stroke:var(--surface-1);stroke-width:2}}svg .wh1{{stroke:var(--c1);stroke-width:2}}svg .whg{{stroke:var(--muted);stroke-width:2}}
svg .hit{{fill:transparent}}svg g:hover .hit{{fill:var(--grid);fill-opacity:.45}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.6}}
</style></head><body><div class="viz-root"><main>
<h1>복리 주기 — 계좌 자산을 언제 다시 반영할까</h1>
<div class="sub">역변동성(S1) · G 0.75 · 교차 증거금 · 47심볼 · 269주 (2021-05 ~ 2026-06) · 매주 신호·비중은 그대로, 명목을 계산하는 자산 B 만 주기를 바꿨다 ·
사전등록 판정 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>판정: 매월 갱신 채택 (사전등록 세 조건 모두 통과).</b> 역사 경로 5년 {h3['mult']:.2f}배 · 연 {h3['cagr'] * 100:.1f}% vs 매주 {h1['mult']:.2f}배 · 연 {h1['cagr'] * 100:.1f}%,
    최대낙폭 {h3['mdd'] * 100:.1f}% vs {h1['mdd'] * 100:.1f}%. 같은 과거 주를 섞은 5년 경로 10,000개에서 매월이 매주를 이긴 비율 <b>{b3['p_beat_c1'] * 100:.0f}%</b>,
    연복리 차이 중앙 {b3['d_cagr_p50'] * 100:+.1f}%p, 낙폭 중앙 {b3['d_mdd_p50'] * 100:+.1f}%p.</li>
<li><b>왜 — 이 전략의 손익은 몇 달 단위로 되돌아온다.</b> 전략 주간 수익의 분산비가 13주 {se['vr']['13']['obs']:.2f} · 26주 {se['vr']['26']['obs']:.2f}
    (순서를 섞으면 0.9 근처, p {se['vr']['13']['p']:.3f} · {se['vr']['26']['p']:.3f}) — 손실 구간 뒤에 이익이 오는 경향. 매주 갱신은 손실 직후 크기를 줄이고
    이익 직후 키우는데, 되돌림이 있으면 그게 <b>'싸게 팔고 비싸게 사는'</b> 셈이 된다. 매월 갱신은 한 달 안에선 크기를 그대로 둬서 그 손해를 덜 본다.</li>
<li><b>순서가 무작위라면 차이가 없다.</b> 주를 하나씩 섞으면(IID) 매월이 이길 확률 {al['1']['월_0']['p_beat'] * 100:.0f}% · 차이 {al['1']['월_0']['d_p50'] * 100:+.2f}%p.
    그래서 매월로 바꿔 잃을 건 거의 없고, 되돌림이 앞으로도 이어지면 연 1 ~ 3%p 를 더 얻는다. 되돌림은 결과를 보고 찾은 성질이라 계속된다는 보장은 없다.</li>
<li><b>대가: 한 달 안에서 실효 G 가 흔들린다</b> — 역사 {h3['g_min']:.2f} ~ {h3['g_max']:.2f} (손실 달엔 커지고 이익 달엔 작아진다), 섞은 경로의 최대 95% {b3['g_max_p95']:.2f}.
    분기 · 매년은 역사에서 더 벌었지만 실효 G 가 1.3 ~ 2.1 까지 커지는 경로가 흔해서(서술) 채택하지 않았다. 갱신 안 함(단리)은 가장 덜 벌고({h6['cagr'] * 100:.1f}%),
    손실이 길면 실효 G 가 계속 커진다.</li>
<li><b>복리를 가장 크게 좌우하는 건 G 자체</b>다. G 0.75 에서 산술 연수익 {x75['arith'] * 100:.1f}% 가 복리로는 {x75['cagr'] * 100:.1f}% — 변동성 끌림 {x75['drag'] * 100:.1f}%p.
    끌림은 G 의 제곱으로 커져 G 1 에서 12%p, 1.5 에서 30%p. 매월 갱신은 이 끌림의 일부를 되찾는 방법이다.</li>
</ul></div>

<h2>1. 누적 자산 (로그 축)</h2>
<div class="card">{equity_chart(w)}</div>

<h2>2. 실효 G — 명목 ÷ 지금 자산</h2>
<div class="card">{geff_chart(w)}</div>
<p>매주 갱신은 늘 0.75(점선). 매월은 한 달 안에서 손실이 나면 위로, 이익이 나면 아래로 흐르다 월초에 0.75 로 돌아온다. 비대칭은 손실 땐 매주 줄여 0.75 를 넘지 않는다.
갱신 안 함은 자산이 불어날수록 계속 작아진다 — 복리를 포기한 대신 낙폭도 작다.</p>

<h2>3. 매주 대비 — 같은 5년 경로에서 짝 비교</h2>
<div class="card">{diff_chart(K)}</div>
<p>과거 주를 4주 블록으로 다시 뽑은 5년 경로 10,000개. 점 = 연복리 차이 중앙, 선 = 5 ~ 95%, 오른쪽 = 매주보다 좋은 경로 비율.</p>

<h2>4. 주기별 표</h2>
<div class="tbl"><table><thead><tr><th>갱신 주기</th><th>횟수</th><th>5년</th><th>연복리</th><th>낙폭</th><th>샤프</th><th>실효 G</th>
<th>섞은 연복리</th><th>섞은 낙폭 중앙·95%</th><th>매주 이김</th><th>섞은 G 최대</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">왼쪽 = 실제 269주 (매월 = 달력 월의 첫 월요일, 분기 = 1·4·7·10월, 매년 = 1월). '섞은' = 4주 블록 부트스트랩 5년 경로 10,000개의 중앙값
(매주 이김 = 같은 경로에서 매주보다 연복리가 높은 비율, 섞은 G 최대 = 경로마다 실효 G 최댓값의 95번째 백분위). 섞은 경로에선 4 · 13 · 52주마다 갱신. 최대낙폭은 장중 저점 포함, 교차 증거금 청산(MMR 2%) 확인 — 역사 경로에서 청산 없음(임계 G {K['gliq_min']:.2f}).</p>

<h2>5. 얼마나 믿을 만한가</h2>
<div class="tbl"><table><thead><tr><th>섞는 블록 길이</th><th>매월 · 블록에 맞춤</th><th>매월 · 2주 어긋남</th><th>분기 · 맞춤</th><th>분기 · 어긋남</th></tr></thead><tbody>{arow}</tbody></table></div>
<p class="note">칸 = 매주보다 좋을 확률 · 연복리 차이 중앙. 블록 1주(완전히 섞음)면 차이가 사라진다 — 효과는 전부 '순서(되돌림)' 에서 나온다. 블록을 길게 해 과거 순서를 더 살릴수록
매월 · 분기의 이점이 뚜렷하다. 4주 블록에서 매월 갱신 시점을 블록 경계와 2주 어긋나게 하면 이점이 줄어든다(58% · +0.3%p) — 한 달 안의 되돌림만으론 약하고(4주 분산비 {se['vr']['4']['obs']:.2f}, p {se['vr']['4']['p']:.2f}),
몇 달 단위 되돌림이 더 뚜렷하다. 주간 자기상관 1 ~ 4주: {' · '.join(f'{a:+.2f}' for a in se['ac'])}.</p>

<h2>6. G 에 따라 (연복리 · 최대낙폭)</h2>
<div class="tbl"><table><thead><tr><th></th><th>매주</th><th>매월</th><th>비대칭</th><th>갱신 안 함</th></tr></thead><tbody>{grow}</tbody></table></div>
<p class="note">G 가 클수록 주기 효과도 커진다(G 1: 매월 +{(gs['1.0']['C3']['cagr'] - gs['1.0']['C1']['cagr']) * 100:.1f}%p). 대신 G 1 매월은 실효 G 가 {gs['1.0']['C3']['g_max']:.2f} 까지 오른다.</p>

<h2>7. 복리와 변동성 끌림</h2>
<div class="card">{drag_chart(K)}</div>
<p>같은 주간 평균 수익이라도 흔들림이 크면 복리가 덜 붙는다(연복리 ≈ 산술 − 분산 ÷ 2). G 를 올리면 산술 수익은 G 에 비례해 늘지만 끌림은 G² 로 늘어 G 1.2 근처부터 연복리가 오히려 준다.</p>

<h2>8. 운용 방법 (매월 갱신)</h2>
<p class="note">① 매달 첫 월요일 00:00 UTC 에 계좌 총자산(미실현 손익 포함, USDT)을 읽어 B 로 저장한다. ② 매주 월요일: 28일 부호와 60일 역변동성 비중 w 를 계산하고,
코인별 목표 명목 = 0.75 × B × w 로 주문한다(B 는 그 달 내내 그대로). ③ 이익금은 인출하지 않는다 — 다음 달 B 에 들어가야 복리가 붙는다. ④ 자산은 거래소 잔고를 그대로 쓰고,
입출금이 있었으면 그 달 B 에 반영한다. 참고: 실효 G 가 1 을 넘으면 B 를 앞당겨 갱신하는 안전장치는 역사 경로에서 한 번도 걸리지 않는다(최대 {h3['g_max']:.2f}) — 걸어도 위 결과는 같다.</p>

<h2>9. 검증</h2>
<p class="note">테스트: 매주 갱신 = 기존 장부, 매월 · 비대칭 · 갱신 안 함의 실효 G 손계산, 청산 · 장중 저점. 독립 재계산(모듈 없이 반복문): 매주 2.716배 · 매월 3.054배(실효 G 0.57 ~ 0.93) · 갱신 안 함 2.428배 — 일치.</p>
</main></div></body></html>"""
    Path("out/tsmom_compound.html").write_text(html, encoding="utf-8")
    print("out/tsmom_compound.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
