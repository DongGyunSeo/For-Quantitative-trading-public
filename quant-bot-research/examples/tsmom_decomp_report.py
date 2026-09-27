"""시계열 추세 수익 분해 — HTML 보고서 (examples/tsmom_decomp.py 산출을 그린다).

입력 runs/tsmom_decomp.json · runs/tsmom_decomp_weekly.csv → out/tsmom_decomp.html
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def svg_open(W: int, H: int, label: str, minw: int = 560) -> str:
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(label)}" '
            f'style="width:100%;min-width:{minw}px;height:auto;display:block;font-family:inherit">')


def bar_up(x: float, y0: float, y1: float, w: float, cls: str, r: float = 3.0) -> str:
    """세로 막대 — 기준선(y0) 반대쪽 끝만 둥글게."""
    h = abs(y1 - y0)
    if h < 0.5:
        return ""
    r = min(r, w / 2, h / 2)
    if y1 < y0:   # 위로
        d = (f"M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} "
             f"Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z")
    else:         # 아래로
        d = (f"M{x:.1f},{y0:.1f} V{y1 - r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} "
             f"Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 - r:.1f} V{y0:.1f} Z")
    return f'<path d="{d}" class="{cls}"/>'


def breadth_chart(hist: list[int]) -> str:
    W, H, l, r, t, b = 720, 300, 56, 16, 30, 58
    n = sum(hist)
    ymax = max(hist) * 1.15
    sy = lambda v: t + (H - t - b) * (1 - v / ymax)
    bw = (W - l - r) / len(hist)
    out = [svg_open(W, H, "주마다 오른 코인 비율의 분포")]
    for v in range(0, int(ymax) + 1, 20):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
    for i, c in enumerate(hist):
        x = l + i * bw + 3
        lo, hi = i * 10, (i + 1) * 10
        edge = i in (0, 1, 8, 9)
        tip = f"{lo}~{hi}% 의 코인이 오른 주: {c}주 ({c / n * 100:.1f}%)"
        out.append(f'<g><title>{escape(tip)}</title><rect x="{l + i * bw:.1f}" y="{t}" width="{bw:.1f}" '
                   f'height="{H - t - b}" class="hit"/>'
                   + bar_up(x, sy(0), sy(c), bw - 6, "s1" if edge else "s1m")
                   + f'<text x="{x + (bw - 6) / 2:.1f}" y="{sy(c) - 6:.1f}" class="val" text-anchor="middle">{c}</text>'
                   + f'<text x="{x + (bw - 6) / 2:.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{lo}~{hi}%</text></g>')
    out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" class="base"/>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 16}" class="tick" text-anchor="middle">'
               f'그 주에 오른 코인 비율 (47개 중) — 막대 높이 = 주 수</text>')
    out.append(f'<text x="{l + 2 * bw:.0f}" y="{t - 10}" class="lg" text-anchor="middle">← 거의 다 같이 내림</text>'
               f'<text x="{l + 8 * bw:.0f}" y="{t - 10}" class="lg" text-anchor="middle">거의 다 같이 오름 →</text>')
    out.append("</svg>")
    return "".join(out)


def cum_chart(w: pd.DataFrame) -> str:
    W, H, l, r, t, b = 820, 390, 58, 150, 34, 40
    cum = pd.DataFrame(dict(gross=w.gross.cumsum(), timing=w.timing.cumsum(), select=w.select.cumsum(),
                            cost=-(w.fee + w.fund).cumsum())) * 100
    lo = min(cum.min().min(), 0) - 10
    hi = cum.max().max() + 10
    xs = np.linspace(l, W - r, len(cum))
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    out = [svg_open(W, H, "주간 수익 누적 합 — 시장 방향, 코인 선별, 비용", 620)]
    step = 50 if hi - lo > 150 else 25
    for v in range(int(np.ceil(lo / step) * step), int(hi) + 1, step):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:+d}</text>')
    years = sorted(set(cum.index.year))
    for y in years:
        k = np.searchsorted(cum.index.year, y)
        out.append(f'<line x1="{xs[k]:.1f}" x2="{xs[k]:.1f}" y1="{t}" y2="{H - b}" class="grid"/>'
                   f'<text x="{xs[k] + 4:.1f}" y="{H - b + 16}" class="tick">{y}</text>')
    series = (("gross", "전략 (수수료 전)", "ink"), ("timing", "① 시장 방향", "s1l"), ("select", "② 코인 선별", "s2l"),
              ("cost", "비용 (수수료+펀딩)", "mutl"))
    lx = l
    for key, name, cls in series:
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="12" y2="12" class="{cls}"/>'
                   f'<text x="{lx + 24}" y="16" class="lg">{escape(name)}</text>')
        lx += 36 + 11 * len(name)
    for key, name, cls in series:
        pts = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, cum[key]))
        out.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
    ends = sorted(((sy(cum[k].iloc[-1]), k, n) for k, n, _ in series), key=lambda z: z[0])
    last_y = -1e9
    for yv, k, n in ends:
        yv = max(yv, last_y + 14)
        last_y = yv
        out.append(f'<text x="{W - r + 8}" y="{yv + 4:.1f}" class="dl">{escape(n)} {cum[k].iloc[-1]:+.0f}</text>')
    # 주별 호버 (세로 띠)
    bw = (W - l - r) / len(cum)
    for i, (ts, row) in enumerate(cum.iterrows()):
        wk = w.iloc[i]
        tip = (f"{ts.date()} 주 · 누적 전략 {row.gross:+.0f} · 시장 방향 {row.timing:+.0f} · 코인 선별 {row.select:+.0f} · "
               f"비용 {row.cost:+.0f} (%p) | 이 주: 시장 {wk.market * 100:+.1f}% · 순노출 {wk.sbar:+.2f}")
        out.append(f'<rect x="{xs[i] - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv">'
                   f'<title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def year_chart(years: dict) -> str:
    ys = sorted(years, key=int)
    W, H, l, r, t, b = 720, 320, 56, 16, 34, 44
    vals = [v for y in ys for v in (years[y]["timing_pct"], years[y]["select_pct"])]
    hi = max(vals) * 1.2
    lo = min(min(vals) * 1.3, -5)
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    gw = (W - l - r) / len(ys)
    bw = min(34, gw * 0.3)
    out = [svg_open(W, H, "연도별 분해 — 시장 방향 대 코인 선별")]
    out.append(f'<rect x="{l}" y="8" width="12" height="12" rx="2" class="s1"/><text x="{l + 18}" y="18" class="lg">① 시장 방향</text>'
               f'<rect x="{l + 120}" y="8" width="12" height="12" rx="2" class="s2"/><text x="{l + 138}" y="18" class="lg">② 코인 선별</text>')
    for v in range(int(np.ceil(lo / 20) * 20), int(hi) + 1, 20):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:+d}</text>')
    for i, y in enumerate(ys):
        d = years[y]
        cx = l + i * gw + gw / 2
        lab = f"{y}" + (" (5월~)" if str(y) == "2021" else " (6월까지)" if str(y) == "2026" else "")
        tip = (f"{lab} · 시장 방향 {d['timing_pct']:+.1f}%p · 코인 선별 {d['select_pct']:+.1f}%p · 비용 {d['cost_pct']:+.1f}%p · "
               f"합 {d['net_pct']:+.1f}%p · 시장 자체 {d['market_pct']:+.1f}%p · 평균 순노출 {d['sbar']:+.2f}")
        out.append(f'<g><title>{escape(tip)}</title><rect x="{l + i * gw:.1f}" y="{t}" width="{gw:.1f}" height="{H - t - b}" class="hit"/>')
        for j, (k, cls) in enumerate((("timing_pct", "s1"), ("select_pct", "s2"))):
            x = cx - bw - 1 + j * (bw + 2)
            v = d[k]
            out.append(bar_up(x, sy(0), sy(v), bw, cls))
            ty = sy(v) - 5 if v >= 0 else sy(v) + 13
            out.append(f'<text x="{x + bw / 2:.1f}" y="{ty:.1f}" class="val" text-anchor="middle">{v:+.0f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{escape(lab)}</text></g>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 6}" class="tick" text-anchor="middle">주간 수익의 단순 합 (%p) — 복리 연수익과 다르다</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_decomp.json").read_text(encoding="utf-8"))
    w = pd.read_csv("runs/tsmom_decomp_weekly.csv", index_col=0, parse_dates=True)
    c, cm, sd, rg = K["components"], K["comove"], K["static_dynamic"], K["regimes"]
    same = cm["breadth_ge80"] + cm["breadth_le20"]
    vshare = K["variance"]["var_timing"] / K["variance"]["var_gross"]
    comp_rows = [("전략 (수수료 전)", c["gross"]), ("① 시장 방향 = 순노출 × 시장", c["timing"]),
                 ("② 코인 선별 = 평균보다 더 롱인 코인이 더 올랐나", c["select"]), ("수수료", c["fee"]),
                 ("펀딩", c["fund"]), ("전략 (비용 후)", c["net"])]
    tr = "".join(f"<tr><td>{escape(n)}</td><td>{v['mean_bps']:+.1f}</td><td>{v['ann_pct']:+.1f}%</td>"
                 f"<td>{v.get('share', 0) * 100:+.0f}%</td><td>{v['t']:+.2f}</td></tr>" for n, v in comp_rows)
    yr = K["years"]
    yrows = "".join(
        f"<tr><td>{y}{' (5월~)' if y == '2021' else ' (6월까지)' if y == '2026' else ''}</td>"
        f"<td>{d['market_pct']:+.0f}</td><td>{d['sbar']:+.2f}</td><td>{d['timing_pct']:+.1f}</td><td>{d['select_pct']:+.1f}</td>"
        f"<td>{d['cost_pct']:+.1f}</td><td>{d['net_pct']:+.1f}</td></tr>" for y, d in sorted(yr.items()))
    rrows = "".join(f"<tr><td>{escape(k)}</td><td>{d['weeks']}</td><td>{d['timing_bps']:+.1f}</td><td>{d['select_bps']:+.1f}</td>"
                    f"<td>{d['gross_bps']:+.1f}</td></tr>" for k, d in rg.items())
    ycorr = " · ".join(f"{y} {v:.2f}" for y, v in sorted(cm["yearly_pair_corr"].items()))
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>시계열 추세 수익 분해</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s1m:#86b6ef;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--s1m:#184f95;--acc:#3987e5}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--s1m:#184f95;--acc:#3987e5}}
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
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:var(--text-secondary);font-weight:600}}
p{{color:var(--text-secondary)}}
.eq{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--surface-1);border:1px solid var(--line);
border-radius:8px;padding:10px 14px;overflow-x:auto;white-space:pre;color:var(--text-primary)}}
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .val{{fill:var(--text-primary);font-size:11px;font-variant-numeric:tabular-nums}}
svg .dl{{fill:var(--text-primary);font-size:12px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .s1{{fill:var(--s1)}}svg .s1m{{fill:var(--s1m)}}svg .s2{{fill:var(--s2)}}svg .hit{{fill:transparent}}
svg g:hover .hit{{fill:var(--grid);fill-opacity:.5}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.6}}
svg .ink{{stroke:var(--text-primary);stroke-width:2}}svg .s1l{{stroke:var(--s1);stroke-width:2}}
svg .s2l{{stroke:var(--s2);stroke-width:2}}svg .mutl{{stroke:var(--muted);stroke-width:2;stroke-dasharray:4 3}}
</style></head><body><div class="viz-root"><main>
<h1>시계열 추세 수익 분해 — 다 같이 움직이는 부분 vs 코인별 부분</h1>
<div class="sub">OKX 47심볼 · 2021-05 ~ 2026-06 · {K['weeks']}주 · 28일 부호 · 동일가중 · 서술(판정 아님)</div>
<div class="box"><ul>
<li><b>코인은 정말 같이 움직인다.</b> 코인 한 개의 주간 변동 중 평균 {cm['r2_mean'] * 100:.0f}% 가 나머지 46개의 평균 움직임으로 설명된다
    (코인끼리 주간 상관 평균 {cm['pair_corr_weekly']:.2f}). {same * 100:.0f}% 의 주에서 47개 중 80% 이상이 같은 방향으로 움직였다.</li>
<li><b>그래서 이 전략의 수익은 사실상 전부 '시장 방향' 에서 나왔다.</b> 수수료 전 주평균 {c['gross']['mean_bps']:+.1f}bps 중
    ① 시장 방향(순노출 × 시장)이 {c['timing']['mean_bps']:+.1f}bps, ② 코인 선별이 {c['select']['mean_bps']:+.1f}bps (t {c['select']['t']:+.2f}) — 0 이다.
    주간 손익 변동의 {vshare * 100:.0f}% 도 ① 에서 나온다.</li>
<li>'평균적으로 숏이었는데 시장이 빠져서' 가 아니다. ① 중 고정된 순숏 기울기 몫은 {sd['timing_static_bps']:+.1f}bps 뿐이고
    {sd['timing_dynamic_bps']:+.1f}bps 가 <b>오를 때 롱 · 내릴 때 숏으로 시기를 맞춘</b> 몫이다.</li>
<li>코인마다 시장 민감도(베타)가 다른 걸 반영해도 같다: 시장 몫 {c['timing_beta']['mean_bps']:+.1f} · 코인 고유 몫 {c['idio_beta']['mean_bps']:+.1f}bps.</li>
<li><b>뜻:</b> 47개로 나눠 들고 있어도 실제로는 <b>'크립토 시장 전체의 한 달 추세' 에 거는 베팅 하나</b>다.
    순노출 부호가 동일가중 지수의 28일 추세 부호와 {cm['signal_agree_index'] * 100:.0f}% 의 주에서 같다.
    분산 효과가 작아 최대낙폭이 −46% 까지 가는 이유이기도 하다.</li>
</ul></div>

<h2>1. 코인은 얼마나 같이 움직이나</h2>
<div class="card">{breadth_chart(cm['breadth_hist'])}</div>
<p>막대 {K['weeks']}개 주를 '그 주에 오른 코인 비율' 로 나눴다. 양 끝(거의 다 같이 내리거나 오름)이 가장 두껍고 가운데는 얇은 U 자다.
양 끝 두 칸씩(80% 이상 같은 방향)은 강조색. 주간 상관 평균 {cm['pair_corr_weekly']:.2f} (일간 {cm['pair_corr_daily']:.2f}), 첫 번째 공통 요인이 전체 변동의
{cm['pc1_share'] * 100:.0f}%. 시장과 가장 따로 노는 코인 {cm['r2_min_coin']} (R² {cm['r2_min']:.2f}), 가장 붙어 다니는 코인 {cm['r2_max_coin']} ({cm['r2_max']:.2f}).
연도별 상관: {ycorr} — 해마다 비슷하게 높다.</p>

<h2>2. 분해 공식</h2>
<div class="eq">전략 수익  = (1/N) Σ s_i · r_i
           = s̄ · m                              ① 시장 방향   s̄ = 순노출(롱 비율 − 숏 비율), m = 47개 평균 수익
           + (1/N) Σ (s_i − s̄)(r_i − m)          ② 코인 선별   평균보다 더 롱인 코인이 평균보다 더 올랐으면 +
           − 수수료 − 펀딩</div>
<p>두 항의 합은 전략 수익과 매주 정확히 같다(검산 완료). ① 은 "다 같이 움직이는 부분을 맞혔나", ② 는 "그걸 빼고 남은 코인별 차이를 맞혔나" 다.</p>
<div class="tbl"><table><thead><tr><th>구성</th><th>주평균 bps</th><th>연 (×52)</th><th>몫</th><th>t</th></tr></thead><tbody>{tr}</tbody></table></div>

<h2>3. 누적으로 보면</h2>
<div class="card">{cum_chart(w)}</div>
<p>주간 수익을 단순 합산한 곡선이다(복리 아님 — 분해가 정확히 더해지도록). 전략 선과 파란 선(시장 방향)이 거의 겹치고,
주황 선(코인 선별)은 0 근처를 오르내린다. 세로 띠에 마우스를 올리면 그 주 값이 나온다.</p>

<h2>4. 연도별</h2>
<div class="card">{year_chart(yr)}</div>
<div class="tbl"><table><thead><tr><th>연도</th><th>시장 자체 합 %p</th><th>평균 순노출</th><th>① 시장 방향</th><th>② 코인 선별</th><th>비용</th><th>전략 합</th></tr></thead>
<tbody>{yrows}</tbody></table></div>
<p>2023 한 해만 코인 선별이 시장 방향만큼 벌었다(+18%p). 나머지 해는 전부 ① 이 만들고 ② 는 0 근처이거나 음수다.
단순 합이라 복리 연수익과 다르다 — 예: 2024 는 합 +21%p 지만 복리로는 −3% (큰 주간 등락의 복리 손실).</p>

<h2>5. 순노출 구간별 (주평균 bps)</h2>
<div class="tbl"><table><thead><tr><th>구간</th><th>주</th><th>① 시장 방향</th><th>② 코인 선별</th><th>합 (수수료 전)</th></tr></thead><tbody>{rrows}</tbody></table></div>
<p>대부분 롱인 주가 수익 대부분을 만든다 — 상승장을 통째로 탄 주들이다.</p>

<h2>6. 그래서 무엇을 뜻하나</h2>
<p><b>이 전략의 엣지(있다면)는 '어떤 코인' 이 아니라 '시장 전체가 어느 쪽으로 가나' 에 있다.</b> 47개 코인 각각의 28일 부호는
대부분 같은 쪽을 가리키고(주의 {cm['abs_sbar_ge_06'] * 100:.0f}% 에서 신호의 80% 이상이 한쪽), 그 합이 곧 지수 추세다.</p>
<p>그렇다면 같은 베팅을 <b>지수 하나(또는 BTC · ETH 같은 유동성 큰 코인)의 추세로</b> 더 싸고 단순하게 할 수 있는지가 자연스러운 다음 질문이다.
다만 이건 새 규칙이라 결과를 보기 전에 사전등록해야 한다 — 여기서는 계산하지 않았다.
또한 이 분해는 표본 안 서술이다. 전략 자체의 유의성(t 1.46)과 lookback 띠 문제는 그대로다.</p>
</main></div></body></html>"""
    Path("out/tsmom_decomp.html").write_text(html, encoding="utf-8")
    print("out/tsmom_decomp.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
