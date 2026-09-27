"""시계열 추세 + 뒤집기 보류 — HTML 보고서. 입력 runs/tsmom_hold*.json · csv → out/tsmom_hold.html"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

SER = (("base", "기준 (28일 부호)", "ink"), ("k0.5", "보류 k = 0.5 ATR", "c1"), ("k1.0", "보류 k = 1 ATR", "c2"),
       ("k2.0", "보류 k = 2 ATR", "c3"))


def svg(W, H, aria, minw=600):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;height:auto;display:block;font-family:inherit">')


def equity_chart(w: pd.DataFrame) -> str:
    W, H, l, r, t, b = 820, 400, 58, 170, 40, 40
    eq = (1 + w).cumprod()
    lo, hi = np.log(0.1), np.log(5.0)
    xs = np.linspace(l, W - r, len(eq))
    sy = lambda v: t + (H - t - b) * (hi - np.log(max(v, 0.1))) / (hi - lo)
    out = [svg(W, H, "누적 자산(로그 축) — 기준 대 뒤집기 보류 k 0.5 · 1 · 2")]
    lx = l
    for key, name, cls in SER:
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="14" y2="14" class="{cls}"/><text x="{lx + 24}" y="18" class="lg">{escape(name)}</text>')
        lx += 44 + 10.5 * len(name)
    for v in (0.1, 0.25, 0.5, 1, 2, 4):
        cls = "base" if v == 1 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}배</text>')
    for y in sorted(set(eq.index.year)):
        k = int(np.searchsorted(eq.index.year, y))
        out.append(f'<line x1="{xs[k]:.1f}" x2="{xs[k]:.1f}" y1="{t}" y2="{H - b}" class="grid"/>'
                   f'<text x="{xs[k] + 4:.1f}" y="{H - b + 16}" class="tick">{y}</text>')
    for key, name, cls in SER:
        pts = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, eq[key]))
        out.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
    ends = sorted(((sy(eq[k].iloc[-1]), k, n) for k, n, _ in SER), key=lambda z: z[0])
    last = -1e9
    for yv, k, n in ends:
        yv = max(yv, last + 15)
        last = yv
        out.append(f'<text x="{W - r + 8}" y="{yv + 4:.1f}" class="dl">{escape(n)} {eq[k].iloc[-1]:.2f}배</text>')
    bw = (W - l - r) / len(eq)
    for i, ts in enumerate(eq.index):
        tip = f"{ts.date()} · " + " · ".join(f"{n} {eq[k].iloc[i]:.2f}배" for k, n, _ in SER)
        out.append(f'<rect x="{xs[i] - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def flip_chart(F: dict) -> str:
    keys = ["보유 방향으로 움직임(<0)", "0~0.5 ATR", "0.5~1 ATR", "1~2 ATR", "2 ATR 이상"]
    labs = ["보유 방향 (<0)", "0 ~ 0.5", "0.5 ~ 1", "1 ~ 2", "2 이상"]
    W, H, l, r, t, b = 680, 310, 60, 20, 40, 56
    hi, lo = 500, -50
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    gw = (W - l - r) / len(keys)
    bw = 46
    out = [svg(W, H, "뒤집힌 직후 1주 수익 — 뒤집힌 주의 역행 크기(ATR)별", 520)]
    for v in range(0, 501, 100):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:+d}</text>')
    u = F["unchanged"]["mean_bps"]
    out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(u):.1f}" y2="{sy(u):.1f}" class="refl"/>'
               f'<line x1="{l}" x2="{l + 18}" y1="12" y2="12" class="refl"/>'
               f'<text x="{l + 24}" y="16" class="lg">안 뒤집힌 주 평균 {u:+.0f}bps</text>'
               f'<rect x="{l + 190}" y="6" width="12" height="12" rx="2" class="bar"/>'
               f'<text x="{l + 208}" y="16" class="lg">뒤집힌 직후 1주 평균</text>')
    for i, (k, lab) in enumerate(zip(keys, labs)):
        v = F[k]
        x = l + i * gw + (gw - bw) / 2
        y1 = sy(v["mean_bps"])
        tip = f"뒤집힌 주의 역행 {lab} ATR · 다음 주 평균 {v['mean_bps']:+.1f}bps (t {v['t']:+.2f}, {v['n']:,}건)"
        h = abs(sy(0) - y1)
        rr = min(3, h / 2)
        out.append(f'<g><title>{escape(tip)}</title><rect x="{l + i * gw:.1f}" y="{t}" width="{gw:.1f}" height="{H - t - b}" class="hit"/>'
                   f'<path d="M{x:.1f},{sy(0):.1f} V{y1 + rr:.1f} Q{x:.1f},{y1:.1f} {x + rr:.1f},{y1:.1f} H{x + bw - rr:.1f} '
                   f'Q{x + bw:.1f},{y1:.1f} {x + bw:.1f},{y1 + rr:.1f} V{sy(0):.1f} Z" class="bar"/>'
                   f'<text x="{x + bw / 2:.1f}" y="{y1 - 6:.1f}" class="val" text-anchor="middle">{v["mean_bps"]:+.0f}</text>'
                   f'<text x="{x + bw / 2:.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{escape(lab)}</text>'
                   f'<text x="{x + bw / 2:.1f}" y="{H - b + 30}" class="tick" text-anchor="middle">{v["n"]:,}건</text></g>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 6}" class="tick" text-anchor="middle">뒤집힌 주에 이전 포지션 방향으로 불리하게 움직인 크기 (주봉 ATR 배수)</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_hold.json").read_text(encoding="utf-8"))
    F = json.loads(Path("runs/tsmom_hold_flips.json").read_text(encoding="utf-8"))
    w = pd.read_csv("runs/tsmom_hold_weekly.csv", index_col=0, parse_dates=True)
    b = K["base"]
    rows = [f"<tr><td>기준 (28일 부호)</td><td>{b['all']['sharpe']:+.2f}</td><td>{b['first']['sharpe']:+.2f}</td><td>{b['second']['sharpe']:+.2f}</td>"
            f"<td>{b['all']['cagr'] * 100:+.1f}%</td><td>{b['all']['mdd'] * 100:.0f}%</td><td>—</td><td>—</td><td>—</td><td>{b['fee_pct_yr']:.2f}%</td><td>—</td></tr>"]
    for k in ("0.5", "1.0", "2.0"):
        o = K[f"k{k}"]
        rows.append(f"<tr><td>보류 k = {float(k):g} ATR</td><td>{o['all']['sharpe']:+.2f}</td><td>{o['first']['sharpe']:+.2f}</td>"
                    f"<td>{o['second']['sharpe']:+.2f}</td><td>{o['all']['cagr'] * 100:+.1f}%</td><td>{o['all']['mdd'] * 100:.0f}%</td>"
                    f"<td>{o['d_mean_bps']:+.0f} (t {o['d_t_boot']:+.2f})</td><td>{o['held_share'] * 100:.0f}%</td>"
                    f"<td>{o['hold_weeks_mean']:.0f}주</td><td>{o['fee_pct_yr']:.2f}%</td><td>{'통과' if o['verdict']['passed'] else '기각'}</td></tr>")
    sens = "".join(f"<tr><td>{lb}일</td><td>{r['base']:+.2f}</td><td>{r['k0.5']:+.2f}</td><td>{r['k1.0']:+.2f}</td><td>{r['k2.0']:+.2f}</td></tr>"
                   for lb, r in K["lookback_sens"].items())
    yrs = K["years"]
    yrow = "".join(f"<tr><td>{y}</td>" + "".join(f"<td>{yrs[c][y] * 100:+.1f}%</td>" for c in ("base", "k0.5", "k1.0", "k2.0")) + "</tr>"
                   for y in yrs["base"])
    o5 = K["k0.5"]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>시계열 추세 뒤집기 보류</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--acc:#3987e5}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--acc:#3987e5}}
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
p{{color:var(--text-secondary)}}
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .val{{fill:var(--text-primary);font-size:11px}}
svg .dl{{fill:var(--text-primary);font-size:12px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .ink{{stroke:var(--text-primary);stroke-width:2}}svg .c1{{stroke:var(--c1);stroke-width:2}}
svg .c2{{stroke:var(--c2);stroke-width:2}}svg .c3{{stroke:var(--c3);stroke-width:2}}
svg .bar{{fill:var(--c1)}}svg .refl{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
svg .hit{{fill:transparent}}svg g:hover .hit{{fill:var(--grid);fill-opacity:.45}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.6}}
</style></head><body><div class="viz-root"><main>
<h1>시계열 추세 + 뒤집기 보류 (주봉 ATR)</h1>
<div class="sub">47심볼 · 2021-05 ~ 2026-06 · 270주 · 롱·숏 대칭 · 주봉 ATR 14주 · 사전등록 판정 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>판정: k = 0.5 · 1 · 2 모두 기각 — 셋 다 기준보다 확실히 나쁘다.</b> 샤프 {b['all']['sharpe']:.2f} →
    {K['k0.5']['all']['sharpe']:+.2f} · {K['k1.0']['all']['sharpe']:+.2f} · {K['k2.0']['all']['sharpe']:+.2f}.
    주간 차이 {o5['d_mean_bps']:+.0f} · {K['k1.0']['d_mean_bps']:+.0f} · {K['k2.0']['d_mean_bps']:+.0f} bps/주 (t {o5['d_t_boot']:+.2f} · {K['k1.0']['d_t_boot']:+.2f} · {K['k2.0']['d_t_boot']:+.2f}).</li>
<li><b>뒤집기 신호는 대부분 조용히 난다.</b> 0.5 ATR 만 걸어도 뒤집기 신호의 {o5['held_share'] * 100:.0f}% 가 보류되고,
    k = 1 이면 {K['k1.0']['held_share'] * 100:.0f}%, k = 2 면 {K['k2.0']['held_share'] * 100:.0f}% — 사실상 안 뒤집는 전략이 된다(평균 보유 {K['k2.0']['hold_weeks_mean']:.0f}주).</li>
<li><b>그런데 이 전략의 돈은 바로 그 '뒤집힌 직후 1주' 에서 나온다.</b> 뒤집힌 주의 다음 주 평균 {F['fresh']['mean_bps']:+.0f}bps
    (t {F['fresh']['t']:+.2f}) vs 안 뒤집힌 주 {F['unchanged']['mean_bps']:+.0f}bps. 조용히(0 ~ 0.5 ATR) 뒤집힌 경우도 {F['0~0.5 ATR']['mean_bps']:+.0f}bps 다.
    보류는 가장 값진 주를 버리고 신호와 반대로 들고 있게 만든다 — k = 0.5 에서 규칙이 바꾼 코인·주는 평균 {o5['diff_gain_bps']:+.0f}bps/주.</li>
<li>lookback 21 · 35 · 56일에서도 모든 k 가 기준보다 나쁘다 — 28일 특유의 우연이 아니다.</li>
<li><b>뜻:</b> 휘둘림을 줄이려는 직관은 자연스럽지만, 이 신호에서는 '작은 움직임에 뒤집힌 것' 이 잡음이 아니라 정보였다.
    (뒤집힌 직후 가치가 크다는 건 결과를 보고 알게 된 사후 관찰이다 — 이를 이용하는 규칙은 새로 사전등록해야 한다.)</li>
</ul></div>

<h2>1. 누적 자산 (로그 축)</h2>
<div class="card">{equity_chart(w)}</div>
<p>세로 띠에 마우스를 올리면 그 주의 네 곡선 값이 나온다. k 가 클수록 뒤집기를 덜 하고, 곡선이 기준에서 멀어진다.</p>

<h2>2. 판정 표</h2>
<div class="tbl"><table><thead><tr><th>규칙</th><th>샤프</th><th>전반</th><th>후반</th><th>연복리</th><th>최대낙폭</th>
<th>주간 차이 bps (t)</th><th>뒤집기 보류</th><th>평균 보유</th><th>수수료 연</th><th>판정</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>판정 기준(사전등록): ① 주간 차이 평균 &gt; 0 · 4주 블록 부트스트랩 t ≥ 2.4 ② 전반·후반 샤프 모두 기준보다 높음 ③ 최대낙폭이 기준보다 5%p 넘게
나빠지지 않음. 세 k 모두 ①②③ 전부 실패. 수수료는 줄지만(연 {b['fee_pct_yr']:.2f}% → {o5['fee_pct_yr']:.2f}% · {K['k1.0']['fee_pct_yr']:.2f}% · {K['k2.0']['fee_pct_yr']:.2f}%) 잃는 수익이 수십 배 크다.</p>

<h2>3. 왜 — 뒤집힌 직후 1주의 가치 (사후 서술)</h2>
<div class="card">{flip_chart(F)}</div>
<p>기준 규칙에서 포지션이 뒤집힌 코인·주({F['fresh']['n']:,}건)의 다음 주 수익을, 뒤집힌 그 주에 이전 포지션 방향으로 얼마나 불리하게 움직였는지(ATR 배수)로 나눴다.
보류 규칙이 '안 뒤집는' 쪽(왼쪽 칸들)도 다음 주 수익이 크다. 점선 = 안 뒤집힌 주 평균. 뒤집힌 뒤 누적은 1주 +161bps · 2주 +221bps · 3주 +250bps 로
2 ~ 3주 안에 대부분 끝난다.</p>

<h2>4. lookback 민감도 (샤프)</h2>
<div class="tbl"><table><thead><tr><th>lookback</th><th>기준</th><th>k 0.5</th><th>k 1</th><th>k 2</th></tr></thead><tbody>{sens}</tbody></table></div>

<h2>5. 연도별 (복리)</h2>
<div class="tbl"><table><thead><tr><th>연도</th><th>기준</th><th>k 0.5</th><th>k 1</th><th>k 2</th></tr></thead><tbody>{yrow}</tbody></table></div>
</main></div></body></html>"""
    Path("out/tsmom_hold.html").write_text(html, encoding="utf-8")
    print("out/tsmom_hold.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
