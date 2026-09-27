"""시계열 추세 — 코인별 수익과 변동성 HTML 보고서 (examples/tsmom_coins.py 산출을 그린다).

입력 runs/tsmom_coins.json · runs/tsmom_coins.csv → out/tsmom_coins.html
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = ("BTC", "ETH", "SOL", "CFX", "UMA", "SNX", "TRX", "IOST")


def scatter(T: pd.DataFrame, ykey: str, *, ylo: float, yhi: float, ystep: float, yfmt, ylab: str,
            err: bool, lines: list[tuple[str, float, float, str]], aria: str) -> str:
    W, H, l, r, t, b = 760, 400, 64, 24, 40, 50
    xlo, xhi = 0.5, 1.45
    sx = lambda v: l + (v - xlo) / (xhi - xlo) * (W - l - r)
    sy = lambda v: t + (yhi - min(max(v, ylo), yhi)) / (yhi - ylo) * (H - t - b)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
           f'style="width:100%;min-width:600px;height:auto;display:block;font-family:inherit">']
    lx = l
    legend = [("dot", "코인 (47)")] + ([("err", "±1 표준오차")] if err else []) + [(c, n) for n, _, _, c in lines]
    for cls, name in legend:
        if cls == "dot":
            out.append(f'<circle cx="{lx + 6}" cy="14" r="4.5" class="pt"/>')
        elif cls == "err":
            out.append(f'<line x1="{lx + 6}" x2="{lx + 6}" y1="6" y2="22" class="eb"/>')
        else:
            out.append(f'<line x1="{lx}" x2="{lx + 16}" y1="14" y2="14" class="{cls}"/>')
        out.append(f'<text x="{lx + 22}" y="18" class="lg">{escape(name)}</text>')
        lx += 36 + 11.5 * len(name)
    v = np.ceil(ylo / ystep) * ystep
    while v <= yhi + 1e-9:
        cls = "base" if abs(v) < 1e-9 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{yfmt(v)}</text>')
        v += ystep
    for xv in np.arange(0.6, 1.45, 0.2):
        out.append(f'<line x1="{sx(xv):.1f}" x2="{sx(xv):.1f}" y1="{t}" y2="{H - b}" class="grid"/>'
                   f'<text x="{sx(xv):.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{xv * 100:.0f}%</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 8}" class="tick" text-anchor="middle">코인 변동성 (연, 일간 수익 기준)</text>')
    out.append(f'<text x="14" y="{(t + H - b) / 2:.0f}" class="tick" text-anchor="middle" '
               f'transform="rotate(-90 14 {(t + H - b) / 2:.0f})">{escape(ylab)}</text>')
    for name, slope, icpt, cls in lines:
        out.append(f'<line x1="{sx(xlo):.1f}" x2="{sx(xhi):.1f}" y1="{sy(icpt + slope * xlo):.1f}" '
                   f'y2="{sy(icpt + slope * xhi):.1f}" class="{cls}"/>')
    for row in T.itertuples(index=False):
        x, y = sx(row.vol), sy(getattr(row, ykey))
        tip = (f"{row.coin} · 변동성 {row.vol * 100:.0f}% · 베타 {row.beta:.2f} · 추세 연수익 {row.ann * 100:+.0f}% "
               f"(±{row.se_ann * 100:.0f}) · 샤프 {row.sharpe:+.2f} · 매수보유 연 {row.bh_ann * 100:+.0f}%"
               + (" · 단독 1배면 파산" if row.busted else ""))
        g = [f'<g><title>{escape(tip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="10" class="hit"/>']
        if err:
            v0, v1 = getattr(row, ykey) - row.se_ann, getattr(row, ykey) + row.se_ann
            g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(v0):.1f}" y2="{sy(v1):.1f}" class="eb"/>')
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="pt"/>')
        if row.coin in LABEL:
            left = row.coin in ("IOST",)
            anchor = ' text-anchor="end"' if left else ""
            g.append(f'<text x="{x - 7 if left else x + 7:.1f}" y="{y - 6:.1f}" class="dl"{anchor}>{row.coin}</text>')
        g.append("</g>")
        out.append("".join(g))
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_coins.json").read_text(encoding="utf-8"))
    T = pd.read_csv("runs/tsmom_coins.csv")
    o, ts, sm, tc, dp = K["obs"], K["tests"], K["summary"], K["terciles"], K["dispersion"]
    cs = ts["common_sharpe_ann"]
    fm = ts["fm"]
    exp_gap = cs * (sm["vol_max"] - sm["vol_min"])
    ch1 = scatter(T, "ann", ylo=-0.6, yhi=2.0, ystep=0.4, yfmt=lambda v: f"{v * 100:+.0f}%",
                  ylab="추세 전략 연수익 (산술, 비용 후)", err=True,
                  lines=[("회귀선", o["slope_ann_vol"], o["icpt_ann_vol"], "fit"),
                         (f"같은 샤프 {cs:.2f} 이면", cs, 0.0, "ref")],
                  aria="코인 변동성 대 코인별 추세 연수익")
    ch2 = scatter(T, "sharpe", ylo=-0.6, yhi=1.4, ystep=0.2, yfmt=lambda v: f"{v:+.1f}",
                  ylab="추세 전략 샤프 (코인 단독)", err=False,
                  lines=[("회귀선", o["slope_sh_vol"], o["icpt_sh_vol"], "fit"), (f"평균 {cs:.2f}", 0.0, cs, "ref")],
                  aria="코인 변동성 대 코인별 샤프")
    trows = "".join(
        f"<tr><td>{escape(k)}</td><td>{v['vol'] * 100:.0f}%</td><td>{v['ann'] * 100:+.1f}%</td>"
        f"<td>{v['vol_strat'] * 100:.0f}%</td><td>{v['sharpe']:+.2f}</td><td>{v['mdd'] * 100:.0f}%</td></tr>"
        for k, v in tc.items())
    crow = []
    for r in T.itertuples(index=False):
        crow.append(f"<tr><td>{r.coin}</td><td>{r.vol * 100:.0f}%</td><td>{r.beta:.2f}</td>"
                    f"<td>{r.ann * 100:+.0f}%</td><td>±{r.se_ann * 100:.0f}</td><td>{r.sharpe:+.2f}</td>"
                    f"<td>{r.bh_ann * 100:+.0f}%</td><td>{r.mkt_bps:+.0f}</td><td>{r.own_bps:+.0f}</td>"
                    f"<td>{r.worst * 100:+.0f}%{' ⚠' if r.busted else ''}</td><td>{r.risk_share * 100:.1f}%</td></tr>")
    busted = " · ".join(sm["busted"])
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>코인별 추세 수익과 변동성</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--eb:#b7d3f6;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--eb:#184f95;--acc:#3987e5}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--eb:#184f95;--acc:#3987e5}}
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
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .dl{{fill:var(--text-primary);font-size:11px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .pt{{fill:var(--s1);stroke:var(--surface-1);stroke-width:2}}svg .eb{{stroke:var(--eb);stroke-width:2}}
svg .fit{{stroke:var(--s2);stroke-width:2}}svg .ref{{stroke:var(--muted);stroke-width:2;stroke-dasharray:5 4}}
svg .hit{{fill:transparent}}svg g:hover .pt{{stroke:var(--text-primary)}}
</style></head><body><div class="viz-root"><main>
<h1>코인별 추세 수익 — 차이는 변동성인가</h1>
<div class="sub">시계열 추세(28일 부호 · 주간) 를 코인 하나씩 명목 1배로 · OKX 47심볼 · 2021-05 ~ 2026-06 · 270주 · 서술(판정 아님)</div>
<div class="box"><ul>
<li><b>코인별 결과:</b> 47개 중 {sm['pos_ann']}개가 비용 후 연수익 양수, {sm['beat_bh_ann']}개가 자기 매수보유보다 높다(연 산술 기준).
    변동성은 연 {sm['vol_min'] * 100:.0f}% (BTC) ~ {sm['vol_max'] * 100:.0f}% (TRB) 로 2.5배 차이.</li>
<li><b>가설과 같은 방향이다.</b> 매주 직전 90일 변동성으로 코인을 비교하면(인과) 변동성이 높을수록 원수익이 높고
    (t {fm['raw']['t']:.2f}), <b>변동성으로 나누면 차이가 사라진다</b>(t {fm['per_vol']['t']:+.2f}).
    코인별 샤프는 변동성과 무관하다(상관 {o['corr_sh_vol']:+.2f}). 샤프의 흩어짐 {ts['sd_sharpe_obs']:.3f} 은
    "모든 코인의 샤프가 같다" 고 가정했을 때의 잡음({ts['sd_sharpe_nullA_mean']:.3f})과 같다(p {ts['p_sd_sharpe_nullA']:.2f}).</li>
<li><b>하지만 증명은 안 된다.</b> 코인 하나의 5년 평균 연수익은 표준오차가 ±{sm['se_ann_med'] * 100:.0f}%p 다.
    같은 샤프 {cs:.2f} 라면 BTC 와 가장 변동성 큰 코인의 기대 차이는 약 {exp_gap * 100:.0f}%p 로 잡음보다 작다.
    그래서 "평균수익이 모두 같다" 도 기각되지 않는다(p {ts['p_sd_ann_nullB']:.2f}). 변동성 기울기의 95% 구간
    [{ts['slope_ann_vol_ci'][0]:+.2f}, {ts['slope_ann_vol_ci'][1]:+.2f}] 도 0 을 포함한다.</li>
<li><b>쓸 수 있는 결론:</b> 코인마다 같은 베팅(시장 추세)을 크기만 다르게 하고 있다는 그림과 모순이 없다 →
    <b>코인 고르기가 아니라 변동성으로 크기를 맞추는 것</b>(위험 균등)이 자연스럽다. 포트폴리오 변동성 타깃은
    이미 샤프 0.64 → 0.69~0.71 로 조금 나았다. 고변동 코인만 고르는 건 근거가 없다(높음 − 낮음 샤프
    {ts['terc_sharpe_diff']:+.2f}, 95% 구간 [{ts['terc_sharpe_diff_ci'][0]:+.2f}, {ts['terc_sharpe_diff_ci'][1]:+.2f}]).</li>
<li><b>코인 하나로는 하지 말 것:</b> 명목 1배 숏으로 {len(sm['busted'])}개 코인({busted})이 한 주에 두 배 넘게 올라
    계좌가 0 아래로 갔다. 47개에 나눠 1/47 씩이라 버틴다.</li>
</ul></div>

<h2>1. 변동성과 연수익</h2>
<div class="card">{ch1}</div>
<p>점 하나 = 코인 하나의 추세 전략 연수익(주평균 × 52, 수수료·펀딩 후). 세로 막대 = ±1 표준오차 — 대부분 ±40~60%p 라
점들의 흩어짐 대부분이 잡음이다. 주황 = 회귀선(변동성 1%p 당 연수익 {o['slope_ann_vol']:+.2f}%p, 상관 {o['corr_ann_vol']:+.2f}),
회색 점선 = 모든 코인의 샤프가 {cs:.2f} 로 같을 때의 기대선. 점에 마우스를 올리면 코인별 값이 나온다.</p>

<h2>2. 변동성과 샤프 (위험 1단위당 수익)</h2>
<div class="card">{ch2}</div>
<p>변동성이 커도 샤프는 오르지도 내리지도 않는다(상관 {o['corr_sh_vol']:+.2f}, 95% 구간
[{ts['corr_sh_vol_ci'][0]:+.2f}, {ts['corr_sh_vol_ci'][1]:+.2f}]). 코인 단독 샤프 평균은 {cs:.2f} 인데 47개를 섞으면 0.64 가 된다 —
코인끼리 상관이 1 이 아니라서 생기는 분산 효과다.</p>

<h2>3. 변동성 3분위로 묶으면</h2>
<div class="tbl"><table><thead><tr><th>묶음</th><th>코인 변동성</th><th>추세 연수익</th><th>묶음 변동성</th><th>샤프</th><th>최대낙폭</th></tr></thead>
<tbody>{trows}</tbody></table></div>
<p>'전구간' 은 5년 전체 변동성으로 나눈 것(사후), '직전90일' 은 매주 그 시점까지의 변동성으로 나눈 것(인과)이다.
고변동 묶음의 원수익이 조금 크지만 묶음 변동성은 52~64% 로 비슷하다 — 묶음 안에서 코인들이 같이 움직여서 개별 변동성 차이가
포트폴리오에선 많이 줄어든다. 위험 기여로 보면 변동성 상위 10개가 포트폴리오 위험의 {dp['top10_vol_risk_share'] * 100:.0f}%,
하위 10개가 {dp['bottom10_vol_risk_share'] * 100:.0f}% (비중은 각각 21%).</p>

<h2>4. 코인별 전체 표 (변동성 순)</h2>
<div class="tbl"><table><thead><tr><th>코인</th><th>변동성</th><th>베타</th><th>추세 연수익</th><th>±SE</th><th>샤프</th>
<th>매수보유 연</th><th>시장 몫 bps/주</th><th>고유 몫 bps/주</th><th>최악의 주</th><th>위험 기여</th></tr></thead>
<tbody>{''.join(crow)}</tbody></table></div>
<p>연수익은 주평균 × 52 (산술). 코인 하나를 명목 1배로 복리 운용하면 ⚠ 표시 코인은 계좌가 0 아래로 가서 복리 수익이 정의되지 않는다.
베타 = 그 코인 주간 수익을 나머지 46개 평균에 회귀한 기울기. 시장 몫 = 포지션 × 베타 × 시장, 고유 몫 = 나머지.
47개 전체로 합치면 시장 몫이 수익의 {sm['mkt_share'] * 100:.0f}% 다(앞 분해와 같은 결론).</p>
</main></div></body></html>"""
    Path("out/tsmom_coins.html").write_text(html, encoding="utf-8")
    print("out/tsmom_coins.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
