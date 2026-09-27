"""체결 시각 지도 — HTML. 입력 runs/tsmom_anchor.json → out/tsmom_anchor.html"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

DAYS = "월화수목금토일"


def svg(W, H, aria, minw=640):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;height:auto;display:block;font-family:inherit">')


def landscape(K: dict) -> str:
    rows = K["et"]
    W, H, l, r, t, b = 900, 400, 50, 20, 44, 48
    lo, hi = 0.2, 1.2
    sx = lambda a: l + (W - l - r) * a / 167
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    out = [svg(W, H, "주 168개 체결 시각(ET)별 샤프 — 금 16:00 ET 와 현행 표시")]
    for d in range(7):
        x0 = sx(d * 24)
        if d >= 5:
            out.append(f'<rect x="{x0:.1f}" y="{t}" width="{sx(min(d * 24 + 24, 167)) - x0:.1f}" height="{H - t - b}" class="wkend"/>')
        out.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{t}" y2="{H - b}" class="grid"/>'
                   f'<text x="{sx(d * 24 + 12):.1f}" y="{H - b + 18}" class="lg" text-anchor="middle">{DAYS[d]}</text>'
                   f'<text x="{x0 + 2:.1f}" y="{H - b + 32}" class="tick">0시</text>'
                   f'<text x="{sx(d * 24 + 12):.1f}" y="{H - b + 32}" class="tick" text-anchor="middle">12시</text>')
    for v in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    v0 = K["v0"]["sharpe"]
    out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v0):.1f}" y2="{sy(v0):.1f}" class="refl"/>')
    pts = " ".join(f"{sx(x['a']):.1f},{sy(x['sharpe']):.1f}" for x in rows)
    out.append(f'<polyline points="{pts}" class="c1" fill="none"/>')
    # 범례
    out.append(f'<line x1="{l}" x2="{l + 18}" y1="14" y2="14" class="c1"/><text x="{l + 24}" y="18" class="lg">그 시각에 신호·체결 (ET 기준, 시각마다 269주)</text>'
               f'<line x1="{l + 330}" x2="{l + 348}" y1="14" y2="14" class="refl"/><text x="{l + 354}" y="18" class="lg">현행 월 00:00 UTC ({v0:.2f})</text>'
               f'<rect x="{l + 520}" y="8" width="12" height="12" class="wkend2"/><text x="{l + 538}" y="18" class="lg">주말(ET)</text>')
    # 표시: 금 16:00 ET · 최고 · 최저 · 현행 위치
    a1 = 4 * 24 + 16
    v1 = rows[a1]["sharpe"]
    out.append(f'<circle cx="{sx(a1):.1f}" cy="{sy(v1):.1f}" r="5.5" class="mk2"/>'
               f'<text x="{sx(a1) + 8:.1f}" y="{sy(v1) - 10:.1f}" class="dl">금 16:00 ET (미국 장마감) {v1:.2f}</text>')
    s = K["et_summary"]
    amax = max(rows, key=lambda x: x["sharpe"])
    amin = min(rows, key=lambda x: x["sharpe"])
    out.append(f'<circle cx="{sx(amax["a"]):.1f}" cy="{sy(amax["sharpe"]):.1f}" r="4" class="mk1"/>'
               f'<text x="{sx(amax["a"]):.1f}" y="{sy(amax["sharpe"]) - 9:.1f}" class="val" text-anchor="middle">최고 {amax["label"]} {amax["sharpe"]:.2f}</text>'
               f'<circle cx="{sx(amin["a"]):.1f}" cy="{sy(amin["sharpe"]):.1f}" r="4" class="mk1"/>'
               f'<text x="{sx(amin["a"]) + 8:.1f}" y="{sy(amin["sharpe"]) + 4:.1f}" class="val">최저 {amin["label"]} {amin["sharpe"]:.2f}</text>')
    a0 = 6 * 24 + 19.5                                  # 월 00:00 UTC = 일 19:00 (EST) · 20:00 (EDT) ET
    out.append(f'<circle cx="{sx(a0):.1f}" cy="{sy(v0):.1f}" r="5" class="hollow"/>'
               f'<text x="{sx(a0) - 8:.1f}" y="{sy(v0) - 10:.1f}" class="dl" text-anchor="end">현행 (일 19~20시 ET)</text>')
    bw = (W - l - r) / 167
    for x in rows:
        tip = (f"{x['label']} ET · 샤프 {x['sharpe']:.2f} · 전반 {x['first']:.2f} · 후반 {x['second']:.2f} · "
               f"연복리 {x['cagr'] * 100:+.0f}% · 최대낙폭 {x['mdd'] * 100:.0f}% · {x['weeks']}주")
        out.append(f'<rect x="{sx(x["a"]) - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_anchor.json").read_text(encoding="utf-8"))
    s, v, tr, mw = K["et_summary"], K["v1_vs_v0"], K["tranche"], K["midweek"]
    v1 = K["v1_rank"]["sharpe"]
    v0 = K["v0"]["sharpe"]
    fz = {x["label"]: x["sharpe"] for x in K["fri_zoom"]}
    y0, y1 = v["years"]["V0"], v["years"]["V1"]
    h0, h1 = v["halves"]["V0"], v["halves"]["V1"]
    vs_rows = (f"<tr><td>샤프</td><td>{v0:.2f}</td><td>{v1:.2f}</td></tr>"
               f"<tr><td>전반 · 후반 샤프</td><td>{h0[0]:.2f} · {h0[1]:.2f}</td><td>{h1[0]:.2f} · {h1[1]:.2f}</td></tr>"
               + "".join(f"<tr><td>{y}</td><td>{y0[y] * 100:+.1f}%</td><td>{y1[y] * 100:+.1f}%</td></tr>" for y in y0)
               + f"<tr><td>2022 ~ 누적</td><td>×{v['mult_2022_on']['V0']:.2f}</td><td>×{v['mult_2022_on']['V1']:.2f}</td></tr>")
    tr_rows = "".join(f"<tr><td>{d} 00:00 UTC 만{' (= 현행)' if d == '월' else ''}</td><td>{x['sharpe']:.2f}</td><td>{x['mdd'] * 100:.0f}%</td></tr>"
                      for d, x in tr["single"].items())
    tr_rows += (f"<tr><td>한 요일만 쓸 때 평균</td><td>{tr['single_mean']['sharpe']:.2f}</td><td>{tr['single_mean']['mdd'] * 100:.0f}%</td></tr>"
                f"<tr><td><b>7개로 나눠 매일 1/7 씩</b></td><td><b>{tr['combined']['sharpe']:.2f}</b></td><td><b>{tr['combined']['mdd'] * 100:.0f}%</b></td></tr>")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>체결 시각 지도</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--bg:#f5f5f3;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--line:#e4e3df;--grid:#e1e0d9;--base:#c3c2b7;--c1:#2a78d6;--c2:#eb6834;--acc:#2a78d6;--wk:#efeee9}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;
--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;
--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926;--acc:#3987e5;--wk:#222220}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--bg:#121211;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--muted:#898781;--line:#34332f;--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926;--acc:#3987e5;--wk:#222220}}
*{{box-sizing:border-box}}body{{margin:0}}
.viz-root{{background:var(--bg);color:var(--text-primary);font:14px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;min-height:100vh}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 56px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:30px 0 10px}}
.sub{{color:var(--text-secondary);margin-bottom:16px}}
.box{{background:var(--surface-1);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 18px}}
.box ul{{padding-left:18px;margin:0}}.box li{{margin:6px 0}}
.card{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}}
.tbl{{background:var(--surface-1);border:1px solid var(--line);border-radius:8px;overflow-x:auto;margin:10px 0}}
.two{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}}.two>div{{min-width:0}}@media (max-width:700px){{.two{{grid-template-columns:minmax(0,1fr)}}}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:var(--text-secondary);font-weight:600}}
p{{color:var(--text-secondary)}}.note{{font-size:13px}}
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .val{{fill:var(--text-primary);font-size:11px}}
svg .dl{{fill:var(--text-primary);font-size:12px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .c1{{stroke:var(--c1);stroke-width:2}}
svg .refl{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
svg .mk2{{fill:var(--c2);stroke:var(--surface-1);stroke-width:2}}svg .mk1{{fill:var(--c1);stroke:var(--surface-1);stroke-width:2}}
svg .hollow{{fill:var(--surface-1);stroke:var(--text-primary);stroke-width:2}}
svg .wkend{{fill:var(--wk)}}svg .wkend2{{fill:var(--wk);stroke:var(--line)}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.7}}
</style></head><body><div class="viz-root"><main>
<h1>체결 시각 지도 — 금 장마감에 체결하면 더 나은가</h1>
<div class="sub">47심볼 · 규칙은 현행 그대로(28일 부호 · 동일가중 롱·숏 · 1주 보유 · 6bps), 신호·체결 시각만 주 168개 시각으로 돌렸다 ·
시각마다 269주 · 사전등록 없는 서술 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>답: 이 표본에선 조금 높았지만(샤프 {v1:.2f} vs 현행 {v0:.2f}) 믿을 만한 차이가 아니다.</b> 주당 {v['d_bps']:+.0f}bps, t {v['t']:+.2f}
    (95% 구간 {v['ci_bps'][0]:+.0f} ~ {v['ci_bps'][1]:+.0f}bps). 이긴 주 {v['win_share'] * 100:.0f}%. 이 크기의 차이를 t 2 로 확인하려면 <b>약 {v['years_for_t2']:.0f}년치</b> 데이터가 필요하다.</li>
<li><b>2021년 한 해가 거의 전부다.</b> 2021 {y1['2021'] * 100:+.0f}% vs {y0['2021'] * 100:+.0f}%. 2021 을 빼면 주당 {v['ex2021_d_bps']:+.0f}bps (t {v['ex2021_t']:+.2f}) —
    2022 년부터 누적은 현행 ×{v['mult_2022_on']['V0']:.2f} vs 금 장마감 ×{v['mult_2022_on']['V1']:.2f}. 후반 샤프도 {h1[1]:.2f} &lt; {h0[1]:.2f}.</li>
<li><b>시각을 한두 시간만 옮겨도 이만큼 변한다.</b> 168개 시각의 샤프 평균 {s['mean']:.2f}, 10 ~ 90% 가 {s['q10']:.2f} ~ {s['q90']:.2f}.
    1시간 옮기면 평균 {s['shift_abs']['1']:.2f}, 4시간 {s['shift_abs']['4']:.2f} 달라진다. 금 15:00 {fz['금 15:00']:.2f} → 16:00 {fz['금 16:00']:.2f} → 18:00 {fz['금 18:00']:.2f}.
    168개 중 {K['v1_rank']['higher']}개 시각이 금 16:00 보다 높다 — '장마감' 이 특별한 시각이라는 흔적은 없다.</li>
<li>(사후 관찰) 요일 수준에선 수 ~ 금 시각이 토 ~ 월 시각보다 좋았던 경향이 있다: 주당 {mw['d_bps']:+.0f}bps (t {mw['t']:+.2f}), 두 반기 모두 양수.
    결과를 보고 찾은 패턴이라 <b>검증할 후보이지 근거가 아니다.</b></li>
<li><b>시각을 고르지 않는 방법 — 나눠 리밸런스.</b> 자본을 7개로 나눠 요일마다 1/7 씩 리밸런스하면 이 표본에서 샤프 {tr['combined']['sharpe']:.2f} ·
    최대낙폭 {tr['combined']['mdd'] * 100:.0f}% (한 요일만 쓸 때 평균 {tr['single_mean']['sharpe']:.2f} · {tr['single_mean']['mdd'] * 100:.0f}%). 기대수익은 같고
    '어느 시각에 걸렸나' 의 운만 줄인다. 쓰려면 운용 명세에 넣어 사전등록한다.</li>
</ul></div>

<h2>1. 주 168개 시각의 샤프 (ET)</h2>
<div class="card">{landscape(K)}</div>
<p>세로 띠에 마우스를 올리면 그 시각의 샤프 · 반기 · 연복리 · 최대낙폭이 나온다. 이웃 시각끼리는 포지션이 거의 같아 곡선이 부드럽지만,
하루 단위로는 {s['min']:.2f} ~ {s['max']:.2f} 사이를 오간다. 현행(월 00:00 UTC)은 ET 로 일 19시(겨울) · 20시(여름)라 격자 위에 정확히 있지 않아 따로 표시했다.
금 16:00 ET = 한국 토 05:00(서머타임) · 06:00, 현행 = 한국 월 09:00.</p>

<div class="two">
<div><h2>2. 금 장마감 vs 현행</h2>
<div class="tbl"><table><thead><tr><th>269주</th><th>현행</th><th>금 장마감</th></tr></thead><tbody>{vs_rows}</tbody></table></div></div>
<div><h2>3. 나눠 리밸런스 (서술)</h2>
<div class="tbl"><table><thead><tr><th>방식 ({tr['weeks']}주)</th><th>샤프</th><th>최대낙폭</th></tr></thead><tbody>{tr_rows}</tbody></table></div></div>
</div>
<p class="note">현행 = 월 00:00 UTC, 금 장마감 = 금 16:00 ET, 전반 = 2021-05 ~ 2023-10. 나눠 리밸런스: 요일별 묶음의 일 손익을 달력 주(월 00:00 UTC ~)로 합쳤다. 일곱 묶음이 모두 돌아가는 2021-05-10 주부터라 현행(월 묶음)이
여기서는 {tr['single']['월']['sharpe']:.2f} 다 — 첫 주(2021-05-03, +14.7%) 하나가 빠진 것만으로 0.68 → {tr['single']['월']['sharpe']:.2f}. 요일 묶음끼리 주간 수익 상관은 평균
{tr['corr_mean']:.2f} 로 높아서 줄어드는 폭이 크진 않다. 합친 포지션은 '지난 7일 동안 매일 낸 신호의 평균' 을 들고 있는 것과 같다.</p>

<h2>4. 어떻게 쟀나</h2>
<p class="note">5분봉에서 매 정시의 가격(그 시각 직전에 끝난 5분봉 종가) 표를 만들고, 각 시각에서 28일 전 같은 시각과 비교해 부호를 내고 같은 시각에 체결,
다음 주 같은 시각까지 보유했다. 시각마다 2021-04-30 00:00 UTC 이후 첫 269주. 검증: 월 00:00 UTC 시각 = 현행 V0, 금 16:00 ET 시각 = 지난 연구의 V1 과
269주 주간 수익이 정확히 같다(스크립트 안 assert). 일 02:00 ET 는 봄 서머타임 전환 날 없는 시각이라 주가 적다. 전반 · 후반 시각별 샤프의 상관은
{mw['halves_corr']:.2f}.</p>
</main></div></body></html>"""
    Path("out/tsmom_anchor.html").write_text(html, encoding="utf-8")
    print("out/tsmom_anchor.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
