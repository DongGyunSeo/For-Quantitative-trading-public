"""금 신호 · 월 시가 · 주말갭 지정가 — HTML 보고서. 입력 runs/tsmom_gap.json · runs/tsmom_gap_weekly.csv → out/tsmom_gapentry.html"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

SER = (("V0", "현행 V0 · 일요일 UTC", "ink"), ("V1", "V1 금 종가 즉시 (서술)", "v1l"),
       ("V2", "V2 금 신호 → 월 시가", "c1"), ("V3m", "V3m 갭 중간 지정가", "c2"), ("V3e", "V3e 갭 끝 지정가", "c3"))


def svg(W, H, aria, minw=600, maxw=None):
    mx = f"max-width:{maxw}px;margin:0 auto;" if maxw else ""
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;{mx}height:auto;display:block;font-family:inherit">')


def vbar(x, w, y0, y1, r=4):
    """세로 막대 — 기준선(y0)에 붙고 데이터 끝(y1)만 둥글게."""
    h = abs(y0 - y1)
    r = min(r, h / 2, w / 2)
    if y1 <= y0:   # 위로
        return (f'M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} '
                f'Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z')
    return (f'M{x:.1f},{y0:.1f} V{y1 - r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} '
            f'Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 - r:.1f} V{y0:.1f} Z')


def hbar(x0, x1, y, h, r=4):
    """가로 막대 — 0(x0)에 붙고 데이터 끝(x1)만 둥글게."""
    w = abs(x1 - x0)
    r = min(r, w / 2, h / 2)
    if x1 >= x0:
        return (f'M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
                f'Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z')
    return (f'M{x0:.1f},{y:.1f} H{x1 + r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
            f'Q{x1:.1f},{y + h:.1f} {x1 + r:.1f},{y + h:.1f} H{x0:.1f} Z')


def equity_chart(w: pd.DataFrame) -> str:
    W, H, l, r, t, b = 840, 420, 58, 200, 58, 40
    eq = (1 + w[[k for k, _, _ in SER]]).cumprod()
    lo, hi = np.log(eq.min().min() * 0.9), np.log(eq.max().max() * 1.1)
    xs = np.linspace(l, W - r, len(eq))
    sy = lambda v: t + (H - t - b) * (hi - np.log(v)) / (hi - lo)
    out = [svg(W, H, "누적 자산(로그 축) — 현행 V0 · V1 · V2 · V3m · V3e")]
    for row, items in enumerate((SER[:3], SER[3:])):
        lx = l
        for key, name, cls in items:
            y = 14 + row * 18
            out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{y}" y2="{y}" class="{cls}"/>'
                       f'<text x="{lx + 24}" y="{y + 4}" class="lg">{escape(name)}</text>')
            lx += 44 + 10.8 * len(name)
    for v in (0.25, 0.5, 1, 2, 4):
        if not (lo <= np.log(v) <= hi):
            continue
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
        out.append(f'<text x="{W - r + 8}" y="{yv + 4:.1f}" class="dl">{escape(n.split(" ")[0])} {eq[k].iloc[-1]:.2f}배</text>')
    bw = (W - l - r) / len(eq)
    for i, ts in enumerate(eq.index):
        tip = f"{ts.date()} 주 · " + " · ".join(f"{n.split(' ')[0]} {eq[k].iloc[i]:.2f}배" for k, n, _ in SER)
        out.append(f'<rect x="{xs[i] - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def attr_chart(A: dict) -> str:
    rows = []
    for v, head in (("V3m", "갭 중간 지정가 (V3m)"), ("V3e", "갭 끝 지정가 (V3e)")):
        a = A[v]["by_act"]
        rows.append(("head", head, None, None, None))
        rows.append(("b1", f"지정가 체결 이득 · {a['limit']['n']:,}건", a["limit"]["contrib_bps_wk"],
                     f"체결된 코인·주 평균 {a['limit']['mean_diff_bps']:+,.0f}bps (월 시가 대비, 뒤집기 2단위)", None))
        rows.append(("b2", f"패스 손실 · {a['skip']['n']:,}건", a["skip"]["contrib_bps_wk"],
                     f"패스한 코인·주 평균 {a['skip']['mean_diff_bps']:+,.0f}bps (옛 포지션을 들고 달아난 주를 맞음)", None))
        rows.append(("bm", "합계 (V3 − V2)", A[v]["total_bps_wk"], "주당 평균 차이 (장 시가·보유 칸의 기여는 ±0.3bps 이하)", None))
    W, l, r, t, rh = 680, 226, 56, 34, 30
    H = t + rh * len(rows) + 30
    lo, hi = -100, 100
    sx = lambda v: l + (W - l - r) * (v - lo) / (hi - lo)
    out = [svg(W, H, "V3 − V2 주간 차이의 출처 — 지정가 체결 이득과 패스 손실 (bps/주)", 560, 820)]
    for v in range(-100, 101, 50):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t - 8}" y2="{H - 30}" class="{cls}"/>'
                   f'<text x="{sx(v):.1f}" y="{t - 14}" class="tick" text-anchor="middle">{v:+d}</text>'.replace(">+0<", ">0<"))
    y = t
    for kind, lab, val, tip, _ in rows:
        if kind == "head":
            out.append(f'<text x="8" y="{y + 19}" class="hd">{escape(lab)}</text>')
            y += rh
            continue
        x0, x1 = sx(0), sx(val)
        out.append(f'<g><title>{escape(lab)} — {val:+.1f}bps/주 · {tip}</title>'
                   f'<rect x="0" y="{y}" width="{W}" height="{rh}" class="hit"/>'
                   f'<text x="22" y="{y + 19}" class="lg">{escape(lab)}</text>'
                   f'<path d="{hbar(x0, x1, y + 6, rh - 12)}" class="{kind}"/>'
                   f'<text x="{x1 + (6 if val >= 0 else -6):.1f}" y="{y + 19}" class="val" '
                   f'text-anchor="{"start" if val >= 0 else "end"}">{val:+.1f}</text></g>')
        y += rh
    out.append(f'<text x="{sx(0):.1f}" y="{H - 8}" class="tick" text-anchor="middle">V2(월 시가 시장가) 대비 주간 순수익 차이에 대한 기여 (bps/주)</text>')
    out.append("</svg>")
    return "".join(out)


def adverse_chart(A: dict) -> str:
    groups = (("mid", "갭 중간에 지정가"), ("edge", "갭 끝(금 종가)에 지정가"))
    W, H, l, r, t, b = 640, 330, 64, 20, 50, 60
    lo, hi = -150, 1400
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    gw = (W - l - r) / 2
    bw = 70
    out = [svg(W, H, "역선택 — 유리한 갭에서 지정가가 채워진 주와 안 채워진 주의 새 방향 수익", 500, 760)]
    for v in (0, 250, 500, 750, 1000, 1250):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{"0" if v == 0 else format(v, "+,d")}</text>')
    ref = A["fav_nd_bps"]
    out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" class="refl"/>'
               f'<rect x="{l}" y="8" width="12" height="12" rx="2" class="b1"/><text x="{l + 18}" y="18" class="lg">채워진 주</text>'
               f'<rect x="{l + 96}" y="8" width="12" height="12" rx="2" class="b2"/><text x="{l + 114}" y="18" class="lg">안 채워진 주 (패스)</text>'
               f'<line x1="{l + 250}" x2="{l + 268}" y1="14" y2="14" class="refl"/>'
               f'<text x="{l + 274}" y="18" class="lg">유리한 갭 전체 평균 {ref:+.0f}bps</text>')
    for gi, (k, name) in enumerate(groups):
        a = A[k]
        cx = l + gi * gw + gw / 2
        for j, (key, cls, lab) in enumerate((("filled", "b1", "채워진 주"), ("unfilled", "b2", "안 채워진 주"))):
            v, tt, n = a[f"{key}_nd_bps"], a[f"{key}_nd_t"], a[f"n_{key}"]
            x = cx - bw - 6 + j * (bw + 12)
            y1 = sy(v)
            tip = f"{name} · {lab}: 월 시가 → 다음 월 시가 새 방향 수익 평균 {v:+,.0f}bps (t {tt:+.2f}, {n:,}건)"
            out.append(f'<g><title>{escape(tip)}</title><rect x="{x - 4:.1f}" y="{t}" width="{bw + 8}" height="{H - t - b}" class="hit"/>'
                       f'<path d="{vbar(x, bw, sy(0), y1)}" class="{cls}"/>'
                       f'<text x="{x + bw / 2:.1f}" y="{(y1 - 6) if v >= 0 else (y1 + 14):.1f}" class="val" text-anchor="middle">{v:+,.0f}</text>'
                       f'<text x="{x + bw / 2:.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{n:,}건</text></g>')
        out.append(f'<text x="{cx:.1f}" y="{H - b + 34}" class="lg" text-anchor="middle">{escape(name)} · 체결률 {a["fill_rate"] * 100:.0f}%</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 6}" class="tick" text-anchor="middle">V2 경로에서 포지션이 바뀌고 갭이 새 방향에 유리했던 코인·주 · 세로 = 새 방향 주간 수익(bps)</text>')
    out.append("</svg>")
    return "".join(out)


def freq_chart(Fq: dict) -> str:
    ks = ["1", "2", "3", "7", "14"]
    W, H, l, r, t, b = 640, 320, 56, 20, 46, 50
    lo, hi = 0.0, 1.1
    sy = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    gw = (W - l - r) / len(ks)
    out = [svg(W, H, "리밸런스 주기별 샤프 — 시작일·요일 평균과 범위, 수수료 전", 500, 760)]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        cls = "base" if v == 0 else "grid"
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{cls}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    out.append(f'<circle cx="{l + 6}" cy="14" r="5" class="dot"/><text x="{l + 16}" y="18" class="lg">수수료 후 (시작일·요일 평균)</text>'
               f'<line x1="{l + 196}" x2="{l + 196}" y1="6" y2="22" class="wh"/><text x="{l + 204}" y="18" class="lg">시작일·요일에 따른 범위</text>'
               f'<circle cx="{l + 372}" cy="14" r="5" class="hollow"/><text x="{l + 382}" y="18" class="lg">수수료 전</text>')
    for i, k in enumerate(ks):
        f = Fq[k]
        cx = l + i * gw + gw / 2
        m, mn, mx, g = f["sharpe_mean"], f["sharpe_min"], f["sharpe_max"], f["gross_sharpe_mean"]
        tip = (f"{k}일마다 리밸런스 · 샤프 {m:.2f} (범위 {mn:.2f} ~ {mx:.2f}, 시작일 {len(f['phases'])}가지) · 수수료 전 {g:.2f} · "
               f"회전 연 {f['turn_yr']:.0f}회 · 수수료 연 {f['fee_pct_yr']:.2f}% · 연복리 {f['cagr_mean'] * 100:+.0f}% · 최대낙폭 {f['mdd_mean'] * 100:.0f}%")
        g_el = [f'<g><title>{escape(tip)}</title><rect x="{cx - gw / 2:.1f}" y="{t}" width="{gw:.1f}" height="{H - t - b}" class="hit"/>']
        if mx > mn:
            g_el.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{sy(mx):.1f}" y2="{sy(mn):.1f}" class="wh"/>'
                        f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{sy(mx):.1f}" y2="{sy(mx):.1f}" class="wh"/>'
                        f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{sy(mn):.1f}" y2="{sy(mn):.1f}" class="wh"/>'
                        f'<text x="{cx + 26:.1f}" y="{sy(mx) + 4:.1f}" class="tick">{mx:.2f}</text>'
                        f'<text x="{cx + 26:.1f}" y="{sy(mn) + 4:.1f}" class="tick">{mn:.2f}</text>')
        g_el.append(f'<circle cx="{cx + 14:.1f}" cy="{sy(g):.1f}" r="5" class="hollow"/>'
                    f'<circle cx="{cx:.1f}" cy="{sy(m):.1f}" r="5.5" class="dot"/>'
                    f'<text x="{cx - 10:.1f}" y="{sy(m) + 4:.1f}" class="val" text-anchor="end">{m:.2f}</text>'
                    f'<text x="{cx:.1f}" y="{H - b + 18}" class="lg" text-anchor="middle">{k}일</text>'
                    f'<text x="{cx:.1f}" y="{H - b + 32}" class="tick" text-anchor="middle">수수료 연 {f["fee_pct_yr"]:.1f}%</text></g>')
        out.append("".join(g_el))
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    K = json.loads(Path("runs/tsmom_gap.json").read_text(encoding="utf-8"))
    w = pd.read_csv("runs/tsmom_gap_weekly.csv", index_col=0, parse_dates=True)
    P, Cm, M, A, Wk, Fq, Fe = K["perf"], K["comp"], K["mode"], K["adverse"], K["weekend"], K["freq"], K["fees"]
    DG = K["delay_grid"]
    dnames = "월화수목금토일"
    grid_rows = "".join(f"<tr><td>{dnames[int(a)]}</td>" + "".join(f"<td>{x:.2f}</td>" for x in DG['sharpe'][a]) + "</tr>"
                        for a in sorted(DG['sharpe'], key=int))
    grid_rows += "<tr><td><b>7요일 평균</b></td>" + "".join(f"<td><b>{x:.2f}</b></td>" for x in DG['mean_sharpe']) + "</tr>"
    grid_rows += "<tr><td>즉시 대비 주간 차이 평균</td>" + "".join(f"<td>{x:+.1f}bps</td>" for x in DG['mean_diff_bps']) + "</tr>"
    P["V1"] = K["V1"]["perf"]
    v1c, v21 = K["V1"]["vs_V0"], K["V1"]["V2_vs_V1"]
    at = K["attr"]
    wk0, wk1, nwk = K["weeks"]

    def ok(b):
        return "○" if b else "✕"

    comp_rows = []
    for key, name, a, bb in (("V2_vs_V0", "① V2 금 신호 → 월 시가 vs 현행 V0", "V2", "V0"),
                             ("V3m_vs_V2", "② V3m 갭 중간 지정가 vs V2", "V3m", "V2"),
                             ("V3e_vs_V2", "③ V3e 갭 끝 지정가 vs V2", "V3e", "V2")):
        c = Cm[key]
        v = c["verdict"]
        comp_rows.append(
            f"<tr><td>{escape(name)}</td><td>{c['d_mean_bps']:+.1f} (t {c['d_t_boot']:+.2f})</td>"
            f"<td>{c['d_ci_bps'][0]:+.0f} ~ {c['d_ci_bps'][1]:+.0f}</td><td>{c['win_share'] * 100:.0f}%</td>"
            f"<td>{P[a]['first']['sharpe']:.2f} vs {P[bb]['first']['sharpe']:.2f}</td>"
            f"<td>{P[a]['second']['sharpe']:.2f} vs {P[bb]['second']['sharpe']:.2f}</td>"
            f"<td>{P[a]['all']['mdd'] * 100:.0f}% vs {P[bb]['all']['mdd'] * 100:.0f}%</td>"
            f"<td>{ok(v['c1'])} {ok(v['c2'])} {ok(v['c3'])}</td><td><b>{'통과' if v['passed'] else '기각'}</b></td></tr>")
    perf_rows = []
    for k, name, _ in SER:
        p = P[k]
        perf_rows.append(f"<tr><td>{escape(name)}</td><td>{p['all']['sharpe']:.2f}</td><td>{p['all']['t']:+.2f}</td>"
                         f"<td>{p['first']['sharpe']:.2f}</td><td>{p['second']['sharpe']:.2f}</td>"
                         f"<td>{p['all']['cagr'] * 100:+.1f}%</td><td>{p['all']['mdd'] * 100:.0f}%</td>"
                         f"<td>{Fe[k]['gross_pct_yr']:+.1f}%</td><td>{Fe[k]['fee_pct_yr']:.2f}%</td></tr>")
    mode_rows = []
    for v, name in (("V3m", "갭 중간 (V3m)"), ("V3e", "갭 끝 (V3e)")):
        m = M[v]
        pen = K["pen"][v]
        mode_rows.append(f"<tr><td>{name}</td><td>{m['per_week']['changes']:.1f}</td><td>{m['share']['limit'] * 100:.0f}%</td>"
                         f"<td>{m['share']['market'] * 100:.0f}%</td><td>{m['share']['skip'] * 100:.0f}%</td>"
                         f"<td>{m['fill_rate'] * 100:.1f}%</td><td>{m['improve_bps']:.0f}</td><td>{m['fill_hours_median']:.1f}시간</td>"
                         f"<td>{m['fill_within_1h'] * 100:.0f}%</td><td>{m['fill_within_24h'] * 100:.0f}%</td>"
                         f"<td>{pen['fill_rate'] * 100:.1f}% · {pen['d_mean_bps']:+.1f}bps</td></tr>")
    freq_rows = "".join(
        f"<tr><td>{k}일</td><td>{f['sharpe_mean']:.2f}</td><td>{f['sharpe_min']:.2f} ~ {f['sharpe_max']:.2f}</td>"
        f"<td>{f['gross_sharpe_mean']:.2f}</td><td>{f['cagr_mean'] * 100:+.0f}%</td><td>{f['mdd_mean'] * 100:.0f}%</td>"
        f"<td>{f['turn_yr']:.0f}</td><td>{f['fee_pct_yr']:.2f}%</td></tr>" for k, f in Fq.items())
    yrs = K["years"]
    cols = [k for k, _, _ in SER]
    yrow = "".join(f"<tr><td>{y}</td>" + "".join(f"<td>{yrs[c][y] * 100:+.1f}%</td>" for c in cols) + "</tr>"
                   for y in yrs["V0"])
    k7 = Fq["7"]
    dows = "월화수목금토일"
    k7s = " · ".join(f"{dows[x['dow']]} {x['sharpe']:.2f}" for x in sorted(k7["phases"], key=lambda z: z["dow"]))
    vo = K["v0_old_vs_trunc"]
    am, ae = A["mid"], A["edge"]
    mm, me = M["V3m"], M["V3e"]
    sm, se = at["V3m"]["by_act"], at["V3e"]["by_act"]

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>금 신호 · 주말갭 지정가</title>
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
p{{color:var(--text-secondary)}}.note{{font-size:13px}}
svg .tick,svg .lg{{fill:var(--text-secondary);font-size:11px}}svg .val{{fill:var(--text-primary);font-size:11px}}
svg .dl{{fill:var(--text-primary);font-size:12px}}svg .hd{{fill:var(--text-primary);font-size:12px;font-weight:600}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .base{{stroke:var(--base);stroke-width:1.5}}
svg .ink{{stroke:var(--text-primary);stroke-width:2}}svg .c1{{stroke:var(--c1);stroke-width:2}}
svg .c2{{stroke:var(--c2);stroke-width:2}}svg .c3{{stroke:var(--c3);stroke-width:2}}
svg .v1l{{stroke:var(--muted);stroke-width:2;stroke-dasharray:6 4}}
svg .b1{{fill:var(--c1)}}svg .b2{{fill:var(--c2)}}svg .bm{{fill:var(--muted)}}
svg .refl{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
svg .dot{{fill:var(--c1);stroke:var(--surface-1);stroke-width:2}}svg .hollow{{fill:var(--surface-1);stroke:var(--text-secondary);stroke-width:2}}
svg .wh{{stroke:var(--c1);stroke-width:2}}
svg .hit{{fill:transparent}}svg g:hover .hit{{fill:var(--grid);fill-opacity:.45}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.6}}
</style></head><body><div class="viz-root"><main>
<h1>시계열 추세 — 금 종가 신호 · 월 시가 · 주말갭 지정가</h1>
<div class="sub">47심볼 · 공통 {nwk}주 ({wk0} ~ {wk1} 월요일) · 28일 부호 · 동일가중 롱·숏 · 시장가 6bps · 지정가 2bps · 사전등록 판정 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>판정: 셋 다 기각.</b> ① 금 신호 → 월 시가(V2) 샤프 {P['V2']['all']['sharpe']:.2f} vs 현행 {P['V0']['all']['sharpe']:.2f} —
    주당 {Cm['V2_vs_V0']['d_mean_bps']:+.0f}bps (t {Cm['V2_vs_V0']['d_t_boot']:+.2f}). ② 갭 중간 지정가(V3m)는 V2 보다 주당 {Cm['V3m_vs_V2']['d_mean_bps']:+.0f}bps
    (t {Cm['V3m_vs_V2']['d_t_boot']:+.2f}) — 방향은 맞지만 기준 t 2.4 에 못 미치고, 현행보다는 여전히 주당 {abs(Cm['V3m_vs_V0']['d_mean_bps']):.0f}bps 낮다.
    ③ 갭 끝 지정가(V3e)는 {Cm['V3e_vs_V2']['d_mean_bps']:+.0f}bps (t {Cm['V3e_vs_V2']['d_t_boot']:+.2f}).</li>
<li><b>월 시가까지 기다리는 것이 손해였다.</b> 금요일에 신호가 바뀐 코인은 주말(금 16:00 → 월 09:30 ET) 동안 새 방향으로 평균
    <b>{Wk['changed_bps']:+.0f}bps</b> (t {Wk['changed_t']:+.2f}) 더 간다 — 안 바뀐 코인은 {Wk['unchanged_bps']:+.0f}bps. 바뀐 것의 {Wk['flip_share'] * 100:.0f}% 가
    뒤집기라서, 기다리는 동안 그 움직임을 <b>반대 포지션</b>으로 맞는다 → 주당 약 {Wk['cost_bps_wk']:.0f}bps.
    (서술: 금 종가에 바로 체결하는 V1 은 샤프 {P['V1']['all']['sharpe']:.2f}, 현행과 차이 {v1c['d_mean_bps']:+.0f}bps/주 · t {v1c['d_t_boot']:+.2f} — 현행과 구별되지 않는다.)
    다만 요일 × 지연 격자(§7)로 보면 지연은 평균적으로 <b>약간</b> 손해(즉시 {DG['mean_sharpe'][0]:.2f} → 2일 {DG['mean_sharpe'][2]:.2f} → 3일 {DG['mean_sharpe'][3]:.2f})이고
    요일마다 크게 흔들린다 — V2 의 손해는 그 분포의 나쁜 쪽 끝이다.</li>
<li><b>지정가는 싸게 사지만, 안 채워지는 주가 하필 가장 좋은 주다(역선택).</b> 갭 중간은 {mm['fill_rate'] * 100:.0f}% 체결 · 평균 {mm['improve_bps']:.0f}bps 싸게,
    갭 끝은 {me['fill_rate'] * 100:.0f}% · {me['improve_bps']:.0f}bps. 그런데 안 채워진 주는 새 방향으로 <b>{am['unfilled_nd_bps']:+,.0f}bps</b>(중간) ·
    {ae['unfilled_nd_bps']:+,.0f}bps(끝) 달아난 주였다 (채워진 주 {am['filled_nd_bps']:+,.0f} · {ae['filled_nd_bps']:+,.0f}bps). 패스하면 그 주를 옛(반대) 포지션으로 들고 있어
    패스 1건이 평균 {sm['skip']['mean_diff_bps']:+,.0f}bps(중간) · {se['skip']['mean_diff_bps']:+,.0f}bps(끝). 합치면 중간 {sm['limit']['contrib_bps_wk']:+.0f} {sm['skip']['contrib_bps_wk']:+.0f} = {at['V3m']['total_bps_wk']:+.0f},
    끝 {se['limit']['contrib_bps_wk']:+.0f} {se['skip']['contrib_bps_wk']:+.0f} = {at['V3e']['total_bps_wk']:+.0f} bps/주.</li>
<li><b>왜 꼭 1주?</b> 일 종가로 1 · 2 · 3 · 7 · 14일마다 리밸런스하면 샤프 {Fq['1']['sharpe_mean']:.2f} · {Fq['2']['sharpe_mean']:.2f} · {Fq['3']['sharpe_mean']:.2f} ·
    {Fq['7']['sharpe_mean']:.2f} · {Fq['14']['sharpe_mean']:.2f}. 수수료 전에도 {Fq['1']['gross_sharpe_mean']:.2f} · {Fq['2']['gross_sharpe_mean']:.2f} · {Fq['3']['gross_sharpe_mean']:.2f} ·
    {Fq['7']['gross_sharpe_mean']:.2f} · {Fq['14']['gross_sharpe_mean']:.2f} 라서, 짧은 주기의 손해는 수수료보다 <b>신호 자체</b>에서 온다 —
    매일 따라가면 연 {Fq['1']['turn_yr']:.0f}회(7일은 {Fq['7']['turn_yr']:.0f}회) 포지션이 바뀌는데, 늘어난 뒤집기가 수수료 전에도 손해였다.
    그러나 7일 안에서도 요일에 따라 {Fq['7']['sharpe_min']:.2f} ~ {Fq['7']['sharpe_max']:.2f} — 요일 선택의 잡음이 주기 차이만큼 크다. 7일은 '증명된 최적' 이 아니라 '잡음 속 무난한 선택' 이다.</li>
<li><b>뜻:</b> 이 신호의 값은 <b>신호가 바뀐 직후</b>에 몰리는 경향이 있다(뒤집힌 직후 1주 +172bps · 금 뒤집기 뒤 주말 {Wk['changed_bps']:+.0f}bps ·
    지연 격자 평균 하락). 뒤집기 보류 · 월 시가 대기 · 갭 지정가 대기 — <b>기다리는 규칙은 모두 졌다.</b> 체결은 신호 확정 직후 시장가가 기본이다.</li>
</ul></div>

<h2>1. 누적 자산 (로그 축)</h2>
<div class="card">{equity_chart(w)}</div>
<p>세로 띠에 마우스를 올리면 그 주의 다섯 곡선 값이 나온다. 점선 V1 은 사전등록 밖 서술(금 종가에 바로 체결) — V2 와 신호가 같고 체결 시각만 다르다.
V1 과 V2 의 간격이 '주말을 기다리는 비용' 이다.</p>

<h2>2. 판정 표</h2>
<div class="tbl"><table><thead><tr><th>비교</th><th>주간 차이 bps (t)</th><th>95% 구간</th><th>이긴 주</th><th>전반 샤프</th><th>후반 샤프</th>
<th>최대낙폭</th><th>①②③</th><th>판정</th></tr></thead><tbody>{''.join(comp_rows)}</tbody></table></div>
<p class="note">사전등록 기준: ① 주간 차이 평균 &gt; 0 · 4주 블록 부트스트랩 t ≥ 2.4 (3개 비교 보정) ② 전반(2021-05 ~ 2023-10) · 후반(2023-11 ~ 2026-06) 샤프 모두 비교 대상보다 높음
③ 최대낙폭이 5%p 넘게 나빠지지 않음. ② V3m 은 ②③ 은 통과했지만 ① 에서 떨어졌다.</p>
<div class="tbl"><table><thead><tr><th>규칙</th><th>샤프</th><th>t</th><th>전반</th><th>후반</th><th>연복리</th><th>최대낙폭</th><th>수수료 전 연수익</th><th>수수료 연</th></tr></thead>
<tbody>{''.join(perf_rows)}</tbody></table></div>

<h2>3. V3 − V2 는 어디서 나오나</h2>
<div class="card">{attr_chart(at)}</div>
<p>코인·주마다 V3 − V2 순수익 차이를 V3 의 행동별로 모았다. 체결되면 월 시가보다 싸게 사고(뒤집기는 2단위라 개선이 두 배) 수수료도 메이커다.
패스하면 옛 포지션을 그대로 들고 있는데, 패스하는 주는 가격이 새 방향으로 달아나서 지정가에 안 닿은 주다 — 반대 포지션으로 그 주를 통째로 맞는다.</p>

<h2>4. 역선택 — 채워진 주 vs 안 채워진 주</h2>
<div class="card">{adverse_chart(A)}</div>
<p>V2 경로에서 포지션이 바뀐 코인·주 {A['changes']:,}건 중 갭이 새 방향에 유리했던 {A['fav_share'] * 100:.0f}%. 막대 = 월 시가 → 다음 월 시가의 새 방향 수익.
지정가는 가격이 새 방향과 <b>반대로</b> 돌아올 때만 채워진다 — 채워진다는 것 자체가 '그 주는 덜 간다' 는 신호다. 채워진 주의 지정가 기준 수익은
{am['filled_fromL_bps']:+.0f}bps(중간) · {ae['filled_fromL_bps']:+.0f}bps(끝)로 괜찮지만, 놓친 주의 크기를 메우지 못한다.
(참고: 갭이 불리했던 주의 새 방향 수익 {A['unfav_nd_bps']:+.0f}bps (t {A['unfav_nd_t']:+.2f}) · 유리했던 주 {A['fav_nd_bps']:+.0f}bps (t {A['fav_nd_t']:+.2f}) — 주말에 새 방향으로 간 코인이 그 주에도 더 갔다. 사후 관찰.)</p>

<h2>5. 진입 방식 (V3)</h2>
<div class="tbl"><table><thead><tr><th>지정가 위치</th><th>변화 / 주</th><th>지정가 체결</th><th>장 시가</th><th>패스</th><th>체결률</th>
<th>개선 bps</th><th>체결 중앙</th><th>1시간 내</th><th>24시간 내</th><th>관통 가정 체결률 · 차이</th></tr></thead><tbody>{''.join(mode_rows)}</tbody></table></div>
<p class="note">'변화 / 주' = 포지션을 바꾸려 한 코인 수(47개 중). '장 시가' = 갭이 새 방향에 불리하거나 0 이라 지정가를 걸 수 없어 월 09:30 ET 시가에 시장가.
'체결률' = 지정가를 건 경우 중 그 주 안에 체결된 비율. '개선' = 월 시가 대비 체결가가 좋은 정도. 체결은 사전등록대로 '닿으면 체결'(프로젝트 표준).
지정가를 1bps 넘어서야 체결되는 보수적 가정으로 바꿔도 체결률 · 결과가 거의 같다 — 체결 가정 탓이 아니다.</p>

<h2>6. 왜 꼭 1주인가 — 리밸런스 주기</h2>
<div class="card">{freq_chart(Fq)}</div>
<div class="tbl"><table><thead><tr><th>주기</th><th>샤프 (평균)</th><th>시작일·요일 범위</th><th>수수료 전 샤프</th><th>연복리</th><th>최대낙폭</th><th>회전 / 연</th><th>수수료 연</th></tr></thead>
<tbody>{freq_rows}</tbody></table></div>
<p class="note">일 종가(UTC) · 28일 부호 · 시장가 · k 일마다 리밸런스, 가능한 시작일을 모두 돌려 평균했다. 7일 = 요일별: {k7s} (일요일 = 현행 V0).
샤프는 주기마다 표본 수가 달라 추정 잡음도 다르다 — 서술이지 판정이 아니다.</p>

<h2>7. 요일 × 지연 격자 (서술)</h2>
<div class="tbl"><table><thead><tr><th>신호 요일 (UTC 일봉 종가)</th><th>즉시 체결</th><th>1일 뒤</th><th>2일 뒤</th><th>3일 뒤</th></tr></thead><tbody>{grid_rows}</tbody></table></div>
<p class="note">일 종가 · 28일 부호 · 시장가. 요일 = 그 요일 UTC 일봉 종가(다음 날 00:00 UTC)에 신호, k 일 뒤 종가에 체결해 1주 보유. 일요일 · 즉시 = 현행 V0.
현행 문서의 '하루 · 이틀 늦어도 괜찮다(0.74 · 0.88)' 는 일요일 줄만 본 것이었다 — 7요일 평균으로는 지연이 조금씩 손해다. 칸마다 잡음이 커서 한 칸을 고르면 안 된다.</p>

<h2>8. 연도별 (복리)</h2>
<div class="tbl"><table><thead><tr><th>연도</th>{''.join(f'<th>{escape(n.split(" ")[0])}</th>' for _, n, _ in SER)}</tr></thead><tbody>{yrow}</tbody></table></div>

<h2>9. 참고 — 현행 샤프가 0.64 가 아니라 {P['V0']['all']['sharpe']:.2f} 인 이유</h2>
<p class="note">비교는 V2 가 만들 수 있는 공통 {nwk}주만 쓴다. 6/29 주는 다음 월 시가가 봉인 창이라 V2 가 없다. 현행 V0 에서 그 주는 분석 창 끝
(7/5 00:05 UTC)까지의 6일짜리 부분 주였고 {vo['last_week_ret'] * 100:+.1f}% 였다 — 270주 샤프 {vo['old']:.2f} 가 269주로는 {vo['trunc']:.2f}.
<b>한 주가 샤프를 0.04 움직인다.</b> 이 전략의 샤프 추정이 얼마나 흔들리는지 보여주는 숫자다.</p>
</main></div></body></html>"""
    Path("out/tsmom_gapentry.html").write_text(html, encoding="utf-8")
    print("out/tsmom_gapentry.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
