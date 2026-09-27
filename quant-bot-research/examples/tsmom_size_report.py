"""사이징 · 교차 증거금 레버리지 — HTML. 입력 runs/tsmom_size/{part1,part2}.json · weights.pkl → out/tsmom_sizing.html"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.research import sizing as S  # noqa: E402

D = Path("runs/tsmom_size")


def svg(W, H, aria, minw=600):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
            f'style="width:100%;min-width:{minw}px;height:auto;display:block;font-family:inherit">')


def g_axis(out, sx, W, l, r, H, b, t, gmax):
    for g in np.arange(0, gmax + 1e-9, 0.5):
        out.append(f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="{t}" y2="{H - b}" class="grid"/>'
                   f'<text x="{sx(g):.1f}" y="{H - b + 16}" class="tick" text-anchor="middle">{g:g}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 4}" class="tick" text-anchor="middle">G = 총명목 ÷ 계좌 자산 (배)</text>')


def mdd_chart(K: dict) -> str:
    hist = [x for x in K["hist"] if x["G"] <= 3.0 + 1e-9]
    boot = [x for x in K["boot"] if x["G"] <= 3.0 + 1e-9]
    W, H, l, r, t, b = 860, 360, 52, 20, 58, 40
    gmax = 3.0
    sx = lambda g: l + (W - l - r) * g / gmax
    sy = lambda v: t + (H - t - b) * v
    out = [svg(W, H, "G 별 최대낙폭 — 역사 경로와 5년 부트스트랩 중앙값 · 95번째 백분위")]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{"0" if v == 0 else f"−{v * 100:.0f}%"}</text>')
    g_axis(out, sx, W, l, r, H, b, t, gmax)
    band = " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p50']):.1f}" for x in boot) + " " + \
        " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p95']):.1f}" for x in reversed(boot))
    out.append(f'<polygon points="{band}" class="band"/>')
    for lim in (0.30, 0.50):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(lim):.1f}" y2="{sy(lim):.1f}" class="refl"/>'
                   f'<text x="{W - r - 4}" y="{sy(lim) - 5:.1f}" class="tick" text-anchor="end">한도 −{lim * 100:.0f}%</text>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p95']):.1f}" for x in boot) + '" class="c1" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_p50']):.1f}" for x in boot) + '" class="c1d" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['mdd_intraweek']):.1f}" for x in hist) + '" class="ink" fill="none"/>')
    # 청산 한계
    for gv, lab in ((K["L1"], f"2배 스트레스 청산 한계 {K['L1']:.2f}"), (K["thresholds"]["close_2"]["min"], f"역사 청산 {K['thresholds']['close_2']['min']:.2f}")):
        out.append(f'<line x1="{sx(gv):.1f}" x2="{sx(gv):.1f}" y1="{t}" y2="{H - b}" class="liq"/>'
                   f'<text x="{sx(gv) + 4:.1f}" y="{t + 12}" class="tick">{escape(lab)}</text>')
    for lim in ("0.3", "0.5"):
        g = K["rec"][lim]
        x = [z for z in boot if abs(z["G"] - g) < 1e-9][0]
        out.append(f'<circle cx="{sx(g):.1f}" cy="{sy(x["mdd_p95"]):.1f}" r="5" class="mk2"/>'
                   f'<text x="{sx(g) + 8:.1f}" y="{sy(x["mdd_p95"]) + 16:.1f}" class="dl">G {g:.2f}</text>')
    out.append(f'<line x1="{l}" x2="{l + 18}" y1="14" y2="14" class="ink"/><text x="{l + 24}" y="18" class="lg">역사 경로 (장중 저점 포함)</text>'
               f'<line x1="{l + 190}" x2="{l + 208}" y1="14" y2="14" class="c1"/><text x="{l + 214}" y="18" class="lg">5년 부트스트랩 95번째 백분위 (판정)</text>'
               f'<line x1="{l + 440}" x2="{l + 458}" y1="14" y2="14" class="c1d"/><text x="{l + 464}" y="18" class="lg">부트스트랩 중앙값</text>'
               f'<circle cx="{l + 6}" cy="36" r="5" class="mk2"/><text x="{l + 16}" y="40" class="lg">권장 G (−30% · −50% 기준)</text>')
    bw = (W - l - r) / len(boot)
    for x, hh in zip(boot, hist):
        tip = (f"G {x['G']:.2f} · 역사 최대낙폭 −{hh['mdd_intraweek'] * 100:.0f}% · 부트스트랩 중앙 −{x['mdd_p50'] * 100:.0f}% · "
               f"95% −{x['mdd_p95'] * 100:.0f}% · 청산 확률 {x['p_liq'] * 100:.1f}%")
        out.append(f'<rect x="{sx(x["G"]) - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def cagr_chart(K: dict) -> str:
    hist = [x for x in K["hist"] if x["G"] <= 3.0 + 1e-9 and x["liq_week"] is None]
    boot = [x for x in K["boot"] if x["G"] <= 3.0 + 1e-9]
    b3 = [x for x in K["sharpe03"]["boot"] if x["G"] <= 3.0 + 1e-9]
    W, H, l, r, t, b = 860, 340, 52, 20, 40, 40
    gmax, lo, hi = 3.0, -0.4, 0.35
    sx = lambda g: l + (W - l - r) * g / gmax
    sy = lambda v: t + (H - t - b) * (hi - max(v, lo)) / (hi - lo)
    out = [svg(W, H, "G 별 연복리 — 역사 경로, 부트스트랩 중앙값, 샤프 0.3 시나리오")]
    for v in (-0.4, -0.2, 0, 0.2):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"base" if v == 0 else "grid"}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v * 100:+.0f}%</text>')
    g_axis(out, sx, W, l, r, H, b, t, gmax)
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['cagr']):.1f}" for x in hist) + '" class="ink" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['cagr_p50']):.1f}" for x in boot) + '" class="c1" fill="none"/>')
    out.append('<polyline points="' + " ".join(f"{sx(x['G']):.1f},{sy(x['cagr_p50']):.1f}" for x in b3) + '" class="c2" fill="none"/>')
    go = K["growth_opt"]["boot_p50"]
    y = [x for x in boot if abs(x["G"] - go) < 1e-9][0]["cagr_p50"]
    out.append(f'<circle cx="{sx(go):.1f}" cy="{sy(y):.1f}" r="5" class="mk1"/><text x="{sx(go):.1f}" y="{sy(y) + 22:.1f}" class="dl" text-anchor="middle">성장 최적 G {go:.2f}</text>')
    g3 = K["sharpe03"]["growth_opt"]
    y3 = [x for x in b3 if abs(x["G"] - g3) < 1e-9][0]["cagr_p50"]
    out.append(f'<circle cx="{sx(g3):.1f}" cy="{sy(y3):.1f}" r="5" class="mk2"/><text x="{sx(g3) + 8:.1f}" y="{sy(y3) + 18:.1f}" class="dl">샤프 0.3 이면 {g3:.2f}</text>')
    out.append(f'<line x1="{l}" x2="{l + 18}" y1="14" y2="14" class="ink"/><text x="{l + 24}" y="18" class="lg">역사 경로 (G 2.6 부터 청산)</text>'
               f'<line x1="{l + 200}" x2="{l + 218}" y1="14" y2="14" class="c1"/><text x="{l + 224}" y="18" class="lg">5년 부트스트랩 중앙값</text>'
               f'<line x1="{l + 380}" x2="{l + 398}" y1="14" y2="14" class="c2"/><text x="{l + 404}" y="18" class="lg">샤프가 0.3 이라면 (중앙값)</text>')
    bw = (W - l - r) / len(boot)
    hd = {round(x["G"], 2): x for x in K["hist"]}
    for x, z in zip(boot, b3):
        hh = hd[round(x["G"], 2)]
        hc = "청산" if hh["liq_week"] else f"{hh['cagr'] * 100:+.1f}%"
        tip = (f"G {x['G']:.2f} · 역사 연복리 {hc} · 부트스트랩 중앙 {x['cagr_p50'] * 100:+.1f}% (5% {x['cagr_p05'] * 100:+.0f}%) · "
               f"5년 손실 확률 {x['p_loss'] * 100:.0f}% · 샤프 0.3 이면 {z['cagr_p50'] * 100:+.1f}%")
        out.append(f'<rect x="{sx(x["G"]) - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def equity_chart(eq: pd.DataFrame) -> str:
    SER = (("0.2", "G 0.2", "c3"), ("0.4", "G 0.4", "c1"), ("1.0", "G 1 (현행)", "ink"), ("2.0", "G 2", "c2"))
    W, H, l, r, t, b = 860, 360, 52, 110, 40, 30
    lo, hi = np.log(0.35), np.log(5.0)
    xs = np.linspace(l, W - r, len(eq))
    sy = lambda v: t + (H - t - b) * (hi - np.log(max(v, 0.35))) / (hi - lo)
    out = [svg(W, H, "G 별 누적 자산 (로그 축) — 0.2 · 0.4 · 1 · 2")]
    for v in (0.5, 1, 2, 4):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="{"base" if v == 1 else "grid"}"/>'
                   f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}배</text>')
    for yv in sorted(set(eq.index.year)):
        k = int(np.searchsorted(eq.index.year, yv))
        out.append(f'<line x1="{xs[k]:.1f}" x2="{xs[k]:.1f}" y1="{t}" y2="{H - b}" class="grid"/><text x="{xs[k] + 4:.1f}" y="{H - b + 16}" class="tick">{yv}</text>')
    lx = l
    for key, name, cls in SER:
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="14" y2="14" class="{cls}"/><text x="{lx + 24}" y="18" class="lg">{escape(name)}</text>')
        lx += 44 + 11 * len(name)
        out.append('<polyline points="' + " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, eq[key])) + f'" class="{cls}" fill="none"/>')
    ends = sorted(((sy(eq[k].iloc[-1]), k, n) for k, n, _ in SER))
    last = -1e9
    for yv, k, n in ends:
        yv = max(yv, last + 15)
        last = yv
        out.append(f'<text x="{W - r + 8}" y="{yv + 4:.1f}" class="dl">{escape(n.split(" (")[0])} {eq[k].iloc[-1]:.2f}배</text>')
    bw = (W - l - r) / len(eq)
    for i, ts in enumerate(eq.index):
        tip = f"{ts.date()} 주 · " + " · ".join(f"{n.split(' (')[0]} {eq[k].iloc[i]:.2f}배" for k, n, _ in SER)
        out.append(f'<rect x="{xs[i] - bw / 2:.1f}" y="{t}" width="{bw:.2f}" height="{H - t - b}" class="hitv"><title>{escape(tip)}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    A = json.loads((D / "part1.json").read_text(encoding="utf-8"))
    K = json.loads((D / "part2.json").read_text(encoding="utf-8"))
    Z = pd.read_pickle(D / "weights.pkl")
    wk = Z["W"]["S0"].index
    eq = pd.DataFrame({f"{g:.1f}": (1 + S.book(Z["W"]["S0"], Z["R"], gross=pd.Series(g, index=wk)).ret).cumprod()
                       for g in (0.2, 0.4, 1.0, 2.0)})
    eq.index = eq.index + pd.Timedelta(days=7)
    Pf, v = A["perf"], A["verdict"]
    names = {"S0": "S0 동일 명목 (현행)", "S1": "S1 역변동성 (위험 균등)", "S2": "S2 변동성 비례",
             "S0+VT": "S0 + 변동성 타깃", "S1+VT": "S1 + 변동성 타깃", "S2+VT": "S2 + 변동성 타깃"}
    rows1 = "".join(
        f"<tr><td>{escape(names[k])}</td><td>{Pf[k]['sharpe']:.3f}</td><td>{Pf[k]['first']:.2f}</td><td>{Pf[k]['second']:.2f}</td>"
        f"<td>{A['at30'][k]['cagr'] * 100:+.1f}%</td><td>{A['at30'][k]['mdd'] * 100:.0f}%</td><td>{A['fee_pct_yr'][k]:.2f}%</td>"
        f"<td>{A['G_mean'][k]:.2f}</td></tr>" for k in names)
    vrow = []
    for key, lab in (("S1_vs_S0", "S1 역변동성 vs S0"), ("S2_vs_S0", "S2 변동성 비례 vs S0 (기준 t ≥ 2)"), ("S0+VT_vs_S0", "S0 + 변동성 타깃 vs S0")):
        x = v[key]
        vrow.append(f"<tr><td>{lab}</td><td>{x['boot']['diff']:+.3f}</td><td>{x['boot']['t']:+.2f}</td>"
                    f"<td>{x['boot']['ci'][0]:+.2f} ~ {x['boot']['ci'][1]:+.2f}</td><td><b>{'채택' if x['adopt'] else '기각'}</b></td></tr>")
    hd = {round(x["G"], 2): x for x in K["hist"]}
    bd = {round(x["G"], 2): x for x in K["boot"]}
    gtab = ""
    for g in (0.2, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        h, bb = hd[g], bd[g]
        liq = h["liq_week"] is not None
        c_txt = "청산" if liq else f"{h['cagr'] * 100:+.1f}%"
        m_txt = "100%" if liq else f"{h['mdd_intraweek'] * 100:.0f}%"
        cls = ' class="hl"' if g in (0.2, 0.4) else ""
        gtab += (f"<tr{cls}><td>{g:g}</td><td>{c_txt}</td><td>{m_txt}</td><td>{h['worst_week'] * 100:.0f}%</td>"
                 f"<td>{bb['cagr_p50'] * 100:+.1f}%</td><td>{bb['mdd_p50'] * 100:.0f}%</td><td>{bb['mdd_p95'] * 100:.0f}%</td>"
                 f"<td>{bb['p_loss'] * 100:.0f}%</td><td>{bb['p_liq'] * 100:.0f}%</td></tr>")
    vt = {round(x["target"], 3): x for x in K["vt"]}
    vtab = "".join(f"<tr><td>{t * 100:g}%</td><td>{vt[t]['G_mean']:.2f}</td><td>{vt[t]['G_max']:.2f}</td><td>{vt[t]['hist_cagr'] * 100:+.1f}%</td>"
                   f"<td>{vt[t]['hist_mdd'] * 100:.0f}%</td><td>{vt[t]['cagr_p50'] * 100:+.1f}%</td><td>{vt[t]['mdd_p95'] * 100:.0f}%</td></tr>"
                   for t in (0.1, 0.2, 0.225, 0.3, 0.4))
    r30, r50 = K["rec"]["0.3"], K["rec"]["0.5"]
    ntab = "".join(f"<tr><td>{L}배</td><td>{(1 - r30 / L) * 100:.0f}%</td><td>{(1 - r50 / L) * 100:.0f}%</td><td>{(1 - 1 / L) * 100:.0f}%</td></tr>"
                   for L in (1, 2, 3, 5, 10))
    th = K["thresholds"]
    bs = K["block_sens"]
    hp = K["hist_pctile"]
    s3 = K["sharpe03"]
    ww = K["worst_week"]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>사이징과 레버리지</title>
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
svg .ink{{stroke:var(--text-primary);stroke-width:2}}svg .c1{{stroke:var(--c1);stroke-width:2}}
svg .c1d{{stroke:var(--c1);stroke-width:2;stroke-dasharray:6 4}}svg .c2{{stroke:var(--c2);stroke-width:2}}svg .c3{{stroke:var(--c3);stroke-width:2}}
svg .band{{fill:var(--c1);fill-opacity:.12}}svg .refl{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
svg .liq{{stroke:var(--c2);stroke-width:1.5;stroke-dasharray:3 3}}
svg .mk1{{fill:var(--c1);stroke:var(--surface-1);stroke-width:2}}svg .mk2{{fill:var(--c2);stroke:var(--surface-1);stroke-width:2}}
svg .hitv{{fill:transparent}}svg .hitv:hover{{fill:var(--grid);fill-opacity:.6}}
</style></head><body><div class="viz-root"><main>
<h1>시계열 추세 — 변동성 사이징과 교차 증거금 레버리지</h1>
<div class="sub">47심볼 · 28일 부호 · 월 00:00 UTC 주간 리밸런스 · 269주 (2021-05 ~ 2026-06) · 되맞춤까지 넣은 실제 거래량 수수료 6bps ·
사전등록 판정 · 봉인 창은 열지 않음</div>
<div class="box"><ul>
<li><b>사이징: 변동성으로 코인 비중을 바꿔도 결과가 사실상 같다 — 사전등록 기준으로 셋 다 채택 안 됨.</b>
    위험 균등(역변동성) 샤프 {Pf['S1']['sharpe']:.3f} vs 현행 {Pf['S0']['sharpe']:.3f} (전반 {Pf['S1']['first']:.2f} &lt; {Pf['S0']['first']:.2f}),
    변동성 비례 {Pf['S2']['sharpe']:.3f} (t {v['S2_vs_S0']['boot']['t']:+.2f}, 후반은 더 낮음), 포트폴리오 변동성 타깃 {Pf['S0+VT']['sharpe']:.3f}
    (후반 {Pf['S0+VT']['second']:.2f} &lt; {Pf['S0']['second']:.2f}). 47개가 한 방향 베팅이라 코인 사이 비중은 위험을 거의 못 바꾼다
    (사전 변동성 {A['exante_vol']['S0'] * 100:.0f}% → {A['exante_vol']['S1'] * 100:.0f}%). <b>결과를 정하는 건 비중이 아니라 총노출 G 다.</b></li>
<li><b>강제청산은 늦게 온다.</b> 역사 경로에서 청산이 나는 G = <b>{th['close_2']['min']:.2f}</b> — {th['close_2']['week']} 주, 숏 44개가 폭락 뒤 반등에 물려
    G = 1 기준 장중 {ww['rmin'] * 100:.1f}% (수요일 00:00 UTC). 장중 최악 손실을 2배로 키워도 버티는 한계는 <b>{K['L1']:.2f}</b>
    (롱은 저가·숏은 고가 동시 + 유지증거금률 5% 로 더 보수적으로 잡아도 {K['L1_variants']['cons_5_x2']:.2f}).</li>
<li><b>낙폭은 훨씬 먼저 온다.</b> 과거 주를 4주씩 섞어 만든 5년 경로 10,000개에서 95% 가 낙폭 −30% 안에 드는 G 는 <b>{K['L2']['0.3']:.2f}</b>,
    −50% 안은 <b>{K['L2']['0.5']:.2f}</b>. 더 긴 블록(12 · 26주)으로 섞으면 {bs['12']['L2_30']:.2f} ~ {bs['26']['L2_30']:.2f} · {bs['12']['L2_50']:.2f} ~ {bs['26']['L2_50']:.2f}.
    실제 역사 경로(G 1 에서 −{hp['1.0']['hist_mdd'] * 100:.0f}%)는 섞은 경로들 중 하위 {hp['1.0']['pctile']:.0f}% — 운이 좋은 편이었다.</li>
<li><b>권장 (사전등록 규칙 = 청산 한계와 낙폭 한계 중 작은 쪽):</b> −30% 기준 <b>G {r30:.2f}</b> (역사 연 {hd[0.2]['cagr'] * 100:.1f}% · 최대낙폭 {hd[0.2]['mdd_intraweek'] * 100:.0f}%),
    −50% 기준 <b>G {r50:.2f}</b> (연 {hd[0.4]['cagr'] * 100:.1f}% · {hd[0.4]['mdd_intraweek'] * 100:.0f}%). <b>레버리지가 필요 없는 수준</b>이다 — 이 전략은 G 1 에서 이미
    연 변동성 {Pf['S0']['vol'] * 100:.0f}%. 교차 증거금에선 잉여 n% 와 레버리지 L 의 조합이 청산과 무관하다 (G = L × (1 − n)): G 0.4 = 잉여 60% + 1배 = 잉여 80% + 2배.</li>
<li><b>수익 극대화(성장 최적) G</b>는 표본 샤프 그대로면 약 {K['growth_opt']['boot_p50']:.2f} 이지만 그때 역사 최대낙폭 −{hd[1.15]['mdd_intraweek'] * 100:.0f}%,
    부트스트랩 95% −{bd[1.15]['mdd_p95'] * 100:.0f}%. 샤프가 0.3 이면 성장 최적이 {s3['growth_opt']:.2f} 로 떨어지고 G 1 의 5년 손실 확률이 반을 넘는다 —
    표본 샤프가 유의하지 않아서(t 1.5) 켈리를 그대로 따르면 위험하다.</li>
</ul></div>

<h2>1. G 별 최대낙폭 — 청산보다 낙폭이 먼저 막는다</h2>
<div class="card">{mdd_chart(K)}</div>
<p>세로 띠에 마우스를 올리면 그 G 의 숫자가 나온다. 파란 띠 = 섞은 5년 경로의 중앙값 ~ 95번째 백분위. 주황 점선 = 청산 한계.
최대낙폭은 장중 저점까지 넣었다 (고점 = 이전 주말 자산의 최고).</p>

<h2>2. G 별 연복리 — 더 걸면 오히려 준다</h2>
<div class="card">{cagr_chart(K)}</div>
<p>변동성이 커질수록 복리에서 새는 몫(변동성 끌림)이 커져, 어느 지점부터는 더 걸수록 연복리가 준다. 역사 경로는 G 1.2 근처가 꼭대기, G 2.6 부터 청산.
주황 선은 주 수익에서 {s3['delta_bps']:.0f}bps 를 빼 샤프를 0.3 으로 낮춘 경우 — 엣지가 작으면 꼭대기가 훨씬 왼쪽으로 온다.</p>

<h2>3. G 별 누적 자산 (로그 축)</h2>
<div class="card">{equity_chart(eq)}</div>

<h2>4. G 표</h2>
<div class="tbl"><table><thead><tr><th>G</th><th>역사 연복리</th><th>역사 최대낙폭</th><th>최악의 주</th><th>5년 중앙 연복리</th>
<th>5년 중앙 낙폭</th><th>95% 낙폭</th><th>5년 손실 확률</th><th>청산 확률</th></tr></thead><tbody>{gtab}</tbody></table></div>
<p class="note">5년 = 과거 269주를 4주 블록으로 다시 뽑은 경로 10,000개. 청산 = 장중 5분마다 자산 ≤ 유지증거금률 2% × 현재 총명목 (OKX 1단계 0.5 ~ 2% 보다 보수적).
G 3 이상은 역사 경로에서 2021-05-24 주에 청산된다.</p>

<h2>5. 교차 증거금 환산 — 잉여 n% · 거래소 레버리지 L</h2>
<div class="tbl"><table><thead><tr><th>거래소 레버리지 L</th><th>G {r30:.2f} 일 때 잉여 n</th><th>G {r50:.2f} 일 때 잉여 n</th><th>G 1 일 때 잉여 n</th></tr></thead><tbody>{ntab}</tbody></table></div>
<p class="note">교차 증거금에선 거래소 레버리지 L 이 개시증거금만 정하고, 청산은 계좌 전체 자산과 유지증거금으로 판단한다(OKX: 유지증거금률 100% 이하에서 청산).
그래서 같은 G 면 어떤 n · L 조합도 청산 위험이 같다. 표는 '레버리지를 L 배로 설정하고 G 만큼 포지션을 열면 개시증거금으로 (100 − n)% 가 묶이고 n% 가 남는다' 는 뜻이다 (n = 1 − G ÷ L).</p>

<h2>6. 사이징 비교 (1부)</h2>
<div class="tbl"><table><thead><tr><th>방식</th><th>샤프</th><th>전반</th><th>후반</th><th>변동성 30% 로 맞춘 연복리</th><th>그때 최대낙폭</th><th>수수료 연</th><th>평균 G</th></tr></thead>
<tbody>{rows1}</tbody></table></div>
<div class="tbl"><table><thead><tr><th>사전등록 판정</th><th>샤프 차이</th><th>블록 부트스트랩 t</th><th>95% 구간</th><th>결과</th></tr></thead><tbody>{''.join(vrow)}</tbody></table></div>
<p class="note">채택 조건: 위험 균등은 전체 · 전반 · 후반 샤프가 모두 현행 이상(이론으로 정한 방향이라 유의성은 요구하지 않음), 변동성 비례는 t ≥ 2 + 두 반기 우위,
변동성 타깃은 전체 · 두 반기 샤프 이상. 셋 다 반기 하나에서 떨어졌다. 전반 · 후반 샤프는 경계의 +10% 주 하나로 0.07 씩 움직일 만큼 흔들린다 —
차이가 전부 잡음 범위라는 뜻이다. 역변동성 60일 대신 20 · 90일로 바꾸면 샤프 {A['window_sens']['S1_20d']['sharpe']:.2f} · {A['window_sens']['S1_90d']['sharpe']:.2f}.
위험 균등은 BTC · ETH · TRX 같은 저변동 코인 비중이 높아진다(평균 TRX {A['s1_weight_mean']['TRX'] * 100:.1f}% · BTC {A['s1_weight_mean']['BTC'] * 100:.1f}% vs 동일 2.1%).</p>

<h2>7. 변동성 타깃으로 레버리지를 조절하면 (서술)</h2>
<div class="tbl"><table><thead><tr><th>목표 변동성</th><th>평균 G</th><th>최대 G</th><th>역사 연복리</th><th>역사 최대낙폭</th><th>5년 중앙 연복리</th><th>95% 낙폭</th></tr></thead><tbody>{vtab}</tbody></table></div>
<p class="note">매주 G = min(3, 목표 ÷ 직전 60일 포트폴리오 변동성). 같은 95% 낙폭에서 고정 G 와 수익이 거의 같다 — 목표 10% ≈ G 0.2, 20 ~ 22.5% ≈ G 0.4.
조용한 시기에 G 를 3배 가까이 올리므로, 조용함 뒤의 급변에 약하다는 점은 고정 G 보다 나쁘다.</p>

<h2>8. 어떻게 쟀나</h2>
<p class="note">장중 경로: 매주 월 00:00 UTC 에 비중을 맞추고 수량을 고정한 채 5분봉 종가로 포트폴리오를 따라갔다(47 × 2016봉 × 269주). 청산 판정은 5분 종가(마크가격 근사)가 기본,
보수적 경로는 롱은 그 봉 저가 · 숏은 고가를 동시에 썼다. 검증: 장중 경로의 주 끝 = 주간 수익, 동일 명목 + 부호 수수료 = 예전 V0 (269주 일치), 2021-05-24 주 장중 최악 −35.45% 는
원자료에서 따로 계산해도 같다, 부트스트랩 분포는 단순 반복문으로 다시 뽑아도 같다(G 1: 중앙 62% · 95% 84%). 한계: 표본에 없는 더 나쁜 주는 만들지 못한다 · 펀딩 이력 없음 ·
자동 감축(ADL) · 거래소 위험은 모형 밖. 출처: <a href="https://www.okx.com/en-us/help/iii-single-currency-margin-cross-margin-trading">OKX 교차 증거금 청산 규칙</a> ·
<a href="https://www.okx.com/en-gb/help/adjustment-of-perpetual-swap-tiered-maintenance-margin-ratio-schedule">OKX 유지증거금률 단계</a>.</p>
</main></div></body></html>"""
    Path("out/tsmom_sizing.html").write_text(html, encoding="utf-8")
    print("out/tsmom_sizing.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
