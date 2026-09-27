"""2단계 — 갭이 뚫린 뒤 CHoCH 에 진입한다.

1단계가 보여준 것 (`results_fvg_excursion.csv`)
-----------------------------------------------
4H FVG 는 **거의 항상 뚫린다** — 12시간 안에 71.9%, 7일 안에 91.9%.
그리고 뚫린 뒤 **대부분 돌아온다** — 12시간 안에 73.9%, 7일 안에 90.3%.
즉 먼 끝 손절은 **반등이 시작되는 자리에 정확히 놓여 있다**. 전제가 틀렸다.

그런데 같은 크기의 **가짜 갭**(무작위 시점)도 89.2% 돌아온다(실측 90.3%).
되돌림 자체는 FVG 의 성질이 아니라 가격의 성질일 수 있다.
그래서 이번 검정에는 **FVG 맥락이 없는 CHoCH 대조군**이 반드시 들어간다.

사전등록 (탐색 전 고정)
-----------------------
**신호**  FVG(4h/1h) 가 먼 끝을 뚫은 뒤, 같은 방향 **CHoCH 확정 봉**에 진입.
        CHoCH 는 확정 스윙(lb=3) 종가 돌파이고 추세 상태가 바뀔 때만 센다.
**CHoCH TF** {5m, 15m, 1h} 셋뿐.
**손절**  갭 첫 터치 ~ 진입 사이의 **스윕 극단**. 버퍼 {0, 0.25 ATR}.
**진입**  CHoCH 봉 종가에 지정가(maker 2bps), 12봉 안 미체결이면 폐기.
**TP**   2R. TP 1R 금지(진입봉 편향).
**규칙**  1R / 일간중앙 ATR >= 2 상시.
**비용**  진입 maker 2 + 승 maker 2 / 패·모호·미해결 taker 6 bps.
**대조군 A** 치환 — 1R bps 분포·방향 고정, 진입 시점만 섞음.
**대조군 B** FVG 없는 순수 CHoCH — 같은 TF 의 **모든** CHoCH 에서, 직전 144봉
        극단을 손절로. **FVG 가 무언가를 더하는지**를 가르는 핵심 대조군.
**주 지표** 4h FVG × CHoCH 15m × 버퍼 0
**판정**  순R > 0 · 월 클러스터 t > 2 · 심볼+ >= 7/10 · 홀드아웃 12개월 양수 ·
        **그리고 대조군 B 를 이길 것**. 다섯 다.
"""
from __future__ import annotations
import zlib
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.data.loader import load_ohlcv
from qbot.research.barrier import control_expectancy, first_passage_r
from qbot.research.execution import ExecConfig, limit_entry_trades
from qbot.research.fvg import FVGConfig, atr5m
from qbot.research.fvg_sweep import choch_on_tf, excursion, sweep_choch

FVG_TFS = ("1h", "4h")
CH_TFS = ("5m", "15m", "1h")
BUFFERS = (0.0, 0.25)
CTRL_LOOKBACK = 144          # 대조군 B 의 손절 창 (12시간)
CFG = ExecConfig(entry_bps=2.0, tp_bps=2.0, sl_bps=6.0, fill_window=12,
                 tp_r=2.0, sl_r=1.0, horizon=2016)
MIN_ROA = 2.0


def plain_choch(df, choch5, atr, lookback=CTRL_LOOKBACK) -> pd.DataFrame:
    """대조군 B — FVG 맥락 없이 모든 CHoCH. 손절은 직전 lookback 봉 극단."""
    hi = df["high"].to_numpy(np.float64); lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    bars = np.flatnonzero(choch5 != 0)
    bars = bars[(bars > lookback) & (bars < len(df) - CFG.horizon - CFG.fill_window - 2)]
    if len(bars) == 0:
        return pd.DataFrame()
    sd = choch5[bars].astype(np.int64)
    lo_r = pd.Series(lo).rolling(lookback).min().to_numpy()[bars]
    hi_r = pd.Series(hi).rolling(lookback).max().to_numpy()[bars]
    stop = np.where(sd > 0, lo_r, hi_r)
    entry = cl[bars]
    risk = np.where(sd > 0, entry - stop, stop - entry)
    ok = np.isfinite(risk) & (risk > 0)
    return pd.DataFrame(dict(bar=bars[ok], side=sd[ok], entry=entry[ok],
                             risk=risk[ok], risk_bps=risk[ok] / entry[ok] * 1e4))


def evaluate(sym, tag, df, t, atr, month, hold_cut, rng, rows, **meta):
    """진입 신호 -> 체결·레이스·비용·대조군."""
    if len(t) == 0:
        return
    hi = df["high"].to_numpy(np.float64); lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64); op = df["open"].to_numpy(np.float64)
    roa = t.risk.to_numpy() / atr[t.bar.to_numpy()]
    t = t[np.isfinite(roa) & (roa >= MIN_ROA)]
    if len(t) < 30:
        return
    bars = t.bar.to_numpy(np.int64)
    keep = bars < len(df) - CFG.horizon - CFG.fill_window - 2
    t = t[keep]
    if len(t) < 30:
        return
    tr = limit_entry_trades(hi, lo, cl, t.bar.to_numpy(np.int64),
                            t.side.to_numpy(np.int64), t.risk.to_numpy(), CFG)
    f = tr[tr.filled].copy()
    if len(f) < 20:
        return
    ctrl, _ = control_expectancy(hi, lo, op, f.fill_bar.to_numpy(),
                                 t.risk_bps.to_numpy()[tr.filled.to_numpy()],
                                 f.side.to_numpy(), n_draw=8, rng=rng,
                                 tp_r=2.0, sl_r=1.0, horizon=CFG.horizon,
                                 skip_entry_bar=False)
    f["month"] = month[f.bar.to_numpy()]
    f["hold"] = f.bar.to_numpy() >= hold_cut
    base = dict(symbol=sym, tag=tag, **meta)
    for seg, m in (("전체", np.ones(len(f), bool)),
                   ("홀드아웃 전", ~f.hold.to_numpy()),
                   ("홀드아웃 12개월", f.hold.to_numpy())):
        s = f[m]
        if len(s) < 15:
            continue
        rows.append(dict(**base, seg=seg, n=len(s), fill=len(f) / len(tr),
                         risk_bps=float(np.median(
                             t.risk_bps.to_numpy()[tr.filled.to_numpy()][m])),
                         gross=float(s.r_gross.mean()), fee=float(s.fee_r.mean()),
                         net=float(s.r_net.mean()),
                         ctrl=ctrl if seg == "전체" else np.nan,
                         win=float((s.r_gross > 0).mean()),
                         zero=float((s.r_gross == 0).mean())))
    for mo, v in f.groupby("month").r_net.mean().items():
        rows.append(dict(**base, seg="월", month=int(mo), month_net=float(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_all")
    ap.add_argument("--out", default="results_fvg_sweep_choch.csv")
    a = ap.parse_args()
    rows = []
    for fp in sorted(Path(a.cache).glob("*_5m_futures.csv")):
        sym = fp.stem.split("_")[0]
        df, _ = load_ohlcv(fp)
        atr = atr5m(df, 14, mode="daily")
        month = df.index.year.to_numpy() * 12 + df.index.month.to_numpy()
        hold_cut = int(df.index.searchsorted(df.index[-1] - pd.Timedelta(days=365)))
        rng = np.random.default_rng(zlib.crc32(sym.encode()))  # hash() 는 프로세스마다 바뀐다
        cache: dict = {}
        ch = {tf: choch_on_tf(df, tf, 3, cache) for tf in CH_TFS}

        for ctf in CH_TFS:                       # 대조군 B — FVG 무관
            evaluate(sym, "순수CHoCH", df, plain_choch(df, ch[ctf], atr),
                     atr, month, hold_cut, rng, rows, fvg_tf="—", ch_tf=ctf, buf=0.0)
        for ftf in FVG_TFS:
            ex = excursion(df, FVGConfig(htf=ftf, atr_mode="daily"), horizon=576)
            for ctf in CH_TFS:
                for buf in BUFFERS:
                    t = sweep_choch(df, ex, ch[ctf], min_depth=0.0, max_wait=576,
                                    buffer_atr=buf, atr=atr)
                    evaluate(sym, "스윕+CHoCH", df, t, atr, month, hold_cut, rng,
                             rows, fvg_tf=ftf, ch_tf=ctf, buf=buf)
        print(f"  {sym} 완료", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows):,} 행 -> {a.out}")


if __name__ == "__main__":
    main()
