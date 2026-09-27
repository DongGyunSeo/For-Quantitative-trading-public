"""1단계 — 반등은 어디서 일어나는가. 기술통계, 판정 없음.

질문: 기존 FVG 셋업은 **"갭 관통 = 손절"** 을 전제로 손절을 먼 끝에 둔다.
그 전제가 맞는가? 갭이 뚫린 뒤 그 아래에서 쓸고 올라오는 일이 흔하다면,
먼 끝 손절은 **반등이 시작되는 자리에서 털리는** 규칙이다.

측정: 확정 FVG 의 가까운 끝을 처음 터치한 뒤 지평(2016봉=7일) 안에서
  depth      = (먼 끝 − 최저) / 갭 크기.  −1 = 갭에 안 들어옴, 0 = 정확히 채움, +1 = 갭만큼 더
  recovered  = 최저점 이후 **가까운 끝**을 종가로 회복했는가
  run_after  = 회복 이후 갭 크기 배수로 얼마나 더 갔나

대조군: 같은 갭 크기 분포를 무작위 시점에 얹은 **가짜 갭**. FVG 자리가
특별한지, 그냥 가격이 원래 평균회귀하는지를 가른다.
"""
from __future__ import annotations
import zlib
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.data.loader import load_ohlcv
from qbot.research.fvg import FVGConfig, atr5m
from qbot.research.fvg_sweep import excursion

TFS = ("1h", "4h")
#: 지평 사다리 — 스윕은 몇 시간, 드리프트는 며칠 단위다
HORIZONS = (48, 144, 576, 2016)
BINS = [(-1.01, -0.5), (-0.5, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 1.0),
        (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
BIN_NAMES = ["갭 절반도 못 옴", "갭 안에서 반등", "0~0.25 관통", "0.25~0.5",
             "0.5~1.0", "1~2", "2~4", "4+"]


def fake_gaps(df, ex, rng, horizon=2016):
    """같은 갭 크기·방향 분포를 무작위 시점에 얹은 대조군."""
    n = len(df)
    hi = df["high"].to_numpy(np.float64); lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    m = len(ex.touch_bar)
    t0 = rng.integers(500, max(n - horizon - 2, 600), m)
    sd = ex.side.copy()
    g = cl[t0] * ex.gap_bps / 1e4
    near = np.where(sd > 0, cl[t0], cl[t0])
    far = np.where(sd > 0, near - g, near + g)
    depth = np.full(m, np.nan); rec = np.zeros(m, bool); run = np.full(m, np.nan)
    for j in range(m):
        s, f = int(t0[j]), min(n, int(t0[j]) + horizon)
        if sd[j] > 0:
            brk = np.flatnonzero(lo[s:f] < far[j])
            b0 = s + int(brk[0]) if len(brk) else s
            back = np.flatnonzero(cl[b0:f] > near[j])
            end = (b0 + int(back[0])) if len(back) else f
            depth[j] = (far[j] - np.min(lo[s:end + 1])) / g[j]
            if len(back):
                rec[j] = True
                run[j] = (np.max(hi[end:f]) - near[j]) / g[j]
        else:
            brk = np.flatnonzero(hi[s:f] > far[j])
            b0 = s + int(brk[0]) if len(brk) else s
            back = np.flatnonzero(cl[b0:f] < near[j])
            end = (b0 + int(back[0])) if len(back) else f
            depth[j] = (np.max(hi[s:end + 1]) - far[j]) / g[j]
            if len(back):
                rec[j] = True
                run[j] = (near[j] - np.min(lo[end:f])) / g[j]
    return depth, rec, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_all")
    ap.add_argument("--out", default="results_fvg_excursion.csv")
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.cache).glob("*_5m_futures.csv")):
        sym = f.stem.split("_")[0]
        df, _ = load_ohlcv(f)
        rng = np.random.default_rng(zlib.crc32(sym.encode()))  # hash() 는 프로세스마다 바뀐다
        for tf in TFS:
            for H in HORIZONS:
                ex = excursion(df, FVGConfig(htf=tf, atr_mode="daily"), horizon=H)
                rows.append(ex.frame().assign(symbol=sym, tf=tf, kind="실측", H=H))
            d, r, ru = fake_gaps(df, ex, rng, horizon=H)
            rows.append(pd.DataFrame(dict(symbol=sym, tf=tf, kind="가짜갭",
                                          depth=d, recovered=r, side=ex.side,
                                          gap_bps=ex.gap_bps, touch_bar=-1,
                                          depth_atr=np.nan, bar_min=-1,
                                          bars_to_recover=-1, run_after=ru, H=H)))
        print(f"  {sym} 완료", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(a.out, index=False)

    for tf in TFS:
      for H in HORIZONS:
        print("\n" + "=" * 96)
        print(f"[{tf} · 지평 {H}봉 = {H*5/60:.0f}시간]  갭 첫 터치 이후의 기하 — 기술통계")
        print("=" * 96)
        g = d[(d.tf == tf) & (d.kind == "실측") & (d.H == H) & d.depth.notna()]
        fk = d[(d.tf == tf) & (d.kind == "가짜갭") & (d.H == H) & d.depth.notna()]
        print(f"  이벤트 {len(g):,}건 · 갭 중앙 {g.gap_bps.median():.0f}bps")
        print(f"  갭을 끝까지 채운 비율 (depth >= 0)   {(g.depth >= 0).mean():>7.1%}"
              f"   (가짜갭 {(fk.depth >= 0).mean():.1%})")
        print(f"  먼 끝을 뚫은 비율   (depth > 0)      {(g.depth > 0).mean():>7.1%}"
              f"   (가짜갭 {(fk.depth > 0).mean():.1%})")
        print(f"\n  {'깊이 구간':<18}{'건수':>8}{'비중':>7}{'회복률':>8}{'가짜갭':>8}"
              f"{'차이':>8}{'회복까지':>9}{'회복후 진행':>11}")
        for (lo_, hi_), nm in zip(BINS, BIN_NAMES):
            s = g[(g.depth >= lo_) & (g.depth < hi_)]
            fs = fk[(fk.depth >= lo_) & (fk.depth < hi_)]
            if len(s) < 20:
                continue
            fr_ = fs.recovered.mean() if len(fs) >= 20 else np.nan
            print(f"  {nm:<18}{len(s):>8,}{len(s)/len(g):>7.1%}{s.recovered.mean():>8.1%}"
                  f"{fr_:>8.1%}{s.recovered.mean()-fr_:>+8.1%}"
                  f"{s[s.recovered].bars_to_recover.median():>9.0f}"
                  f"{s[s.recovered].run_after.median():>11.2f}")
        brk = g[g.depth > 0]; fbrk = fk[fk.depth > 0]
        print(f"\n  ** 먼 끝을 뚫은 {len(brk):,}건 중 가까운 끝까지 회복: "
              f"{brk.recovered.mean():.1%}  (가짜갭 {fbrk.recovered.mean():.1%})")
        print(f"     회복 후 진행 중앙 {brk[brk.recovered].run_after.median():.2f}갭 "
              f"(가짜갭 {fbrk[fbrk.recovered].run_after.median():.2f}갭)")


if __name__ == "__main__":
    main()
