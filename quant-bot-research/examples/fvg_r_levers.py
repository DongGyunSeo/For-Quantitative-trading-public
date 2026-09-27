"""FVG — TF 를 올리지 않고 1R 을 키우는 세 레버.

배경
----
`규칙적용-전수감사-및-FVG기준선.md`: 측정해상도 규칙을 걸면 FVG 는 1h~4h 에서
치환 대조군 대비 **+0.06~0.07R** 이 일관되게 나온다. 그런데 비용이 거의 같은
크기라 4H 가 손익분기, 1D 가 +0.009R 이다.

`1H레그-1단계결과-기각.md`: **TF 사다리는 이미 기각됐다.** 15m→1H 로 올렸더니
수수료 −33%, 총R −31% 로 **둘이 같은 비율로 줄어** 순R 이 제자리였다.
그래서 이번에는 **TF 를 고정**하고 다른 레버를 쓴다.

`추세추종-FVG-IS결과-기각.md` 가 남긴 다음 가설 #2 를 그대로 실행한다:
*"1R ≥ 손익분기 bps × k 라는 **사전** 게이트를 규칙의 일부로 넣는다
(사후 분위 선택이 아니라)."*

사전등록 (탐색 전 고정)
-----------------------
**레버**
  L1 진입 깊이 ``entry_level`` ∈ {0.0, 0.25, 0.5(기존), 0.75}
     0 = 갭의 가까운 끝. 1R = 갭 전체라 mid 의 2배.
  L2 손절 확장 ``stop_ext``    ∈ {0.0(기존), 0.25, 0.5, 1.0} — 먼 끝에서 갭×m 만큼 더.
  L3 사전 1R 게이트 ``min_risk_bps`` ∈ {0, 40, 60, 80, 120, 160}

**고정**  TF ∈ {1h, 4h} 둘뿐 (감사에서 엣지가 안정적이던 구간). TP 2R / SL 1R.
        측정해상도 규칙 1R/일간중앙ATR ≥ 2 는 항상 켠다. 추적 지평 2016봉.
        TP 1R 금지 — 진입봉 편향에 취약하다(추세추종 FVG 문서).

**비용**  진입 maker 2bps + 승 maker 2bps / 패·모호·미해결 taker 6bps.
        거래마다 ``fee_R = (2 + 청산bps) / 1R(bps)``.

**대조군** 치환(1R bps 분포·방향·심볼 고정, 진입 시점만 섞음) 심볼당 6회.

**주 지표** ``4h × entry_level 0.5 × stop_ext 0 × 게이트 120bps``
        — 비용 산술로 정한 단 하나의 칸이다. 감사의 4H 는 엣지 +0.072R /
        비용 0.079R 이었고, 1R 을 84 → 120bps 로 올리면 비용이 0.056R 로 내려간다.
        **엣지가 유지되기만 하면** 순 +0.016R 이 된다. 그게 유지되는지가 질문의 전부다.

**판정**  순R > 0 · 월 클러스터 t > 2 · 심볼+ >= 7/10 · 홀드아웃 12개월도 양수.
        나머지 191 칸은 **기술(description)이지 판정이 아니다.**

**홀드아웃** 마지막 12개월(2025-07~2026-07)은 따로 보고한다.

정직 고지
---------
배관 점검으로 BTC 한 심볼만 먼저 돌려봤고, 주 지표 칸의 gross 가 음수였다.
주 지표는 그 전에 비용 산술로 정한 것이고 판정 기준도 바꾸지 않는다.
"""
from __future__ import annotations
import zlib

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.data.loader import load_ohlcv
from qbot.research.barrier import control_expectancy
from qbot.research.fvg import FVGConfig, run_fvg

TFS = ("1h", "4h")
LEVELS = (0.0, 0.25, 0.5, 0.75)
EXTS = (0.0, 0.25, 0.5, 1.0)
GATES = (0.0, 40.0, 60.0, 80.0, 120.0, 160.0)
ENTRY_BPS, TP_BPS, SL_BPS = 2.0, 2.0, 6.0
NDRAW = 6
MIN_N = 50


def fee_r(r: np.ndarray, risk_bps: np.ndarray) -> np.ndarray:
    """거래별 수수료 R. 승리만 maker 청산, 나머지는 전부 taker."""
    exit_bps = np.where(r > 0, TP_BPS, SL_BPS)
    return (ENTRY_BPS + exit_bps) / risk_bps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_all")
    ap.add_argument("--out", default="results_fvg_levers.csv")
    a = ap.parse_args()

    rows = []
    for f in sorted(Path(a.cache).glob("*_5m_futures.csv")):
        sym = f.stem.split("_")[0]
        df, _ = load_ohlcv(f)
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        o = df["open"].to_numpy(np.float64)
        month = df.index.year.to_numpy() * 12 + df.index.month.to_numpy()
        hold_cut = int(df.index.searchsorted(df.index[-1] - pd.Timedelta(days=365)))
        rng = np.random.default_rng(zlib.crc32(sym.encode()))  # hash() 는 프로세스마다 바뀐다

        for tf in TFS:
            for lvl in LEVELS:
                for ext in EXTS:
                    res = run_fvg(df, FVGConfig(htf=tf, entry_level=lvl, stop_ext=ext,
                                                min_r_over_atr=2.0, atr_mode="daily",
                                                tp_r=2.0, sl_r=1.0, horizon=2016))
                    if len(res) == 0:
                        continue
                    fr = fee_r(res.r, res.risk_bps)
                    net = res.r - fr
                    mth = month[res.entry_bar]
                    hold = res.entry_bar >= hold_cut
                    for g in GATES:
                        k = res.risk_bps >= g
                        if k.sum() < MIN_N:
                            continue
                        ctrl, _ = control_expectancy(
                            h, l, o, res.entry_bar[k], res.risk_bps[k], res.side[k],
                            n_draw=NDRAW, rng=rng, tp_r=2.0, sl_r=1.0,
                            horizon=2016, skip_entry_bar=False)
                        base = dict(symbol=sym, tf=tf, lvl=lvl, ext=ext, gate=g)
                        for seg, m in (("전체", np.ones(k.sum(), bool)),
                                       ("홀드아웃 전", ~hold[k]),
                                       ("홀드아웃 12개월", hold[k])):
                            if m.sum() < 20:
                                continue
                            rows.append(dict(
                                **base, seg=seg, n=int(m.sum()),
                                risk_bps=float(np.median(res.risk_bps[k][m])),
                                gross=float(res.r[k][m].mean()),
                                ctrl=ctrl if seg == "전체" else np.nan,
                                fee=float(fr[k][m].mean()),
                                net=float(net[k][m].mean()),
                                win=float((res.r[k][m] > 0).mean()),
                                zero=float((res.r[k][m] == 0).mean())))
                        # 월별 순R (클러스터 t 용)
                        mm = pd.Series(net[k], index=mth[k]).groupby(level=0).mean()
                        for mo, v in mm.items():
                            rows.append(dict(**base, seg="월", month=int(mo),
                                             month_net=float(v)))
        print(f"  {sym} 완료", flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n{len(rows):,} 행 -> {a.out}")


if __name__ == "__main__":
    main()
