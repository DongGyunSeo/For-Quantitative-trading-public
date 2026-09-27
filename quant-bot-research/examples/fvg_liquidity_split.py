"""보조 절단 검사 — 유동성 3분위가 연도·홀드아웃에서도 버티는가.

**사후(post-hoc) 절단이다.** 47심볼 보고서의 보조 표에서 4h 게이트150 이
얇음 −0.065 / 중간 +0.077 (t 3.26) / 두꺼움 +0.063 (t 2.18) 로 갈렸다.
이걸 후보로 삼기 전에 **같은 잣대(연도별 부호·홀드아웃)** 로 먼저 본다.

두 가지 분위 정의
  static : 봉인 전 전체 기간 5분 중앙 거래대금으로 한 번 나눈 것 (미래 정보 섞임)
  pit    : 진입 **전날까지** 30일 일간 거래대금 중앙값의 횡단면 순위 (그 시점에 알 수 있던 것만)

게이트는 다섯 개 그대로, 분위도 3개 고정. 여기서 새 컷을 찾지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research.yardstick import tstat

SEAL = pd.Timestamp("2026-07-05 00:00", tz="UTC")
GATES = (100.0, 125.0, 150.0, 175.0, 200.0)
LV = ("얇음", "중간", "두꺼움")
PANEL = Path("runs/u47_dailyqv.csv")


def daily_panel() -> pd.DataFrame:
    if PANEL.exists():
        p = pd.read_csv(PANEL, index_col=0, parse_dates=True)
        return p
    cols = {}
    for sym, f in symbol_files("cache_okx").items():
        df, _ = load_ohlcv(f)
        df = df[df.index <= SEAL]
        cols[sym] = (df.volume * df.close).resample("1D").sum()
        print(f"  {sym}", flush=True)
    p = pd.DataFrame(cols)
    p.to_csv(PANEL)
    return p


def pit_tercile(p: pd.DataFrame) -> pd.DataFrame:
    """날짜 d 의 분위 = d−1 까지 30일 중앙 거래대금의 횡단면 3분위."""
    med = p.rolling(30, min_periods=20).median().shift(1)
    r = med.rank(axis=1, pct=True)
    lab = pd.DataFrame(np.select([r <= 1 / 3, r <= 2 / 3, r > 2 / 3], [0, 1, 2], -1),
                       index=r.index, columns=r.columns)
    return lab.where(r.notna(), -1)


def year_line(s: pd.DataFrame, yrs) -> tuple[list[str], int]:
    cells, neg = [], 0
    for y in yrs:
        ys = s[s.year == y]
        if len(ys) >= 15:
            x = ys.net.mean()
            neg += x < 0
            cells.append((f"**{x:+.3f}**" if x < 0 else f"{x:+.3f}") + f" ({len(ys)})")
        else:
            cells.append(f"— ({len(ys)})")
    return cells, neg


def main() -> None:
    t = pd.read_csv("runs/u47_trades.csv", parse_dates=["time"])
    info = pd.read_csv("runs/u47_info.csv").set_index("symbol")
    q = info.med_quote_vol.rank(method="first")
    st = pd.qcut(q, 3, labels=list(LV))
    t["static"] = t.symbol.map(st).astype(str)

    p = daily_panel()
    lab = pit_tercile(p)
    day = t.time.dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    li = lab.stack()
    li.index.names = ["day", "symbol"]
    li = li.rename("pit").reset_index()
    li["day"] = pd.to_datetime(li.day).dt.tz_localize(None)
    t = t.assign(day=day).merge(li, on=["day", "symbol"], how="left")
    t["pit"] = t.pit.map({0: LV[0], 1: LV[1], 2: LV[2]}).fillna("미정")

    out = []
    out.append("#### 정적 분위 구성 (봉인 전 5분 중앙 거래대금)")
    out.append("")
    for lv in LV:
        ss = sorted(st[st == lv].index, key=lambda s: -info.med_quote_vol[s])
        out.append(f"- **{lv}** ({len(ss)}): " + ", ".join(ss))
    out.append("")

    yrs = sorted(t.year.unique())
    for mode in ("static", "pit"):
        out.append(f"#### 분위 정의 = {mode} · 게이트150 · 연도별 순R (괄호 = 거래 수)")
        out.append("")
        out.append("| TF | 분위 | 거래수 | 순R | t(월) | " + " | ".join(str(y) for y in yrs)
                   + " | 음수 해 | 홀드아웃 순R (n) |")
        out.append("|---|---|---:|---:|---:|" + "---:|" * (len(yrs) + 2))
        for tf in ("4h", "1h"):
            for lv in LV:
                s = t[(t.tf == tf) & (t.risk_bps >= 150) & (t[mode] == lv)]
                mm = s.groupby("month").net.mean()
                _, tv, _ = tstat(mm)
                cells, neg = year_line(s, yrs)
                h = s[s.holdout]
                out.append(f"| {tf} | {lv} | {len(s):,} | {s.net.mean():+.4f} | {tv:.2f} | "
                           + " | ".join(cells) + f" | {neg} | {h.net.mean():+.4f} ({len(h)}) |")
        out.append("")

    out.append("#### 중간+두꺼움 합산 · 게이트 다섯 개 (static / pit)")
    out.append("")
    out.append("| TF | 정의 | 게이트 | 거래수 | gross | 순R | t(월) | 음수 해 | 홀드아웃 순R |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for tf in ("4h", "1h"):
        for mode in ("static", "pit"):
            for gt in GATES:
                s = t[(t.tf == tf) & (t.risk_bps >= gt) & t[mode].isin(["중간", "두꺼움"])]
                mm = s.groupby("month").net.mean()
                _, tv, _ = tstat(mm)
                _, neg = year_line(s, yrs)
                h = s[s.holdout]
                out.append(f"| {tf} | {mode} | {int(gt)} | {len(s):,} | {s.gross.mean():+.4f} | "
                           f"{s.net.mean():+.4f} | {tv:.2f} | {neg} | {h.net.mean():+.4f} |")
    out.append("")

    # 두꺼움 안에서 기존9 vs 신규 — 독립 재현 여부
    orig = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "TRX"}
    out.append("#### 정적 중간+두꺼움 안에서 기존9 / 신규 · 게이트150")
    out.append("")
    out.append("| TF | 코호트 | 심볼 | 거래수 | 순R | t(월) | " + " | ".join(str(y) for y in yrs) + " | 음수 해 |")
    out.append("|---|---|---:|---:|---:|---:|" + "---:|" * (len(yrs) + 1))
    for tf in ("4h", "1h"):
        for nm, m in (("기존9", t.symbol.isin(orig)), ("신규", ~t.symbol.isin(orig))):
            s = t[(t.tf == tf) & (t.risk_bps >= 150) & t.static.isin(["중간", "두꺼움"]) & m]
            mm = s.groupby("month").net.mean()
            _, tv, _ = tstat(mm)
            cells, neg = year_line(s, yrs)
            out.append(f"| {tf} | {nm} | {s.symbol.nunique()} | {len(s):,} | {s.net.mean():+.4f} | {tv:.2f} | "
                       + " | ".join(cells) + f" | {neg} |")
    out.append("")
    Path("out/fvg_liquidity_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
