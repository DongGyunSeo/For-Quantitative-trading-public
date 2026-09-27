"""수집 데이터 감사 — **표시만 하고 제거하지 않는다.**

두 가지를 본다.

``audit``   : 파일마다 manifest 와 행 수·시작·끝·공백·이상봉을 대조하고,
              분석을 오염시킬 수 있는 봉(극단 스파이크, 시가 점프, 긴 무거래 구간,
              평평한 봉 비율)을 센다. 결과는 CSV 로 남기고 **데이터는 건드리지 않는다.**

``overlap`` : 기존 캐시와 새 캐시가 겹치는 구간에서 OHLCV 가 같은지 본다.
              같은 거래소·같은 봉이면 가격은 **완전히 같아야** 한다.

사용
----
    python tools/data_audit.py audit   --cache cache_okx --out results_data_audit.csv
    python tools/data_audit.py overlap --old cache_all --new cache_okx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.data.loader import load_ohlcv, symbol_files
from qbot.research.fvg import atr5m

#: 극단봉 문턱 — 일간중앙 ATR14 배수. 표시용이며 분석에서 쓰지 않는다.
SPIKE_ATR = 20.0      # 봉 범위가 평소 5분 ATR 의 20배
WICK_ATR = 10.0       # 꼬리 하나가 10배 이상인데
WICK_BODY_ATR = 2.0   #   몸통은 2배 미만 (찍고 돌아온 봉)
JUMP_ATR = 5.0        # 시가가 직전 종가에서 5배 이상 떨어져 열림
STEP_US = 300_000_000


def longest_run(mask: np.ndarray) -> int:
    """True 가 연속되는 최장 길이."""
    if not mask.any():
        return 0
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def audit_one(sym: str, path: Path, man: dict | None) -> tuple[dict, list[dict]]:
    df, q = load_ohlcv(path)
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    atr = atr5m(df, 14, mode="daily")

    rng = (h - l) / atr
    body = np.abs(c - o) / atr
    wick = np.maximum(h - np.maximum(o, c), np.minimum(o, c) - l) / atr
    jump = np.zeros_like(o)
    jump[1:] = np.abs(o[1:] - c[:-1]) / atr[1:]

    spike = rng > SPIKE_ATR
    wickrev = (wick > WICK_ATR) & (body < WICK_BODY_ATR)
    jmp = jump > JUMP_ATR

    row = dict(symbol=sym, rows=q.n, start=str(q.start), end=str(q.end), gaps=q.gaps,
               max_gap=q.max_gap, dupes=q.dupes, bad_ohlc=q.bad_ohlc,
               off_grid=int((df.index.asi8 % STEP_US != 0).sum()),
               nonpos=int((l <= 0).sum()),
               zero_vol=q.zero_volume, zero_vol_run=longest_run(v <= 0),
               flat_pct=float((h == l).mean() * 100),
               flat_run=longest_run(h == l),
               spike=int(spike.sum()), wick_revert=int(wickrev.sum()), jump=int(jmp.sum()),
               max_range_atr=float(np.nanmax(rng)),
               px_first=float(c[0]), px_last=float(c[-1]),
               med_quote_vol=float(np.median(v * c)))
    if man is not None:
        f = man.get("final", {})
        row["man_ok"] = bool(man.get("done") and f.get("ok")
                             and f.get("rows") == q.n
                             and f.get("start") == str(q.start)
                             and f.get("end") == str(q.end)
                             and f.get("gaps") == q.gaps
                             and f.get("bad_ohlc") == q.bad_ohlc)
    else:
        row["man_ok"] = None

    events = []
    for name, m, score in (("spike", spike, rng), ("wick_revert", wickrev, wick), ("jump", jmp, jump)):
        idx = np.flatnonzero(m)
        for i in idx[np.argsort(-score[idx])][:3]:
            events.append(dict(symbol=sym, kind=name, time=str(df.index[i]), x_atr=float(score[i]),
                               open=o[i], high=h[i], low=l[i], close=c[i],
                               prev_close=float(c[i - 1]) if i else np.nan))
    return row, events


def cmd_audit(a) -> None:
    files = symbol_files(a.cache)
    mp = Path(a.cache) / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    rows, events = [], []
    for sym, f in files.items():
        r, e = audit_one(sym, f, manifest.get(sym))
        rows.append(r)
        events += e
        print(f"  {sym:<6} {r['rows']:>8,} 공백 {r['gaps']} 이상봉 {r['bad_ohlc']} "
              f"manifest {'일치' if r['man_ok'] else '불일치' if r['man_ok'] is False else '-'} "
              f"스파이크 {r['spike']} 꼬리 {r['wick_revert']} 점프 {r['jump']} "
              f"평평 {r['flat_pct']:.1f}%", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(events).to_csv(a.events, index=False)
    print(f"\n{len(rows)}심볼 -> {a.out} · 극단봉 표본 {len(events)} -> {a.events}")


def compare(old: pd.DataFrame, new: pd.DataFrame, until: pd.Timestamp | None) -> dict:
    if until is not None:
        old = old[old.index <= until]
        new = new[new.index <= until]
    both = old.index.intersection(new.index)
    a, b = old.loc[both], new.loc[both]
    out = dict(old_rows=len(old), new_rows=len(new), common=len(both),
               only_old=len(old.index.difference(new.index)),
               only_new=len(new.index.difference(old.index)),
               first_only_new=str(new.index.difference(old.index)[:3].tolist()))
    for k in ("open", "high", "low", "close"):
        x, y = a[k].to_numpy(), b[k].to_numpy()
        diff = x != y
        out[f"{k}_mismatch"] = int(diff.sum())
        out[f"{k}_max_rel"] = float(np.max(np.abs(x - y) / np.abs(x))) if len(x) else np.nan
    vx, vy = a.volume.to_numpy(), b.volume.to_numpy()
    ok = (vx > 0) & (vy > 0)
    out["vol_equal_pct"] = float(np.isclose(vx, vy, rtol=1e-9, atol=0).mean() * 100)
    out["vol_ratio_med"] = float(np.median(vy[ok] / vx[ok])) if ok.any() else np.nan
    out["vol_ratio_p01"] = float(np.quantile(vy[ok] / vx[ok], 0.01)) if ok.any() else np.nan
    out["vol_ratio_p99"] = float(np.quantile(vy[ok] / vx[ok], 0.99)) if ok.any() else np.nan
    # 가격이 다르면 어디서 다른지 첫 몇 개
    d = (a[["open", "high", "low", "close"]].to_numpy() != b[["open", "high", "low", "close"]].to_numpy()).any(1)
    out["first_px_diff"] = str([str(t) for t in both[d][:3]])
    # 볼륨이 다르기 시작하는 시점 — 수정 반영 여부 판단용
    vd = ~np.isclose(vx, vy, rtol=1e-9, atol=0)
    out["vol_diff_first"] = str(both[vd][0]) if vd.any() else ""
    out["vol_diff_last"] = str(both[vd][-1]) if vd.any() else ""
    return out


def cmd_overlap(a) -> None:
    old, new = symbol_files(a.old), symbol_files(a.new)
    until = pd.Timestamp(a.until, tz="UTC") if a.until else None
    rows = []
    for sym in sorted(set(old) & set(new)):
        o, _ = load_ohlcv(old[sym])
        n, _ = load_ohlcv(new[sym])
        r = dict(symbol=sym, **compare(o, n, until))
        rows.append(r)
        print(f"  {sym:<5} 공통 {r['common']:>8,}  기존만 {r['only_old']:>6,}  새것만 {r['only_new']:>6,}  "
              f"가격불일치 O{r['open_mismatch']} H{r['high_mismatch']} L{r['low_mismatch']} C{r['close_mismatch']}  "
              f"볼륨일치 {r['vol_equal_pct']:.2f}%  비율중앙 {r['vol_ratio_med']:.6g}", flush=True)
    print("  기존에만 있는 심볼:", sorted(set(old) - set(new)))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"-> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--cache", default="cache_okx")
    p.add_argument("--out", default="results_data_audit.csv")
    p.add_argument("--events", default="results_data_audit_events.csv")
    p = sub.add_parser("overlap")
    p.add_argument("--old", default="cache_all")
    p.add_argument("--new", default="cache_okx")
    p.add_argument("--until", default="")
    p.add_argument("--out", default="results_data_overlap.csv")
    a = ap.parse_args()
    {"audit": cmd_audit, "overlap": cmd_overlap}[a.cmd](a)


if __name__ == "__main__":
    main()
