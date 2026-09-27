#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE 6 — swing 트레일 vs 균일 트레일(W=1.5) EV 비교 (stage4c 봉내순서 프레임).
로컬:  python stage6_swing_trail.py
필요:  ml_train_*.csv, tp_train_*.csv, {SYMBOL}_5m_futures.csv (같은 폴더 or OHLCV_DIR)

══════════════════════════════════════════════════════════════════
swing 트레일 정의 (형님 설계, long 기준 / short 미러):
  1) 진입 이후 5m fractal swing 감지 (swings.py 동일: left >=, right >,
     lookback LB). 스윙 i는 i+LB봉 "종가"에서만 확정 → lookahead 없음.
  2) HH = 가격이 "마지막 확정 swing high"를 돌파 (BOS/CHoCH).
     → 그 순간 직전 확정 swing low를 SL로 lock (ratchet: 위로만).
  3) HH가 나오지 않은 swing low는 SL 대상 아님.
     같은 swing high는 1회만 소모(broken flag) → 다음 lock은
     "새로 확정된" swing high 돌파가 있어야 함.
  4) 돌파 판정 2모드 동시 측정:
       swing_hb: high 돌파 (봉내 즉시, intrabar)
       swing_cb: close 돌파 (struct_event.py 방식, 봉마감 확정)
     → cb는 SL 갱신이 항상 봉마감이라 봉내 순서 착시가 구조적으로 작아야 함.

비교 프레임 (stage4c와 동일):
  - 3패턴: fav(낙관, 유리극값 먼저) / adv(비관, 불리극값 먼저) /
           candle(현실추정: 양봉=low→high→close, 음봉=high→low→close)
  - HORIZON=576봉, 초기 SL=-1R, TP 없음(트레일 단독 EV 프레임)
  - 착시폭 = EV(fav) - EV(adv). 작을수록 5분봉 측정 신뢰 높음.
  - eff_W = 청산 시점 (running peak - SL)/risk → stage4b c_w(폭 반비례 비용)
    프레임으로 슬리피지 비교 가능 (좁을수록 비용↑).

entry/risk 역산 (rebuild_trail_path와 동일):
  cand_dist  = dist_atr * entry_atr
  entry_price= candidate_price - dir*cand_dist   (long=-, short=+)
  risk       = cand_dist / rr_ratio
  → 트레이드별 후보 여러 개의 median (float 노이즈 방지)

검증: replay MFE ≈ ml_train.mfe_r_clean — 시작봉 포함/제외 2규약을
      자동 테스트해 더 맞는 쪽 채택, 일치율 출력. 80% 미만이면 경고.

출력:
  stage6_summary.csv  전략×패턴 요약 (EV/중앙값/SL률/timeout률/보유봉/eff_W)
  stage6_trades.csv   트레이드별 R (전략×패턴 컬럼) — 후속 분석용
"""
import sys, os, glob
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ────────────────────────── 설정 ──────────────────────────
OHLCV_DIR   = "."      # {SYM}_5m_futures.csv 폴더
HORIZON     = 576      # backtest _RR_SWEEP_HORIZON과 동일해야 함
LB          = 3        # fractal lookback (backtest TRAIL_SWING_LB=3)
SWING_BACK  = 1        # 1=직전 확정 swing low, 2=그 전 것(더 여유, Group B 성격)
W_UNIFORM   = 0.3      # 균일 트레일 기준폭 (stage4b 확정값)
MFE_TOL     = 0.05     # 검증 허용오차 (R)
PATTERNS    = ["fav", "adv", "candle"]
STRATS      = ["swing_hb", "swing_cb", "uniform"]

# ────────────────────────── 데이터 로드 ──────────────────────────

def load_ohlcv(sym: str):
    path = os.path.join(OHLCV_DIR, f"{sym.upper()}_5m_futures.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    pos = {t: i for i, t in enumerate(df["time"].values)}
    return {"o": o, "h": h, "l": l, "c": c, "pos": pos, "n": len(df)}


def fractal_pivots(h: np.ndarray, l: np.ndarray, lb: int):
    """전체 시리즈 1회 벡터화 fractal 감지 (swings.py find_swings 동일 규약).
    swing_high[i]: h[i] >= max(h[i-lb:i]) AND h[i] > max(h[i+1:i+1+lb])
    swing_low[i] : l[i] <= min(l[i-lb:i]) AND l[i] < min(l[i+1:i+1+lb])
    확정 시점 = i + lb (그 봉 종가)."""
    hs = pd.Series(h); ls = pd.Series(l)
    left_hmax  = hs.rolling(lb).max().shift(1)             # max(h[i-lb..i-1])
    right_hmax = hs[::-1].rolling(lb).max().shift(1)[::-1] # max(h[i+1..i+lb])
    left_lmin  = ls.rolling(lb).min().shift(1)
    right_lmin = ls[::-1].rolling(lb).min().shift(1)[::-1]

    sh = (hs >= left_hmax) & (hs > right_hmax)
    sl = (ls <= left_lmin) & (ls < right_lmin)
    sh = sh.fillna(False).values
    sl = sl.fillna(False).values
    return sh, sl


def reconstruct_entry_risk(ml: pd.DataFrame, tp: pd.DataFrame, sym: str):
    """entry_price/risk 복원 — 2점해법 우선 (entry_atr 양자화 우회).

    근거: ml_train의 entry_atr은 round(,4) 저장 → DOGE/TRX 등 저가코인에서
    ATR(~0.0002)이 유효숫자 1개 이하로 뭉개지고 0.0도 발생 (risk=0 크래시 원인).
    candidate_price는 풀 정밀도, rr_ratio는 round(,4) 비율(상대오차 ~1e-5)이므로
    p_i = entry + sgn*risk*rr_i 관계에서 후보쌍으로 직접 해법:
      risk  = (p_j - p_i) / (sgn*(rr_j - rr_i))   (전체 쌍 median, robust)
      entry = median(p_i - sgn*risk*rr_i)
    후보 1개/rr 동일 트레이드만 구 atr 방식 fallback."""
    need = {"entry_time", "candidate_price", "rr_ratio"}
    if not need.issubset(tp.columns):
        print(f"  [{sym}] tp_train 컬럼 부족: {need - set(tp.columns)} → 스킵")
        return None
    t = tp.copy()
    if "direction" not in t.columns:
        dmap = ml.drop_duplicates("entry_time").set_index("entry_time")["direction"]
        t["direction"] = t["entry_time"].map(dmap)
    t = t.dropna(subset=["direction", "rr_ratio"])
    t = t[t["rr_ratio"] > 0]

    amap = ml.drop_duplicates("entry_time").set_index("entry_time")["entry_atr"]

    ent, rsk = {}, {}
    n_2pt = n_atr = n_drop = 0
    for et, g in t.groupby("entry_time"):
        sgn = 1.0 if str(g["direction"].iloc[0]).lower() == "long" else -1.0
        p = g["candidate_price"].astype(float).values
        rr = g["rr_ratio"].astype(float).values
        risks = []
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                drr = rr[j] - rr[i]
                if abs(drr) > 1e-6:
                    risks.append((p[j] - p[i]) / (sgn * drr))
        risk = float(np.median(risks)) if risks else np.nan
        if np.isfinite(risk) and risk > 0:
            ent[et] = float(np.median(p - sgn * risk * rr))
            rsk[et] = risk
            n_2pt += 1
            continue
        # fallback: 구 atr 방식 (후보 1개 등)
        atr = amap.get(et, np.nan)
        da = g["dist_atr"].astype(float).values if "dist_atr" in g.columns else None
        if da is not None and np.isfinite(atr) and atr > 0:
            cd = da * atr
            r2 = float(np.median(cd / rr))
            if np.isfinite(r2) and r2 > 0:
                ent[et] = float(np.median(p - sgn * cd))
                rsk[et] = r2
                n_atr += 1
                continue
        n_drop += 1

    out = ml.copy()
    out["entry_px"] = out["entry_time"].map(ent)
    out["risk"] = out["entry_time"].map(rsk)
    out = out[(out["risk"].notna()) & (out["risk"] > 0)].reset_index(drop=True)
    print(f"  [{sym}] entry/risk 복원: 2점해법 {n_2pt} / atr-fallback {n_atr} / "
          f"제외 {n_drop + (len(ml) - n_2pt - n_atr - n_drop)}건")
    return out


# ────────────────────────── replay 코어 ──────────────────────────

def bar_order(o_i, c_i, pattern, direction):
    """봉내 이벤트 순서 반환: ('H','L') 또는 ('L','H').
    fav: 유리극값 먼저 (long=H, short=L) / adv: 불리 먼저
    candle: 양봉(close>=open)=L→H, 음봉=H→L (방향 무관 물리 순서)"""
    if pattern == "candle":
        return ("L", "H") if c_i >= o_i else ("H", "L")
    fav_first = (pattern == "fav")
    if direction == "long":
        return ("H", "L") if fav_first else ("L", "H")
    else:
        return ("L", "H") if fav_first else ("H", "L")


def replay_swing(o, h, l, c, start, end, entry, risk, direction,
                 sh_flag, sl_flag, pattern, break_mode, lb, back):
    """swing 트레일 replay. 반환: (pnl_r, exit_type, bars, eff_w)"""
    is_long = (direction == "long")
    sgn = 1.0 if is_long else -1.0
    sl = entry - sgn * risk                      # 초기 SL = -1R
    peak = entry                                  # 유리극값 (eff_W 계산용)

    last_ref = None        # 마지막 확정 기준스윙 (long=swing high, short=swing low)
    ref_broken = True      # 새 기준스윙 확정 전까지 BOS 불가
    locks = []             # 확정된 lock 후보 (long=swing low 가격들, 최신순 아님→append)

    for t in range(start + 1, end + 1):
        ht, lt, ot, ct = h[t], l[t], o[t], c[t]

        def sl_hit(px_extreme):
            return (px_extreme <= sl) if is_long else (px_extreme >= sl)

        def try_lock():
            nonlocal sl
            if len(locks) >= back:
                cand = locks[-back]
                if (is_long and cand > sl) or ((not is_long) and cand < sl):
                    sl = cand

        def bos_check(px):
            nonlocal ref_broken
            if ref_broken or last_ref is None:
                return
            broke = (px > last_ref) if is_long else (px < last_ref)
            if broke:
                ref_broken = True
                try_lock()

        first, second = bar_order(ot, ct, pattern, direction)
        for ev in (first, second):
            if ev == "H":
                if is_long:
                    peak = max(peak, ht)
                    if break_mode == "high":
                        bos_check(ht)
                else:
                    if sl_hit(ht):
                        r = (entry - sl) / risk
                        effw = abs(peak - sl) / risk
                        return r, "sl", t - start, effw
            else:  # "L"
                if is_long:
                    if sl_hit(lt):
                        r = (sl - entry) / risk
                        effw = abs(peak - sl) / risk
                        return r, "sl", t - start, effw
                else:
                    peak = min(peak, lt)
                    if break_mode == "high":   # short의 intrabar 돌파 = low 돌파
                        bos_check(lt)

        # ── 봉마감: 스윙 확정 + close-break BOS
        i_conf = t - lb
        if i_conf > start:
            if is_long:
                if sh_flag[i_conf]:
                    last_ref = h[i_conf]; ref_broken = False
                if sl_flag[i_conf]:
                    locks.append(l[i_conf])
            else:
                if sl_flag[i_conf]:
                    last_ref = l[i_conf]; ref_broken = False
                if sh_flag[i_conf]:
                    locks.append(h[i_conf])
        if break_mode == "close":
            bos_check(ct)

    r = sgn * (c[end] - entry) / risk
    effw = abs(peak - sl) / risk
    return r, "timeout", end - start, effw


def replay_uniform(o, h, l, c, start, end, entry, risk, direction, pattern, W):
    """균일 트레일 W (act 무관 — stage4 결론). SL = clamp(peak - W*risk, 초기SL~)."""
    is_long = (direction == "long")
    sgn = 1.0 if is_long else -1.0
    init_sl = entry - sgn * risk
    sl = init_sl
    peak = entry

    for t in range(start + 1, end + 1):
        ht, lt, ot, ct = h[t], l[t], o[t], c[t]
        first, second = bar_order(ot, ct, pattern, direction)
        for ev in (first, second):
            if ev == "H":
                if is_long:
                    if ht > peak:
                        peak = ht
                        sl = max(sl, peak - W * risk)   # ratchet
                else:
                    if ht >= sl:
                        return (entry - sl) / risk, "sl", t - start, abs(peak - sl) / risk
            else:
                if is_long:
                    if lt <= sl:
                        return (sl - entry) / risk, "sl", t - start, abs(peak - sl) / risk
                else:
                    if lt < peak:
                        peak = lt
                        sl = min(sl, peak + W * risk)
    r = sgn * (c[end] - entry) / risk
    return r, "timeout", end - start, abs(peak - sl) / risk


# ────────────────────────── 검증 (베이스 일치) ──────────────────────────

def validate_convention(trades, ohlcv, sym):
    """replay MFE vs mfe_r_clean — 시작봉 포함/제외 자동 판별."""
    h = ohlcv["h"]; l = ohlcv["l"]
    errs = {"excl": [], "incl": []}
    n_test = min(400, len(trades))
    for _, row in trades.head(n_test).iterrows():
        start = row["_start"]; end = row["_end"]
        entry = row["entry_px"]; risk = row["risk"]
        sgn = 1.0 if row["direction"] == "long" else -1.0
        init_sl = entry - sgn * risk
        mfe_csv = row.get("mfe_r_clean", np.nan)
        if not np.isfinite(mfe_csv):
            continue
        for key, s0 in (("excl", start + 1), ("incl", start)):
            best = 0.0
            for t in range(s0, end + 1):
                if sgn > 0:
                    best = max(best, (h[t] - entry) / risk)
                    if l[t] <= init_sl:
                        break
                else:
                    best = max(best, (entry - l[t]) / risk)
                    if h[t] >= init_sl:
                        break
            errs[key].append(abs(best - mfe_csv))
    res = {}
    for k, v in errs.items():
        v = np.array(v)
        res[k] = (np.median(v) if len(v) else np.inf,
                  float((v < MFE_TOL).mean()) if len(v) else 0.0)
    conv = "excl" if res["excl"][0] <= res["incl"][0] else "incl"
    med, rate = res[conv]
    print(f"  [{sym}] 검증: 시작봉 {'제외' if conv=='excl' else '포함'} 채택 "
          f"| median |Δmfe|={med:.4f}R, 일치율(<{MFE_TOL}R)={rate*100:.1f}%"
          + ("  ⚠️ 80% 미만 — entry/risk 정의 어긋남 의심!" if rate < 0.8 else ""))
    return conv, rate


# ────────────────────────── 메인 ──────────────────────────

def main():
    ml_files = sorted(glob.glob("ml_train_*.csv"))
    ml_files = [f for f in ml_files if "with_tp" not in f and "_trail" not in f]
    if not ml_files:
        print("ml_train_*.csv 없음"); return

    all_rows = []
    low_match_syms = []

    for mf in ml_files:
        ml = pd.read_csv(mf)
        if ml.empty:
            continue
        sym = str(ml["symbol"].iloc[0])
        tpf = mf.replace("ml_train_", "tp_train_")
        if not os.path.exists(tpf):
            print(f"  [{sym}] {tpf} 없음 → 스킵"); continue
        tp = pd.read_csv(tpf)

        ohlcv = load_ohlcv(sym)
        if ohlcv is None:
            print(f"  [{sym}] OHLCV 없음 ({sym.upper()}_5m_futures.csv) → 스킵"); continue

        tr = reconstruct_entry_risk(ml, tp, sym)
        if tr is None or tr.empty:
            continue

        # entry_time → OHLCV 위치
        et = pd.to_datetime(tr["entry_time"], utc=True).values
        tr["_start"] = [ohlcv["pos"].get(t, -1) for t in et]
        n_miss = (tr["_start"] < 0).sum()
        if n_miss:
            print(f"  [{sym}] entry_time→OHLCV 매칭 실패 {n_miss}건 제외")
        tr = tr[tr["_start"] >= 0].reset_index(drop=True)
        tr["_end"] = np.minimum(tr["_start"] + HORIZON, ohlcv["n"] - 1)
        tr = tr[tr["_end"] > tr["_start"] + LB].reset_index(drop=True)
        if tr.empty:
            continue

        conv, rate = validate_convention(tr, ohlcv, sym)
        if rate < 0.8:
            low_match_syms.append(sym)
        start_off = 0 if conv == "excl" else -1   # replay는 start+1부터 → incl이면 start-1로 보정

        sh_flag, sl_flag = fractal_pivots(ohlcv["h"], ohlcv["l"], LB)
        o, h, l, c = ohlcv["o"], ohlcv["h"], ohlcv["l"], ohlcv["c"]

        print(f"  [{sym}] replay {len(tr)}건 × {len(STRATS)}전략 × {len(PATTERNS)}패턴 ...")
        for _, row in tr.iterrows():
            start = int(row["_start"]) + start_off
            end = int(row["_end"])
            entry, risk, d = row["entry_px"], row["risk"], row["direction"]
            rec = {"symbol": sym, "entry_time": row["entry_time"], "direction": d,
                   "exit_reason_orig": row.get("exit_reason", ""),
                   "mfe_r_clean": row.get("mfe_r_clean", np.nan)}
            for pat in PATTERNS:
                r1 = replay_swing(o, h, l, c, start, end, entry, risk, d,
                                  sh_flag, sl_flag, pat, "high", LB, SWING_BACK)
                r2 = replay_swing(o, h, l, c, start, end, entry, risk, d,
                                  sh_flag, sl_flag, pat, "close", LB, SWING_BACK)
                r3 = replay_uniform(o, h, l, c, start, end, entry, risk, d, pat, W_UNIFORM)
                for name, res in (("swing_hb", r1), ("swing_cb", r2), ("uniform", r3)):
                    rec[f"{name}_{pat}_r"]    = round(res[0], 4)
                    rec[f"{name}_{pat}_exit"] = res[1]
                    rec[f"{name}_{pat}_bars"] = res[2]
                    rec[f"{name}_{pat}_effw"] = round(res[3], 3)
            all_rows.append(rec)

    if not all_rows:
        print("결과 없음"); return
    df = pd.DataFrame(all_rows)
    df.to_csv("stage6_trades.csv", index=False)

    # ────────── 요약 ──────────
    print("\n" + "=" * 100)
    print(f"STAGE 6 결과 — {len(df)}건 | LB={LB} back={SWING_BACK} W={W_UNIFORM} "
          f"horizon={HORIZON} | swing_cb=close돌파(권장 후보), swing_hb=high돌파")
    print("=" * 100)

    summary = []
    header = (f"{'전략':<10s}{'패턴':<8s}{'EV(R)':>8s}{'중앙값':>8s}{'SL률%':>7s}"
              f"{'TO률%':>7s}{'R≥0%':>7s}{'평균봉':>8s}{'effW':>7s}")
    print(header); print("-" * len(header))
    for s in STRATS:
        for pat in PATTERNS:
            r = df[f"{s}_{pat}_r"]
            ex = df[f"{s}_{pat}_exit"]
            line = {"strategy": s, "pattern": pat,
                    "ev": r.mean(), "median": r.median(),
                    "sl_pct": (ex == "sl").mean() * 100,
                    "to_pct": (ex == "timeout").mean() * 100,
                    "ge0_pct": (r >= 0).mean() * 100,
                    "bars": df[f"{s}_{pat}_bars"].mean(),
                    "effw": df[f"{s}_{pat}_effw"].mean()}
            summary.append(line)
            print(f"{s:<10s}{pat:<8s}{line['ev']:>8.4f}{line['median']:>8.3f}"
                  f"{line['sl_pct']:>7.1f}{line['to_pct']:>7.1f}{line['ge0_pct']:>7.1f}"
                  f"{line['bars']:>8.1f}{line['effw']:>7.2f}")
        gap = (df[f"{s}_fav_r"].mean() - df[f"{s}_adv_r"].mean())
        print(f"{'':<10s}→ 착시폭(fav-adv) = {gap:+.4f}R\n")

    pd.DataFrame(summary).to_csv("stage6_summary.csv", index=False)

    # ────────── 심볼별 (candle=현실추정) ──────────
    print("심볼별 EV (candle 패턴, 현실추정):")
    print(f"{'심볼':<7s}" + "".join(f"{s:>12s}" for s in STRATS) + f"{'건수':>8s}")
    for sym, g in df.groupby("symbol"):
        print(f"{sym:<7s}" + "".join(f"{g[f'{s}_candle_r'].mean():>12.4f}" for s in STRATS)
              + f"{len(g):>8d}")
    tot = "".join(f"{df[f'{s}_candle_r'].mean():>12.4f}" for s in STRATS)
    print(f"{'전체':<7s}{tot}{len(df):>8d}")

    if low_match_syms:
        print(f"\n⚠️ 검증 일치율 80% 미만 심볼: {low_match_syms} — 해당 심볼 수치 신뢰 주의 "
              f"(TRX/DOGE 등 OHLCV 품질 이슈와 동일 원인일 수 있음)")

    print("\n판정 가이드:")
    print("  1) swing 착시폭 < uniform 착시폭 → '봉내착시 작다' 가설 확인")
    print("  2) candle 패턴 EV: swing_cb vs uniform W=1.5 직접 비교 (stage4 기준 +0.25~0.28R)")
    print("  3) effW: swing이 더 넓으면 stage4b c_w 비용 프레임에서도 유리")
    print("  저장: stage6_summary.csv / stage6_trades.csv")


if __name__ == "__main__":
    main()