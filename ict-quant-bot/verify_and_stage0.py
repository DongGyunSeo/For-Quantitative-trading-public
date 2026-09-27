#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 collect 검증 + STAGE 0 (트레일 폭 분포) 분석.
로컬에서 실행:  python verify_and_stage0.py
  - ml_train_*.csv (심볼별, with_tp 제외) 만 있으면 됨.
  - mfe_r_clean, mdd_1r/2r/4r/8r 컬럼 필요 (새 ml_data.py 수집본).
"""
import pandas as pd, numpy as np, glob, sys

# ── 데이터 경로 (필요시 수정)
GLOB = "ml_train_*.csv"
RR_PCT_FROM_TP = True   # R%(비용계산용) 복원에 tp_train 필요. 없으면 자동 skip.

def load():
    fs = [f for f in sorted(glob.glob(GLOB)) if "with_tp" not in f]
    if not fs:
        sys.exit(f"❌ {GLOB} 없음. 작업 폴더에서 실행하세요.")
    df = pd.concat([pd.read_csv(c) for c in fs], ignore_index=True)
    print(f"파일 {len(fs)}개 | trade {len(df):,} | 심볼 {sorted(df.symbol.unique())}")
    return df

def verify(df):
    print("\n" + "="*70)
    print("  [검증 1] 컬럼 존재 + forward-sim 정합")
    print("="*70)
    need = ["mfe_r_clean","mdd_1r","mdd_2r","mdd_4r","mdd_8r"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"❌ 누락 컬럼: {miss}  (새 ml_data.py로 재수집 필요)")
    print("  컬럼 OK:", need)
    wr1 = (df.mfe_r_clean >= 1).mean()*100
    print(f"  mfe_r_clean>=1.0 = {wr1:.2f}%   (RR1 collect 58.6%와 일치하면 forward-sim 정합)")

    print("\n" + "="*70)
    print("  [검증 2] mdd 정합성 (방향 분해) — 버그 vs 정의적 점프")
    print("="*70)
    bad_total = 0
    for b, lvl in [("mdd_1r",1),("mdd_2r",2),("mdd_4r",4),("mdd_8r",8)]:
        caseA = ((df.mfe_r_clean>=lvl)&(df[b]<0)).sum()   # 도달했는데 mdd없음 = 밴드점프(정의적)
        caseB = ((df.mfe_r_clean<lvl)&(df[b]>=0)).sum()    # 미도달인데 mdd있음 = 진짜버그
        bad_total += caseB
        hit = df.loc[df[b]>=0, b]
        rng = f"min={hit.min():.2f} med={hit.median():.2f} max={hit.max():.2f}" if len(hit) else "도달0"
        print(f"  {b}: 점유 {len(hit):5d} | 점프(A) {caseA:4d} | 버그(B) {caseB:4d} | 되돌림R[{rng}]")
    print(f"  → 버그(B) 합계 = {bad_total}  {'✅ 정상' if bad_total==0 else '❌ 조사 필요'}")
    print("    (A는 큰 runner가 밴드를 한 봉에 관통 = 정의적, 정상)")

def stage0(df):
    print("\n" + "="*70)
    print("  [STAGE 0] 밴드별 mdd 분포 — 트레일 폭 후보")
    print("="*70)
    print("  ※ mdd = 그 밴드 도달 후 다음 레벨 전까지 running-peak 대비 최대 되돌림(R)")
    print("  ※ 트레일 폭을 분위로 잡으면: 폭 < 그 분위 → 그 % trade는 꼬리 전에 털림\n")
    bands = [("mdd_1r","1R→2R"),("mdd_2r","2R→4R"),("mdd_4r","4R→8R"),("mdd_8r","8R+")]
    print(f"  {'밴드':<8}{'n':>6}{'median':>9}{'80%ile':>9}{'90%ile':>9}{'95%ile':>9}{'max':>8}")
    rows = {}
    for b, lab in bands:
        h = df.loc[df[b]>=0, b]
        if len(h)==0:
            print(f"  {lab:<8}{0:>6}{'-':>9}"); continue
        q = h.quantile([.5,.8,.9,.95]).values
        rows[b] = q
        print(f"  {lab:<8}{len(h):>6}{q[0]:>9.2f}{q[1]:>9.2f}{q[2]:>9.2f}{q[3]:>9.2f}{h.max():>8.2f}")
    print("\n  해석: 예) 4R밴드 80%ile이 1.8R이면, 4R 이후 트레일을 peak−1.8R보다")
    print("        타이트하게 잡으면 8R 가는 trade의 20%를 그 되돌림에 뺏김.")

    # mdd × is_tail : 꼬리는 더 깊게 되돌리며 가는가
    print("\n" + "-"*70)
    print("  [STAGE 0b] mdd × 꼬리(8R+) — 꼬리는 더 깊게 되돌리나?")
    print("-"*70)
    df = df.copy(); df["is_tail"] = (df.mfe_r_clean>=8).astype(int)
    for b, lab in [("mdd_1r","1R→2R"),("mdd_2r","2R→4R"),("mdd_4r","4R→8R")]:
        sub = df[df[b]>=0]
        if sub.is_tail.nunique()<2: 
            print(f"  {lab}: 표본 부족"); continue
        t = sub.loc[sub.is_tail==1, b]; nt = sub.loc[sub.is_tail==0, b]
        print(f"  {lab}: 꼬리 median={t.median():.2f}(n={len(t)}) vs 비꼬리 median={nt.median():.2f}(n={len(nt)})")
    print("  → 꼬리 되돌림이 더 깊으면: 단일 고정폭은 꼬리를 일찍 자름 → 밴드별 가변폭 필요")

    # 심볼별 안정성 (4R밴드 80%ile)
    print("\n" + "-"*70)
    print("  [STAGE 0c] 심볼별 트레일 폭 안정성 (mdd_2r 80%ile, R)")
    print("-"*70)
    for s, g in df.groupby("symbol"):
        h = g.loc[g.mdd_2r>=0, "mdd_2r"]
        if len(h)>=20:
            print(f"  {s}: 80%ile={h.quantile(.8):.2f}  median={h.median():.2f}  (n={len(h)})")
    print("  → 심볼별 편차 크면 심볼별 폭, 작으면 통합 단일폭")

def cost_check(df):
    """R%(비용) 복원은 tp_train의 candidate_price/dist_atr/rr_ratio + entry_atr 필요."""
    print("\n" + "="*70)
    print("  [참고] net-of-cost — R%% 복원 (tp_train 있으면)")
    print("="*70)
    tps = [f for f in sorted(glob.glob("tp_train_*.csv"))]
    if not tps:
        print("  tp_train_*.csv 없음 → R%% 복원 skip (STAGE 0엔 불필요)")
        return
    tp = pd.concat([pd.read_csv(c) for c in tps], ignore_index=True)
    tp1 = tp.sort_values("rr_ratio").drop_duplicates(["symbol","entry_time"], keep="first")
    m = df.copy()
    m["entry_time"] = pd.to_datetime(m.entry_time, utc=True)
    tp1 = tp1.copy(); tp1["entry_time"] = pd.to_datetime(tp1.entry_time, utc=True)
    j = m[["symbol","entry_time","entry_atr","direction"]].merge(
        tp1[["symbol","entry_time","candidate_price","dist_atr","rr_ratio"]],
        on=["symbol","entry_time"], how="inner")
    j["cand_dist"] = j.dist_atr*j.entry_atr
    j["risk"] = j.cand_dist/j.rr_ratio
    j["entry_price"] = np.where(j.direction=="long", j.candidate_price-j.cand_dist, j.candidate_price+j.cand_dist)
    j["R_pct"] = j.risk/j.entry_price*100
    j = j[(j.R_pct>0.05)&(j.R_pct<20)]
    print(f"  R 1개 % 크기: median={j.R_pct.median():.3f}%  IQR[{j.R_pct.quantile(.25):.3f}, {j.R_pct.quantile(.75):.3f}]")
    print(f"  → 비용(R) ≈ 왕복%% / R%%.  maker익절 0.04%% / SL·트레일 taker 0.07%%")
    print(f"     median R%%={j.R_pct.median():.3f} 기준 트레일 exit 비용 ≈ {0.07/j.R_pct.median():.3f}R")

if __name__ == "__main__":
    df = load()
    verify(df)
    stage0(df)
    cost_check(df)
    print("\n✅ 완료. 버그(B)=0 이고 mfe_r_clean>=1≈58%% 면 수집 정상.")
