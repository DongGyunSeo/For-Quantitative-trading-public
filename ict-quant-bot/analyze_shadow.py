""" 
analyze_shadow.py — SHADOW_TRAIL=1로 수집한 CSV의 trail off vs on 직접 비교

SHADOW_TRAIL=1 + TRAIL_MODE=off로 collect_ml_data.py를 한 번 돌리면
각 trade에 다음이 함께 기록됨:
  - pnl_r          : primary 청산 (trail OFF)
  - shadow_pnl_r   : 같은 trade가 trail ON이었다면의 청산
  - shadow_exit_reason

같은 진입 → 두 청산이므로 trade 시퀀스 갈라짐 문제 없음.
청산 후 가격도 백테스트가 끝까지 추적하므로 정확.

사용:
    python analyze_shadow.py                     # ml_train_*.csv 자동
    python analyze_shadow.py ml_train_with_tp.csv
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import pandas as pd


def load(path_or_glob: str) -> pd.DataFrame:
    if path_or_glob and Path(path_or_glob).exists():
        df = pd.read_csv(path_or_glob)
        print(f"  📂 {path_or_glob}  ({len(df):,}건)")
    else:
        files = [p for p in glob.glob("ml_train_*.csv")
                 if not Path(p).stem.endswith("_traon")]
        if not files:
            print("❌ ml_train_*.csv 없음")
            sys.exit(1)
        dfs = [pd.read_csv(p) for p in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"  📂 {len(files)}개 파일 ({len(df):,}건)")
    return df


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    df = load(path)

    if "shadow_pnl_r" not in df.columns:
        print("\n❌ shadow_pnl_r 컬럼 없음.")
        print("   SHADOW_TRAIL=1 설정 후 collect_ml_data.py 재수집 필요:")
        print("     $env:SHADOW_TRAIL = \"1\"")
        print("     python collect_ml_data.py")
        sys.exit(1)

    # shadow가 기록된 trade만
    valid = df[df["shadow_pnl_r"].notna()].copy()
    print(f"\n  shadow 기록된 trade: {len(valid):,} / {len(df):,}")
    if len(valid) == 0:
        print("  ⚠️ shadow_pnl_r이 모두 비어있음. SHADOW_TRAIL=1로 수집했는지 확인")
        sys.exit(1)

    valid["pnl_off"] = valid["pnl_r"]
    valid["pnl_on"] = valid["shadow_pnl_r"]
    valid["pnl_off_c3"] = valid["pnl_off"].clip(-1, 3)
    valid["pnl_on_c3"] = valid["pnl_on"].clip(-1, 3)
    valid["delta"] = valid["pnl_on"] - valid["pnl_off"]

    # ── 전체 비교
    print(f"\n{'='*78}")
    print(f"  📊 Trail OFF vs ON (같은 trade, 같은 진입)")
    print(f"{'='*78}")
    print(f"  {'':30}{'OFF':>12}{'ON':>12}{'Δ':>12}")
    print(f"  {'-'*30}{'-'*12}{'-'*12}{'-'*12}")
    print(f"  {'EV (raw)':30}{valid['pnl_off'].mean():>+12.4f}{valid['pnl_on'].mean():>+12.4f}"
          f"{valid['pnl_on'].mean()-valid['pnl_off'].mean():>+12.4f}")
    print(f"  {'EV (clip3)':30}{valid['pnl_off_c3'].mean():>+12.4f}{valid['pnl_on_c3'].mean():>+12.4f}"
          f"{valid['pnl_on_c3'].mean()-valid['pnl_off_c3'].mean():>+12.4f}")
    wr_off = 100*(valid['pnl_off']>0).mean()
    wr_on = 100*(valid['pnl_on']>0).mean()
    print(f"  {'WR (pnl>0)':30}{wr_off:>11.2f}%{wr_on:>11.2f}%{wr_on-wr_off:>+11.2f}p")
    sh_off = valid['pnl_off_c3'].mean()/valid['pnl_off_c3'].std() if valid['pnl_off_c3'].std()>0 else 0
    sh_on = valid['pnl_on_c3'].mean()/valid['pnl_on_c3'].std() if valid['pnl_on_c3'].std()>0 else 0
    print(f"  {'Sharpe (c3)':30}{sh_off:>+12.3f}{sh_on:>+12.3f}{sh_on-sh_off:>+12.3f}")

    # ── 핵심 결론
    delta_ev = valid['pnl_on'].mean() - valid['pnl_off'].mean()
    delta_c3 = valid['pnl_on_c3'].mean() - valid['pnl_off_c3'].mean()
    print(f"\n  ★ Trail ON의 영향: EV {delta_ev:+.4f}R (raw), {delta_c3:+.4f}R (clip3)")
    if delta_c3 > 0.03:
        print(f"    🟢 Trail ON이 더 좋음 → 실거래 trail 유지가 유리")
    elif delta_c3 < -0.03:
        print(f"    🔴 Trail ON이 edge를 죽임 → trail 완화/제거 검토")
        print(f"       (단, trail 없으면 리스크 관리 부담 ↑)")
    else:
        print(f"    ⚪ 영향 미미 → trail은 edge와 무관, 다른 곳이 문제")

    # ── trail이 죽인 winner / 살린 케이스
    print(f"\n{'─'*78}\n  🎯 Trail이 trade를 바꾼 영향 분류\n{'─'*78}")
    killed = valid[valid["delta"] < -0.1]
    saved = valid[valid["delta"] > 0.1]
    same = valid[valid["delta"].abs() <= 0.1]
    print(f"  🔴 trail이 깎음 (Δ<-0.1R): {len(killed):>5}건  "
          f"ΣΔ={killed['delta'].sum():>+8.1f}R  평균={killed['delta'].mean():+.3f}R")
    print(f"  🟢 trail이 살림 (Δ>+0.1R): {len(saved):>5}건  "
          f"ΣΔ={saved['delta'].sum():>+8.1f}R  평균={saved['delta'].mean():+.3f}R")
    print(f"  ⚪ 동일       (|Δ|≤0.1R): {len(same):>5}건")
    print(f"\n  순효과: {killed['delta'].sum()+saved['delta'].sum():+.1f}R "
          f"({(killed['delta'].sum()+saved['delta'].sum())/len(valid):+.4f}R/trade)")

    # 큰 winner가 trail에 깎인 케이스
    print(f"\n  🔍 trail에 깎인 큰 winner (off>=3R인데 on에서 작아짐):")
    big = killed[killed["pnl_off"] >= 3.0].sort_values("delta").head(10)
    if len(big) > 0:
        print(f"     {'심볼':<6}{'dir':<6}{'off':>8}{'on':>8}{'Δ':>8}{'peak_r':>9}{'shadow_exit':>14}")
        for _, r in big.iterrows():
            print(f"     {r['symbol']:<6}{r['direction']:<6}{r['pnl_off']:>+7.2f}R"
                  f"{r['pnl_on']:>+7.2f}R{r['delta']:>+7.2f}R{r.get('peak_r',0):>+8.2f}R"
                  f"{str(r['shadow_exit_reason']):>14}")
        print(f"     ... 총 {len(killed[killed['pnl_off']>=3.0])}건 (off>=3R 깎임)")
    else:
        print(f"     없음")

    # ── 좋은 setup 한정
    print(f"\n{'─'*78}\n  📋 좋은 setup (best_tp_score>=0.5)에서 trail 영향\n{'─'*78}")
    if "best_tp_score" in valid.columns:
        f = valid[valid["best_tp_score"] >= 0.5]
        if len(f) > 50:
            print(f"  n={len(f)}")
            print(f"  OFF EV_c3 = {f['pnl_off_c3'].mean():+.4f}R")
            print(f"  ON  EV_c3 = {f['pnl_on_c3'].mean():+.4f}R")
            print(f"  Δ        = {f['pnl_on_c3'].mean()-f['pnl_off_c3'].mean():+.4f}R")

    # ── 방향별
    print(f"\n{'─'*78}\n  📋 방향별 trail 영향\n{'─'*78}")
    for d in ["long", "short"]:
        sd = valid[valid["direction"] == d]
        if len(sd) < 20: continue
        print(f"  {d.upper():<6}: OFF EV_c3={sd['pnl_off_c3'].mean():+.4f}R → "
              f"ON EV_c3={sd['pnl_on_c3'].mean():+.4f}R  "
              f"Δ={sd['pnl_on_c3'].mean()-sd['pnl_off_c3'].mean():+.4f}R")

    # ── shadow exit reason 분포
    print(f"\n{'─'*78}\n  🚪 Shadow (trail ON) exit reason 분포\n{'─'*78}")
    print(valid["shadow_exit_reason"].value_counts().to_string())

    print(f"\n{'='*78}")


if __name__ == "__main__":
    main()
