"""
prepare_entry_data_notp.py — [임시] TP 모델 없이 ml_train_*.csv 단순 결합
═══════════════════════════════════════════════════════════════════
prepare_entry_data.py 의 TP 예측/조인 단계를 건너뛰고,
ml_train_*.csv 들을 그대로 concat 하여 ml_train_with_tp.csv 로 출력한다.

용도:
    RR1(=TP 1R 고정) 실험에선 TP selector가 무력화되고, best_tp_score는
    Entry 50피처에 포함되지 않으므로 TP 모델을 돌릴 필요가 없다.
    → tp_model.json / tp_train_*.csv 없이 Entry 학습 데이터를 만든다.

차이 (vs prepare_entry_data.py):
    - tp_model.json 로드/예측 없음
    - tp_train_*.csv 불필요
    - best_tp_score = 0.5 상수로만 채움 (스키마 호환용, 정보량 0, AUC≈0.500)
      → 이 컬럼은 어차피 get_feature_columns()의 50피처에 없어 학습에 미사용

사용법:
    python prepare_entry_data_notp.py
    # 그 다음 기존 그대로:
    # python train_entry.py --split
    # python ml_univariate_analysis.py
"""
import sys
import glob
import pandas as pd

# Windows 터미널 이모지 출력 보호
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# best_tp_score 상수값 (스키마 호환용 — 학습 미사용)
BEST_TP_SCORE_CONST = 0.5
OUTPUT_PATH = "ml_train_with_tp.csv"


def main():
    # ── ml_train_*.csv 탐색 (with_tp / shadow 파생본 제외)
    ml_csvs = sorted(glob.glob("ml_train_*.csv"))
    ml_csvs = [p for p in ml_csvs if "with_tp" not in p and "_shadow" not in p]

    if not ml_csvs:
        print("❌ ml_train_*.csv 파일을 찾을 수 없습니다.")
        print("   → collect_ml_data.py 로 백테스트를 먼저 실행하세요.")
        sys.exit(1)

    print("═" * 60)
    print("  🔗 [임시] TP 모델 없이 ml_train 단순 결합")
    print("═" * 60)
    print(f"  발견된 ML CSV: {ml_csvs}")
    print(f"  출력 파일    : {OUTPUT_PATH}")

    # ── 로드 + concat
    dfs = []
    for p in ml_csvs:
        d = pd.read_csv(p)
        dfs.append(d)
        print(f"     - {p}  ({len(d):,}행)")
    ml_all = pd.concat(dfs, ignore_index=True)

    # ── best_tp_score 상수 채움 (있으면 덮어씀)
    if "best_tp_score" in ml_all.columns:
        ml_all = ml_all.drop(columns=["best_tp_score"])
    ml_all["best_tp_score"] = BEST_TP_SCORE_CONST

    # ── 저장
    ml_all.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  ✅ 결합 완료: {len(ml_all):,}행, {len(ml_all.columns)}열")
    print(f"     best_tp_score = {BEST_TP_SCORE_CONST} 상수 (TP 모델 미사용)")
    if "symbol" in ml_all.columns:
        print(f"     심볼: {sorted(ml_all['symbol'].unique())}")
    print(f"  💾 저장: {OUTPUT_PATH}")
    print(f"\n  🚀 다음 단계:")
    print(f"     python train_entry.py --split")
    print("═" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  🔥 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
