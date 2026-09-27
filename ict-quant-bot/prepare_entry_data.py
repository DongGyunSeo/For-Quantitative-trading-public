"""
prepare_entry_data.py — TP 모델 예측값을 Entry 학습 데이터에 결합
═══════════════════════════════════════════════════════════════════
사용법:
    python prepare_entry_data.py

흐름:
    1) tp_model.json 로드
    2) tp_train_*.csv 에 대해 일괄 예측 (score_candidates)
    3) trade_id별 best_tp_score 추출 (= 가장 높은 점수의 TP 후보)
    4) ml_train_*.csv 에 trade_id로 LEFT JOIN
    5) ml_train_with_tp.csv 출력

※ tp_model.json 이 존재해야 합니다 (1차 TP 학습 완료 후 실행)
"""
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Windows 환경에서 터미널 출력 시 UnicodeEncodeError(이모지 등) 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def merge_tp_scores(
    tp_csv_paths: list,
    ml_csv_paths: list,
    tp_model_path: str = "tp_model.json",
    output_path: str = "ml_train_with_tp.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    TP 모델의 예측 스코어를 Entry 학습 데이터에 결합.

    Parameters
    ----------
    tp_csv_paths : list of str
        tp_train_*.csv 파일 경로 목록
    ml_csv_paths : list of str
        ml_train_*.csv 파일 경로 목록
    tp_model_path : str
        학습된 TP 모델 파일 경로
    output_path : str
        결합된 결과 CSV 출력 경로
    verbose : bool
        진행 상황 출력 여부

    Returns
    -------
    pd.DataFrame
        best_tp_score 컬럼이 추가된 Entry 학습 데이터
    """
    # ── 1. TP 모델 로드
    if not Path(tp_model_path).exists():
        raise FileNotFoundError(
            f"TP 모델을 찾을 수 없습니다: {tp_model_path}\n"
            "  → tp_selector.py 의 train_tp_model() 을 먼저 실행하세요."
        )

    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost 가 필요합니다: pip install xgboost")

    from tp_selector import get_tp_feature_columns

    model = xgb.XGBClassifier()
    model.load_model(tp_model_path)
    tp_feature_cols = get_tp_feature_columns()

    if verbose:
        print()
        print("═" * 60)
        print("  🔗 TP Score → Entry Data 결합 파이프라인")
        print("═" * 60)
        print(f"  TP 모델    : {tp_model_path}")
        print(f"  TP CSV     : {tp_csv_paths}")
        print(f"  ML CSV     : {ml_csv_paths}")
        print(f"  출력 파일  : {output_path}")

    # ── 2. TP 학습 데이터 로드 + 예측
    tp_dfs = [pd.read_csv(p) for p in tp_csv_paths]
    tp_all = pd.concat(tp_dfs, ignore_index=True)

    if verbose:
        print(f"\n  📂 TP 데이터: {len(tp_all):,}행 (후보 레코드)")

    # 누락 피처 0으로 채우기
    for col in tp_feature_cols:
        if col not in tp_all.columns:
            tp_all[col] = 0

    X_tp = tp_all[tp_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    tp_scores = model.predict_proba(X_tp)[:, 1].clip(0, 1)
    tp_all["tp_score"] = np.round(tp_scores, 4)

    if verbose:
        print(f"  🤖 TP 예측 완료: score 범위 [{tp_scores.min():.3f}, {tp_scores.max():.3f}]")

    # ── 3. trade_id별 best_tp_score 추출
    if "trade_id" not in tp_all.columns:
        raise ValueError("tp_train CSV에 trade_id 컬럼이 없습니다.")

    best_tp = (
        tp_all.groupby("trade_id")["tp_score"]
        .max()
        .reset_index()
        .rename(columns={"tp_score": "best_tp_score"})
    )

    if verbose:
        print(f"  📊 trade_id별 best_tp_score: {len(best_tp):,}건")
        print(f"     분포: mean={best_tp['best_tp_score'].mean():.3f}  "
              f"std={best_tp['best_tp_score'].std():.3f}  "
              f"median={best_tp['best_tp_score'].median():.3f}")

    # ── 4. ml_train 로드 + LEFT JOIN
    ml_dfs = [pd.read_csv(p) for p in ml_csv_paths]
    ml_all = pd.concat(ml_dfs, ignore_index=True)

    if verbose:
        print(f"\n  📂 ML 데이터: {len(ml_all):,}행 (Entry 학습 레코드)")

    if "trade_id" not in ml_all.columns:
        raise ValueError("ml_train CSV에 trade_id 컬럼이 없습니다.")

    # 기존 best_tp_score 컬럼이 있으면 제거 (재실행 대비)
    if "best_tp_score" in ml_all.columns:
        ml_all = ml_all.drop(columns=["best_tp_score"])

    # trade_id 형식 통일:
    #   TP CSV: trade_id = "BTC_2" (symbol + "_" + 숫자)
    #   ML CSV: trade_id = 2, symbol = "BTC" (별도 컬럼)
    # → ML CSV에 복합키를 생성하여 TP CSV와 매칭
    best_tp["trade_id"] = best_tp["trade_id"].astype(str)

    if "symbol" in ml_all.columns:
        # ML CSV: symbol + "_" + trade_id → TP CSV 형식과 동일하게
        ml_all["_join_key"] = ml_all["symbol"].astype(str) + "_" + ml_all["trade_id"].astype(str)
    else:
        ml_all["_join_key"] = ml_all["trade_id"].astype(str)

    # LEFT JOIN: ml_train의 모든 행을 유지, 매칭되는 best_tp_score를 붙임
    best_tp = best_tp.rename(columns={"trade_id": "_join_key"})
    merged = ml_all.merge(best_tp, on="_join_key", how="left")
    merged = merged.drop(columns=["_join_key"])

    # JOIN 미매칭 건은 중앙값으로 대체 (보수적 대입)
    n_missing = merged["best_tp_score"].isna().sum()
    if n_missing > 0:
        median_score = best_tp["best_tp_score"].median()
        merged["best_tp_score"] = merged["best_tp_score"].fillna(median_score)
        if verbose:
            print(f"  ⚠️  JOIN 미매칭 {n_missing}건 → median({median_score:.3f})으로 대체")

    # ── 5. 저장
    merged.to_csv(output_path, index=False)

    if verbose:
        matched = len(merged) - n_missing
        print(f"\n  ✅ 결합 완료: {len(merged):,}행 (매칭 {matched:,}건)")
        print(f"  💾 저장: {output_path}")
        print(f"\n  🚀 다음 단계:")
        print(f"     python train_entry.py --csv {output_path}")
        print("═" * 60)

    return merged


def main():
    """자동 탐색 후 결합 실행."""
    tp_csvs = sorted(glob.glob("tp_train_*.csv"))
    ml_csvs = sorted(glob.glob("ml_train_*.csv"))

    # ml_train_with_tp.csv 는 ml_train 원본이 아니므로 제외
    ml_csvs = [p for p in ml_csvs if "with_tp" not in p]

    if not tp_csvs:
        print("❌ tp_train_*.csv 파일을 찾을 수 없습니다.")
        print("   → collect_ml_data.py 로 백테스트를 먼저 실행하세요.")
        sys.exit(1)

    if not ml_csvs:
        print("❌ ml_train_*.csv 파일을 찾을 수 없습니다.")
        print("   → collect_ml_data.py 로 백테스트를 먼저 실행하세요.")
        sys.exit(1)

    tp_model = "tp_model.json"
    if not Path(tp_model).exists():
        print(f"❌ TP 모델을 찾을 수 없습니다: {tp_model}")
        print("   → 먼저 TP 모델 학습을 실행하세요:")
        print("      python -c \"from tp_selector import train_tp_model; "
              "train_tp_model(tp_train_*.csv 경로 리스트)\"")
        sys.exit(1)

    print(f"🔍 발견된 TP CSV: {tp_csvs}")
    print(f"🔍 발견된 ML CSV: {ml_csvs}")

    merge_tp_scores(
        tp_csv_paths=tp_csvs,
        ml_csv_paths=ml_csvs,
        tp_model_path=tp_model,
        output_path="ml_train_with_tp.csv",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  🔥 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
