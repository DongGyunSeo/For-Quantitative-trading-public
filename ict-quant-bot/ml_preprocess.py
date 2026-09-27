"""
ml_preprocess.py — ML 학습용 데이터 전처리 파이프라인
═══════════════════════════════════════════════════════════
사용법:
    from ml_preprocess import load_and_preprocess

    # 단일 에셋
    X, y, df = load_and_preprocess("ml_train_btc.csv")

    # 멀티에셋
    X, y, df = load_and_preprocess(
        ["ml_train_btc.csv", "ml_train_eth.csv", "ml_train_sol.csv"]
    )
"""
import pandas as pd
from typing import List, Tuple, Union
from ml_data import get_feature_columns


# ──────────────────────────────────────────────────────────
# 피처 컬럼 정의
# ──────────────────────────────────────────────────────────

def get_ml_features() -> List[str]:
    """
    ML 모델에 실제 사용할 피처 목록.
    get_feature_columns()를 그대로 사용.
    (삭제 피처는 ml_data.py에서 이미 제거됨)
    """
    return get_feature_columns()


# ──────────────────────────────────────────────────────────
# 메인 전처리 함수
# ──────────────────────────────────────────────────────────

def load_and_preprocess(
    csv_paths: Union[str, List[str]],
    target_col: str = "label",
    sort_by_time: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    CSV 파일(들) → 전처리된 (X, y, df_full) 반환.

    Parameters
    ----------
    csv_paths : str 또는 List[str]
        ml_train_*.csv 파일 경로(들)
    target_col : str
        라벨 컬럼 이름 ("label" = 이진, "pnl_r" = 회귀)
    sort_by_time : bool
        시간순 정렬 (TimeSeriesSplit에 필수)
    verbose : bool
        진행 상황 출력

    Returns
    -------
    X : pd.DataFrame  — 피처 (학습 준비 완료)
    y : pd.Series      — 라벨
    df_full : pd.DataFrame — 원본 + 전처리 결과 (메타 정보 포함)
    """
    # ── 1. CSV 로드 + 합산
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if verbose:
            print(f"  📂 로드: {path}  ({len(df)}건)")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"  📊 합산: {len(df_all)}건 (심볼: {df_all['symbol'].unique().tolist()})")

    # ── 2. 시간순 정렬
    if sort_by_time and "entry_time" in df_all.columns:
        df_all["entry_time"] = pd.to_datetime(df_all["entry_time"])
        df_all = df_all.sort_values("entry_time").reset_index(drop=True)

    # ── 3. NaN 처리
    feature_cols = get_ml_features()

    # pd_pos: NaN → 중앙값 대체
    if "pd_pos" in df_all.columns:
        median_pd = df_all["pd_pos"].median()
        n_nan = df_all["pd_pos"].isna().sum()
        df_all["pd_pos"] = df_all["pd_pos"].fillna(median_pd)
        if verbose and n_nan > 0:
            print(f"  🔧 pd_pos NaN {n_nan}건 → median({median_pd:.2f})으로 대체")

    # 나머지 피처 NaN → 0으로 대체 (one-hot, binary 등)
    for col in feature_cols:
        if col in df_all.columns:
            n_nan = df_all[col].isna().sum()
            if n_nan > 0:
                df_all[col] = df_all[col].fillna(0)
                if verbose:
                    print(f"  🔧 {col} NaN {n_nan}건 → 0으로 대체")

    # ── 4. 피처/라벨 분리
    # 누락된 피처 컬럼 확인
    missing_cols = [c for c in feature_cols if c not in df_all.columns]
    if missing_cols:
        if verbose:
            print(f"  ⚠️  누락 피처 (0으로 생성): {missing_cols}")
        for c in missing_cols:
            df_all[c] = 0

    X = df_all[feature_cols].copy()
    y = df_all[target_col].copy()

    # ── 5. 데이터 품질 리포트
    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  📋 데이터 요약")
        print(f"  {'─'*50}")
        print(f"  총 샘플: {len(X)}")
        print(f"  피처 수: {len(feature_cols)}")
        print(f"  라벨 분포: WIN={int(y.sum())} ({y.mean()*100:.1f}%)  "
              f"LOSS={int(len(y)-y.sum())} ({(1-y.mean())*100:.1f}%)")
        print(f"  기간: {df_all['entry_time'].min()} ~ {df_all['entry_time'].max()}")
        if "symbol" in df_all.columns:
            for sym in df_all["symbol"].unique():
                n = (df_all["symbol"] == sym).sum()
                wr = df_all.loc[df_all["symbol"] == sym, "label"].mean() * 100
                print(f"    {sym}: {n}건  WR={wr:.1f}%")
        print(f"  {'─'*50}\n")

    return X, y, df_all


# ──────────────────────────────────────────────────────────
# 유틸: 학습/검증 분할 (시간 기반)
# ──────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_end_date: str = None,
    test_months: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    시간 기반 train/test 분할.

    Parameters
    ----------
    df : pd.DataFrame
        entry_time 컬럼이 있는 전체 데이터
    train_end_date : str
        학습 종료 날짜 (이 날짜 이전 = train, 이후 = test)
        None이면 test_months로 자동 계산
    test_months : int
        test에 사용할 최근 개월 수

    Returns
    -------
    df_train, df_test
    """
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    if train_end_date:
        cutoff = pd.Timestamp(train_end_date, tz="UTC")
    else:
        cutoff = df["entry_time"].max() - pd.DateOffset(months=test_months)

    # timezone 호환
    if df["entry_time"].dt.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    elif df["entry_time"].dt.tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    train = df[df["entry_time"] <= cutoff].reset_index(drop=True)
    test = df[df["entry_time"] > cutoff].reset_index(drop=True)

    print(f"  📅 Train: {len(train)}건  (~{cutoff.strftime('%Y-%m-%d')})")
    print(f"  📅 Test:  {len(test)}건  ({cutoff.strftime('%Y-%m-%d')}~)")

    return train, test


# ──────────────────────────────────────────────────────────
# 실행 예시
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import glob

    # CSV 파일 자동 탐색
    csvs = sorted(glob.glob("ml_train_*.csv"))
    if not csvs:
        print("❌ ml_train_*.csv 파일을 찾을 수 없습니다.")
        print("   먼저 백테스트를 실행해서 CSV를 생성하세요.")
        sys.exit(1)

    print(f"🔍 발견된 CSV: {csvs}\n")
    X, y, df = load_and_preprocess(csvs)
    print(f"✅ 전처리 완료: X.shape={X.shape}, y.shape={y.shape}")
