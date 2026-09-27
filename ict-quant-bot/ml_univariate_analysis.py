"""
ml_univariate_analysis.py — 피처별 단변량 예측력 진단 (Long/Short 분리)
═══════════════════════════════════════════════════════════
각 피처가 단독으로 WIN/LOSS를 구분하는 능력을 측정합니다.
Long과 Short을 분리하여 분석합니다.

출력:
  1) 피처별 AUC (0.5 = 랜덤, 1.0 = 완벽)
  2) 피처별 WIN vs LOSS 분포 차이 (KS statistic)
  3) 상위 피처의 분포 비교 (평균, 중앙값)
  4) 상관관계가 높은 피처 쌍 (다중공선성)
  5) Long vs Short 피처 예측력 비교

사용법:
    python ml_univariate_analysis.py
"""
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import glob
import numpy as np
import pandas as pd

# ── 피처 목록
from ml_data import get_feature_columns
from ml_preprocess import load_and_preprocess


def univariate_auc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """각 피처의 단변량 AUC 계산."""
    from sklearn.metrics import roc_auc_score
    results = []
    for col in X.columns:
        vals = X[col].values
        # 분산이 0이면 (상수) → AUC = 0.5
        if np.std(vals) < 1e-10:
            results.append({"feature": col, "auc": 0.5, "note": "constant"})
            continue
        try:
            auc = roc_auc_score(y, vals)
            # AUC < 0.5이면 반전 (음의 상관)
            auc_adj = max(auc, 1 - auc)
            direction = "+" if auc >= 0.5 else "-"
            results.append({
                "feature": col, "auc": round(auc_adj, 4),
                "raw_auc": round(auc, 4), "direction": direction
            })
        except ValueError:
            results.append({"feature": col, "auc": 0.5, "note": "error"})
    
    df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


def ks_test(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """WIN vs LOSS 분포 차이 (KS statistic)."""
    from scipy.stats import ks_2samp
    results = []
    win_mask = y == 1
    for col in X.columns:
        win_vals = X.loc[win_mask, col].values
        loss_vals = X.loc[~win_mask, col].values
        if len(win_vals) < 5 or len(loss_vals) < 5:
            results.append({"feature": col, "ks_stat": 0.0, "p_value": 1.0})
            continue
        stat, pval = ks_2samp(win_vals, loss_vals)
        results.append({
            "feature": col,
            "ks_stat": round(stat, 4),
            "p_value": round(pval, 6),
        })
    return pd.DataFrame(results).sort_values("ks_stat", ascending=False).reset_index(drop=True)


def distribution_comparison(X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """상위 피처의 WIN vs LOSS 분포 비교."""
    win_mask = y == 1
    results = []
    for col in X.columns:
        win_vals = X.loc[win_mask, col]
        loss_vals = X.loc[~win_mask, col]
        results.append({
            "feature": col,
            "win_mean": round(win_vals.mean(), 4),
            "loss_mean": round(loss_vals.mean(), 4),
            "diff_mean": round(win_vals.mean() - loss_vals.mean(), 4),
            "win_median": round(win_vals.median(), 4),
            "loss_median": round(loss_vals.median(), 4),
            "win_std": round(win_vals.std(), 4),
            "loss_std": round(loss_vals.std(), 4),
        })
    return pd.DataFrame(results)


def high_correlation_pairs(X: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """상관계수가 threshold 이상인 피처 쌍."""
    corr = X.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(float(corr.iloc[i, j]), 4),
                })
    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    return pd.DataFrame(pairs).sort_values("correlation", ascending=False).reset_index(drop=True)


def analyze_direction(X: pd.DataFrame, y: pd.Series, direction: str,
                      n_samples: int, baseline_wr: float):
    """한 방향(Long/Short)에 대한 전체 단변량 분석 실행."""

    print(f"\n{'━' * 70}")
    print(f"  📊 {direction.upper()} 단변량 분석  ({n_samples}건, 진입OK율(non-SL)={baseline_wr*100:.1f}%)")
    print(f"{'━' * 70}")

    # ── 1. 단변량 AUC
    print(f"\n  1️⃣  피처별 단변량 AUC")
    print(f"  {'─'*65}")
    auc_df = univariate_auc(X, y)
    for _, row in auc_df.iterrows():
        delta = row["auc"] - 0.5
        bar = "█" * int(delta * 100)
        sig = "✅" if delta >= 0.03 else ("⚠️" if delta >= 0.015 else "  ")
        d = row.get("direction", "")
        print(f"  {sig} {row['feature']:35s}  AUC={row['auc']:.4f}  (Δ={delta:+.4f} {d})  {bar}")

    # ── 2. KS Test
    print(f"\n  2️⃣  WIN vs LOSS 분포 차이 (KS test, Top 15)")
    print(f"  {'─'*65}")
    ks_df = ks_test(X, y)
    for _, row in ks_df.head(15).iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
        print(f"  {row['feature']:35s}  KS={row['ks_stat']:.4f}  p={row['p_value']:.6f}  {sig}")

    # ── 3. 분포 비교 (상위 AUC 피처)
    top_features = auc_df.head(10)["feature"].tolist()
    dist_df = distribution_comparison(X, y)
    dist_top = dist_df[dist_df["feature"].isin(top_features)].copy()
    dist_top = dist_top.set_index("feature").loc[top_features].reset_index()

    print(f"\n  3️⃣  상위 10 피처: WIN vs LOSS 평균 비교")
    print(f"  {'─'*65}")
    print(f"  {'Feature':35s} {'WIN mean':>10} {'LOSS mean':>10} {'Diff':>10}")
    print(f"  {'─'*67}")
    for _, row in dist_top.iterrows():
        print(f"  {row['feature']:35s} {row['win_mean']:>10.4f} {row['loss_mean']:>10.4f} {row['diff_mean']:>+10.4f}")

    # ── 4. 요약
    useful = auc_df[auc_df["auc"] >= 0.53]
    marginal = auc_df[(auc_df["auc"] >= 0.515) & (auc_df["auc"] < 0.53)]
    useless = auc_df[auc_df["auc"] < 0.515]

    print(f"\n  📋 {direction.upper()} 요약:")
    print(f"     유용 (AUC ≥ 0.53)   : {len(useful)}개  {useful['feature'].tolist()[:8]}")
    print(f"     미약 (0.515~0.53)   : {len(marginal)}개")
    print(f"     무의미 (AUC < 0.515) : {len(useless)}개")

    return auc_df


def main():
    # fib_tail_score 있는 최신 collect 사용 (with_tp/augmented는 fib_tail 없어 제외)
    csvs = sorted(f for f in glob.glob("ml_train_*.csv")
                  if "with_tp" not in f and "augmented" not in f and "_trail" not in f)
    if not csvs:
        print("❌ ml_train_*.csv 파일이 없습니다.")
        sys.exit(1)

    print("═" * 70)
    print("  📊 단변량 피처 예측력 분석 — SL 회피 라벨 (entry 모델 일관)")
    print("═" * 70)
    print(f"  📂 파일: {csvs}")

    X, y, df_full = load_and_preprocess(csvs, verbose=True)

    # ── SL 회피 라벨로 y 교체 (entry 모델과 동일: y=1=SL아님/진입OK, y=0=exit_reason"sl")
    if "exit_reason" not in df_full.columns:
        print("  ❌ exit_reason 컬럼 없음 — SL 회피 라벨 불가")
        sys.exit(1)
    y = (df_full["exit_reason"] != "sl").astype(int).reset_index(drop=True)
    print(f"\n  🎯 SL 회피 라벨: 진입OK={int(y.sum())} ({y.mean()*100:.1f}%)  "
          f"SL={int(len(y)-y.sum())} ({(1-y.mean())*100:.1f}%)")

    # ── direction 컬럼 확인
    if "direction" not in df_full.columns:
        print("  ❌ 'direction' 컬럼 없음 — 분리 분석 불가")
        sys.exit(1)

    # ── 다중공선성 (전체 데이터, 방향 무관)
    print(f"\n{'─'*70}")
    print(f"  🔗 다중공선성 (|상관계수| ≥ 0.85) — 전체 데이터")
    print(f"{'─'*70}")
    corr_df = high_correlation_pairs(X, threshold=0.85)
    if corr_df.empty:
        print("  ✅ 없음")
    else:
        for _, row in corr_df.iterrows():
            print(f"  {row['feature_1']:30s}  ↔  {row['feature_2']:30s}  r={row['correlation']:.4f}")

    # ── Long/Short 분리 분석
    feature_cols = get_feature_columns()
    auc_results = {}

    # ── 0. 합병(MERGED) 분석 먼저 — 분리 편차 크면 합병이 안정적
    print(f"\n\n{'━' * 70}")
    print(f"  0️⃣  MERGED (Long+Short 통합) 단변량 — 표본 2배, 안정적")
    print(f"{'━' * 70}")
    auc_merged = analyze_direction(X.reset_index(drop=True), y.reset_index(drop=True),
                                   "merged", len(X), y.mean())
    auc_results["merged"] = auc_merged

    for direction in ["long", "short"]:
        mask = df_full["direction"] == direction
        X_dir = X.loc[mask].reset_index(drop=True)
        y_dir = y.loc[mask].reset_index(drop=True)

        n = len(X_dir)
        wr = y_dir.mean()

        auc_df = analyze_direction(X_dir, y_dir, direction, n, wr)
        auc_results[direction] = auc_df

    # ── 5. Long vs Short 비교 테이블
    print(f"\n\n{'━' * 70}")
    print(f"  5️⃣  Long vs Short 피처 예측력 비교")
    print(f"{'━' * 70}")

    long_auc = auc_results["long"].set_index("feature")["auc"]
    short_auc = auc_results["short"].set_index("feature")["auc"]
    merged_auc = auc_results["merged"].set_index("feature")["auc"]

    compare = pd.DataFrame({
        "long_auc": long_auc,
        "short_auc": short_auc,
    }).fillna(0.5)
    compare["long_delta"] = compare["long_auc"] - 0.5
    compare["short_delta"] = compare["short_auc"] - 0.5
    compare["diff"] = compare["short_auc"] - compare["long_auc"]
    compare["max_auc"] = compare[["long_auc", "short_auc"]].max(axis=1)
    compare = compare.sort_values("max_auc", ascending=False).reset_index()

    print(f"  {'Feature':30s} {'Long AUC':>10} {'Short AUC':>10} {'Diff':>8}  Note")
    print(f"  {'─'*75}")
    for _, row in compare.iterrows():
        # 어느 방향에서 더 유용한지 표시
        note = ""
        if row["long_auc"] >= 0.53 and row["short_auc"] < 0.53:
            note = "Long only"
        elif row["short_auc"] >= 0.53 and row["long_auc"] < 0.53:
            note = "Short only"
        elif row["long_auc"] >= 0.53 and row["short_auc"] >= 0.53:
            note = "Both ✅"
        elif row["long_auc"] < 0.515 and row["short_auc"] < 0.515:
            note = "×"

        l_mark = "✅" if row["long_auc"] >= 0.53 else ("⚠" if row["long_auc"] >= 0.515 else " ")
        s_mark = "✅" if row["short_auc"] >= 0.53 else ("⚠" if row["short_auc"] >= 0.515 else " ")

        print(f"  {row['feature']:30s} {l_mark}{row['long_auc']:.4f}  {s_mark}{row['short_auc']:.4f}  {row['diff']:>+.4f}  {note}")

    # ── 6. 방향별 추천 피처
    print(f"\n{'━' * 70}")
    print(f"  📋 방향별 피처 추천 요약")
    print(f"{'━' * 70}")

    for direction in ["long", "short"]:
        auc_df = auc_results[direction]
        useful = auc_df[auc_df["auc"] >= 0.53]["feature"].tolist()
        marginal = auc_df[(auc_df["auc"] >= 0.515) & (auc_df["auc"] < 0.53)]["feature"].tolist()
        useless = auc_df[auc_df["auc"] < 0.515]["feature"].tolist()
        print(f"\n  {direction.upper()}:")
        print(f"     유용 ({len(useful)}개): {useful}")
        print(f"     미약 ({len(marginal)}개): {marginal}")
        print(f"     무의미 ({len(useless)}개): {len(useless)}개 생략")

    # 3중 무의미한 피처 (long/short/merged 모두 < 0.515) — 가장 안전한 제거 후보
    both_useless = []
    for feat in feature_cols:
        l = long_auc.get(feat, 0.5)
        s = short_auc.get(feat, 0.5)
        m = merged_auc.get(feat, 0.5)
        if l < 0.515 and s < 0.515 and m < 0.515:
            both_useless.append(feat)

    if both_useless:
        print(f"\n  🗑️  3중 무의미 (long/short/merged 모두 AUC<0.515) — 안전 제거 후보 {len(both_useless)}개:")
        for f in both_useless:
            l = long_auc.get(f, 0.5); s = short_auc.get(f, 0.5); m = merged_auc.get(f, 0.5)
            print(f"     {f:30s}  L={l:.4f}  S={s:.4f}  M={m:.4f}")

    # merged 기준 유용/미약 (합병 모델 채택 시 참고)
    print(f"\n  📊 MERGED 기준 (합병 모델 채택 시):")
    mu = auc_results["merged"]
    print(f"     유용(≥0.53): {mu[mu['auc']>=0.53]['feature'].tolist()}")
    print(f"     미약(0.515~0.53): {mu[(mu['auc']>=0.515)&(mu['auc']<0.53)]['feature'].tolist()}")

    print("═" * 70)


if __name__ == "__main__":
    main()
