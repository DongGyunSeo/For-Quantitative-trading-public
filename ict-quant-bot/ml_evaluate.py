"""
ml_evaluate.py — ML 모델 평가 + 시각화 + 과적합 진단
═══════════════════════════════════════════════════════════
사용법:
    from ml_evaluate import evaluate_model
    evaluate_model(
        csv_paths=["ml_train_btc.csv", "ml_train_eth.csv", "ml_train_sol.csv"],
        model_path="ml_entry_model.json"
    )
"""
import numpy as np
import pandas as pd
from typing import List, Union
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # GUI 없이 저장용
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve,
        classification_report, confusion_matrix,
        precision_score, recall_score
    )
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml_preprocess import load_and_preprocess, get_ml_features


# ──────────────────────────────────────────────────────────
# 메인 평가 함수
# ──────────────────────────────────────────────────────────

def evaluate_model(
    csv_paths: Union[str, List[str]],
    model_path: str = "entry_model.json",
    save_dir: str = "ml_results",
    threshold: float = None,
) -> dict:
    """
    학습된 모델의 종합 평가.

    출력:
    1. CV 성능 + 과적합 진단
    2. Feature Importance (bar chart)
    3. ROC Curve
    4. Precision-Recall Curve
    5. Calibration Plot (확률 보정)
    6. Threshold별 거래 시뮬레이션
    7. 시간대별 / 레벨별 성능 분석
    """
    if not HAS_XGB or not HAS_SKLEARN:
        raise ImportError("xgboost, scikit-learn 필요")

    # 결과 저장 디렉토리
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # ── 1. 데이터 + 모델 로드
    X, y, df = load_and_preprocess(csv_paths, verbose=True)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # 메타데이터
    import json
    import pickle
    meta_path = model_path.replace(".json", "_meta.json")
    meta = {}
    if Path(meta_path).exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if threshold is None:
        threshold = meta.get("threshold", 0.40)

    temperature = meta.get("temperature", 0.75)

    # ── 2. 전체 예측
    y_raw = model.predict_proba(X)[:, 1]
    
    # Isotonic Calibration 로드
    cal_path = model_path.replace(".json", "_calibrator.pkl")
    calibrator = None
    if Path(cal_path).exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
            
    if calibrator is not None:
        y_cal = calibrator.predict(y_raw)
    else:
        y_cal = y_raw
        
    # Temperature Scaling 적용
    y_cal = np.clip(y_cal, 1e-7, 1 - 1e-7)
    logits = np.log(y_cal / (1 - y_cal))
    y_proba = 1.0 / (1.0 + np.exp(-logits / temperature))
    
    y_pred = (y_proba >= threshold).astype(int)

    # ── 3. TimeSeriesSplit 재현 (train vs val AUC 비교 = 과적합 진단)
    print("\n" + "=" * 60)
    print("  📊 과적합 진단 (Train vs Val AUC)")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=5)
    train_aucs, val_aucs = [], []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, min_child_weight=10,
            gamma=1.0, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0, learning_rate=0.05,
            random_state=42, verbosity=0, use_label_encoder=False,
            early_stopping_rounds=30,
            scale_pos_weight=meta.get("scale_pos_weight", 1.0),
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        tr_proba = fold_model.predict_proba(X_tr)[:, 1]
        vl_proba = fold_model.predict_proba(X_vl)[:, 1]

        try:
            tr_auc = roc_auc_score(y_tr, tr_proba)
            vl_auc = roc_auc_score(y_vl, vl_proba)
        except ValueError:
            tr_auc = vl_auc = 0.5

        train_aucs.append(tr_auc)
        val_aucs.append(vl_auc)

        gap = tr_auc - vl_auc
        status = "✅" if gap < 0.10 else "⚠️" if gap < 0.15 else "❌"
        print(f"  Fold {fold_i+1}: Train AUC={tr_auc:.3f}  Val AUC={vl_auc:.3f}  "
              f"Gap={gap:.3f} {status}")

    avg_gap = np.mean(train_aucs) - np.mean(val_aucs)
    print(f"\n  {'─'*50}")
    print(f"  평균 Gap: {avg_gap:.3f}", end="  ")
    if avg_gap < 0.10:
        print("✅ 과적합 없음")
    elif avg_gap < 0.15:
        print("⚠️ 경미한 과적합 — 모니터링 필요")
    else:
        print("❌ 과적합 의심 — 파라미터 조정 필요")

    # ── 4. Classification Report
    print(f"\n{'='*60}")
    print("  📋 Classification Report (전체 데이터, threshold=0.5)")
    print("=" * 60)
    print(classification_report(y, y_pred, target_names=["LOSS", "WIN"]))

    # ── 5. Threshold별 시뮬레이션
    print(f"{'='*60}")
    print("  🎯 Threshold별 거래 시뮬레이션")
    print("=" * 60)

    th_results = _threshold_simulation(y, y_proba, df)

    # ── 6. 시각화
    if HAS_PLT:
        _plot_all(y, y_proba, model, X, df, threshold, save_path, th_results)
        print(f"\n  📊 차트 저장 위치: {save_path}/")

    # ── 7. 레벨별 / 시간대별 분석
    _segment_analysis(df, y_proba, threshold)

    return {
        "y_proba": y_proba,
        "train_aucs": train_aucs,
        "val_aucs": val_aucs,
        "avg_gap": avg_gap,
        "threshold_results": th_results,
    }


# ──────────────────────────────────────────────────────────
# Threshold 시뮬레이션
# ──────────────────────────────────────────────────────────

def _threshold_simulation(y, y_proba, df) -> list:
    """threshold별 거래 필터링 시뮬레이션."""
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = []

    print(f"  {'Th':>5} {'거래수':>6} {'WIN':>5} {'LOSS':>5} "
          f"{'승률':>7} {'총PnL':>10} {'PF':>7} {'Avg R':>7}")
    print(f"  {'─'*60}")

    has_pnl = "pnl_usdt" in df.columns
    has_pnl_r = "pnl_r" in df.columns

    for th in thresholds:
        mask = y_proba >= th
        n = mask.sum()
        if n == 0:
            continue

        y_f = y.values[mask]
        wins = int(y_f.sum())
        losses = n - wins
        wr = wins / n * 100

        total_pnl = df.loc[mask, "pnl_usdt"].sum() if has_pnl else 0
        avg_r = df.loc[mask, "pnl_r"].mean() if has_pnl_r else 0

        # Profit Factor
        gross_win = df.loc[mask & (df["label"] == 1), "pnl_usdt"].sum() if has_pnl else 0
        gross_loss = abs(df.loc[mask & (df["label"] == 0), "pnl_usdt"].sum()) if has_pnl else 1
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        print(f"  {th:>5.2f} {n:>6} {wins:>5} {losses:>5} "
              f"{wr:>6.1f}% {total_pnl:>10.1f} {pf:>7.2f} {avg_r:>7.2f}")

        results.append({
            "threshold": th, "n_trades": n, "wins": wins, "losses": losses,
            "win_rate": round(wr, 1), "total_pnl": round(total_pnl, 2),
            "profit_factor": round(pf, 2), "avg_r": round(avg_r, 2),
        })

    # 기준선 (필터 없음)
    total = len(y)
    base_wr = y.mean() * 100
    base_pnl = df["pnl_usdt"].sum() if has_pnl else 0
    print(f"  {'─'*60}")
    print(f"  {'전체':>5} {total:>6} {int(y.sum()):>5} "
          f"{total-int(y.sum()):>5} {base_wr:>6.1f}% {base_pnl:>10.1f}")

    return results


# ──────────────────────────────────────────────────────────
# 세그먼트 분석
# ──────────────────────────────────────────────────────────

def _segment_analysis(df, y_proba, threshold):
    """레벨별, 시간대별, 방향별 성능 분석."""
    df = df.copy()
    df["p_win"] = y_proba
    df["ml_pass"] = y_proba >= threshold

    print(f"\n{'='*60}")
    print("  📊 세그먼트별 분석")
    print("=" * 60)

    # ── 레벨별 (one-hot에서 역추출)
    lv_cols = [c for c in df.columns if c.startswith("lv_")]
    if lv_cols:
        print(f"\n  [레벨 타입별]")
        print(f"  {'레벨':>16} {'전체':>5} {'WR_전체':>8} {'ML통과':>6} {'WR_ML':>8} {'개선':>6}")
        print(f"  {'─'*55}")

        for lv in lv_cols:
            mask = df[lv] == 1
            if mask.sum() == 0:
                continue
            wr_all = df.loc[mask, "label"].mean() * 100
            ml_mask = mask & df["ml_pass"]
            n_ml = ml_mask.sum()
            wr_ml = df.loc[ml_mask, "label"].mean() * 100 if n_ml > 0 else 0

            diff = wr_ml - wr_all
            icon = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            print(f"  {lv:>16} {mask.sum():>5} {wr_all:>7.1f}% "
                  f"{n_ml:>6} {wr_ml:>7.1f}% {diff:>+5.1f}% {icon}")

    # ── 방향별
    print(f"\n  [방향별]")
    for dir_val, dir_name in [(1, "LONG"), (0, "SHORT")]:
        mask = df["is_long"] == dir_val
        if mask.sum() == 0:
            continue
        wr_all = df.loc[mask, "label"].mean() * 100
        ml_mask = mask & df["ml_pass"]
        n_ml = ml_mask.sum()
        wr_ml = df.loc[ml_mask, "label"].mean() * 100 if n_ml > 0 else 0
        print(f"  {dir_name:>8}: 전체 {mask.sum()}건 WR={wr_all:.1f}%  "
              f"→ ML통과 {n_ml}건 WR={wr_ml:.1f}%")

    # ── HTF alignment별
    print(f"\n  [HTF Alignment별]")
    for aligned_val, label in [(1, "HTF 일치"), (0, "HTF 불일치")]:
        mask = df["htf_4h_aligned"] == aligned_val
        if mask.sum() == 0:
            continue
        wr_all = df.loc[mask, "label"].mean() * 100
        ml_mask = mask & df["ml_pass"]
        n_ml = ml_mask.sum()
        wr_ml = df.loc[ml_mask, "label"].mean() * 100 if n_ml > 0 else 0
        print(f"  {label:>12}: 전체 {mask.sum()}건 WR={wr_all:.1f}%  "
              f"→ ML통과 {n_ml}건 WR={wr_ml:.1f}%")


# ──────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────

def _plot_all(y, y_proba, model, X, df, threshold, save_path, th_results):
    """6개 차트를 2×3 그리드에 그려서 저장."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ML Entry Filter — Evaluation Report", fontsize=14, fontweight="bold")

    # 1. ROC Curve
    ax = axes[0, 0]
    try:
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_val = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, "ROC 계산 불가", ha="center", va="center")

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    try:
        prec, rec, _ = precision_recall_curve(y, y_proba)
        ax.plot(rec, prec, color="#4CAF50", linewidth=2)
        ax.axhline(y.mean(), color="gray", linestyle="--", alpha=0.5,
                   label=f"Baseline (WR={y.mean()*100:.1f}%)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, "PR 계산 불가", ha="center", va="center")

    # 3. Feature Importance
    ax = axes[0, 2]
    feature_cols = get_ml_features()
    imp = model.feature_importances_
    fi = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:12]
    names = [f[0] for f in fi][::-1]
    vals = [f[1] for f in fi][::-1]
    colors = ["#FF9800" if v >= 0.05 else "#78909C" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_title("Feature Importance (Top 12)")
    ax.grid(alpha=0.3, axis="x")

    # 4. Calibration Plot
    ax = axes[1, 0]
    try:
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=8)
        ax.plot(prob_pred, prob_true, "o-", color="#9C27B0", linewidth=2, label="Model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.set_xlabel("Predicted P(win)")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title("Calibration Plot")
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, "보정 계산 불가", ha="center", va="center")

    # 5. P(win) 분포
    ax = axes[1, 1]
    ax.hist(y_proba[y == 1], bins=20, alpha=0.6, color="#4CAF50", label="WIN", density=True)
    ax.hist(y_proba[y == 0], bins=20, alpha=0.6, color="#F44336", label="LOSS", density=True)
    ax.axvline(threshold, color="orange", linestyle="--", linewidth=2,
               label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("P(win)")
    ax.set_ylabel("Density")
    ax.set_title("P(win) Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. Threshold vs Metrics
    ax = axes[1, 2]
    if th_results:
        ths = [r["threshold"] for r in th_results]
        wrs = [r["win_rate"] for r in th_results]
        nts = [r["n_trades"] for r in th_results]

        ax2 = ax.twinx()
        ax.plot(ths, wrs, "o-", color="#2196F3", linewidth=2, label="Win Rate %")
        ax2.bar(ths, nts, alpha=0.3, color="#FF9800", width=0.03, label="Trade Count")
        ax.axvline(threshold, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Win Rate %", color="#2196F3")
        ax2.set_ylabel("Trade Count", color="#FF9800")
        ax.set_title("Threshold Impact")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = save_path / "ml_evaluation_report.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  💾 평가 리포트 저장: {out_path}")


# ──────────────────────────────────────────────────────────
# SHAP 분석 (선택적)
# ──────────────────────────────────────────────────────────

def shap_analysis(
    csv_paths: Union[str, List[str]],
    model_path: str = "entry_model.json",
    save_dir: str = "ml_results",
    max_display: int = 15,
):
    """
    SHAP 값 분석 (pip install shap 필요).

    SHAP은 각 피처가 개별 예측에 기여한 정도를 보여줍니다.
    Feature Importance보다 더 정확한 피처 영향 분석을 제공합니다.
    """
    try:
        import shap
    except ImportError:
        print("❌ shap 미설치. 설치: pip install shap")
        return

    X, y, df = load_and_preprocess(csv_paths, verbose=False)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(str(save_path / "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 SHAP Summary 저장: {save_path / 'shap_summary.png'}")

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(str(save_path / "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 SHAP Bar 저장: {save_path / 'shap_bar.png'}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import glob
    import os

    # 보통 Entry 분석에는 ml_train_with_tp.csv를 사용, 없으면 전체 취합
    if os.path.exists("ml_train_with_tp.csv"):
        csvs = ["ml_train_with_tp.csv"]
    else:
        csvs = sorted(glob.glob("ml_train_*.csv"))
    model_path = "entry_model.json"

    if not csvs:
        print("❌ ml_train_*.csv 파일 없음")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"❌ 모델 파일 없음: {model_path}")
        print("   먼저 ml_model.py를 실행하세요.")
        sys.exit(1)

    print(f"🔍 CSV: {csvs}")
    print(f"🤖 모델: {model_path}\n")

    evaluate_model(csvs, model_path)
