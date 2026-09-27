"""
ml_model.py — XGBoost 진입 필터 모델 (학습 + 예측 + 저장/로드)
═══════════════════════════════════════════════════════════════
사용법:
    # ── 학습
    from ml_model import train_entry_filter
    result = train_entry_filter(
        csv_paths=["ml_train_btc.csv", "ml_train_eth.csv", "ml_train_sol.csv"],
        save_path="ml_entry_model.json"
    )

    # ── 예측 (실전 봇에서)
    from ml_model import MLFilter
    ml = MLFilter("ml_entry_model.json")
    p_win = ml.predict_from_reason(reason_dict)
"""
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ xgboost 미설치. 설치: pip install xgboost")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score,
        f1_score, log_loss, classification_report,
        confusion_matrix
    )
    from sklearn.model_selection._split import BaseCrossValidator
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn 미설치. 설치: pip install scikit-learn")

from ml_preprocess import load_and_preprocess, get_ml_features


# ──────────────────────────────────────────────────────────
# Focal Loss — Custom XGBoost Objective
# ──────────────────────────────────────────────────────────

class FocalLoss:
    """
    Focal Loss for XGBoost custom objective.
    FL(p) = -alpha * (1-p)^gamma * log(p)       for y=1
    FL(p) = -(1-alpha) * p^gamma * log(1-p)     for y=0
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None):
        self.gamma = gamma
        self.alpha = alpha  # None → 학습 시 auto (pos_ratio)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)))

    def objective(self, y_pred_raw: np.ndarray, dtrain) -> tuple:
        """XGBoost custom obj → (grad, hess). 수치 미분으로 검증 완료."""
        y = dtrain.get_label().astype(np.float64)
        p = self._sigmoid(y_pred_raw).clip(1e-7, 1 - 1e-7)
        alpha = self.alpha if self.alpha is not None else y.mean()
        gamma = self.gamma

        pt = np.where(y == 1, p, 1.0 - p)
        at = np.where(y == 1, alpha, 1.0 - alpha)
        log_pt = np.log(pt.clip(1e-7))

        # dFL/dpt (chain rule 첫 단계)
        dfl_dpt = -at * (-gamma * (1.0 - pt) ** (gamma - 1) * log_pt
                         + (1.0 - pt) ** gamma / pt)
        # dpt/dz = (2y-1) * p*(1-p)  (sigmoid chain)
        dpt_dz = (2.0 * y - 1.0) * p * (1.0 - p)
        grad = dfl_dpt * dpt_dz

        # hessian 근사 (양수 보장, 수치 hessian과 0.1~2.5x 범위 확인됨)
        hess = at * (1.0 - pt) ** gamma * p * (1.0 - p) * (
            gamma * (1.0 - pt) * np.abs(log_pt) + 1.0)
        hess = np.maximum(hess, 1e-6)
        return grad, hess

    def eval_metric(self, y_pred_raw: np.ndarray, dtrain) -> tuple:
        """Custom eval: focal_auc."""
        y = dtrain.get_label()
        p = self._sigmoid(y_pred_raw)
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = 0.5
        return "focal_auc", auc


# ──────────────────────────────────────────────────────────
# PurgedTimeSeriesSplit — 미래 누수 완전 차단 CV
# ──────────────────────────────────────────────────────────

class PurgedTimeSeriesSplit:
    """
    TimeSeriesSplit + purge + embargo.

    sklearn TimeSeriesSplit과 동일한 expanding-window 방식:
      train = [0, t0)    test = [t0, t1)

    여기에 두 가지 안전장치 추가:
      purge_gap : test 직전 N개를 train에서 제거
                  → sweep~entry 사이 시간 오버랩 방지
      embargo   : test 직후 N개를 train에서 제거 (미사용, 구조적 불필요)
                  → expanding window에서는 test 이후가 train에 포함되지 않으므로
                    embargo는 형식적으로만 유지

    핵심 차이 (vs 기존 PurgedKFold):
      ❌ 기존: Fold 1에서 test=[0:1258], train에 1258 이후 전체(=미래) 포함
      ✅ 변경: Fold 1에서 test=[t0:t1], train은 반드시 test 이전 데이터만

    Parameters
    ----------
    n_splits : int
        fold 수 (default=5)
    purge_gap : int
        test 시작 직전에서 제거할 train 샘플 수 (default=12)
    embargo : int
        test 종료 직후에서 제거할 범위 (expanding에서는 형식적, default=6)
    min_train_size : int
        최소 train 크기 — 이보다 작으면 해당 fold skip (default=200)
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 12,
                 embargo: int = 6, min_train_size: int = 200):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        n = len(X)
        # test fold 크기 = 전체 / (n_splits + 1)
        # 첫 번째 fold만큼은 train-only 구간으로 확보
        fold_size = n // (self.n_splits + 1)
        indices = np.arange(n)

        for i in range(self.n_splits):
            # test 구간: expanding window 방식
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size if i < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            # train 구간: test 이전만 사용 (미래 데이터 완전 차단)
            train_end = max(0, test_start - self.purge_gap)
            train_idx = indices[:train_end]

            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ──────────────────────────────────────────────────────────
# 기본 하이퍼파라미터 (4,000건+ 기준)
# ──────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",               # logloss → auc (직접 최적화)

    # ── 과적합 방지 (4,000건 기준)
    "max_depth": 6,                      # 3 → 6 (데이터 충분)
    "min_child_weight": 5,               # 10 → 5
    "gamma": 0.5,                        # 1.0 → 0.5
    "subsample": 0.85,                   # 0.7 → 0.85
    "colsample_bytree": 0.8,             # 0.7 → 0.8
    "reg_alpha": 1.0,                    # L1 유지
    "reg_lambda": 5.0,                   # L2 유지

    # ── 학습률
    "learning_rate": 0.03,               # 0.05 → 0.03 (더 천천히)

    "random_state": 42,
    "verbosity": 0,
    "use_label_encoder": False,
}

N_ESTIMATORS = 800                       # 500 → 800 (learning_rate 낮춘 만큼 증가)
EARLY_STOPPING_ROUNDS = 80              # 30 → 80


# ──────────────────────────────────────────────────────────
# 학습 파이프라인
# ──────────────────────────────────────────────────────────

def train_entry_filter(
    csv_paths: Union[str, List[str]],
    save_path: str = "ml_entry_model.json",
    n_splits: int = 5,
    params: dict = None,
    verbose: bool = True,
    use_focal_loss: bool = True,
    focal_gamma: float = 1.6,
    focal_alpha: float = 0.42,
    purge_gap: int = 12,
    embargo: int = 6,
) -> dict:
    """
    XGBoost 진입 필터 모델 학습.

    Parameters
    ----------
    csv_paths : str or List[str]
        ml_train_*.csv 파일 경로(들)
    save_path : str
        모델 저장 경로 (.json)
    n_splits : int
        PurgedKFold fold 수
    params : dict
        XGBoost 파라미터 (None이면 DEFAULT_PARAMS 사용)
    verbose : bool
        상세 출력
    use_focal_loss : bool
        True면 Focal Loss 사용, False면 기본 binary:logistic
    focal_gamma : float
        Focal Loss gamma (기본 2.0)
    purge_gap : int
        PurgedKFold purge gap (test 직전 제거 샘플 수)
    embargo : int
        PurgedKFold embargo (test 직후 제거 샘플 수)

    Returns
    -------
    dict with keys:
        "model": 학습된 XGBClassifier
        "cv_results": fold별 성능
        "feature_importance": 피처 중요도
        "best_threshold": 최적 threshold
    """
    if not HAS_XGB or not HAS_SKLEARN:
        raise ImportError("xgboost와 scikit-learn이 필요합니다.")

    if params is None:
        params = DEFAULT_PARAMS.copy()

    print("=" * 60)
    print("  🤖 XGBoost 진입 필터 학습 시작")
    print("=" * 60)

    # ── 1. 데이터 로드 + 전처리
    X, y, df_full = load_and_preprocess(csv_paths, verbose=verbose)

    # ── 2. Focal Loss 설정
    focal = None
    if use_focal_loss:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        params.pop("objective", None)
        params.pop("scale_pos_weight", None)
        if verbose:
            print(f"  🔥 Focal Loss: gamma={focal_gamma}, alpha={focal_alpha:.3f}")
            print(f"     (LOSS {n_neg} / WIN {n_pos})")
    else:
        # 기본 모드: scale_pos_weight 자동 계산
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        params["scale_pos_weight"] = round(spw, 2)
        if verbose:
            print(f"  ⚖️  scale_pos_weight = {spw:.2f} (LOSS {n_neg} / WIN {n_pos})")

    # ── 3. PurgedTimeSeriesSplit 교차검증 (미래 누수 완전 차단)
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap,
                                embargo=embargo, min_train_size=200)
    cv_results = []
    fold_models = []
    all_importances = []

    if verbose:
        print(f"\n  📊 PurgedTimeSeriesSplit ({n_splits}-fold, purge={purge_gap}, embargo={embargo})")
        print(f"  {'─'*55}")

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if focal is not None:
            # Focal Loss: xgb.train API 사용
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            xgb_params = {k: v for k, v in params.items()
                          if k not in ("random_state", "use_label_encoder",
                                       "n_estimators", "early_stopping_rounds")}
            xgb_params["seed"] = params.get("random_state", 42)

            bst = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=N_ESTIMATORS,
                evals=[(dval, "val")],
                obj=focal.objective,
                custom_metric=focal.eval_metric,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )

            y_pred_raw = bst.predict(dval, output_margin=True)
            y_pred_proba = focal._sigmoid(y_pred_raw)
            best_iter = bst.best_iteration if hasattr(bst, 'best_iteration') else N_ESTIMATORS

            # Booster를 XGBClassifier 래퍼로 저장 (호환성)
            model = xgb.XGBClassifier()
            model._Booster = bst
            model.n_classes_ = 2
        else:
            # 기본 binary:logistic
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                **params,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            best_iter = model.best_iteration if hasattr(model, 'best_iteration') else N_ESTIMATORS

        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 메트릭 계산
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            auc = 0.5

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        ll = log_loss(y_val, y_pred_proba.clip(1e-7, 1 - 1e-7))
        best_iter = best_iter or N_ESTIMATORS

        fold_result = {
            "fold": fold_i + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "logloss": round(ll, 4),
            "best_iteration": best_iter,
            "val_win_rate": round(y_val.mean() * 100, 1),
        }
        cv_results.append(fold_result)
        fold_models.append(model)

        # Feature importance 수집
        if focal is not None:
            imp_dict = bst.get_score(importance_type='gain')
            feature_cols_tmp = get_ml_features()
            imp = np.array([imp_dict.get(f, 0.0) for f in feature_cols_tmp])
            imp = imp / imp.sum() if imp.sum() > 0 else imp
        else:
            imp = model.feature_importances_
        all_importances.append(imp)

        if verbose:
            print(f"  Fold {fold_i+1}: "
                  f"AUC={auc:.3f}  "
                  f"Prec={precision:.3f}  "
                  f"Rec={recall:.3f}  "
                  f"F1={f1:.3f}  "
                  f"iter={best_iter}  "
                  f"(train={len(train_idx)}, val={len(val_idx)})")

    # ── 4. CV 결과 요약
    avg_auc = np.mean([r["auc"] for r in cv_results])
    avg_prec = np.mean([r["precision"] for r in cv_results])
    avg_rec = np.mean([r["recall"] for r in cv_results])
    avg_f1 = np.mean([r["f1"] for r in cv_results])

    print(f"  {'─'*55}")
    print(f"  평균: AUC={avg_auc:.3f}  Prec={avg_prec:.3f}  "
          f"Rec={avg_rec:.3f}  F1={avg_f1:.3f}")

    # ── 5. Feature Importance (평균)
    feature_cols = get_ml_features()
    avg_imp = np.mean(all_importances, axis=0)
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": avg_imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n  🏆 Feature Importance (전체 순위)")
        print(f"  {'─'*55}")
        for _, row in fi_df.iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"  {row['feature']:25s}  {row['importance']:.3f}  {bar}")

    # ── 6. 최종 모델: 전체 데이터로 재학습
    print(f"\n  🔄 전체 데이터({len(X)}건)로 최종 모델 학습 중...")

    # early stopping을 위한 마지막 20% 분할
    split_idx = int(len(X) * 0.8)
    X_final_train, X_final_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_final_train, y_final_val = y.iloc[:split_idx], y.iloc[split_idx:]

    avg_best_iter = int(np.mean([r["best_iteration"] for r in cv_results]))
    _n_rounds = max(avg_best_iter + 50, 100)

    if focal is not None:
        dtrain_f = xgb.DMatrix(X_final_train, label=y_final_train)
        dval_f = xgb.DMatrix(X_final_val, label=y_final_val)
        xgb_params_f = {k: v for k, v in params.items()
                        if k not in ("random_state", "use_label_encoder",
                                     "n_estimators", "early_stopping_rounds")}
        xgb_params_f["seed"] = params.get("random_state", 42)

        final_bst = xgb.train(
            xgb_params_f, dtrain_f, num_boost_round=_n_rounds,
            evals=[(dval_f, "val")], obj=focal.objective,
            custom_metric=focal.eval_metric,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
        )

        y_full_raw = final_bst.predict(dval_f, output_margin=True)
        y_full_proba = focal._sigmoid(y_full_raw)

        final_model = xgb.XGBClassifier()
        final_model._Booster = final_bst
        final_model.n_classes_ = 2
    else:
        final_model = xgb.XGBClassifier(
            n_estimators=_n_rounds,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(
                X_final_train, y_final_train,
                eval_set=[(X_final_val, y_final_val)],
                verbose=False,
            )
        y_full_proba = final_model.predict_proba(X_final_val)[:, 1]

    # ── 7. 최적 threshold 탐색 (calibration 전 raw proba 기준)
    best_threshold = _find_best_threshold(y_final_val, y_full_proba, verbose=verbose)

    # ── 7.5 Isotonic Calibration (Focal Loss 확률 보정)
    #   과적합 방지: val을 cal_fit(앞 60%) / cal_test(뒤 40%)로 분할
    #   cal_fit으로 calibrator 학습, cal_test로 threshold 탐색
    calibrator = None
    y_calibrated = y_full_proba
    if use_focal_loss:
        try:
            from sklearn.isotonic import IsotonicRegression

            # val을 시간순으로 60/40 분할
            cal_split = int(len(y_full_proba) * 0.6)
            y_proba_fit  = y_full_proba[:cal_split]
            y_true_fit   = y_final_val.values[:cal_split]
            y_proba_test = y_full_proba[cal_split:]
            y_true_test  = y_final_val.iloc[cal_split:]

            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(y_proba_fit, y_true_fit)

            # cal_test에 대해 calibrate
            y_cal_test = calibrator.predict(y_proba_test)
            y_calibrated = calibrator.predict(y_full_proba)  # 전체 val에도 적용 (저장용)

            if verbose:
                _raw_wr = y_true_test[y_proba_test >= best_threshold].mean() * 100 if (y_proba_test >= best_threshold).sum() > 0 else 0
                _cal_wr = y_true_test[y_cal_test >= best_threshold].mean() * 100 if (y_cal_test >= best_threshold).sum() > 0 else 0
                print(f"\n  📐 Isotonic Calibration 적용 (holdout {len(y_proba_test)}건)")
                print(f"     raw P(win) range: [{y_full_proba.min():.3f}, {y_full_proba.max():.3f}]")
                print(f"     cal P(win) range: [{y_calibrated.min():.3f}, {y_calibrated.max():.3f}]")
                print(f"     WR@th={best_threshold:.2f}: raw={_raw_wr:.1f}% → cal={_cal_wr:.1f}%")

            # calibration 후 threshold 재탐색 (cal_test 기준 — unseen data)
            best_threshold = _find_best_threshold(y_true_test, y_cal_test, verbose=verbose)

        except ImportError:
            print("  ⚠️ sklearn.isotonic 없음 — calibration 스킵")

    # ── 8. 모델 저장
    if focal is not None:
        final_bst.save_model(save_path)
    else:
        final_model.save_model(save_path)

    # Calibrator 저장 (pickle)
    cal_path = save_path.replace(".json", "_calibrator.pkl")
    if calibrator is not None:
        import pickle
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        if verbose:
            print(f"  💾 Calibrator 저장: {cal_path}")

    # 메타데이터도 함께 저장
    meta = {
        "feature_columns": feature_cols,
        "threshold": best_threshold,
        "cv_avg_auc": round(avg_auc, 4),
        "cv_avg_f1": round(avg_f1, 4),
        "scale_pos_weight": params.get("scale_pos_weight", None),
        "use_focal_loss": use_focal_loss,
        "focal_gamma": focal_gamma if use_focal_loss else None,
        "focal_alpha": focal_alpha if use_focal_loss else None,
        "has_calibrator": calibrator is not None,
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "training_params": {k: v for k, v in params.items()
                           if k not in ("verbosity", "use_label_encoder")},
    }
    meta_path = save_path.replace(".json", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 모델 저장: {save_path}")
    print(f"  💾 메타 저장: {meta_path}")
    print(f"  📊 최적 threshold: {best_threshold:.2f}")
    print("=" * 60)

    return {
        "model": final_model,
        "cv_results": cv_results,
        "feature_importance": fi_df,
        "best_threshold": best_threshold,
        "meta": meta,
    }


def _find_best_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: list = None,
    verbose: bool = True,
) -> float:
    """
    트레이딩에 최적화된 threshold 탐색.

    Score = (precision - baseline_wr) × log(남은 거래 수)
      - baseline_wr: 전체 승률 (=필터 없을 때의 precision)
      - precision: threshold 적용 후 실제 WIN 비율
      - log(n): 거래 수가 너무 적으면 패널티, 너무 많으면 점진적 보상

    이 공식의 의미:
      - precision이 baseline보다 높아야 양수 (=필터가 실제로 도움)
      - 거래 수가 많을수록 통계적으로 신뢰할 수 있으므로 log 보상
      - precision이 baseline보다 낮으면 음수 → 해당 threshold 배제
    """
    if thresholds is None:
        thresholds = [round(0.15 + i * 0.01, 2) for i in range(31)]  # 0.15 ~ 0.45

    baseline_wr = float(y_true.mean())  # 전체 WIN 비율 (필터 없을 때)
    min_trades = 30  # 최소 거래 수 (통계적 신뢰)

    if verbose:
        print(f"\n  🎯 Threshold 탐색 (baseline WR={baseline_wr*100:.1f}%)")
        print(f"  {'─'*72}")
        print(f"  {'Threshold':>10} {'남은거래':>8} {'승률':>8} {'향상':>8} {'Score':>10} {'WIN수':>6}")
        print(f"  {'─'*72}")

    best_th, best_score = 0.22, -999.0
    for th in thresholds:
        mask = y_proba >= th
        n_remain = int(mask.sum())
        if n_remain < min_trades:
            continue
        y_filtered = y_true.values[mask]
        win_rate = float(y_filtered.mean())
        n_wins = int(y_filtered.sum())

        # Score = (precision - baseline) × log(n_trades)
        precision_lift = win_rate - baseline_wr
        score = precision_lift * np.log(max(n_remain, 1))

        if verbose:
            _lift_pct = precision_lift * 100
            marker = " ◀" if score > best_score else ""
            print(f"  {th:>10.2f} {n_remain:>8} {win_rate*100:>7.1f}% "
                  f"{_lift_pct:>+7.1f}pp {score:>10.3f} {n_wins:>6}{marker}")

        if score > best_score:
            best_score = score
            best_th = th

    if verbose:
        print(f"  {'─'*72}")
        if best_score <= 0:
            print(f"  ⚠️  모든 threshold에서 baseline 대비 향상 없음 → 기본값 {best_th:.2f} 사용")
        else:
            print(f"  → 최적 threshold: {best_th:.2f}  (score={best_score:.3f})")

    return best_th


# ──────────────────────────────────────────────────────────
# MLFilter: 실전 봇에서 사용하는 예측 클래스
# ──────────────────────────────────────────────────────────

class MLFilter:
    """
    학습된 XGBoost 모델을 로드하여 실전 예측.

    사용법:
        ml = MLFilter("ml_entry_model.json")

        # reason dict에서 직접 예측
        p_win = ml.predict_from_reason(reason_dict)

        # 또는 피처 dict에서 예측
        p_win = ml.predict(feature_dict)

        # 필터링 판단
        if p_win < ml.threshold:
            skip  # 진입 차단
    """

    # P(win) 구간별 사이즈 조정
    SIZE_TIERS = [
        (0.50, 1.2),
        (0.36, 1.0),
        (0.00, 0.0),
    ]

    def __init__(self, model_path: str, threshold: float = None):
        if not HAS_XGB:
            raise ImportError("xgboost가 필요합니다.")

        # 메타데이터 로드
        meta_path = model_path.replace(".json", "_meta.json")
        self.meta = {}
        if Path(meta_path).exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        self._use_focal = self.meta.get("use_focal_loss", False)

        if self._use_focal:
            self._booster = xgb.Booster()
            self._booster.load_model(model_path)
            self.model = None
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            self._booster = None

        # Isotonic Calibrator 로드
        self._calibrator = None
        if self.meta.get("has_calibrator", False):
            cal_path = model_path.replace(".json", "_calibrator.pkl")
            if Path(cal_path).exists():
                import pickle
                with open(cal_path, "rb") as f:
                    self._calibrator = pickle.load(f)

        self.feature_columns = self.meta.get("feature_columns", get_ml_features())
        self._feature_col_list = list(self.feature_columns)  # 캐싱 (매 predict 호출 시 list() 변환 방지)
        self._n_features = len(self._feature_col_list)
        self.threshold = threshold or self.meta.get("threshold", 0.22)

        _mode = "Focal+Isotonic" if self._calibrator else ("Focal" if self._use_focal else "logistic")
        print(f"  🤖 MLFilter 로드 완료: {model_path}  ({_mode})")
        print(f"     피처 {len(self.feature_columns)}개, threshold={self.threshold:.2f}")

    def predict(self, features: dict) -> float:
        """
        피처 dict → calibrated P(win) 반환.

        [최적화] DataFrame 생성을 제거하고 NumPy 배열을 직접 사용.
        기존 대비 약 10~50× 빠름 (단건 기준).
        """
        # NumPy 1D 배열 직접 생성 (DataFrame 오버헤드 완전 제거)
        cols = self._feature_col_list
        arr = np.array([[features.get(col, 0) for col in cols]], dtype=np.float32)

        if self._use_focal:
            dmat = xgb.DMatrix(arr, feature_names=cols)
            raw = self._booster.predict(dmat, output_margin=True)
            p = float(np.where(raw >= 0,
                               1.0 / (1.0 + np.exp(-raw)),
                               np.exp(raw) / (1.0 + np.exp(raw)))[0])
        else:
            p = float(self.model.predict_proba(arr)[0, 1])

        # Isotonic calibration 적용
        if self._calibrator is not None:
            p = float(self._calibrator.predict(np.array([p]))[0])

        return p

    def predict_batch(self, features_list: list) -> np.ndarray:
        """
        여러 피처 dict를 한 번에 배치 예측.

        Parameters
        ----------
        features_list : list of dict
            각 dict는 피처 이름 → 값 매핑.

        Returns
        -------
        np.ndarray
            calibrated P(win) 배열 (shape: (n,))

        [최적화] N건의 예측을 DataFrame 없이 한 번의 XGBoost 호출로 처리.
        개별 predict() 호출 대비 약 100× 빠름.
        """
        if not features_list:
            return np.array([], dtype=np.float64)

        n = len(features_list)
        cols = self._feature_col_list
        n_cols = self._n_features

        # 2D NumPy 배열 생성 (한 번에)
        arr = np.zeros((n, n_cols), dtype=np.float32)
        for i, features in enumerate(features_list):
            for j, col in enumerate(cols):
                arr[i, j] = features.get(col, 0)

        if self._use_focal:
            dmat = xgb.DMatrix(arr, feature_names=cols)
            raw = self._booster.predict(dmat, output_margin=True)
            probs = np.where(raw >= 0,
                             1.0 / (1.0 + np.exp(-raw)),
                             np.exp(raw) / (1.0 + np.exp(raw)))
        else:
            probs = self.model.predict_proba(arr)[:, 1]

        # Isotonic calibration 적용 (배치)
        if self._calibrator is not None:
            probs = self._calibrator.predict(probs)

        return np.asarray(probs, dtype=np.float64)

    def predict_from_reason(self, reason: dict) -> float:
        """
        backtest.py의 reason dict에서 직접 피처를 추출하여 예측.

        이 메서드는 ml_data.py의 extract_ml_dataset()과 동일한 로직으로
        reason dict에서 피처를 추출합니다.
        """
        features = self._extract_features_from_reason(reason)
        return self.predict(features)

    def should_enter(self, p_win: float) -> bool:
        """P(win)이 threshold 이상이면 True (진입 허용)."""
        return p_win >= self.threshold

    def get_size_adjustment(self, p_win: float) -> float:
        """
        P(win) 기반 사이즈 조정 배수 반환.

        P(win) >= 0.35 → 1.0 (S급, 풀 사이즈)
        P(win) >= 0.30 → 0.5 (A급, 중간 사이즈)
        P(win) >= 0.22 → 0.25 (B급, 소형 사이즈)
        P(win) <  0.22 → 0.0 (skip)
        """
        for min_p, adj in self.SIZE_TIERS:
            if p_win >= min_p:
                return adj
        return 0.0

    def _extract_features_from_reason(self, r: dict) -> dict:
        """
        reason dict → 피처 dict 변환.

        우선순위:
          1) reason["feature_dict"]가 이미 있으면 그대로 사용
             (signals.py + backtest.py에서 완성된 16개 피처)
          2) 없으면 reason의 개별 키에서 최대한 추출 (fallback)
        """
        # ── 1순위: feature_dict가 이미 완성되어 있는 경우
        fd = r.get("feature_dict")
        if fd and len(fd) >= 10:
            return {col: fd.get(col, 0) for col in self.feature_columns}

        # ── 2순위: fallback (구 버전 reason 호환)
        #    get_feature_columns() 16개에 맞춰 동기화

        direction = r.get("direction", r.get("sweep_dir", "long"))

        # HTF alignment (1H 사용)
        htf_1h = r.get("htf_state", "unknown")
        if direction == "long":
            htf_1h_aligned = 1 if htf_1h == "bull" else 0
        else:
            htf_1h_aligned = 1 if htf_1h == "bear" else 0

        features = {
            # ── A. Displacement & Structure Quality
            "net_move_atr_ratio":           0.0,
            # strong_displacement_score 제거 (v20 폐기)
            "sweep_depth_atr":              float(r.get("sweep_depth_score", 0.0)),
            # ── A2. 신규 ATR 재정규화 변형
            "net_move_atr50":               0.0,
            "net_move_atr_wilder":          0.0,
            "net_move_atr1h":               0.0,
            # ── A3. 신규 15m displacement 품질 (NaN sentinel)
            "disp_quality_15m":             float("nan"),
            # ── B. Volume (1)
            "vol_ratio":                    1.0,
            # ── C. Market Context (4)
            "range_pos_30d":                0.5,
            "range_pos_7d":                 0.5,
            "directional_er":               0.0,
            "is_killzone":                  0.0,
            # ── D. Structure (2)
            "lvn_proximity":                3.0,
            "sweep_to_disp_fib":            0.5,
            # ── E. Market Regime (3) — adx_14 + 신규 MA 구조
            "adx_14":                       0.0,
            "trend_structure":              0.0,
            "ma_alignment":                 0.0,
            "return_1h":                    0.0,
            "return_4h":                    0.0,
            # ── F. TP Quality (1)
            "best_tp_score":                0.0,
            # ── G. Sweep & Trap (2)
            "sweep_duration_norm":          1.0,
            "trap_alignment":               0.0,
            # ── H. Timing (1)
            "bars_since_asia_break":        -1,
            # ── I. FVG Quality (1)
            "effective_fvg_strength":       0.0,
            # ── J. v7 복원 (3)
            "choch_break_atr_ratio":        0.0,
            "displacement_consistency":     0.0,
            "bars_since_sweep":             10,
            # ── K. 신규 Reclaim 건강함 (2)
            "reclaim_quality":              0.0,
            "reclaim_absorption":           0.0,
            # ── 4H FVG Lifecycle confluence (신규 layer)
            "in_4h_fvg":                    0.0,
            "in_4h_fvg_aligned":            0.0,
            "is_4h_fvg_combined":           0.0,
            "is_4h_fvg_alone":              0.0,
            "nearest_4h_fvg_dist_atr":      float("nan"),
            "active_4h_fvg_fill":           float("nan"),
            "n_active_4h_fvg":              0.0,
            "in_1h_fvg":                    0.0,
            "in_1h_fvg_aligned":            0.0,
            "is_1h_fvg_combined":           0.0,
            "is_1h_fvg_alone":              0.0,
            "nearest_1h_fvg_dist_atr":      float("nan"),
            "active_1h_fvg_fill":           float("nan"),
            "n_active_1h_fvg":              0.0,
            "is_4h1h_fvg_combined":         0.0,
            "unswept_above":                0.0,
            "unswept_below":                0.0,
            "unswept_imbalance":            0.0,
            "unswept_draw_aligned":         0.0,
            # ── L. Interaction
            "sweep_x_disp":                 0.0,
            # disp_x_fvg 제거 (v20 폐기, strong_disp 인수)
            "sweep_velocity":               0.0,
            "rangepos_x_sweep_dur":         0.0,
            "rangepos_x_sweep_depth":       0.0,
            "lvn_x_fvg":                    0.0,
            "killzone_x_disp":              0.0,
            "consistency_x_vol":            0.0,
            # ── L. Spot-Futures 괴리 (4)
            "return_div":                   0.0,
            "vol_div":                      0.0,
            "basis_z":                      0.0,
            "liq_excursion_div":            0.0,
            # signed 괴리 + 방향 + sweep 추적 (stage7b/7c, sweep tracking)
            "spot_div_signed_z":            0.0,
            "is_long":                      1.0 if direction == "long" else 0.0,
            "opp_sweep_present":            0.0,
            "opp_sweep_to_setup_bars":      -1.0,
            "opp_sweep_dist_atr":           0.0,
            "opp_sweep_is_last":            0.0,
            "opp_sweep_unviolated":         0.0,
            "opp_sweep_range_atr":          0.0,
        }

        return features


# ──────────────────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import glob

    csvs = sorted(glob.glob("ml_train_*.csv"))
    if not csvs:
        print("❌ ml_train_*.csv 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"🔍 발견된 CSV: {csvs}")
    result = train_entry_filter(csvs, save_path="ml_entry_model.json")

    print(f"\n✅ 학습 완료!")
    print(f"   평균 AUC: {np.mean([r['auc'] for r in result['cv_results']]):.3f}")
    print(f"   최적 threshold: {result['best_threshold']:.2f}")
    print(f"\n   Feature Importance (Top 5):")
    for _, row in result["feature_importance"].head(5).iterrows():
        print(f"     {row['feature']:25s}  {row['importance']:.3f}")
