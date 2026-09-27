""" 분리
train_entry.py — Entry Filter 학습 (VP Regression / Binary Classification) 
═══════════════════════════════════════════════════════════════════
사용법:
    python train_entry.py --csv ml_train_with_tp.csv --save entry_model.json

CSV에 vp_score 컬럼이 있으면 → XGBRegressor (VP Score regression)
없으면 → XGBClassifier (binary classification) fallback

VP Regression:
  Target: log1p(vp_score) — vp_score = vp_consumption / (1 + adversity) × (1 + disp_broken)
  평가: Spearman correlation + AUC (binary label 기준)
  Threshold: raw pred 기준 binary WR 최적화

Binary Classification:
  Target: label (1=TP, 0=SL)
  후처리: Isotonic Calibration + Temperature Scaling

Early Stopping: fold별 adaptive (train_size // 80, 30~100)
"""
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection._split import BaseCrossValidator
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml_data import get_feature_columns


# ──────────────────────────────────────────────────────────
# Focal Loss (Custom Objective for XGBoost)
# ──────────────────────────────────────────────────────────

def focal_loss_objective(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss for XGBoost custom objective.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
      p = sigmoid(z)  (raw logit → probability)
      p_t = p if y=1, else (1-p)
      alpha_t = alpha if y=1, else (1-alpha)

    Gradient derivation:
      dFL/dz = dFL/dp * dp/dz
      dp/dz = p * (1 - p)  (sigmoid derivative)

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 = standard CE, 2 = strong focus on hard samples.
    alpha : float
        Class balance weight for positive class (0~1).
        0.25 = positive class gets 25% weight (적을수록 negative에 집중).
    """
    def _objective(y_true, y_pred):
        # y_pred = raw logit (XGBoost custom obj receives raw margin)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # p_t, alpha_t
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** gamma

        # dFL/dp
        # = -alpha_t * [ gamma * (1-p_t)^(gamma-1) * log(p_t) * (-1) + (1-p_t)^gamma * (1/p_t) ]
        # Simplified gradient via chain rule:
        #   grad = alpha_t * (1-p_t)^gamma * (gamma * p_t * log(p_t) / (1-p_t) + p_t - y_true)
        #        * ... but cleaner to use the direct form:

        # Direct form (correct chain rule):
        # dFL/dz = (y - p) * focal_weight * [gamma * log(p_t) + (1-p_t)^(-1) * (1-p_t)]
        # Simplified correct gradient:
        grad = (p - y_true) * focal_weight * (
            gamma * (1 - p_t) * np.log(p_t + 1e-10) + p_t
        ) / p_t

        # Wait — let's use the verified, simpler form:
        # grad = alpha_t * ( gamma * (1-p_t)^(gamma) * p_t * log(p_t) + (1-p_t)^(gamma+1) )
        #        * sign(p - y_true)
        # But the cleanest verified derivation is:
        #
        # For y=1:  loss = -alpha * (1-p)^gamma * log(p)
        #   dL/dz = -alpha * [ -gamma*(1-p)^(gamma-1) * p*(1-p) * log(p)
        #                      + (1-p)^gamma * (1/p) * p*(1-p) ]
        #         = -alpha * (1-p)^gamma * (1-p) * [ -gamma * log(p) + (1-p)/p * ... ]
        #
        # Correct simplified (numerically stable):
        grad = focal_weight * (p - y_true)
        # This is the standard gradient with focal weighting.
        # (1-p_t)^gamma down-weights easy samples, alpha_t balances classes.

        # Hessian (second derivative) — diagonal approximation
        hess = focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)  # numerical stability

        return grad, hess

    return _objective


def focal_loss_eval(gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss evaluation metric for early stopping.
    XGBoost >= 2.0: returns (name, value) from feval(y_pred, DMatrix)
    """
    def _eval(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t)
        return "focal_loss", float(loss.mean())
    return _eval


# ──────────────────────────────────────────────────────────
# CV
# ──────────────────────────────────────────────────────────

class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, purge_gap=12, embargo=6, min_train_size=200):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_gap
            test_end = min(test_start + fold_size, n)
            test_start_adj = test_start + self.embargo if test_start + self.embargo < test_end else test_start
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start_adj, test_end)
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ──────────────────────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────────────────────

ENTRY_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    # scale_pos_weight: 학습 시 자동 계산 (n_neg / n_pos)
    "max_depth": 4,
    "min_child_weight": 3,
    "gamma": 0.3,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "learning_rate": 0.03,
    "random_state": 42,
    "verbosity": 0,
}
N_ESTIMATORS = 800
EARLY_STOPPING = 80

# Temperature Scaling
DEFAULT_TEMPERATURE = 0.75  # 0.6 ~ 0.9 사이 tuning


# ──────────────────────────────────────────────────────────
# Temperature Scaling
# ──────────────────────────────────────────────────────────

def _temperature_scale(probs: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Isotonic 보정 후 Temperature Scaling으로 부드럽게."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1 - probs))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


# ──────────────────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────────────────

def train_entry_split(
    csv_paths: Union[str, List[str]],
    n_splits: int = 5,
    params: dict = None,
    verbose: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Long/Short 분리 학습.
    entry_model_long.json + entry_model_short.json 생성.
    """
    if params is None:
        params = ENTRY_PARAMS.copy()

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    if "direction" not in df.columns:
        raise ValueError("CSV에 'direction' 컬럼이 없습니다.")

    results = {}
    for direction in ["long", "short"]:
        sub = df[df["direction"] == direction].reset_index(drop=True)
        n = len(sub)
        if n < 100:
            print(f"\n  ⚠️ {direction} 데이터 부족 ({n}건) → 학습 스킵")
            continue

        print(f"\n{'▶' * 30}")
        print(f"  📌 {direction.upper()} 모델 학습 ({n}건)")
        print(f"{'▶' * 30}")

        save_path = f"entry_model_{direction}.json"
        res = train_entry(
            csv_paths=csv_paths,
            save_path=save_path,
            n_splits=n_splits,
            params=params.copy(),
            verbose=verbose,
            temperature=temperature,
            direction_filter=direction,
        )
        results[direction] = res

    print(f"\n{'═' * 60}")
    print(f"  🎉 Long/Short 분리 학습 완료")
    for d, r in results.items():
        print(f"     {d:5s}: AUC={r['meta']['cv_avg_auc']:.4f}  "
              f"threshold={r['best_threshold']:.4f}  "
              f"n={r['meta']['n_samples']}")
    print(f"{'═' * 60}")

    return results


def train_entry(
    csv_paths: Union[str, List[str]],
    save_path: str = "entry_model.json",
    n_splits: int = 5,
    params: dict = None,
    verbose: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    direction_filter: str = None,
) -> dict:
    """
    Entry Filter 학습 (binary classification).

    direction_filter: "long" or "short" → 해당 방향만 학습.
                      None → 전체 데이터 학습 (기존 동작).
    """
    if not HAS_XGB or not HAS_SKLEARN:
        raise ImportError("xgboost, scikit-learn 필요")
    if params is None:
        params = ENTRY_PARAMS.copy()

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    # 방향 필터
    if direction_filter is not None:
        df = df[df["direction"] == direction_filter].reset_index(drop=True)

    feature_cols = get_feature_columns()

    # ── Target 선택 (기본 = SL 회피)
    #    (기본)                : SL 회피 (y=1=SL아님/진입OK, y=0=exit_reason"sl")  ★
    #    ENTRY_TARGET=vp       : vp_score regression (구 기본, 무신호 AUC~0.50)
    #    ENTRY_TARGET=label    : tp/sl label (y=1=TP, y=0=SL)
    #    (ENTRY_BINARY=1 도 label 모드로 호환 유지)
    #
    #    SL 회피 근거(가지9): exit_reason="sl"만 타깃하면 OOF AUC 0.641, 진입70%그룹
    #    SL률 43→37%. timeout 섞으면 AUC만 오르고(꼬리신호 재탕) SL식별 변질되므로
    #    SL만 순수 타깃. 역할분리: entry=SL회피, trail=러너 포획.
    import os as _os_te
    _entry_target = _os_te.environ.get("ENTRY_TARGET", "sl_avoid").lower()
    _force_binary = _os_te.environ.get("ENTRY_BINARY", "0") in ("1", "true", "True")
    if _force_binary:
        _entry_target = "label"
    _use_sl_avoid = (_entry_target == "sl_avoid")
    _use_vp = (_entry_target == "vp")

    if _use_sl_avoid:
        if "exit_reason" not in df.columns:
            raise ValueError("기본 SL회피 모드인데 exit_reason 컬럼 없음 "
                             "(구버전 CSV면 ENTRY_TARGET=label 사용)")
        print("  🎯 기본 SL 회피 학습 (y=1=진입OK, y=0=SL)")
        use_regression = False
        y_col = "sl_avoid"
        # y=1: SL 아님(진입 적합), y=0: exit_reason=="sl"(회피 대상). timeout/tp는 모두 1.
        y = (df["exit_reason"] != "sl").astype(int)
    else:
        use_regression = _use_vp and "vp_score" in df.columns and df["vp_score"].notna().sum() > 100
        if _use_vp:
            print("  🔧 ENTRY_TARGET=vp → vp_score regression")
        else:
            print("  🔧 ENTRY_TARGET=label → tp/sl binary")
        if use_regression:
            y_col = "vp_score"
            y = df[y_col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
            # log transform으로 분포 정규화 (score 범위가 0~80+)
            y = np.log1p(y)
        else:
            y_col = "label"
            y = df[y_col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️ 누락 피처 0으로 채움: {missing}")
        for c in missing:
            df[c] = 0

    # v3: NaN은 XGBoost native handling으로 처리 (fillna 금지)
    #     inf만 NaN으로 정규화 (XGBoost는 NaN을 자동 branch로 학습)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # NaN 비율 리포트 (진단용)
    nan_summary = X.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    if len(nan_cols) > 0:
        print(f"  ℹ️ NaN sentinel (XGBoost native handling):")
        for c, n in nan_cols.items():
            pct = 100.0 * n / len(X)
            print(f"     {c:<32s}  {n:>6,d}건 ({pct:.1f}%)")

    # binary label 통계 (진단용) — SL회피 모드는 y(sl_avoid) 기준으로 재계산
    if _use_sl_avoid:
        n_pos = int(y.sum())                 # y=1: SL 아님(진입OK)
        n_neg = len(y) - n_pos               # y=0: SL
        wr = n_pos / len(y) * 100
    elif "label" in df.columns:
        _label = df["label"].astype(int)
        n_pos = int(_label.sum())
        n_neg = len(_label) - n_pos
        wr = n_pos / len(_label) * 100
    else:
        n_pos = 0; n_neg = len(y); wr = 0.0

    if use_regression:
        # Regression: scale_pos_weight 불필요
        params.pop("scale_pos_weight", None)
        params.pop("eval_metric", None)
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"

        print("=" * 60)
        print(f"  🤖 Entry Filter 학습 ({len(feature_cols)}피처, VP Regression)")
        print("=" * 60)
        if verbose:
            print(f"  데이터: {len(X)}건, 피처: {len(feature_cols)}개")
            print(f"  Target: vp_score (log1p transformed)")
            print(f"  Binary WR (참고): TP={n_pos}건 ({wr:.1f}%)  SL={n_neg}건 ({100-wr:.1f}%)")
            print(f"  vp_score 분포: mean={y.mean():.3f}  median={y.median():.3f}  "
                  f"std={y.std():.3f}  range=[{y.min():.2f}, {y.max():.2f}]")
    else:
        # Classification fallback
        if "scale_pos_weight" not in params:
            spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
            params["scale_pos_weight"] = spw

        print("=" * 60)
        print(f"  🤖 Entry Filter 학습 ({len(feature_cols)}피처, binary classification)")
        print("=" * 60)
        if verbose:
            print(f"  데이터: {len(X)}건, 피처: {len(feature_cols)}개")
            if _use_sl_avoid:
                print(f"  Label: 진입OK={n_pos}건 ({wr:.1f}%)  SL={n_neg}건 ({100-wr:.1f}%)")
            else:
                print(f"  Label: TP={n_pos}건 ({wr:.1f}%)  SL={n_neg}건 ({100-wr:.1f}%)")
            print(f"  scale_pos_weight: {params.get('scale_pos_weight', 1.0)}")

    if verbose:
        # 진단: 피처-target 상관 Top 5
        corrs = X.corrwith(y).abs().sort_values(ascending=False)
        print(f"\n  📊 피처-Target 상관 Top 5:")
        for col, r in corrs.head(5).items():
            print(f"     {col:30s}  |r|={r:.4f}")

    # ── CV
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=12, embargo=6, min_train_size=200)
    cv_results = []
    all_imp = []

    if verbose:
        print(f"\n  📊 PurgedTimeSeriesSplit ({n_splits}-fold)")
        print(f"  {'─'*65}")

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        tr_wr = y_tr.mean() * 100
        val_wr = y_val.mean() * 100

        # Adaptive early stopping: 데이터 크기에 비례
        _es = max(30, min(100, len(tr_idx) // 80))

        if use_regression:
            m = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                early_stopping_rounds=_es,
                **params,
            )
        else:
            m = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                early_stopping_rounds=_es,
                use_label_encoder=False,
                **params,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        if use_regression:
            preds = m.predict(X_val)
        else:
            preds = m.predict_proba(X_val)[:, 1]
        bi = m.best_iteration if hasattr(m, 'best_iteration') else N_ESTIMATORS

        if use_regression:
            # Regression 평가: spearman 상관 + binary label 기준 AUC
            from scipy.stats import spearmanr
            _spear, _ = spearmanr(preds, y_val)
            _spear = _spear if np.isfinite(_spear) else 0.0

            # preds → binary label 변환하여 AUC 계산
            if "label" in df.columns:
                _y_bin = df.iloc[val_idx]["label"].astype(int).values
                try:
                    auc = roc_auc_score(_y_bin, preds)
                except ValueError:
                    auc = 0.5
            else:
                auc = 0.5

            # RMSE
            _rmse = float(np.sqrt(np.mean((preds - y_val.values) ** 2)))

            cv_results.append({"fold": fold_i+1, "auc": round(auc, 4),
                               "spearman": round(_spear, 4),
                               "rmse": round(_rmse, 4),
                               "best_iter": bi})
            all_imp.append(m.feature_importances_)

            if verbose:
                print(f"  Fold {fold_i+1}: AUC={auc:.3f}  Spear={_spear:.3f}  "
                      f"RMSE={_rmse:.3f}  iter={bi}  ES={_es}  "
                      f"train={len(X_tr)}({tr_wr:.1f}%)  val={len(X_val)}({val_wr:.1f}%)  "
                      f"pred_range=[{preds.min():.3f},{preds.max():.3f}]")
        else:
            try:
                auc = roc_auc_score(y_val, preds)
            except ValueError:
                auc = 0.5

            y_pred_bin = (preds >= 0.5).astype(int)
            tp_cnt = int(((y_pred_bin == 1) & (y_val == 1)).sum())
            fp_cnt = int(((y_pred_bin == 1) & (y_val == 0)).sum())
            fn_cnt = int(((y_pred_bin == 0) & (y_val == 1)).sum())
            prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
            rec = tp_cnt / (tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else 0.0

            cv_results.append({"fold": fold_i+1, "auc": round(auc, 4),
                               "precision": round(prec, 4), "recall": round(rec, 4),
                               "best_iter": bi})
            all_imp.append(m.feature_importances_)

            if verbose:
                n_pos_pred = int((preds >= 0.5).sum())
                print(f"  Fold {fold_i+1}: AUC={auc:.3f}  Prec={prec:.3f}  "
                      f"Rec={rec:.3f}  iter={bi}  ES={_es}  "
                      f"train={len(X_tr)}({tr_wr:.1f}%)  val={len(X_val)}({val_wr:.1f}%)  "
                      f"pred_range=[{preds.min():.3f},{preds.max():.3f}]  "
                      f"pred>=0.5: {n_pos_pred}")

    avg_auc = np.mean([r["auc"] for r in cv_results])
    if use_regression:
        avg_spear = np.mean([r.get("spearman", 0) for r in cv_results])
        avg_rmse = np.mean([r.get("rmse", 0) for r in cv_results])
        avg_prec = 0.0
        avg_rec = 0.0
        print(f"  {'─'*65}")
        print(f"  평균: AUC={avg_auc:.3f}  Spearman={avg_spear:.3f}  RMSE={avg_rmse:.3f}")
    else:
        avg_prec = np.mean([r["precision"] for r in cv_results])
        avg_rec = np.mean([r["recall"] for r in cv_results])
        print(f"  {'─'*65}")
        print(f"  평균: AUC={avg_auc:.3f}  Prec={avg_prec:.3f}  Rec={avg_rec:.3f}")

    # Feature importance
    avg_imp_vals = np.mean(all_imp, axis=0)
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": avg_imp_vals}
                         ).sort_values("importance", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"\n  🏆 Feature Importance (Top 15)")
        print(f"  {'─'*65}")
        for _, row in fi_df.head(15).iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"  {row['feature']:30s}  {row['importance']:.3f}  {bar}")

    # ── 최종 모델 (80/20 split)
    print(f"\n  🔄 전체 데이터로 최종 모델 학습...")
    split_idx = int(len(X) * 0.8)
    X_final_train, X_final_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_final_train, y_final_val = y.iloc[:split_idx], y.iloc[split_idx:]

    avg_bi = int(np.mean([r["best_iter"] for r in cv_results]))

    if use_regression:
        final = xgb.XGBRegressor(
            n_estimators=max(avg_bi + 50, 100),
            early_stopping_rounds=EARLY_STOPPING,
            **params,
        )
    else:
        final = xgb.XGBClassifier(
            n_estimators=max(avg_bi + 50, 100),
            early_stopping_rounds=EARLY_STOPPING,
            use_label_encoder=False,
            **params,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X_final_train, y_final_train,
                  eval_set=[(X_final_val, y_final_val)],
                  verbose=False)

    if use_regression:
        y_raw_pred = final.predict(X_final_val)
    else:
        y_raw_pred = final.predict_proba(X_final_val)[:, 1]

    if use_regression:
        # Regression: calibration 불필요, threshold는 raw score 기준
        calibrator = None
        if verbose:
            print(f"\n  📐 Regression 모드 — Calibration 생략")
            print(f"     raw pred range: [{y_raw_pred.min():.3f}, {y_raw_pred.max():.3f}]")

        # threshold 탐색: raw pred 기준으로 binary label WR 최적화
        _y_bin = df.iloc[split_idx:]["label"].astype(int).values
        cal_split = int(len(y_raw_pred) * 0.6)
        y_for_threshold = y_raw_pred[cal_split:]
        y_true_for_threshold = pd.Series(_y_bin[cal_split:])
        best_th = _find_best_threshold(y_true_for_threshold, y_for_threshold, verbose=verbose)
    else:
        # Classification: Isotonic Calibration
        cal_split = int(len(y_raw_pred) * 0.6)
        calibrator = None
        y_calibrated = y_raw_pred

        try:
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(y_raw_pred[:cal_split], y_final_val.values[:cal_split])
            y_calibrated = calibrator.predict(y_raw_pred)

            y_smoothed = _temperature_scale(y_calibrated, temperature)

            if verbose:
                print(f"\n  📐 Isotonic Calibration + Temperature Scaling (T={temperature})")
                print(f"     raw  range: [{y_raw_pred.min():.3f}, {y_raw_pred.max():.3f}]")
                print(f"     cal  range: [{y_calibrated.min():.3f}, {y_calibrated.max():.3f}]")
                print(f"     smooth range: [{y_smoothed.min():.3f}, {y_smoothed.max():.3f}]")
        except Exception as e:
            print(f"  ⚠️ Calibration 실패: {e}")

        y_for_threshold = y_raw_pred[cal_split:]
        y_true_for_threshold = y_final_val.iloc[cal_split:]
        best_th = _find_best_threshold(y_true_for_threshold, y_for_threshold, verbose=verbose)

    # ── 저장
    final.save_model(save_path)

    # Calibrator 저장 (classification만)
    cal_path = save_path.replace(".json", "_calibrator.pkl")
    if calibrator is not None:
        import pickle
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        if verbose:
            print(f"  💾 Calibrator: {cal_path}")

    meta = {
        "feature_columns": feature_cols,
        "threshold": best_th,
        "temperature": temperature,
        "direction": direction_filter,
        "model_type": "entry_filter_vp_regression" if use_regression else "entry_filter_binary",
        "label_type": "vp_score" if use_regression else "binary_tp_sl",
        "has_calibrator": calibrator is not None,
        "use_regression": int(use_regression),
        "cv_avg_auc": round(avg_auc, 4),
        "cv_avg_precision": round(avg_prec, 4),
        "cv_avg_recall": round(avg_rec, 4),
        "n_samples": len(X),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_features": len(feature_cols),
        "training_params": {k: v for k, v in params.items() if k != "verbosity"},
    }
    meta_path = save_path.replace(".json", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 모델: {save_path}")
    print(f"  💾 메타: {meta_path}")
    print(f"  📊 Threshold: {best_th:.2f}")
    print("=" * 60)

    return {"model": final, "cv_results": cv_results, "feature_importance": fi_df,
            "best_threshold": best_th, "calibrator": calibrator, "meta": meta}


# ──────────────────────────────────────────────────────────
# Threshold 탐색
# ──────────────────────────────────────────────────────────

def _find_best_threshold(
    y_true: pd.Series,
    y_pred: np.ndarray,
    thresholds: list = None,
    verbose: bool = True,
) -> float:
    """
    Binary classification 예측 확률 기반 threshold 탐색.
    Score = (filtered_wr - baseline_wr) × log(n_remain)

    thresholds가 None이면 예측값 분포에 맞게 자동 생성.
    """
    if thresholds is None:
        # 예측값의 실제 분포에 맞게 탐색 범위 자동 설정
        p_min, p_max = float(np.percentile(y_pred, 10)), float(np.percentile(y_pred, 95))
        step = max((p_max - p_min) / 30, 0.005)
        thresholds = [round(p_min + i * step, 4) for i in range(31)]
        thresholds = [t for t in thresholds if t > 0.01]  # 너무 낮은 값 제거

    y_vals = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    baseline_wr = float(y_vals.mean())
    n_total = len(y_vals)
    min_trades = max(30, int(n_total * 0.05))  # 최소 5% 이상 남아야 함

    if verbose:
        print(f"\n  🎯 Threshold 탐색 (baseline WR={baseline_wr*100:.1f}%, "
              f"pred range=[{y_pred.min():.3f}, {y_pred.max():.3f}])")
        print(f"  {'─'*78}")
        print(f"  {'Threshold':>10} {'남은거래':>8} {'WR':>8} {'향상':>8} {'Score':>10}")
        print(f"  {'─'*78}")

    best_th, best_score = thresholds[0], -999.0
    found_any = False
    for th in thresholds:
        mask = y_pred >= th
        n = int(mask.sum())
        if n < min_trades:
            continue
        filtered = y_vals[mask]
        wr = float(filtered.mean())
        lift = wr - baseline_wr
        score = lift * np.log(max(n, 1))

        if verbose and lift > 0:
            marker = " ◀" if score > best_score else ""
            print(f"  {th:>10.4f} {n:>8} {wr*100:>7.1f}% "
                  f"{lift*100:>+7.1f}pp {score:>10.3f}{marker}")
        if score > best_score and lift > 0:
            best_score = score
            best_th = th
            found_any = True

    if verbose:
        print(f"  {'─'*78}")
        if found_any:
            print(f"  → 최적 threshold: {best_th:.4f}  (score={best_score:.3f})")
        else:
            # 향상 없으면 중앙값 사용
            best_th = round(float(np.median(y_pred)), 4)
            print(f"  ⚠️ 향상 없음 → median 기반 기본값 {best_th:.4f}")
    elif not found_any:
        best_th = round(float(np.median(y_pred)), 4)

    return best_th


# ──────────────────────────────────────────────────────────
# EntryFilter: 실전 봇에서 사용
# ──────────────────────────────────────────────────────────

class EntryFilter:
    """
    Entry Filter — Long/Short 분리 모델 지원.

    사용법 A (분리 모델):
        ef = EntryFilter("entry_model_long.json", "entry_model_short.json")
        score = ef.predict(feature_dict, direction="long")

    사용법 B (통합 모델, 하위 호환):
        ef = EntryFilter("entry_model.json")
        score = ef.predict(feature_dict)
    """

    SIZE_TIERS = [
        (0.80, 1.2),   # S급: raw P >= 0.80 → 120%
        (0.75, 1.0),   # A급: raw P >= threshold → 풀사이즈
        (0.00, 0.0),   # skip
    ]

    def __init__(self, model_path: str, model_path_short: str = None, threshold: float = None):
        if not HAS_XGB:
            raise ImportError("xgboost 필요")

        self._models = {}       # {"long": model, "short": model} or {"all": model}
        self._meta = {}         # {"long": meta, "short": meta} or {"all": meta}
        self._calibrators = {}  # {"long": cal, ...}
        self._thresholds = {}   # {"long": th, ...}

        if model_path_short is not None:
            # 분리 모델 모드
            for direction, path in [("long", model_path), ("short", model_path_short)]:
                self._load_single(direction, path, threshold)
            self._mode = "split"
            print(f"  🤖 EntryFilter 로드: SPLIT 모드")
            for d in ["long", "short"]:
                n_feat = len(self._meta[d].get("feature_columns", []))
                th = self._thresholds[d]
                print(f"     {d:5s}: {n_feat}피처, threshold={th:.4f}")
        else:
            # 통합 모델 모드 (하위 호환)
            self._load_single("all", model_path, threshold)
            self._mode = "unified"
            n_feat = len(self._meta["all"].get("feature_columns", []))
            th = self._thresholds["all"]
            print(f"  🤖 EntryFilter 로드: UNIFIED 모드  {n_feat}피처, threshold={th:.4f}")

        # 외부에서 접근하는 threshold (통합 모드 호환)
        self.threshold = self._thresholds.get("all", 
                         self._thresholds.get("long", 0.5))
        self.feature_columns = self._meta.get("all", self._meta.get("long", {})).get(
            "feature_columns", get_feature_columns())
        self.temperature = self._meta.get("all", self._meta.get("long", {})).get(
            "temperature", DEFAULT_TEMPERATURE)

    def _load_single(self, key: str, path: str, threshold_override: float = None):
        """단일 모델 로드."""
        meta_path = path.replace(".json", "_meta.json")
        meta = {}
        if Path(meta_path).exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        model = xgb.XGBClassifier()
        model.load_model(path)

        calibrator = None
        if meta.get("has_calibrator", False):
            cal_path = path.replace(".json", "_calibrator.pkl")
            if Path(cal_path).exists():
                import pickle
                with open(cal_path, "rb") as f:
                    calibrator = pickle.load(f)

        self._models[key] = model
        self._meta[key] = meta
        self._calibrators[key] = calibrator
        self._thresholds[key] = threshold_override or meta.get("threshold", 0.5)

    def _resolve_key(self, direction: str = None) -> str:
        """direction → 모델 key 결정."""
        if self._mode == "unified":
            return "all"
        if direction in self._models:
            return direction
        # fallback: long 모델로
        return "long"

    def predict(self, features: dict, direction: str = None) -> float:
        """피처 dict → P(TP) raw score."""
        key = self._resolve_key(direction)
        model = self._models[key]
        feat_cols = self._meta[key].get("feature_columns", get_feature_columns())

        row = {col: features.get(col, 0) for col in feat_cols}
        X = pd.DataFrame([row])[feat_cols]
        raw = float(model.predict_proba(X)[0, 1])
        return raw

    def should_enter(self, score: float, direction: str = None) -> bool:
        key = self._resolve_key(direction)
        return score >= self._thresholds[key]

    def get_size_adjustment(self, score: float) -> float:
        for min_s, adj in self.SIZE_TIERS:
            if score >= min_s:
                return adj
        return 0.0


if __name__ == "__main__":
    import glob as _glob
    parser = argparse.ArgumentParser(description="Train Entry Filter")
    parser.add_argument("--csv", nargs="+", default=None,
                        help="학습 CSV 파일 경로 (기본: ml_train_*.csv 자동 탐색)")
    parser.add_argument("--save", default="entry_model.json",
                        help="모델 저장 경로 (통합 모드 --merged 일 때만)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--merged", action="store_true",
                        help="(호환용) Long/Short 통합 — 기본 동작이므로 불필요")
    parser.add_argument("--split", action="store_true",
                        help="Long/Short 분리 학습 (실험용; 기본은 통합 단일 모델)")
    args = parser.parse_args()

    if args.csv:
        csv_paths = args.csv
    else:
        _exclude = {"ml_train_with_tp.csv", "ml_train_augmented.csv"}
        csv_paths = sorted(
            p for p in _glob.glob("ml_train_*.csv")
            if p.replace("\\", "/").split("/")[-1] not in _exclude
        )
    if not csv_paths:
        print("❌ ml_train_*.csv 파일을 찾을 수 없습니다.")
        sys.exit(1)
    print(f"🔍 학습 CSV: {csv_paths}")

    if args.split:
        print("  ✂️ Long/Short 분리 학습 (실험용)")
        train_entry_split(csv_paths=csv_paths, n_splits=args.folds,
                          temperature=args.temperature)
    else:
        print("  📦 통합 단일 모델 (기본; long/short 합병 — SL회피 단변량서 방향 편차 0.0005)")
        train_entry(csv_paths=csv_paths, save_path=args.save,
                    n_splits=args.folds, temperature=args.temperature)
