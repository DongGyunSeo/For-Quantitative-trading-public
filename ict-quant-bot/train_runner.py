#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_runner.py — runner(exit 타이밍) 모델 학습.

입력: ml_runner_v2.csv (prepare_runner_data_v2.py 생성, 이벤트 기반)
출력: dt별 모델 3종 (이진 P(fav>=adv) + 회귀 fav,adv).

핵심: group=trade_id GroupKFold — 같은 트레이드의 봉 샘플이 train/val에
  섞이면 누수(인접 봉은 trail_path 공유). 반드시 트레이드 단위 분리.

라벨 (dt ∈ {6,12,24}):
  y_win_dt : 이진 (위가 아래보다 멀리)
  fav_dt   : 회귀 (현재 대비 추가 상승 여지 R)
  adv_dt   : 회귀 (현재 대비 추가 까임 여지 R)

exit 룰 연결: P(win) 낮거나 E[adv] > E[fav]면 (부분)청산.
  EV ≈ P(win)*E[fav] - (1-P(win))*E[adv].
"""
import os, sys
import numpy as np, pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

try:
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, mean_absolute_error
except ImportError as e:
    print("필요 패키지:", e); sys.exit(1)

DT_LIST = [6, 12, 24]

# 동적 + 정적 feature (라벨/식별 컬럼 제외)
DROP = {"symbol", "trade_id", "entry_time", "direction", "exit_reason",
        "setup_level_type", "eval_t", "eval_5m_idx", "fav_next", "adv_next",
        "eval_swing_kind", "extreme_5m_idx"}
LABEL_PREFIX = ("y_win_", "fav_", "adv_")


def get_features(df):
    feats = []
    for c in df.columns:
        if c in DROP:
            continue
        if c.startswith(LABEL_PREFIX):
            continue
        if df[c].dtype.kind in "biufc":   # 숫자형만
            feats.append(c)
    return feats


def train_one(df, feats, target, groups, is_binary):
    sub = df.dropna(subset=[target])
    X = sub[feats].values
    y = sub[target].values
    g = groups[sub.index]
    gkf = GroupKFold(n_splits=5)
    scores = []
    importances = np.zeros(len(feats))
    for tr, va in gkf.split(X, y, g):
        if is_binary:
            if len(np.unique(y[tr])) < 2:
                continue
            m = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                eval_metric="logloss", early_stopping_rounds=40, n_jobs=4)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            p = m.predict_proba(X[va])[:, 1]
            scores.append(roc_auc_score(y[va], p))
        else:
            m = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                eval_metric="mae", early_stopping_rounds=40, n_jobs=4)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            p = m.predict(X[va])
            scores.append(-mean_absolute_error(y[va], p))  # 음수 MAE (높을수록 좋음)
        importances += m.feature_importances_
    return np.mean(scores) if scores else np.nan, importances / max(len(scores), 1)


def main():
    import glob as _glob
    # 입력 우선순위: runner_train_*.csv (collect 직접생성) > ml_runner_v2.csv (prepare) > ml_runner.csv
    _rt = [f for f in _glob.glob("runner_train_*.csv") if "shadow" not in f and "traon" not in f]
    if _rt:
        df = pd.concat([pd.read_csv(f) for f in sorted(_rt)], ignore_index=True).reset_index(drop=True)
        print(f"입력: runner_train_*.csv {len(_rt)}개 (collect 직접생성)")
    elif os.path.exists("ml_runner_v2.csv"):
        df = pd.read_csv("ml_runner_v2.csv").reset_index(drop=True)
        print("입력: ml_runner_v2.csv")
    elif os.path.exists("ml_runner.csv"):
        df = pd.read_csv("ml_runner.csv").reset_index(drop=True)
        print("입력: ml_runner.csv")
    else:
        print("runner 데이터 없음 — collect_ml_data 또는 prepare_runner_data_v2 먼저"); return
    # 라벨 자동 감지: 이벤트(next) 우선 + dt 백업
    targets = []
    if "y_win_next" in df.columns:
        targets.append(("y_win_next", "fav_next", "adv_next"))
    for dt in DT_LIST:
        if f"y_win_{dt}" in df.columns:
            targets.append((f"y_win_{dt}", f"fav_{dt}", f"adv_{dt}"))
    feats = get_features(df)
    groups = df["trade_id"].values if "trade_id" in df else np.arange(len(df))
    div_feats = [f for f in feats if f.startswith("div_") or f.startswith("rsi")]
    print(f"샘플 {len(df)} | feature {len(feats)}개 | 트레이드 {pd.Series(groups).nunique()}개")
    print(f"  divergence/RSI feature: {div_feats}")
    print(f"  전체 feature: {feats}\n")

    print("="*78)
    print(f"{'타깃':<16s}{'AUC':>8s}  {'Top importance (★=div/rsi)':<46s}")
    print("="*78)
    for yc, favc, advc in targets:
        if yc not in df.columns or df[yc].dropna().nunique() < 2:
            continue
        auc, imp = train_one(df, feats, yc, groups, True)
        order = np.argsort(imp)[::-1][:6]
        tops = ", ".join((("★" if feats[i] in div_feats else "") +
                          f"{feats[i]}({imp[i]:.02f})") for i in order)
        print(f"{yc:<16s}{auc:>8.4f}  {tops}")
        # 회귀 (있으면)
        if favc in df.columns and advc in df.columns:
            mae_f, _ = train_one(df, feats, favc, groups, False)
            mae_a, _ = train_one(df, feats, advc, groups, False)
            print(f"  {favc} MAE={-mae_f:.4f}   {advc} MAE={-mae_a:.4f}")
        # divergence 그룹만의 기여 (div feature 빼면 AUC 얼마 떨어지나)
        if div_feats:
            nondiv = [f for f in feats if f not in div_feats]
            auc_nd, _ = train_one(df, nondiv, yc, groups, True)
            print(f"  └ div/rsi 제외 AUC={auc_nd:.4f}  (Δ={auc-auc_nd:+.4f} = divergence 순기여)")
    print("="*78)
    print("판정:")
    print("  - y_win_next AUC > 0.55 → swing 시점 exit 신호 有 (이벤트 기반)")
    print("  - div/rsi 제외 시 AUC 하락폭(Δ)이 크면 → divergence가 진짜 기여 ✅")
    print("  - Δ≈0 → divergence는 상태 feature에 흡수 (entry RSI 추산 폐기와 같은 운명)")

    # ── EV exit용 모델 저장 (dt=12 주력: 자기상관 거품 아님 확인됨, 표본 큼)
    _save_ev_models(df, feats, groups)


def _save_ev_models(df, feats, groups, dt=12):
    """EV exit용 3모델 저장: P(win_dt), E[fav_dt], E[adv_dt]. 전체데이터 학습."""
    import json
    yc, favc, advc = f"y_win_{dt}", f"fav_{dt}", f"adv_{dt}"
    if yc not in df.columns:
        print(f"\n[EV모델] y_win_{dt} 없음 — 저장 스킵"); return
    sub = df.dropna(subset=[yc]).reset_index(drop=True)
    X = sub[feats].fillna(0).values
    print(f"\n[EV모델] dt={dt} 3모델 학습·저장 (n={len(sub)})")
    # P(win)
    mw = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        eval_metric='logloss', n_jobs=4)
    mw.fit(X, sub[yc].values)
    mw.save_model(f"runner_ev_pwin_dt{dt}.json")
    # E[fav], E[adv] (회귀, 있으면)
    for tgt, name in [(favc, "fav"), (advc, "adv")]:
        if tgt in sub.columns and sub[tgt].notna().sum() > 300:
            s2 = sub.dropna(subset=[tgt])
            mr = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                eval_metric='mae', n_jobs=4)
            mr.fit(s2[feats].fillna(0).values, s2[tgt].values)
            mr.save_model(f"runner_ev_{name}_dt{dt}.json")
    # feature 목록 저장 (backtest가 순서 맞춰 입력)
    with open(f"runner_ev_features.json", "w") as f:
        json.dump({"features": feats, "dt": dt}, f)
    print(f"  저장: runner_ev_pwin_dt{dt}.json, runner_ev_fav_dt{dt}.json, "
          f"runner_ev_adv_dt{dt}.json, runner_ev_features.json")
    print(f"  → backtest가 로드해 EV = P*E[fav] - (1-P)*E[adv] 계산, 임계 이하 청산")

if __name__ == "__main__":
    main()
