"""   get_feature_columns
ml_data.py — ML 학습 데이터 피처 정의 및 추출
═══════════════════════════════════════════════════════════════════
파이프라인:
  1차: TP ML  → tp_selector.py 에서 학습
  2차: Entry ML → train_entry.py 에서 학습

사용처:
  1) get_feature_columns()      — Entry 필터 피처 목록 (17개)
  2) extract_ml_dataset()       — Trade 리스트 → ML CSV DataFrame

변경 이력:
  v16: 32→17개 (Long/Short 분리 단변량 AUC 분석 기반)
  삭제: vol_ratio, range_pos_30d, sweep_duration_norm, bars_since_asia_break,
        bars_since_sweep, sweep_velocity, rangepos_x_sweep_depth,
        consistency_x_vol, return_div, vol_div, basis_z, liq_excursion_div,
        effective_fvg_strength, killzone_x_disp, sweep_x_disp
  v4: 16→14개 (다중공선성 제거)
  삭제: choch_x_liquidity (strong_displacement_score와 r=0.87)
        vp_confluence_x_displacement (vp_confluence와 r=0.90)
  v3: 25→16개 (다중공선성/노이즈 14개 삭제, VP 피처 5개 추가)
  삭제: choch_break_atr_ratio, net_move_x_pdpos (r=1.0 쌍둥이)
        net_move_x_vol (vol_ratio와 r=0.91)
        bars_since_sweep, recent_sweep_count (노이즈)
        choch_is_close_beyond, is_long, nasdaq_divergence_sweep_pct_atr (노이즈)
        hour_sin, hour_cos, dow (시간 → VP 대체)
        is_daily_level, is_ny_session_level (세션 더미)
  추가: vp_confluence, lvn_proximity, sweep_to_disp_fib
"""
from typing import List

import pandas as pd
import numpy as np


def get_feature_columns() -> List[str]:
    """Entry 필터 피처 (14개).

    v19 변경 (26→14개):
      VP regression + binary AUC 교차 분석 기반.
      두 기준 모두 무의미한 12개 제거.
      regression target(vp_score) 상관 |r| ≥ 0.03 또는 binary AUC |Δ| ≥ 0.015 유지.
    """
    return [
        # ── A. Displacement (1) — regression 상관 상위
        "net_move_atr_ratio",          # r=0.126  AUC=0.414  (기존 atr20 정규화)
        "sweep_depth_atr",             # r=0.100  AUC=0.420
        # strong_displacement_score 제거 (v20): 잘못된 철학 합성, 역방향 신호
        # ── A2. 신규 ATR 재정규화 변형 (sweet spot 흩어짐 ATR 원인 검증)
        "net_move_atr50",              # 신규: 50봉 단순평균 ATR 정규화
        "net_move_atr_wilder",         # 신규: Wilder ATR 정규화
        "net_move_atr1h",              # 신규: 1h ATR 정규화 (안정적)
        # ── A3. 신규 15m displacement 품질 (ATR 무관, body/wick)
        "disp_quality_15m",            # 신규: 15m 봉 방향성 품질 (NaN sentinel)
        # ── B. Market Context (1)
        "directional_er",              # r=0.062  AUC=0.528
        # ── C. Structure (2) — regression 상관 1위/2위
        "sweep_to_disp_fib",           # r=0.223  AUC=0.595 ★
        "lvn_proximity",               # r=0.141  AUC=0.510  (v3: NaN sentinel)
        "lvn_available",               # v3 신규: LVN 발견 여부 (0/1)
        # ── D. Market Regime (2) — adx_14 vs trend_structure 비교용 둘 다 유지
        "adx_14",                      # r=0.070  AUC=0.464  (대체 후보, 비교용 유지)
        "trend_structure",             # 신규: 1h MA 구조 (adx 대체, 방향보정)
        "ma_alignment",                # 신규: 1h MA 배열 정렬도 (-1/0/+1)
        "return_4h",                   # r=0.014  AUC=0.508 (regression FI 상위)
        # ── E. Sweep (1)
        "trap_alignment",              # r=0.022  AUC=0.521
        # ── F. Interaction (2)
        "lvn_x_fvg",                   # r=0.066  AUC=0.495  (v3: NaN 전파)
        # disp_x_fvg 제거 (v20): strong_displacement_score 인수 → 연쇄 폐기
        "rangepos_x_sweep_dur",        # r=0.041  AUC=0.525
        # ── G. Reclaim (3) — 신규 건강함 피처
        "reclaim_vol_spike",           # r=0.008  AUC=0.478 (기존, 비교용 유지)
        "reclaim_quality",             # 신규: 거부강도/체류페널티 (move/depth)/(sqrt(bars)+1)
        "reclaim_absorption",          # 신규: vol/speed 흡수 결합
        # ── H. FVG Quality (1)
        "fvg_composite_score",         # r=0.019  AUC=0.519
        "fib_tail_score",              # fib 꼬리 multiplier — 사이징 입력
        # ── I. Spot-Futures signed 괴리 (1) — stage7b/7c 검증 통과
        #     구 5종(sweep_z/reclaim_vol_z/slope×2/combined)은 abs 방향소거 +
        #     z-of-z 노이즈증폭으로 AUC 0.50 → 폐기. signed판 LONG AUC 0.5277,
        #     11/11 일관, R%-매칭 통과 (fib·risk_pct 분위내 유지)
        "spot_div_signed_z",           # sweep봉 signed bps/rolling_std, +=선물 깊은 wick=청산헌트
        "is_long",                     # 방향 플래그 (7c E2 재도입: 방향 비대칭 신호 상호작용 통로)
        # ── J. 반대편 유동성 선행회수 (sweep 추적, 신규 평가) — ICT 반대편 회수 후 반전.
        #     진입 셋업 sweep 직전 반대방향 독립 sweep이 있었나. 검증 후 거취 결정.
        "opp_sweep_present",           # 반대방향 선행 sweep 존재 (0/1)
        "opp_sweep_to_setup_bars",     # 반대 sweep → 셋업 sweep 5m봉수 (-1=없음)
        "opp_sweep_dist_atr",          # 반대 sweep extreme ~ 진입가 거리 / 4h atr (NaN=없음)
        "opp_sweep_is_last",           # 큐 바로직전 이벤트가 반대방향 (직접 전달; 동방향 끼면 0)
        "opp_sweep_unviolated",        # 반대 sweep 후 셋업까지 extreme 미재침범 (레인지 보존; NaN=구간없음)
        "opp_sweep_range_atr",         # 반대 extreme ~ 셋업 extreme / 4h atr (전달 레인지 높이; NaN=없음)
        # ── J. Basis (3) — 신규 평가 (중기 30h, basis_features.py)
        "basis_z",                     # 베이시스 Z-score (현물-선물 괴리 수준)
        "basis_momentum",              # 1h 베이시스 변화 (방향)
        "spot_lead",                   # 현물 선행 (스마트머니 신호)
        # ── K. Liquidation Heatmap (2) — 신규 평가 (14일 청산 풀 근사)
        "sweep_liq_consumed",          # sweep이 먹은 청산 풀 강도 (셋업 진정성)
        "liq_imbalance_hm",            # 위/아래 청산 풀 불균형 (방향 동력)
        # ── L. 4H FVG Lifecycle confluence (신규 layer — 3D 맥락)
        # ── L. 4H FVG Lifecycle confluence: 전량 삭제 (도구2: SL회피·러너 양쪽 무력,
        #    sweep 기준 재측정서도 +0.9~−1.5pp, idm×FVG 단변량 0.6233=구분불가).
        #    tracker 인프라(fvg_lifecycle.py)는 보존, 학습 feature만 제거.
        # (4H FVG 7개 삭제: in_4h_fvg, in_4h_fvg_aligned, is_4h_fvg_combined,
        #  is_4h_fvg_alone, nearest_4h_fvg_dist_atr, active_4h_fvg_fill, n_active_4h_fvg)
        # ── M. 1H FVG Lifecycle confluence: 전량 삭제 (4H와 동일 무력).
        # (1H FVG 7개 + 4h1h 결합 1개 삭제: in_1h_fvg, in_1h_fvg_aligned,
        #  is_1h_fvg_combined, is_1h_fvg_alone, nearest_1h_fvg_dist_atr,
        #  active_1h_fvg_fill, n_active_1h_fvg, is_4h1h_fvg_combined)
        # ── N. 미회수 유동성 장부 (swing 기반 liquidity, draw-on-liquidity)
        "unswept_above",               # 현재가 위 미회수 유동성 개수
        "unswept_below",               # 현재가 아래 미회수 유동성 개수
        "unswept_imbalance",           # (above-below)/total, +면 위쪽 쏠림
        "unswept_draw_aligned",        # 진입 방향 정합 draw (long=+imbalance, short=-)
        # ── O. TP모델 → Entry 이주: 게이트2 검증 결과 freshness만 채택.
        #    displacement_toward_candidate(net_move별칭 Δ+0.0007), entry_path_vp_integral
        #    (코어흡수 Δ+0.0010), htf_4h_aligned(Δ+0.0004) 삭제 — 전부 코어 중복.
        "level_freshness_score",       # 셋업 레벨 나이: max(0, 1-나이h/168)
        # ── O2. 레벨 그룹 더미 (freshness와 상호작용; 그룹별 성격차 학습)
        #    검증: 더미4+freshness Δ+0.0021 (depth3), 8/11심볼 양수.
        #    pw는 오래돼도 강함(SL회피67%, fresh중앙0.62) vs asia 신선해도 약함(55%).
        "is_pd",                       # 전일 H/L (pdh/pdl)
        "is_pw",                       # 전주 H/L (pwh/pwl)
        "is_asia",                     # asia 세션 H/L
        "is_eq",                       # equal high/low (eqh/eql)
    ]




def extract_ml_dataset(
    trades: list,
    symbol: str = "BTC",
    timeout_win_threshold: float = 0.5,
    timeout_loss_threshold: float = -0.3,
    drop_dead_zone: bool = False,
) -> pd.DataFrame:
    """
    Trade 리스트 → ML 학습용 DataFrame 변환.

    라벨링 (v3 — timeout threshold 파라미터화):
      - TP hit → label=1
      - SL hit → label=0
      - timeout + pnl_r >= timeout_win_threshold  → label=1
      - timeout + pnl_r <= timeout_loss_threshold → label=0
      - timeout + 그 사이 (dead zone):
          * drop_dead_zone=False (기본): label=0으로 분류
          * drop_dead_zone=True       : 샘플 제외

    Parameters
    ----------
    trades : list
        backtest trade 객체 리스트.
    symbol : str
    timeout_win_threshold : float, default 0.5
        timeout 시 이 R 이상이면 WIN으로 학습. (기존 1.5 → 0.5로 완화)
        환경변수 ML_TIMEOUT_WIN으로 override 가능 (collect_ml_data.py에서 전달).
    timeout_loss_threshold : float, default -0.3
        timeout 시 이 R 이하이면 LOSS로 학습.
    drop_dead_zone : bool, default False
        True면 dead zone (loss_thr < pnl_r < win_thr) 샘플 제외.

    peak_r: CSV에 저장 (분석용), 라벨링에는 미사용.
    """
    feature_cols = get_feature_columns()
    rows = []

    # 진단 카운터
    _n_tp, _n_sl = 0, 0
    _n_timeout_win, _n_timeout_loss, _n_timeout_dead = 0, 0, 0
    _n_no_fd = 0  # feature_dict 누락 카운터

    for t in trades:
        reason = t.reason or {}
        fd = reason.get("feature_dict", {})

        if not fd:
            _n_no_fd += 1
            continue

        # ── 메타 정보
        entry_price = float(t.entry_price) if t.entry_price else 0.0
        sl_price = float(t.initial_sl) if t.initial_sl else float(t.sl)
        tp_price = float(t.tp) if t.tp else 0.0
        peak_px = float(t.peak_px) if t.peak_px else entry_price
        pnl_usdt = float(t.pnl_usdt)

        # risk_dist (1R = SL까지 거리)
        risk_dist = abs(entry_price - sl_price)

        # pnl_r (청산가 기준)
        if risk_dist > 0 and entry_price > 0:
            if t.direction == "long":
                pnl_r = ((float(t.exit_price or entry_price) - entry_price) / risk_dist)
            else:
                pnl_r = ((entry_price - float(t.exit_price or entry_price)) / risk_dist)
        else:
            pnl_r = 0.0

        # peak_r (미실현 최대 R — 분석용, 라벨링 미사용)
        if risk_dist > 0 and entry_price > 0 and peak_px > 0:
            if t.direction == "long":
                peak_r = (peak_px - entry_price) / risk_dist
            else:
                peak_r = (entry_price - peak_px) / risk_dist
        else:
            peak_r = 0.0

        # ── 라벨 결정 (v3: 파라미터화)
        if t.exit_reason == "tp":
            label = 1.0
            _n_tp += 1
        elif t.exit_reason == "sl":
            label = 0.0
            _n_sl += 1
        elif t.exit_reason in ("timeout", "time_out_atr", "timeout_atr"):
            if pnl_r >= timeout_win_threshold:
                label = 1.0
                _n_timeout_win += 1
            elif pnl_r <= timeout_loss_threshold:
                label = 0.0
                _n_timeout_loss += 1
            else:
                # dead zone
                _n_timeout_dead += 1
                if drop_dead_zone:
                    continue
                label = 0.0  # 기본: LOSS로 분류
        else:
            continue

        # ── VP Score (regression target)
        # score = vp_consumption / (1 + vp_adversity)
        # + disp_extreme_broken 가산 (α=1.0)
        _vp_cons = float(getattr(t, 'vp_consumption', 0.0))
        _max_adv_px = float(getattr(t, 'max_adverse_px', 0.0))
        _vp_adv = abs(_max_adv_px - entry_price) / risk_dist if risk_dist > 0 else 0.0
        _deb = 1.0 if getattr(t, 'disp_extreme_broken', False) else 0.0

        _vp_base = _vp_cons / (1 + _vp_adv)
        _vp_score = _vp_base * (1 + _deb)  # α=1.0: broken이면 2배

        row = {
            "symbol":      symbol,
            "trade_id":    t.trade_id,
            "entry_time":  str(t.entry_time),
            "direction":   t.direction,
            "label":       label,
            "vp_score":    round(_vp_score, 4),
            "pnl_usdt":    round(pnl_usdt, 2),
            "pnl_r":       round(pnl_r, 2),
            "peak_r":      round(peak_r, 2),
            "mfe_r_clean": round(float(getattr(t, "mfe_r_clean", 0.0)), 4),
            "mdd_1r":      round(float(getattr(t, "mdd_1r", -1.0)), 4),
            "mdd_2r":      round(float(getattr(t, "mdd_2r", -1.0)), 4),
            "mdd_4r":      round(float(getattr(t, "mdd_4r", -1.0)), 4),
            "mdd_8r":      round(float(getattr(t, "mdd_8r", -1.0)), 4),
            "trail_path":  getattr(t, "trail_path", "") or "",
            "exit_reason":  t.exit_reason or "",
            "entry_atr":   round(float(t.entry_atr), 4) if t.entry_atr else 0.0,
            # ── Shadow exit (trail ON 동시 기록) — SHADOW_TRAIL=1일 때만 유효
            "shadow_pnl_r":      (round(float(t.shadow_pnl_r), 2)
                                  if getattr(t, "shadow_pnl_r", None) is not None else None),
            "shadow_exit_reason": getattr(t, "shadow_exit_reason", None) or "",
            # ── Timeout 스윕 (사후 4h/6h/8h/12h 재라벨링용)
            "bars_to_tp":  int(getattr(t, "bars_to_tp", -1)),
            "bars_to_sl":  int(getattr(t, "bars_to_sl", -1)),
            "r_at_48":     round(float(getattr(t, "r_at_48", 0.0)), 3),
            "r_at_72":     round(float(getattr(t, "r_at_72", 0.0)), 3),
            "r_at_96":     round(float(getattr(t, "r_at_96", 0.0)), 3),
            "r_at_144":    round(float(getattr(t, "r_at_144", 0.0)), 3),
            "ratr_at_48":  round(float(getattr(t, "ratr_at_48", 0.0)), 3),
            "ratr_at_72":  round(float(getattr(t, "ratr_at_72", 0.0)), 3),
            "ratr_at_96":  round(float(getattr(t, "ratr_at_96", 0.0)), 3),
            "ratr_at_144": round(float(getattr(t, "ratr_at_144", 0.0)), 3),
            # VP scoring (분석/향후 regression target용)
            "vp_consumption": round(float(getattr(t, 'vp_consumption', 0.0)), 4),
            "vp_adversity":   round(float(abs(getattr(t, 'max_adverse_px', 0.0) - entry_price) / risk_dist) if risk_dist > 0 else 0.0, 4),
            "disp_extreme_broken": int(getattr(t, 'disp_extreme_broken', False)),
            # ── 4H FVG confluence 진단용 (feature 아님, 가설 A/B 분석용)
            #    feature_dict(fd)에서 추출. 없으면 NaN.
            "combined_4h_fvg_fill":    fd.get("combined_4h_fvg_fill", np.nan),
            "combined_4h_fvg_touches": fd.get("combined_4h_fvg_touches", np.nan),
            "touch_to_entry_bars":     fd.get("touch_to_entry_bars", np.nan),
        }

        # ── 피처 추출
        # NaN sentinel 피처: 누락/미존재 시 0이 아니라 NaN (XGBoost native 처리)
        _nan_sentinel_cols = {"disp_quality_15m", "lvn_proximity", "lvn_x_fvg"}
        # (4H/1H FVG nan sentinel 4종 제거 — feature 삭제 동기화)
        # ── 레벨 그룹 더미 4종: setup_level_type 문자열에서 파생 (feature 루프 전)
        _slt = str(fd.get("setup_level_type", "") or "")
        fd["is_pd"]   = 1 if _slt in ("pdh", "pdl") else 0
        fd["is_pw"]   = 1 if _slt in ("pwh", "pwl") else 0
        fd["is_asia"] = 1 if _slt.startswith("asia") else 0
        fd["is_eq"]   = 1 if _slt in ("eqh", "eql") else 0

        for col in feature_cols:
            if col in _nan_sentinel_cols:
                row[col] = fd.get(col, np.nan)
            else:
                row[col] = fd.get(col, 0)

        # ── 4H FVG 진단 feature (meta, 학습 제외) — 약한 원인 분석
        for _dk in ["diag_dist_atr", "diag_state_code", "diag_bars_since_touch",
                    "diag_bars_since_first_touch", "diag_fill_ratio", "diag_zone_pos",
                    "diag_size_atr", "diag_n_same_dir_alive",
                    "diag_sweep_zone_pos", "diag_sweep_dist_atr", "diag_sweep_in_zone",
                    "opp_sweep_extreme_raw"]:
            row[_dk] = fd.get(_dk, np.nan)
        # 셋업 레벨 type (문자열 진단 — 레벨 오염 사후 식별)
        row["setup_level_type"] = fd.get("setup_level_type", "")

        rows.append(row)

    # ── feature_dict 누락 진단
    if _n_no_fd > 0:
        print(f"  ⚠️ [{symbol:<5s}] feature_dict 누락: {_n_no_fd}/{len(trades)}건 스킵됨!")

    # ── 라벨링 진단 출력 (심볼별)
    _n_total = _n_tp + _n_sl + _n_timeout_win + _n_timeout_loss + _n_timeout_dead
    if _n_total > 0:
        _n_win = _n_tp + _n_timeout_win
        _n_loss = _n_sl + _n_timeout_loss + (0 if drop_dead_zone else _n_timeout_dead)
        _wr = 100.0 * _n_win / (_n_win + _n_loss) if (_n_win + _n_loss) > 0 else 0.0
        _dead_str = "drop" if drop_dead_zone else "loss"
        print(f"  [{symbol:<5s}] labels: tp={_n_tp} sl={_n_sl} "
              f"to_win={_n_timeout_win}(>={timeout_win_threshold:+.2f}R) "
              f"to_loss={_n_timeout_loss}(<={timeout_loss_threshold:+.2f}R) "
              f"to_dead={_n_timeout_dead}({_dead_str}) "
              f"→ WR={_wr:.1f}%")

    if not rows:
        meta_cols = ["symbol", "trade_id", "entry_time", "direction",
                     "label", "vp_score", "pnl_usdt", "pnl_r", "peak_r",
                     "mfe_r_clean", "mdd_1r", "mdd_2r", "mdd_4r", "mdd_8r", "trail_path",
                     "exit_reason", "entry_atr",
                     "shadow_pnl_r", "shadow_exit_reason",
                     "bars_to_tp", "bars_to_sl", "r_at_48", "r_at_72", "r_at_96", "r_at_144",
                     "ratr_at_48", "ratr_at_72", "ratr_at_96", "ratr_at_144",
                     "vp_consumption", "vp_adversity", "disp_extreme_broken",
                     "combined_4h_fvg_fill", "combined_4h_fvg_touches", "touch_to_entry_bars", "diag_dist_atr", "diag_state_code", "diag_bars_since_touch", "diag_fill_ratio", "diag_zone_pos", "diag_size_atr", "diag_n_same_dir_alive", "diag_bars_since_first_touch", "diag_sweep_zone_pos", "diag_sweep_dist_atr", "diag_sweep_in_zone", "opp_sweep_extreme_raw", "setup_level_type"]
        return pd.DataFrame(columns=meta_cols + feature_cols)

    df = pd.DataFrame(rows)

    meta_cols = ["symbol", "trade_id", "entry_time", "direction",
                 "label", "vp_score", "pnl_usdt", "pnl_r", "peak_r",
                 "mfe_r_clean", "mdd_1r", "mdd_2r", "mdd_4r", "mdd_8r", "trail_path",
                 "exit_reason", "entry_atr",
                 "shadow_pnl_r", "shadow_exit_reason",
                     "bars_to_tp", "bars_to_sl", "r_at_48", "r_at_72", "r_at_96", "r_at_144",
                     "ratr_at_48", "ratr_at_72", "ratr_at_96", "ratr_at_144",
                 "vp_consumption", "vp_adversity", "disp_extreme_broken",
                 "combined_4h_fvg_fill", "combined_4h_fvg_touches", "touch_to_entry_bars", "diag_dist_atr", "diag_state_code", "diag_bars_since_touch", "diag_fill_ratio", "diag_zone_pos", "diag_size_atr", "diag_n_same_dir_alive", "diag_bars_since_first_touch", "diag_sweep_zone_pos", "diag_sweep_dist_atr", "diag_sweep_in_zone", "opp_sweep_extreme_raw", "setup_level_type"]
    ordered = meta_cols + feature_cols
    for c in ordered:
        if c not in df.columns:
            df[c] = 0
    df = df[ordered]

    return df
