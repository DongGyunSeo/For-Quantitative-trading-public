# backtest.py ML_data
"""
ICT 전략 백테스터 — incremental update 버전

★ ML 학습용 세팅 (원래값 → 변경값):
  - TRAIL_ACTIVATE_R:   2.0 → 1000.0  (트레일링 비활성화)
  - TRAIL_ACTIVATE_R_A: 1.5 → 1000.0  (트레일링 비활성화)
  - TRAIL_BREAKEVEN_R:  1.0 → 1000.0  (본절 SL 비활성화)
  - BREAKEVEN_R (내부): 1.0 → 1000.0  (본절 SL 비활성화)
  - timeout: 고정 60봉 → displacement 기반 동적 계산
      timeout_bars = max(24, min(144, net_move_atr_ratio*8 + bars_since_sweep*0.8))
  - TP fallback: 후보 없을 시 1.5R 고정 TP 설정

★ Swing 변경:
  - find_swings → find_swings_enhanced (TF별 dynamic lookback)
  - 5m/15m: lb_low=2, lb_high=4 (평균 lb≈3)
  - 1h/4h:  lb_low=3, lb_high=5 (평균 lb≈4)

★ Liquidity 변경:
  - NY H/L 삭제, IDM 추가
  - liq_pool.tick()에 df_15m_struct 전달

  복구: 위 상수들을 원래값으로 되돌리고, timeout/TP fallback 블록 제거

변경사항 (원본):
  - FVG_LOOKBACK_BARS 상수 누락 수정
  - incremental update 적용 (engine.py 동일 방식)
  - 거래 빈도 완화
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd
import numpy as np

from swings import find_swings_enhanced
from pivots import build_pivots, compress_pivots
from timeframes import resample_5m_to_15m_1h, resample_5m_to_4h
from liquidity import SessionLiquidityPool        # 세션 기반 유동성 풀 (v3.0)
from struct_event import detect_structure_events_close_protected
from pd_zones import add_pd_bounds_from_strong_weak, DealingRangeTracker
from order_blocks import (
    build_order_blocks_from_bos,
    mark_invalidation_by_close,
    select_latest_two_obs,
)
from fvg import detect_fvgs_lookback
from fvg_lifecycle import FVGLifecycleTracker
from scoring import compute_total_score, calc_risk_capped_position_value
from signals import SweepReclaimOTE

# ── ML 필터 (선택적 로드)
try:
    from train_entry import EntryFilter as _EntryFilter
    HAS_ML = True
except ImportError:
    HAS_ML = False

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
LEVERAGE         = 30
MAX_RISK_FRAC    = 0.05
TAKER_FEE_RATE   = 0.0005
FVG_LOOKBACK_BARS = 200      # ← 누락됐던 상수
WARMUP_BARS      = 4032   # 14일치 (PWH/PWL 계산용)
TAIL_LEN_FACTOR  = 2         # tail = lookback * TAIL_LEN_FACTOR + 5

# 완화 파라미터
SWEEP_BUFFER_PCT  = 0.0005   # pivot level에 ±0.05% 버퍼 (sweep 감지 완화)
RECLAIM_SECONDS   = 1800     # STANDBY timeout 30분 (6봉)
OB_WAIT_BARS      = 6        # OB 없을 때 reset 대신 최대 6봉 대기
ARMED_TIMEOUT_MULT = 1.5     # CHOCH/BOS 평균 간격 × 배수 → ARMED timeout
ARMED_TIMEOUT_MIN  = 14      # ARMED timeout 최소값 (5m봉, ~1h 10m)
ARMED_TIMEOUT_MAX  = 14      # ARMED timeout 최대값 (5m봉, ~1h 10m)

# ── 데이터 크기 관리 (메모리 누수 방지)
# 5m 기준: 1일=288봉, 1주=2016봉
OBS_MAX_AGE_BARS  = 2016       # 5m OB: 1주일 이전 생성 + invalidated → 삭제
OBS_MAX_ROWS      = 2016 * 2   # 5m OB: 총 행 수 상한 (2주치)
STRUCT_MAX_ROWS   = 2016 * 2   # struct DataFrame: 최대 2주치 행

# HTF OB 만료 (1H/4H 전용 — 4주)
OBS_HTF_MAX_AGE_BARS_1H = 24 * 28     # 1H OB: 4주 = 672봉
OBS_HTF_MAX_AGE_BARS_4H = 6 * 28      # 4H OB: 4주 = 168봉
OBS_HTF_MAX_ROWS        = 500         # HTF OB 총 행 수 상한



# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:     int
    direction:    str
    entry_price:  float
    sl:           float
    tp:           Optional[float]
    size_usdt:    float
    entry_time:   pd.Timestamp
    exit_time:    Optional[pd.Timestamp] = None
    exit_price:   Optional[float]        = None
    exit_reason:  Optional[str]          = None
    pnl_usdt:     float                  = 0.0
    equity_after: float                  = 0.0
    reason:       Optional[dict]         = None   # 진입 근거
    max_unrealized_pnl: float            = 0.0    # 포지션 유지 중 최고 미실현 손익
    trailing_activated: bool             = False  # 트레일링 스탑 활성화 여부
    entry_bar_idx:      int              = 0      # 진입 봉 인덱스 (trail SL용)
    sl_history: Optional[list]           = None   # SL 이동 이력 [(time, sl), ...]
    peak_px:    float                    = 0.0    # 진입 후 최고(long)/최저(short) 가격 추적
    entry_atr:  float                    = 0.0    # 진입 시점 5m ATR(14) — MFE 계산용
    initial_sl: float                    = 0.0    # 초기 SL (트레일링 전) — MFE 계산용
    # ── MFE/trail (러너 라벨 + 트레일 EV 사후재구성용; 매봉 누적, 청산 시 확정)
    mfe_r_clean: float                   = 0.0    # 초기SL 터치 전까지 clean MFE (R단위)
    _mfe_best_r: float                   = 0.0    # 내부: 진행중 best favorable R (clean)
    _sl_touched: bool                    = False  # 내부: 초기SL 터치 여부 (clean 종료조건)
    trail_path: str                      = ""     # 봉별 (favR:advR) 경로 압축 (트레일 replay용)
    _trail_buf: Optional[list]           = None   # 내부: trail_path 누적 버퍼
    # ── swing_cb 트레일 상태 (stage6 replay의 last_ref/ref_broken/locks 라이브化)
    _cb_last_ref: float                  = 0.0    # 마지막 확정 기준스윙 (long=swing high)
    _cb_has_ref: bool                    = False  # 기준스윙 존재 여부
    _cb_ref_broken: bool                 = True   # BOS 가능 여부 (새 ref 확정 전 True)
    _cb_ref_idx: int                     = -1     # 현재 기준스윙의 pivot idx (중복갱신 방지)
    _cb_locks: Optional[list]            = None   # (미사용, 호환 보존)

    # ── Timeout 스윕용: 각 timeout 지점의 상태 (한 번 수집으로 4h~12h 비교)
    #   bars_to_tp/sl: TP/SL 도달까지 봉 수 (미도달 시 -1)
    #   r_at_N       : N봉 시점의 R-multiple (그 시점 종가 기준)
    #   timeout을 짧게 줬을 때의 label을 사후 재계산하는 데 사용.
    bars_to_tp:  int                     = -1
    bars_to_sl:  int                     = -1
    r_at_48:     float                   = 0.0    # 4h
    r_at_72:     float                   = 0.0    # 6h
    r_at_96:     float                   = 0.0    # 8h
    r_at_144:    float                   = 0.0    # 12h
    # ATR 정규화 버전: (close-entry)/ATR — SL 길이에 독립적인 움직임 품질
    #   timeout label을 상위 n% 컷으로 매길 때 사용 (SL 타이트 outlier 방지)
    ratr_at_48:  float                   = 0.0
    ratr_at_72:  float                   = 0.0
    ratr_at_96:  float                   = 0.0
    ratr_at_144: float                   = 0.0
    # VP Scoring fields
    vp_consumption:  float               = 0.0    # Σ(crossed VP density) / mean_density
    vp_prev_close:   float               = 0.0    # 이전 봉 close (bar-by-bar 누적용)
    max_adverse_px:  float               = 0.0    # 최대 불리 가격 (adversity 계산용)
    disp_extreme:    float               = 0.0    # displacement extreme 가격
    disp_extreme_broken: bool            = False  # disp_extreme 돌파 여부

    # ── Shadow exit (대체 청산 방식 동시 추적)
    # 실제 청산(primary)과 별개로 "trail 설정이 달랐다면 어디서 청산됐을지"를
    # 같은 trade에서 동시에 계산. 한 번의 진입 → 두 가지 청산 결과 기록.
    # 진입/trade 시퀀스는 primary 기준으로 고정되므로 두 시나리오가 갈라지지 않음.
    shadow_enabled:       bool           = False  # shadow 추적 여부
    shadow_sl:            float          = 0.0    # shadow 청산용 SL (별도 trail 로직)
    shadow_trailing_activated: bool      = False
    shadow_breakeven_set: bool           = False
    shadow_peak_px:       float          = 0.0
    shadow_exit_price:    Optional[float] = None
    shadow_exit_time:     Optional[pd.Timestamp] = None
    shadow_exit_reason:   Optional[str]  = None
    shadow_pnl_r:         Optional[float] = None

    def __post_init__(self):
        if self.sl_history is None:
            self.sl_history = []
        if self._trail_buf is None:
            self._trail_buf = []
        if self._cb_locks is None:
            self._cb_locks = []


@dataclass
class BacktestResult:
    trades:         List[Trade]
    equity_curve:   List[float]
    final_equity:   float
    initial_equity: float

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity - self.initial_equity) / self.initial_equity * 100

    @property
    def win_trades(self):
        return [t for t in self.trades if t.pnl_usdt > 0]

    @property
    def lose_trades(self):
        return [t for t in self.trades if t.pnl_usdt <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.win_trades) / len(self.trades) * 100 if self.trades else 0.0

    @property
    def avg_win(self) -> float:
        return sum(t.pnl_usdt for t in self.win_trades) / len(self.win_trades) if self.win_trades else 0.0

    @property
    def avg_loss(self) -> float:
        return sum(t.pnl_usdt for t in self.lose_trades) / len(self.lose_trades) if self.lose_trades else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak, max_dd = self.equity_curve[0], 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100)
        return max_dd

    @property
    def profit_factor(self) -> float:
        gross_win  = sum(t.pnl_usdt for t in self.win_trades)
        gross_loss = abs(sum(t.pnl_usdt for t in self.lose_trades))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

CACHE_DIR = "cache_5m"  # 캐시 디렉토리


def fetch_historical_5m(
    exchange: ccxt.Exchange,
    symbol: str,
    months: int = 3,
    end_date: str = None,  # "2025-03-25" 형식, None이면 현재
) -> pd.DataFrame:
    import time as _time
    import os

    if end_date is not None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    since_dt = end_dt - timedelta(days=30 * months)
    since_ms = int(since_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    # ── 캐시 파일명 결정
    # "BTC/USDT:USDT" → "BTC_5m_futures.csv",  "BTC/USDT" → "BTC_5m_spot.csv"
    _sym_short = symbol.split("/")[0]
    _is_futures = ":USDT" in symbol or "SWAP" in symbol.upper()
    _suffix = "futures" if _is_futures else "spot"
    os.makedirs(CACHE_DIR, exist_ok=True)
    _cache_path = os.path.join(CACHE_DIR, f"{_sym_short}_5m_{_suffix}.csv")

    # ── 캐시 확인: 파일이 있고, 기간을 커버하면 로드
    if os.path.exists(_cache_path):
        try:
            df_cache = pd.read_csv(_cache_path)
            df_cache["time"] = pd.to_datetime(df_cache["time"], utc=True)
            cache_start = df_cache["time"].iloc[0]
            cache_end = df_cache["time"].iloc[-1]

            # 캐시 커버리지: 요청 기간의 95% 이상 커버하면 OK
            requested_span = (end_dt - since_dt).total_seconds()
            cached_span = (min(cache_end, end_dt) - max(cache_start, since_dt)).total_seconds()
            coverage = cached_span / requested_span if requested_span > 0 else 0

            if coverage >= 0.95:
                df_out = df_cache[(df_cache["time"] >= since_dt) &
                                  (df_cache["time"] <= end_dt)].reset_index(drop=True)
                print(f"[data] {symbol} 캐시 로드: {_cache_path}  ({len(df_out):,}봉, 커버리지 {coverage*100:.0f}%)")
                return df_out
            else:
                print(f"[data] {symbol} 캐시 커버리지 부족 ({coverage*100:.0f}%) → 재다운로드")
        except Exception as e:
            print(f"[data] 캐시 로드 실패: {e} → 재다운로드")

    # ── API 다운로드
    _end_str = end_dt.strftime('%Y-%m-%d %H:%M')
    print(f"[data] {symbol} 5m 데이터 로드 중 ({months}개월) ...")
    print(f"       시작: {since_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"       종료: {_end_str} UTC")

    all_bars, limit = [], 300
    _max_retries = 5
    _retry_count = 0

    while True:
        try:
            _time.sleep(0.1)
            bars = exchange.fetch_ohlcv(symbol, timeframe="5m", since=since_ms, limit=limit)
            _retry_count = 0
        except Exception as e:
            _retry_count += 1
            if _retry_count >= _max_retries:
                print(f"\n[data] ⚠️ 연속 {_max_retries}회 에러 → 중단: {e}")
                break
            _wait = min(2 ** _retry_count + 1, 15)  # 3, 5, 9, 15초
            print(f"       [data] API 재시도 {_retry_count}/{_max_retries} ({_wait}s 대기) — {e}", end="\r")
            _time.sleep(_wait)
            continue

        if not bars:
            break

        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts >= end_ms:
            break
        if end_date is None and last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 5 * 60 * 1000:
            break
        since_ms  = last_ts + 1
        loaded_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"       로드 중 ... {loaded_dt.strftime('%Y-%m-%d %H:%M')}  ({len(all_bars):,}봉)", end="\r")

    df = pd.DataFrame(all_bars, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["open","high","low","close"])

    if end_date is not None:
        df = df[df["time"] <= end_dt].reset_index(drop=True)

    if len(df) > 0:
        print(f"\n[data] 로드 완료: {len(df):,}봉  "
              f"({df['time'].iloc[0].strftime('%Y-%m-%d')} ~ {df['time'].iloc[-1].strftime('%Y-%m-%d')})")
        # ── 캐시 저장
        try:
            df.to_csv(_cache_path, index=False)
            print(f"[data] 캐시 저장: {_cache_path}")
        except Exception as e:
            print(f"[data] 캐시 저장 실패: {e}")
    else:
        print(f"\n[data] ⚠️ 데이터 없음")

    return df


# ──────────────────────────────────────────────
# Incremental State (backtest용)
# ──────────────────────────────────────────────

class _IncrState:
    """
    백테스트용 incremental state.
    engine.py의 IncrementalState와 동일한 로직.
    """

    def __init__(self, window_5m: pd.DataFrame, lb: int, eq_band_pct: float = 0.003,
                 min_eq_gap_bars_15m: int = 0,
                 min_eq_gap_bars_1h: int = 0):
        self.lb                  = lb
        self.eq_band_pct         = eq_band_pct
        self.min_eq_gap_bars_15m = min_eq_gap_bars_15m
        self.min_eq_gap_bars_1h  = min_eq_gap_bars_1h

        # ── TF별 dynamic lookback 설정
        # 5m/15m: 평균 lb≈3 (lb_low=2, lb_high=4)
        # 1h/4h:  평균 lb≈4 (lb_low=3, lb_high=5)
        self._lb_5m  = {"lb_low": 2, "lb_high": 4}   # 평균 lb≈3
        self._lb_15m = {"lb_low": 2, "lb_high": 4}   # 평균 lb≈3
        self._lb_1h  = {"lb_low": 3, "lb_high": 5}   # 평균 lb≈4
        self._lb_4h  = {"lb_low": 3, "lb_high": 5}   # 평균 lb≈4

        # tail 길이는 최대 lb_high 기준으로 계산 (5가 최대)
        _max_lb = 5
        self._tail_len           = _max_lb * TAIL_LEN_FACTOR + 5

        # 초기 full 계산
        df_15m, df_1h = resample_5m_to_15m_1h(window_5m, time_col="time", tz="UTC", drop_incomplete=True)
        df_4h = resample_5m_to_4h(window_5m, time_col="time", tz="UTC", drop_incomplete=True)

        self.df_5m  = window_5m.copy()
        self.df_15m = df_15m.copy()
        self.df_1h  = df_1h.copy()
        self.df_4h  = df_4h.copy()

        self.df_5m_struct,  self.pivots_5m  = self._full_tf(window_5m)
        self.df_15m_struct, self.pivots_15m = self._full_tf(df_15m)
        self.df_1h_struct,  self.pivots_1h  = self._full_tf(df_1h)
        self.df_4h_struct,  self.pivots_4h  = self._full_tf(df_4h)

        self.df_1h_struct = add_pd_bounds_from_strong_weak(self.df_1h_struct, fallback_to_last_swing=True)
        last_1h = self.df_1h_struct.iloc[-1]
        self._legacy_pd_low  = float(last_1h["pd_low"])  if pd.notna(last_1h.get("pd_low"))  else None
        self._legacy_pd_high = float(last_1h["pd_high"]) if pd.notna(last_1h.get("pd_high")) else None

        # v2: DealingRangeTracker (BOS origin + IDM sweep confirmation)
        self.pd_tracker = DealingRangeTracker()
        self.pd_tracker.update(self.df_1h_struct, self.pivots_1h)
        dr_low, dr_high = self.pd_tracker.get_dealing_range()
        self.pd_low  = dr_low  if dr_low  is not None else self._legacy_pd_low
        self.pd_high = dr_high if dr_high is not None else self._legacy_pd_high

        self.htf_state = str(self.df_1h_struct["structure_state"].iloc[-1])
        # 4H 방향 상태
        self.htf_4h_state = str(self.df_4h_struct["structure_state"].iloc[-1]) if len(self.df_4h_struct) > 0 else "unknown"

        self.obs_5m_df = self._build_obs(self.df_5m_struct)
        self.obs_1h_df = self._build_obs(self.df_1h_struct)
        self.obs_4h_df = self._build_obs(self.df_4h_struct)

        # HTF FVG (1H, 4H)
        self.fvgs_1h = self._build_fvgs(self.df_1h_struct)
        self.fvgs_4h = self._build_fvgs(self.df_4h_struct)

        # 세션 기반 유동성 풀 초기화
        self.liq_pool  = SessionLiquidityPool(
            eq_band_pct    = eq_band_pct,
        )
        # 워밍업 데이터를 pool에 피드 (히스토리 구축)
        for _, row in window_5m.iterrows():
            self.liq_pool.feed_bar(row.to_dict())
        # 마지막 봉 기준으로 초기 레벨 계산
        if not window_5m.empty:
            last_bar = window_5m.iloc[-1].to_dict()
            ts_last  = pd.Timestamp(last_bar["time"])
            if ts_last.tzinfo is None:
                ts_last = ts_last.tz_localize("UTC")
            today = ts_last.normalize()
            # PDH/PDL, PWH/PWL 초기 계산
            from liquidity import build_pdhl, build_pwhl, build_session_hl
            import pandas as _pd
            for lv in build_pdhl(self.liq_pool._df_5m_history, today):
                self.liq_pool._levels.append(lv)
            for lv in build_pwhl(self.liq_pool._df_5m_history, today):
                self.liq_pool._levels.append(lv)
            # Asia H/L 초기화: warmup 마지막 봉 시각 기준
            # 05:00(pre_london) 이후면 당일 아시아 완료 → today 사용
            # 05:00 이전(asia 세션 중)이면 전날 아시아 사용
            from liquidity import _session_of as _sess
            _last_sess = _sess(ts_last)
            _asia_date = today if _last_sess != "asia" else today - _pd.Timedelta(days=1)
            asia_lv = build_session_hl(self.liq_pool._df_5m_history, "asia", _asia_date)
            if not asia_lv:  # fallback: 반대 날짜 시도
                _fallback = today - _pd.Timedelta(days=1) if _last_sess != "asia" else today
                asia_lv = build_session_hl(self.liq_pool._df_5m_history, "asia", _fallback)
            for lv in asia_lv:
                self.liq_pool._levels.append(lv)
            # London H/L: v2.2에서 제거 확정 (confluence뿐 아니라 sweep 레벨 생성 자체 차단).
            #   기존엔 여기서 _levels에 append돼 sweep 후보로 부활 → 셋업/큐 오염.
            #   제거 사유: london 세션 H/L이 crypto 24h 시장에서 변별 미미 + 노이즈.
            # (london 생성 블록 비활성화)
            # EQL/EQH 초기 계산
            self.liq_pool._rebuild_eq_levels(self.pivots_15m, today)

        self._last_15m_time = df_15m["time"].iloc[-1] if len(df_15m) else None
        self._last_1h_time  = df_1h["time"].iloc[-1]  if len(df_1h)  else None
        self._last_4h_time  = df_4h["time"].iloc[-1]  if len(df_4h)  else None

        # ── 4H FVG Lifecycle Tracker (confluence 맥락용)
        # 환경변수로 파라미터 스윕 가능
        self.fvg_tracker_4h = FVGLifecycleTracker(
            tf="4h",
            n_depart=int(os.getenv("FVG4H_N_DEPART", "5")),
            n_depart_arm=int(os.getenv("FVG4H_N_DEPART_ARM", "4")),
            timeout_bars=int(os.getenv("FVG4H_TIMEOUT", "360")),
            fill_dead=float(os.getenv("FVG4H_FILL_DEAD", "0.5")),
            min_size_atr=float(os.getenv("FVG4H_MIN_SIZE_ATR", "0.3")),
            max_slots=int(os.getenv("FVG4H_MAX_SLOTS", "20")),
        )
        # 워밍업: 현재까지의 4H 봉들로 tracker 초기화 (FVG 히스토리 확보)
        self._fvg4h_known_idx = 0  # 마지막으로 detect한 4H 길이
        self._prime_fvg_tracker_4h()

        # ── 1H FVG Lifecycle Tracker (confluence 맥락용, 1H 검증)
        # timeout 60일=1440 1H봉. 떠남봉수는 4H 비율 유지(시작값 4)
        self.fvg_tracker_1h = FVGLifecycleTracker(
            tf="1h",
            n_depart=int(os.getenv("FVG1H_N_DEPART", "5")),
            n_depart_arm=int(os.getenv("FVG1H_N_DEPART_ARM", "4")),
            timeout_bars=int(os.getenv("FVG1H_TIMEOUT", "720")),
            fill_dead=float(os.getenv("FVG1H_FILL_DEAD", "0.5")),
            min_size_atr=float(os.getenv("FVG1H_MIN_SIZE_ATR", "0.3")),
            max_slots=int(os.getenv("FVG1H_MAX_SLOTS", "40")),
        )
        self._fvg1h_known_idx = 0
        self._prime_fvg_tracker_1h()

    def _atr_4h(self) -> float:
        """4H Wilder-근사 ATR(14). FVG min_size 판정용."""
        df = self.df_4h
        if df is None or len(df) < 15:
            return 0.0
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        tr = np.maximum(h[1:]-l[1:],
                        np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        return float(tr[-14:].mean()) if len(tr) >= 14 else 0.0

    def _prime_fvg_tracker_4h(self):
        """워밍업 구간 4H 봉들을 tracker에 순차 공급 (시뮬 시작 전 FVG 히스토리)."""
        df = self.df_4h
        if df is None or len(df) < 3:
            return
        all_fvgs = detect_fvgs_lookback(df, lookback_bars=len(df),
                                        fill_mode="wick", remove_filled=False,
                                        min_size=0.0)
        atr = self._atr_4h()
        for i in range(len(df)):
            new = all_fvgs[all_fvgs["created_idx"] == i] if len(all_fvgs) else None
            if new is not None and len(new) and atr > 0:
                self.fvg_tracker_4h.add_new_fvgs(new, atr=atr, cur_idx=i)
            row = df.iloc[i]
            self.fvg_tracker_4h.update(i, bar_low=float(row["low"]),
                                       bar_high=float(row["high"]),
                                       bar_close=float(row["close"]))
        self._fvg4h_known_idx = len(df)

    def _atr_1h(self) -> float:
        """1H Wilder-근사 ATR(14). 1H FVG min_size 판정용."""
        df = self.df_1h
        if df is None or len(df) < 15:
            return 0.0
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        tr = np.maximum(h[1:]-l[1:],
                        np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        return float(tr[-14:].mean()) if len(tr) >= 14 else 0.0

    def _prime_fvg_tracker_1h(self):
        """워밍업 구간 1H 봉들로 1H tracker 초기화."""
        df = self.df_1h
        if df is None or len(df) < 3:
            return
        all_fvgs = detect_fvgs_lookback(df, lookback_bars=len(df),
                                        fill_mode="wick", remove_filled=False,
                                        min_size=0.0)
        atr = self._atr_1h()
        for i in range(len(df)):
            new = all_fvgs[all_fvgs["created_idx"] == i] if len(all_fvgs) else None
            if new is not None and len(new) and atr > 0:
                self.fvg_tracker_1h.add_new_fvgs(new, atr=atr, cur_idx=i)
            row = df.iloc[i]
            self.fvg_tracker_1h.update(i, bar_low=float(row["low"]),
                                       bar_high=float(row["high"]),
                                       bar_close=float(row["close"]))
        self._fvg1h_known_idx = len(df)

    def get_liq_all(self, bar_idx: int = 0) -> list:
        """SessionLiquidityPool에서 현재 활성 레벨 반환."""
        return self.liq_pool.get_levels(current_bar_idx=bar_idx, do_merge=True)

    @property
    def liq_all(self) -> list:
        return self.liq_pool.get_levels(do_merge=True)

    # ── 업데이트 ──

    def update(self, new_bar: pd.Series):
        self.df_5m = self._append(self.df_5m, new_bar)

        new_15m, new_1h = resample_5m_to_15m_1h(
            self.df_5m, time_col="time", tz="UTC", drop_incomplete=True
        )
        new_4h = resample_5m_to_4h(
            self.df_5m, time_col="time", tz="UTC", drop_incomplete=True
        )


        # 5m incremental
        self._update_incremental_5m()

        # 15m
        if len(new_15m) > 0:
            latest = new_15m["time"].iloc[-1]
            if self._last_15m_time is None or latest > self._last_15m_time:
                self._last_15m_time = latest
                self.df_15m = new_15m
                self._update_incremental_tf("15m")

        # 1h
        if len(new_1h) > 0:
            latest = new_1h["time"].iloc[-1]
            if self._last_1h_time is None or latest > self._last_1h_time:
                self._last_1h_time = latest
                self.df_1h = new_1h
                self._update_1h_full()
                # ── 1H FVG Lifecycle tracker 전진 (새 1H봉 닫힘)
                self._advance_fvg_tracker_1h()

        # 4h — structure_state 업데이트
        if len(new_4h) > 0:
            latest_4h = new_4h["time"].iloc[-1]
            if not hasattr(self, "_last_4h_time") or self._last_4h_time is None or latest_4h > self._last_4h_time:
                self._last_4h_time = latest_4h
                self.df_4h = new_4h
                self._update_4h_state()
                # ── 4H FVG Lifecycle tracker 전진 (새 4H봉 닫힘)
                self._advance_fvg_tracker_4h()

        # 세션 유동성 풀 tick (매 5m 봉마다)
        bar_idx = len(self.df_5m) - 1
        self.liq_pool.tick(
            bar        = new_bar.to_dict(),
            pivots_15m = self.pivots_15m,
            current_bar_idx = float(bar_idx),
            df_15m_struct = self.df_15m_struct,
        )

    def get_obs_5m(self):
        return select_latest_two_obs(self.obs_5m_df, t_now=self.df_5m["time"].iloc[-1])

    def get_obs_1h(self):
        return select_latest_two_obs(self.obs_1h_df, t_now=self.df_5m["time"].iloc[-1])

    def get_obs_4h(self):
        return select_latest_two_obs(self.obs_4h_df, t_now=self.df_5m["time"].iloc[-1])

    def get_fvgs_1h(self):
        """현재 유효한 1H FVG DataFrame 반환."""
        return self.fvgs_1h

    def get_fvgs_4h(self):
        """현재 유효한 4H FVG DataFrame 반환."""
        return self.fvgs_4h

    # ── 내부 메서드 ──

    def _update_incremental_5m(self):
        tail = self.df_5m.tail(self._tail_len + 5).copy()  # lb_high=4 기준 여유
        tail_sw = find_swings_enhanced(tail, **self._lb_5m)
        p_raw   = build_pivots(tail_sw)
        if not p_raw.empty:
            p_c = compress_pivots(p_raw, mode="zigzag")
            offset = len(self.df_5m) - len(tail)
            p_c["idx"] += offset
            self.pivots_5m = self._merge_pivots(self.pivots_5m, p_c)

        # tail reset_index + pivots를 상대 idx로 변환
        tail_offset = len(self.df_5m) - len(tail)
        tail_reset  = tail.reset_index(drop=True)
        if not self.pivots_5m.empty:
            p_for_struct = self.pivots_5m.copy()
            p_for_struct["idx"] = p_for_struct["idx"] - tail_offset
            p_for_struct = p_for_struct[
                (p_for_struct["idx"] >= 0) & (p_for_struct["idx"] < len(tail_reset))
            ].reset_index(drop=True)
        else:
            p_for_struct = pd.DataFrame(columns=["idx","time","kind","price"])
        tail_struct = detect_structure_events_close_protected(tail_reset, p_for_struct)
        self.df_5m_struct = self._merge_struct(self.df_5m_struct, tail_struct, len(tail))
        self._update_obs("obs_5m_df", tail_struct, self.df_5m)

    def _update_incremental_tf(self, tf: str):
        df = self.df_15m if tf == "15m" else self.df_1h
        _lb_cfg = self._lb_15m if tf == "15m" else self._lb_1h
        tail = df.tail(self._tail_len + 5).copy()
        tail_sw = find_swings_enhanced(tail, **_lb_cfg)
        p_raw   = build_pivots(tail_sw)
        if not p_raw.empty:
            p_c = compress_pivots(p_raw, mode="zigzag")
            offset = len(df) - len(tail)
            p_c["idx"] += offset
            if tf == "15m":
                self.pivots_15m = self._merge_pivots(self.pivots_15m, p_c)
            else:
                self.pivots_1h = self._merge_pivots(self.pivots_1h, p_c)

        # tail reset_index + pivots를 상대 idx로 변환
        tail_offset  = len(df) - len(tail)
        tail_reset   = tail.reset_index(drop=True)
        pivots_src   = self.pivots_15m if tf == "15m" else self.pivots_1h
        if not pivots_src.empty:
            p_for_struct = pivots_src.copy()
            p_for_struct["idx"] = p_for_struct["idx"] - tail_offset
            p_for_struct = p_for_struct[
                (p_for_struct["idx"] >= 0) & (p_for_struct["idx"] < len(tail_reset))
            ].reset_index(drop=True)
        else:
            p_for_struct = pd.DataFrame(columns=["idx","time","kind","price"])
        tail_struct = detect_structure_events_close_protected(tail_reset, p_for_struct)
        if tf == "15m":
            self.df_15m_struct = self._merge_struct(self.df_15m_struct, tail_struct, len(tail))
            pass  # EQL/EQH/Trendline은 liq_pool.tick()이 아시아 종료 시 갱신
        else:
            self.df_1h_struct = self._merge_struct(self.df_1h_struct, tail_struct, len(tail))

    def _update_1h_full(self):
        self._update_incremental_tf("1h")

        self.df_1h_struct = add_pd_bounds_from_strong_weak(self.df_1h_struct, fallback_to_last_swing=True)
        last_1h = self.df_1h_struct.iloc[-1]
        self._legacy_pd_low  = float(last_1h["pd_low"])  if pd.notna(last_1h.get("pd_low"))  else None
        self._legacy_pd_high = float(last_1h["pd_high"]) if pd.notna(last_1h.get("pd_high")) else None

        # v2: DealingRangeTracker 증분 업데이트
        self.pd_tracker.update(self.df_1h_struct, self.pivots_1h)
        dr_low, dr_high = self.pd_tracker.get_dealing_range()
        self.pd_low  = dr_low  if dr_low  is not None else self._legacy_pd_low
        self.pd_high = dr_high if dr_high is not None else self._legacy_pd_high

        self.htf_state = str(self.df_1h_struct["structure_state"].iloc[-1])

        # 1H 봉 마감마다 EQL/EQH 갱신
        if not self.df_1h.empty:
            _ts_1h = pd.Timestamp(self.df_1h["time"].iloc[-1])
            if _ts_1h.tzinfo is None:
                _ts_1h = _ts_1h.tz_localize("UTC")
            self.liq_pool.refresh_eq_levels(self.pivots_15m, _ts_1h.normalize(), current_ts=_ts_1h)

        self._update_obs("obs_1h_df", self.df_1h_struct.tail(self._tail_len + 5), self.df_1h,
                         max_age_bars=OBS_HTF_MAX_AGE_BARS_1H, max_rows=OBS_HTF_MAX_ROWS)
        self.fvgs_1h = self._build_fvgs(self.df_1h)

    def _update_4h_state(self):
        """4H 봉 structure_state + OB + FVG 업데이트."""
        if len(self.df_4h) < 7:
            return
        df, pivots = self._full_tf(self.df_4h)
        if len(df) == 0:
            return
        self.df_4h_struct = df
        self.pivots_4h    = pivots
        self.htf_4h_state = str(df["structure_state"].iloc[-1])
        self._update_obs("obs_4h_df", df.tail(self._tail_len + 5), self.df_4h,
                         max_age_bars=OBS_HTF_MAX_AGE_BARS_4H, max_rows=OBS_HTF_MAX_ROWS)
        self.fvgs_4h = self._build_fvgs(self.df_4h)

    def _advance_fvg_tracker_4h(self):
        """새 4H봉 닫힘 시 tracker 전진. 전체 detect 방식 (검증된 방식 A).
        incremental(최근10봉) 방식은 created_idx 보정 오류로 ARMED 전이가 깨져
        폐기. 4H봉은 수가 적어 전체 detect 비용 감당 가능."""
        df = self.df_4h
        if df is None or len(df) < 3:
            return
        cur_idx = len(df) - 1
        atr = self._atr_4h()
        if atr <= 0:
            return
        # 전체 detect — created_idx가 절대 인덱스로 정확히 매겨짐
        all_fvgs = detect_fvgs_lookback(df, lookback_bars=len(df),
                                        fill_mode="wick", remove_filled=False,
                                        min_size=0.0)
        # 마지막 봉에서 새로 생성된 FVG만 공급 (add_new_fvgs가 중복 방지)
        if len(all_fvgs):
            new_fvgs = all_fvgs[all_fvgs["created_idx"] >= self._fvg4h_known_idx]
            if len(new_fvgs):
                self.fvg_tracker_4h.add_new_fvgs(new_fvgs, atr=atr, cur_idx=cur_idx)
        self._fvg4h_known_idx = len(df)
        row = df.iloc[-1]
        self.fvg_tracker_4h.update(cur_idx, bar_low=float(row["low"]),
                                   bar_high=float(row["high"]),
                                   bar_close=float(row["close"]))

    def _advance_fvg_tracker_1h(self):
        """새 1H봉 닫힘 시 1H tracker 전진. 전체 detect 방식 (4H와 동일)."""
        df = self.df_1h
        if df is None or len(df) < 3:
            return
        cur_idx = len(df) - 1
        atr = self._atr_1h()
        if atr <= 0:
            return
        all_fvgs = detect_fvgs_lookback(df, lookback_bars=len(df),
                                        fill_mode="wick", remove_filled=False,
                                        min_size=0.0)
        if len(all_fvgs):
            new_fvgs = all_fvgs[all_fvgs["created_idx"] >= self._fvg1h_known_idx]
            if len(new_fvgs):
                self.fvg_tracker_1h.add_new_fvgs(new_fvgs, atr=atr, cur_idx=cur_idx)
        self._fvg1h_known_idx = len(df)
        row = df.iloc[-1]
        self.fvg_tracker_1h.update(cur_idx, bar_low=float(row["low"]),
                                   bar_high=float(row["high"]),
                                   bar_close=float(row["close"]))

    def _update_obs(self, attr: str, tail_struct: pd.DataFrame, df_full: pd.DataFrame,
                    max_age_bars: int = None, max_rows: int = None):
        """
        OB 증분 업데이트 + 오래된 OB 정리.
        정리 기준:
          - invalidated=True AND formed_at이 max_age_bars 이상 오래됨 → 삭제
          - 총 행 수가 max_rows 초과 시 가장 오래된 것부터 삭제
        """
        if max_age_bars is None:
            max_age_bars = OBS_MAX_AGE_BARS
        if max_rows is None:
            max_rows = OBS_MAX_ROWS

        existing = getattr(self, attr)
        new_obs  = build_order_blocks_from_bos(tail_struct, time_col="time")
        if not new_obs.empty:
            if not existing.empty:
                existing_times = set(existing["formed_at"].tolist())
                new_obs = new_obs[~new_obs["formed_at"].isin(existing_times)]
            if not new_obs.empty:
                existing = pd.concat([existing, new_obs], ignore_index=True)
        if not existing.empty:
            active = existing[existing["invalidated"] == False].copy()
            if not active.empty:
                active = mark_invalidation_by_close(df_full, active, time_col="time")
                existing.loc[existing["invalidated"] == False, "invalidated"] = active["invalidated"].values
        # ── 오래된 OB 정리
        if not existing.empty and "formed_at" in existing.columns:
            # 1) invalidated + 오래됨 → 삭제
            cutoff_idx = max(0, len(df_full) - max_age_bars)
            if len(df_full) > max_age_bars:
                try:
                    cutoff_time = pd.Timestamp(df_full["time"].iloc[cutoff_idx], tz="UTC") \
                        if df_full["time"].iloc[0] is not None else None
                    if cutoff_time is not None:
                        formed = pd.to_datetime(existing["formed_at"], utc=True)
                        old_and_dead = (existing["invalidated"] == True) & (formed < cutoff_time)
                        existing = existing[~old_and_dead].reset_index(drop=True)
                except Exception:
                    pass
            # 2) 총 행 수 상한
            if len(existing) > max_rows:
                existing = existing.iloc[-max_rows:].reset_index(drop=True)
        setattr(self, attr, existing)

    def _build_obs(self, df_struct: pd.DataFrame) -> pd.DataFrame:
        obs = build_order_blocks_from_bos(df_struct, time_col="time")
        if obs.empty:
            return obs
        return mark_invalidation_by_close(df_struct, obs, time_col="time")

    @staticmethod
    def _build_fvgs(df: pd.DataFrame) -> pd.DataFrame:
        """
        HTF FVG 탐색 (1H/4H용).
        만료 조건: FVG mid를 wick으로라도 터치했으면 삭제.
        (기본 detect_fvgs_lookback은 zone edge 기준이므로 별도 처리)
        """
        if df is None or len(df) < 3:
            return pd.DataFrame(columns=["fvg_type","created_idx","created_time",
                                         "zone_low","zone_high","size","mid","filled"])

        # filled=False로 전부 가져온 뒤, mid 기준으로 직접 필터
        fvg_df = detect_fvgs_lookback(
            df, lookback_bars=len(df),
            fill_mode="wick", remove_filled=False, min_size=0.0,
        )
        if fvg_df.empty:
            return fvg_df

        # mid 컬럼 추가
        fvg_df["mid"] = (fvg_df["zone_low"] + fvg_df["zone_high"]) / 2.0

        # mid 터치 판정: 생성 이후 wick이 mid를 넘었으면 삭제
        idxs = df.index.to_list()
        pos = {idx: i for i, idx in enumerate(idxs)}
        highs = df["high"].astype(float).to_list()
        lows = df["low"].astype(float).to_list()

        keep = []
        for r_i in range(len(fvg_df)):
            created_idx = fvg_df.iloc[r_i]["created_idx"]
            start = pos.get(created_idx, None)
            if start is None:
                keep.append(True)
                continue

            mid = float(fvg_df.iloc[r_i]["mid"])
            typ = fvg_df.iloc[r_i]["fvg_type"]
            mid_touched = False

            for j in range(start + 1, len(idxs)):
                if typ == "bull":
                    # bull FVG: 가격이 내려와서 mid 이하 터치
                    if lows[j] <= mid:
                        mid_touched = True
                        break
                else:
                    # bear FVG: 가격이 올라와서 mid 이상 터치
                    if highs[j] >= mid:
                        mid_touched = True
                        break

            keep.append(not mid_touched)

        fvg_df = fvg_df[keep].reset_index(drop=True)
        return fvg_df

    @staticmethod
    def _full_tf(df: pd.DataFrame):
        if len(df) < 7:
            empty_p = pd.DataFrame(columns=["idx","time","kind","price"])
            return df.copy(), empty_p
        sw = find_swings_enhanced(df, lb_low=3, lb_high=5)  # 4H: 평균 lb≈4
        p  = build_pivots(sw)
        pc = compress_pivots(p, mode="zigzag") if not p.empty else p
        st = detect_structure_events_close_protected(df, pc)
        return st, pc

    @staticmethod
    def _append(df: pd.DataFrame, bar: pd.Series, max_len: int = 9000) -> pd.DataFrame:
        out = pd.concat([df, pd.DataFrame([bar])], ignore_index=True)
        if len(out) > max_len:
            out = out.iloc[len(out) - max_len:].reset_index(drop=True)
        return out

    @staticmethod
    def _merge_pivots(existing: pd.DataFrame, new_p: pd.DataFrame,
                      max_pivots: int = 200) -> pd.DataFrame:
        """pivot 병합 후 최대 max_pivots개만 유지 (오래된 것부터 제거)."""
        if existing.empty:
            return new_p.copy()
        if new_p.empty:
            return existing.copy()
        kept = existing[existing["idx"] < new_p["idx"].min()]
        merged = pd.concat([kept, new_p], ignore_index=True).sort_values("idx").reset_index(drop=True)
        if len(merged) > max_pivots:
            merged = merged.iloc[-max_pivots:].reset_index(drop=True)
        return merged

    @staticmethod
    def _merge_struct(existing: pd.DataFrame, tail: pd.DataFrame, tail_len: int) -> pd.DataFrame:
        if existing.empty:
            return tail.copy()
        head = existing.iloc[:max(0, len(existing) - tail_len)]
        merged = pd.concat([head, tail], ignore_index=True).reset_index(drop=True)
        # 크기 상한: STRUCT_MAX_ROWS 초과 시 오래된 것부터 제거
        if len(merged) > STRUCT_MAX_ROWS:
            merged = merged.iloc[-STRUCT_MAX_ROWS:].reset_index(drop=True)
        return merged


# ─────────────────────────────────────────────────────────────────────
# Trail/BE 환경 설정 (환경변수로 toggle 가능)
# ─────────────────────────────────────────────────────────────────────
# TRAIL_MODE 환경변수:
#   "off" (기본): trail/BE 비활성 (1000R) — 학습 데이터 수집용 (raw EV 측정)
#   "on"        : trail/BE 활성 (실거래 환경과 동일) — environment parity 검증용
#
# 동균님 직접 검증 시나리오:
#   1) collect_ml_data.py 두 번 실행:
#        TRAIL_MODE=off → ml_train_*_traoff.csv  (현재 default)
#        TRAIL_MODE=on  → ml_train_*_traon.csv   (실거래 환경)
#   2) 같은 setup이 trail 유무에 따라 어떻게 끝나는지 직접 비교
import os as _os
_TRAIL_MODE = _os.environ.get("TRAIL_MODE", "off").lower()

if _TRAIL_MODE == "swing_cb":
    # stage6 확정: swing 트레일 (close 돌파 BOS → 직전 swing low lock).
    #   봉내 착시 구조적 0 (close 돌파는 봉마감에만), LB3/back1.
    #   c_w 0.1~0.2 현실영역서 uniform 대비 +0.012~0.033R.
    TRAIL_ACTIVATE_R   = 0.0    # swing 트레일은 BOS 시점에 자동 활성 (R 게이트 불요)
    TRAIL_ACTIVATE_R_A = 0.0
    TRAIL_BREAKEVEN_R  = 1000.0  # BE 비활성 (swing lock이 대체)
    _BREAKEVEN_R       = 1000.0
    print(f"[backtest] 🔧 TRAIL_MODE=swing_cb (close돌파 BOS swing trail, LB3/back1)")
elif _TRAIL_MODE == "on":
    # 실거래 환경 (backtest_oos.py와 동일)
    TRAIL_ACTIVATE_R   = 2.0
    TRAIL_ACTIVATE_R_A = 1.5
    TRAIL_BREAKEVEN_R  = 1.0
    _BREAKEVEN_R       = 1.0
    print(f"[backtest] 🔧 TRAIL_MODE=on  (실거래 환경: trail 2R/1.5R, BE 1R)")
else:
    # 학습 데이터 수집용 (raw EV)
    TRAIL_ACTIVATE_R   = 1000.0
    TRAIL_ACTIVATE_R_A = 1000.0
    TRAIL_BREAKEVEN_R  = 1000.0
    _BREAKEVEN_R       = 1000.0
    print(f"[backtest] 🔧 TRAIL_MODE=off (학습용: trail/BE 비활성)")

TRAIL_SWING_BACK_A     = 3     # Group A: 스윙 3개 기준
TRAIL_FALLBACK_PCT = 0.008 # 스윙 없을 때 최고점 대비 0.8% 되돌림
TRAIL_SWING_LB    = 3     # 스윙 감지 lookback (봉 수)


TRAIL_SWING_BACK = 4   # Group B: 4개 스윙 기준 (3→4, 더 여유)

# ── swing_cb 트레일 (stage6 확정): close 돌파 BOS → 직전 swing low lock
TRAIL_CB_LB   = 3      # fractal lookback (피벗 i는 i+LB봉 종가에서 확정)
TRAIL_CB_BACK = 1      # lock 시 직전(1) swing low 사용 (back=2는 더 깊은 lock)

# ── Shadow trail 추적 toggle
# SHADOW_TRAIL=1이면 각 trade에 대해 "trail ON이었다면 어디서 청산됐을지"를
# 동시에 계산하여 shadow_pnl_r 등에 기록. primary는 TRAIL_MODE 설정대로 진행.
# → 한 번의 collect로 trail off/on 두 결과를 같은 trade에서 비교 가능.
_SHADOW_TRAIL = _os.environ.get("SHADOW_TRAIL", "0") in ("1", "true", "True")
# shadow 비교 시 primary/shadow가 공유할 고정 timeout (기본 144봉 = 12시간)
# 환경변수 SHADOW_TIMEOUT_BARS로 조정 가능
SHADOW_TIMEOUT_BARS = int(_os.environ.get("SHADOW_TIMEOUT_BARS", "144"))
# timeout 스윕 모드: 144봉 고정 + 각 시점 R 기록 → 사후 4h~12h 비교
_TIMEOUT_SWEEP = _os.environ.get("TIMEOUT_SWEEP", "0") in ("1", "true", "True")
if _SHADOW_TRAIL:
    print(f"[backtest] 🔧 SHADOW_TRAIL=1 (각 trade에 trail ON shadow 청산 동시 기록)")
    print(f"[backtest]    primary/shadow 공유 timeout = {SHADOW_TIMEOUT_BARS}봉 "
          f"({SHADOW_TIMEOUT_BARS*5//60}h{SHADOW_TIMEOUT_BARS*5%60}m)  [방법 A: 공정 비교]")


def _wilder_rsi_vec(close, period=14):
    """Wilder RSI 벡터화 (lfilter, SMA seed). prepare_runner_data_v2와 동일."""
    from scipy.signal import lfilter
    n = len(close)
    if n <= period:
        return np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    a = 1.0 / period

    def _avg(x):
        out = np.full(len(x), np.nan)
        seed = x[1:period+1].mean()
        out[period] = seed
        tail = x[period+1:]
        if len(tail):
            out[period+1:] = lfilter([a], [1.0, -(1 - a)], tail, zi=[seed * (1 - a)])[0]
        return out

    ag, al = _avg(gain), _avg(loss)
    rsi = np.full(n, np.nan)
    valid = al > 1e-12
    rs = np.where(valid, ag / np.where(valid, al, 1.0), 0.0)
    rf = np.where(valid, 100.0 - 100.0 / (1.0 + rs), 100.0)
    m = ~np.isnan(ag)
    rsi[m] = rf[m]
    return rsi


def _sma_vec(arr, period):
    return pd.Series(arr).rolling(period, min_periods=period).mean().values


def _collect_out_dir() -> str:
    """수집 결과(ml_train/runner_train CSV) 저장 폴더 = 프로젝트 폴더 내 collect_result/.
    프로젝트 폴더는 이 파일(__file__) 기준 고정 → 워커 cwd 무관. .gitignore 대상."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_result")
    os.makedirs(d, exist_ok=True)
    return d


# runner 데이터 상수 (prepare_runner_data_v2와 동일)
RUNNER_RSI_PERIOD = 14
RUNNER_RSI_MA = 9
RUNNER_SLOPE_K = 3
RUNNER_RSI_PCT_LB = 200

# runner 평가시점: True면 L+H 합본(방향 무관, 모든 15m swing에서 평가),
#   False면 기존 방향별(long=swing low, short=swing high)만.
#   합본은 eval 밀도 ~2배 → forward window 중첩 → 자기상관 부활.
#   판정은 AUC 아닌 EV로, CV embargo >= 12, runner_dt_check 재실행 필수.
RUNNER_EVAL_BOTH_KINDS = True

# runner 평가 기준봉 (option A/B):
#   True(B)  = 확정봉(extreme + lb)에서 trigger/feature/label 전부 측정.
#              live servable + 낙관편향 제거(extreme→확정 bounce가 라벨에서 빠짐).
#              단 신호 stale(bounce 후라 div 일부 해소).
#   False(A) = extreme 봉에서 측정. 신호 날카롭지만 라벨이 capturable EV 과대평가,
#              live에선 그 시점에 swing인지 모르므로 unservable.
#   RUNNER_CONFIRM_LB15 = 확정 lookback(15m봉 단위). 실제 lb는 ATR기반 dynamic 2~4인데
#   pivots_15m에 per-pivot lb가 없어 고정 근사. 고변동(lb2) 피벗은 1봉 늦게,
#   저변동(lb4) 피벗은 1봉 빠르게 어긋남(수용). 정밀화하려면 detect때 lb 저장 필요.
RUNNER_EVAL_AT_CONFIRM = True
RUNNER_CONFIRM_LB15 = 3


def extract_runner_training_data(trades, df_5m, pivots_5m, pivots_15m, df_15m, symbol):
    """collect 시 backtest 자원으로 runner 학습 데이터 직접 생성 (방향 A).

    prepare_runner_data_v2 로직 이식. backtest는 이미 df_5m/pivots를 가지므로
    OHLCV 재읽기·재리샘플·재swing 불필요 (그 부분이 prepare_v2의 병목이었음).

    평가시점 = 15m swing 확정봉(extreme+lb, option B 기본) 또는 extreme봉(option A).
    divergence = 5m swing 기반 raw 상태변수.
    라벨 = 다음 15m swing까지 fav/adv (이벤트) + dt 백업.
    """
    if df_5m is None or len(df_5m) < 300 or pivots_15m is None or pivots_15m.empty:
        return pd.DataFrame()
    close5 = df_5m["close"].values
    high5 = df_5m["high"].values
    low5 = df_5m["low"].values
    times5 = df_5m["time"].values if "time" in df_5m else np.arange(len(df_5m))
    n5 = len(close5)

    # 5m RSI + 모멘텀
    rsi5 = _wilder_rsi_vec(close5, RUNNER_RSI_PERIOD)
    rsi_ma5 = _sma_vec(rsi5, RUNNER_RSI_MA)
    # 15m RSI (df_15m close)
    close15 = df_15m["close"].values if (df_15m is not None and "close" in df_15m) else None
    if close15 is not None and len(close15) > RUNNER_RSI_PERIOD:
        rsi15 = _wilder_rsi_vec(close15, RUNNER_RSI_PERIOD)
        rsi_ma15 = _sma_vec(rsi15, RUNNER_RSI_MA)
        t15_arr = df_15m["time"].values if "time" in df_15m else np.arange(len(close15))
    else:
        rsi15 = rsi_ma15 = t15_arr = None

    # 15m swing → 5m idx 평가시점 (kind별 + 합본).
    #   option B: eval = 확정봉(i15 + lb). option A: eval = extreme(lb=0).
    ev5_by_kind = {"L": [], "H": []}
    ev5_kind_map = {}      # eval 5m idx → swing kind ("L"/"H"). 합본 분석용 태그.
    ev5_extreme_map = {}   # eval 5m idx → extreme 5m idx. B의 bounce(=제거된 낙관편향) 측정용.
    _conf_lb = RUNNER_CONFIRM_LB15 if RUNNER_EVAL_AT_CONFIRM else 0
    n15 = len(t15_arr) if t15_arr is not None else 0
    for _, pv in pivots_15m.iterrows():
        k = pv["kind"]
        if k not in ev5_by_kind:
            continue
        i15 = int(pv["idx"])
        if t15_arr is None or i15 >= n15:
            continue
        ex5i = int(np.searchsorted(times5, t15_arr[i15], side="right") - 1)  # extreme 5m
        i15_eval = i15 + _conf_lb                                            # 확정 15m봉
        if i15_eval >= n15:
            continue                                                        # 데이터 끝에서 확정 못함 → drop
        e5i = int(np.searchsorted(times5, t15_arr[i15_eval], side="right") - 1)  # eval(확정) 5m
        if e5i >= 0:
            ev5_by_kind[k].append(e5i)
            ev5_kind_map[e5i] = k       # 충돌(같은 5m봉) 시 마지막 우선 — 드묾
            ev5_extreme_map[e5i] = ex5i
    ev5_by_kind["L"] = np.array(sorted(set(ev5_by_kind["L"])), dtype=int)
    ev5_by_kind["H"] = np.array(sorted(set(ev5_by_kind["H"])), dtype=int)
    # 합본: L+H union (방향 무관). nxt(다음 eval)는 자동으로 "다음 swing 둘 중 가까운 것"이 됨.
    ev5_all = np.array(sorted(set(ev5_by_kind["L"].tolist()
                                  + ev5_by_kind["H"].tolist())), dtype=int)

    # 5m swing (divergence)
    piv5_L = pivots_5m[pivots_5m["kind"] == "L"].sort_values("idx") if not pivots_5m.empty else pivots_5m
    piv5_H = pivots_5m[pivots_5m["kind"] == "H"].sort_values("idx") if not pivots_5m.empty else pivots_5m
    L_idx = piv5_L["idx"].values if len(piv5_L) else np.array([])
    L_px = piv5_L["price"].values if len(piv5_L) else np.array([])
    H_idx = piv5_H["idx"].values if len(piv5_H) else np.array([])
    H_px = piv5_H["price"].values if len(piv5_H) else np.array([])

    def _rolling_pct(arr, idx, val, lb=RUNNER_RSI_PCT_LB):
        lo = max(0, idx - lb)
        w = arr[lo:idx+1]
        w = w[~np.isnan(w)]
        return 0.5 if len(w) < 10 else float((w < val).mean())

    DT_BACKUP = [12, 24]
    rows = []
    for t in trades:
        e5 = getattr(t, "entry_bar_idx", -1)
        if e5 < 0 or e5 >= n5:
            continue
        is_long = (t.direction == "long")
        risk = abs(t.entry_price - t.initial_sl) if getattr(t, "initial_sl", 0) else None
        if not risk or risk <= 0:
            continue
        entry_px = t.entry_price
        tp_str = getattr(t, "trail_path", "") or ""
        hold = tp_str.count(";") + 1 if len(tp_str) > 3 else 999
        exit5 = e5 + hold

        # 5m divergence 상태 (진입 이후 같은 kind)
        if is_long:
            p_idx, p_px = L_idx, L_px
        else:
            p_idx, p_px = H_idx, H_px
        mask = p_idx > e5
        a_idx, a_px = p_idx[mask], p_px[mask]
        div_hist = []
        for j in range(1, len(a_idx)):
            pidx, ppx = int(a_idx[j]), float(a_px[j])
            ppidx, pppx = int(a_idx[j-1]), float(a_px[j-1])
            if pidx >= len(rsi5) or ppidx >= len(rsi5):
                continue
            pr, ppr = rsi5[pidx], rsi5[ppidx]
            if np.isnan(pr) or np.isnan(ppr):
                continue
            d_price = (ppx - pppx) / risk
            d_rsi = pr - ppr
            ds = (d_rsi / 10.0) - d_price if is_long else (-d_rsi / 10.0) + d_price
            div_hist.append((pidx, round(ds, 4)))

        # 평가시점 (e5+4 ~ exit5). 합본이면 L+H union, 아니면 방향별.
        if RUNNER_EVAL_BOTH_KINDS:
            ev_arr = ev5_all
        else:
            ev_arr = ev5_by_kind["L" if is_long else "H"]
        lo = int(np.searchsorted(ev_arr, e5 + 4, side="left"))
        hi = int(np.searchsorted(ev_arr, exit5, side="left"))
        pts = ev_arr[lo:hi]
        for pi in range(len(pts)):
            ev5 = int(pts[pi])
            nxt = int(pts[pi+1]) if pi+1 < len(pts) else None
            if is_long:
                cur_fav = (high5[e5:ev5+1].max() - entry_px) / risk
                cur_adv = (entry_px - low5[e5:ev5+1].min()) / risk
            else:
                cur_fav = (entry_px - low5[e5:ev5+1].min()) / risk
                cur_adv = (high5[e5:ev5+1].max() - entry_px) / risk

            divs = [d for (ix, d) in div_hist if ix <= ev5]
            if divs:
                d_lat = divs[-1]; d_avg = float(np.mean(divs[-5:]))
                d_blend = (d_avg + d_lat) / 2.0
                d_fresh = ev5 - max(ix for (ix, d) in div_hist if ix <= ev5)
            else:
                d_lat = d_blend = 0.0; d_fresh = 999

            rn = rsi5[ev5] if ev5 < len(rsi5) else np.nan
            rpct = _rolling_pct(rsi5, ev5, rn) if not np.isnan(rn) else 0.5
            k = RUNNER_SLOPE_K
            ma_dev = (rn - rsi_ma5[ev5]) if (ev5 < len(rsi_ma5) and not np.isnan(rsi_ma5[ev5]) and not np.isnan(rn)) else 0.0
            slope = (rn - rsi5[ev5-k]) if (ev5-k >= 0 and not np.isnan(rsi5[ev5-k]) and not np.isnan(rn)) else 0.0
            if ev5-2*k >= 0 and not np.isnan(rsi5[ev5-2*k]) and not np.isnan(rsi5[ev5-k]) and not np.isnan(rn):
                slope = slope; accel = slope - (rsi5[ev5-k] - rsi5[ev5-2*k])
            else:
                accel = 0.0
            # 15m RSI
            if t15_arr is not None:
                i15 = int(np.searchsorted(t15_arr, times5[ev5], side="right") - 1)
                r15 = rsi15[i15] if (0 <= i15 < len(rsi15)) else np.nan
                r15pct = _rolling_pct(rsi15, i15, r15) if (i15 >= 0 and not np.isnan(r15)) else 0.5
                r15_ma = (r15 - rsi_ma15[i15]) if (0 <= i15 < len(rsi_ma15) and not np.isnan(rsi_ma15[i15]) and not np.isnan(r15)) else 0.0
                r15_sl = (r15 - rsi15[i15-k]) if (i15-k >= 0 and not np.isnan(rsi15[i15-k]) and not np.isnan(r15)) else 0.0
                if i15-2*k >= 0 and not np.isnan(rsi15[i15-2*k]) and not np.isnan(rsi15[i15-k]) and not np.isnan(r15):
                    r15_ac = r15_sl - (rsi15[i15-k] - rsi15[i15-2*k])
                else:
                    r15_ac = 0.0
            else:
                r15pct = 0.5; r15_ma = r15_sl = r15_ac = 0.0

            rec = {"symbol": symbol, "trade_id": getattr(t, "trade_id", f"{symbol}_{e5}"),
                   "direction": t.direction, "exit_reason": getattr(t, "exit_reason", ""),
                   "eval_5m_idx": ev5, "elapsed_bars": ev5 - e5,
                   "extreme_5m_idx": ev5_extreme_map.get(ev5, ev5),
                   "eval_swing_kind": ev5_kind_map.get(ev5, "?"),
                   "cur_fav_r": round(cur_fav, 4), "cur_adv_r": round(cur_adv, 4),
                   "rsi_pct_now": round(rpct, 4), "rsi_ma_dev": round(float(ma_dev), 4),
                   "rsi_slope": round(float(slope), 4), "rsi_accel": round(float(accel), 4),
                   "rsi15_pct_now": round(r15pct, 4), "rsi15_ma_dev": round(float(r15_ma), 4),
                   "rsi15_slope": round(float(r15_sl), 4), "rsi15_accel": round(float(r15_ac), 4),
                   "div_blend": round(d_blend, 4), "div_latest": round(d_lat, 4),
                   "div_freshness": int(min(d_fresh, 999))}
            # 라벨: 다음 15m swing(nxt, exit5 이내)
            fe = min(nxt, exit5) if nxt is not None else None
            if fe is not None and ev5 < fe < n5:
                if is_long:
                    ff = (high5[ev5+1:fe+1].max() - close5[ev5]) / risk
                    fa = (close5[ev5] - low5[ev5+1:fe+1].min()) / risk
                else:
                    ff = (close5[ev5] - low5[ev5+1:fe+1].min()) / risk
                    fa = (high5[ev5+1:fe+1].max() - close5[ev5]) / risk
                rec["fav_next"] = round(float(ff), 4)
                rec["adv_next"] = round(float(max(fa, 0)), 4)
                rec["y_win_next"] = 1 if ff >= max(fa, 0) else 0
            else:
                rec["fav_next"] = rec["adv_next"] = rec["y_win_next"] = np.nan
            for dt in DT_BACKUP:
                end = ev5 + dt
                if end < n5:
                    if is_long:
                        ff = (high5[ev5+1:end+1].max() - close5[ev5]) / risk
                        fa = (close5[ev5] - low5[ev5+1:end+1].min()) / risk
                    else:
                        ff = (close5[ev5] - low5[ev5+1:end+1].min()) / risk
                        fa = (high5[ev5+1:end+1].max() - close5[ev5]) / risk
                    rec[f"y_win_{dt}"] = 1 if ff >= max(fa, 0) else 0
                    rec[f"fav_{dt}"] = round(float(ff), 4)        # EV용 회귀 타깃
                    rec[f"adv_{dt}"] = round(float(max(fa, 0)), 4)
                else:
                    rec[f"y_win_{dt}"] = np.nan
                    rec[f"fav_{dt}"] = rec[f"adv_{dt}"] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


_RUNNER_EV_MODELS = None  # lazy 로드 캐시


def _load_ev_models():
    """runner EV 모델 로드 (P win, E fav, E adv) + feature 목록. 1회만."""
    global _RUNNER_EV_MODELS
    if _RUNNER_EV_MODELS is not None:
        return _RUNNER_EV_MODELS
    import json, os as _os
    if not _os.path.exists("runner_ev_features.json"):
        _RUNNER_EV_MODELS = {"ok": False}
        return _RUNNER_EV_MODELS
    try:
        import xgboost as _xgb
        with open("runner_ev_features.json") as f:
            meta = json.load(f)
        dt = meta["dt"]
        mw = _xgb.XGBClassifier(); mw.load_model(f"runner_ev_pwin_dt{dt}.json")
        mf = _xgb.XGBRegressor(); mf.load_model(f"runner_ev_fav_dt{dt}.json")
        ma = _xgb.XGBRegressor(); ma.load_model(f"runner_ev_adv_dt{dt}.json")
        _RUNNER_EV_MODELS = {"ok": True, "features": meta["features"], "dt": dt,
                             "pwin": mw, "fav": mf, "adv": ma}
    except Exception as e:
        print(f"[EV] 모델 로드 실패: {e}")
        _RUNNER_EV_MODELS = {"ok": False}
    return _RUNNER_EV_MODELS


def _runner_ev_at(feat_row: dict):
    """현재 상태 feature로 EV = P*E[fav] - (1-P)*E[adv] 계산. None이면 미사용."""
    m = _load_ev_models()
    if not m.get("ok"):
        return None
    x = np.array([[float(feat_row.get(f, 0.0) or 0.0) for f in m["features"]]])
    p = float(m["pwin"].predict_proba(x)[0, 1])
    ef = float(m["fav"].predict(x)[0])
    ea = float(m["adv"].predict(x)[0])
    return {"ev": p * ef - (1 - p) * ea, "p_win": p, "e_fav": ef, "e_adv": ea}


def _update_swing_cb_trail(t, cur_idx: int, bar_close: float,
                           pivots_5m, lb: int = TRAIL_CB_LB,
                           back: int = TRAIL_CB_BACK) -> None:
    """swing_cb 트레일 (stage6 확정) 라이브化. 매봉 close 후 호출.

    stage6 replay_swing 이식 (pivots DataFrame 인터페이스):
      - 피벗은 i+lb봉에서 확정 (pivots_5m의 idx는 이미 확정 시점 기준).
        진입(entry_bar_idx) 이후 idx ≤ cur_idx-lb 인 피벗만 사용 (lookahead 없음).
      - long: 기준스윙 = swing high(kind H). close가 마지막 H 돌파(BOS) →
        직전(back) swing low(kind L)로 SL ratchet(위로만).
      - short 대칭. 봉마감 close 돌파라 봉내 착시 0.
    t.sl 직접 갱신.
    """
    if pivots_5m is None or pivots_5m.empty:
        return
    is_long = (t.direction == "long")
    ref_kind = "H" if is_long else "L"
    lock_kind = "L" if is_long else "H"
    conf_idx = cur_idx - lb   # 이 봉까지 확정된 피벗만

    # 진입 이후 & 확정된 피벗
    pv = pivots_5m[(pivots_5m["idx"] > t.entry_bar_idx) & (pivots_5m["idx"] <= conf_idx)]
    if pv.empty:
        return
    refs = pv[pv["kind"] == ref_kind]
    locks = pv[pv["kind"] == lock_kind]["price"].tolist()

    # 마지막 기준스윙 갱신 (가장 최근 확정)
    if not refs.empty:
        new_ref = float(refs.iloc[-1]["price"])
        new_ref_idx = int(refs.iloc[-1]["idx"])
        if new_ref_idx != t._cb_ref_idx:   # 새 기준스윙 등장
            t._cb_last_ref = new_ref; t._cb_has_ref = True
            t._cb_ref_broken = False; t._cb_ref_idx = new_ref_idx

    # close 돌파 BOS → 직전 lock으로 ratchet
    if t._cb_has_ref and not t._cb_ref_broken:
        broke = (bar_close > t._cb_last_ref) if is_long else (bar_close < t._cb_last_ref)
        if broke and len(locks) >= back:
            t._cb_ref_broken = True
            cand = locks[-back]   # 직전(back=1) swing low/high
            if (is_long and cand > t.sl) or ((not is_long) and cand < t.sl):
                t.sl = cand


def _find_trail_sl(direction: str, pivots_5m: pd.DataFrame,
                   current_sl: float, current_px: float,
                   entry_bar_idx: int = 0,
                   swing_back: int = TRAIL_SWING_BACK) -> Optional[float]:
    """
    진입 이후 생긴 스윙 저점(long) / 스윙 고점(short)을 새 SL 후보로 반환.
    swing_back=2 → 직전 2개 중 오래된 것 (Group A: 더 일찍 청산)
    swing_back=3 → 직전 3개 중 오래된 것 (Group B: 더 여유)
    """
    if pivots_5m is None or pivots_5m.empty:
        return None

    if direction == "long":
        p = pivots_5m[(pivots_5m["kind"] == "L") & (pivots_5m["idx"] > entry_bar_idx)]
        candidates = p[(p["price"] > current_sl) & (p["price"] < current_px)]["price"]
        if candidates.empty:
            return None
        top = candidates.nlargest(swing_back)
        return float(top.iloc[-1])

    else:
        p = pivots_5m[(pivots_5m["kind"] == "H") & (pivots_5m["idx"] > entry_bar_idx)]
        candidates = p[(p["price"] > current_px) & (p["price"] < current_sl)]["price"]
        if candidates.empty:
            return None
        top = candidates.nsmallest(swing_back)
        return float(top.iloc[-1])


def _find_liq_trail_sl(direction: str, liq_levels: list,
                       current_sl: float, h: float, l: float) -> Optional[float]:
    """
    포지션 방향과 반대편 유동성 레벨을 가격이 통과했을 때
    해당 레벨을 새 trail SL로 반환.

    Long:  고점 레벨(EQH, pdh, pwh, london_high, asia_high 등)을
           현재 봉 고점(h)이 통과 → zone_low를 SL 후보로
    Short: 저점 레벨(EQL, pdl, pwl, london_low, asia_low 등)을
           현재 봉 저점(l)이 통과 → zone_high를 SL 후보로

    여러 레벨이 통과됐으면 가장 유리한 것(long=가장 높은, short=가장 낮은) 사용.
    """
    HIGH_TYPES = {"eqh", "swing_high",
                  "pdh", "pwh", "asia_high"}  # v2.2: london/ny 제거
    LOW_TYPES  = {"eql", "swing_low",
                  "pdl", "pwl", "asia_low"}   # v2.2: london/ny 제거

    candidates = []

    if direction == "long":
        # 위쪽 레벨을 고점이 통과했으면 zone_low를 SL 후보로
        for lv in liq_levels:
            lt = lv.get("level_type", "")
            if lt not in HIGH_TYPES:
                continue
            zh = float(lv["zone_high"])
            zl = float(lv["zone_low"])
            sl_candidate = zl  # zone_low (또는 단일 가격이면 price_center)
            # 가격이 레벨을 통과(h > zone_high)하고 SL 후보가 현재 SL보다 유리(높음)
            if h > zh and sl_candidate > current_sl:
                candidates.append(sl_candidate)
        return float(max(candidates)) if candidates else None

    else:  # short
        # 아래쪽 레벨을 저점이 통과했으면 zone_high를 SL 후보로
        for lv in liq_levels:
            lt = lv.get("level_type", "")
            if lt not in LOW_TYPES:
                continue
            zh = float(lv["zone_high"])
            zl = float(lv["zone_low"])
            sl_candidate = zh  # zone_high (또는 단일 가격이면 price_center)
            # 가격이 레벨을 통과(l < zone_low)하고 SL 후보가 현재 SL보다 유리(낮음)
            if l < zl and sl_candidate < current_sl:
                candidates.append(sl_candidate)
        return float(min(candidates)) if candidates else None


class _PositionSim:
    """멀티 포지션 관리. 동시 여러 포지션 보유 가능."""

    MAX_POSITIONS = 5  # 동시 최대 포지션 수

    def __init__(self):
        self.positions: List[Trade] = []

    @property
    def active(self):
        """호환성: 포지션이 있으면 첫 번째 반환 (레거시 코드용)."""
        return self.positions[0] if self.positions else None

    @property
    def n_open(self) -> int:
        return len(self.positions)

    def open(self, trade: Trade):
        self.positions.append(trade)

    def can_open(self, new_plan: dict) -> bool:
        """
        동시진입 필터:
        1. 같은 level_type 금지
        2. 같은 방향이고, 기존 포지션의 sweep 이후
           가격이 반대로 sweep(= 다른 레벨을 건드림) 없이 진입 → 금지
           여기서는 단순화: 같은 방향의 포지션이 이미 있고,
           새 sweep_extreme이 기존 포지션의 sweep_extreme과
           같은 방향(둘 다 하락 or 둘 다 상승)이면 차단.
           "되돌림" = 반대 방향 sweep이 발생했다는 것 = 다른 셋업.
        3. 최대 포지션 수 제한
        """
        if self.n_open >= self.MAX_POSITIONS:
            return False

        new_reason = new_plan.get("_reason", {}) or {}
        new_level = new_reason.get("swept_level", {}) or {}
        new_level_type = new_level.get("level_type", "")
        new_direction = new_plan.get("direction", "")
        new_sweep_ext = new_plan.get("sweep_extreme", 0)

        for t in self.positions:
            t_reason = t.reason or {}
            t_level = t_reason.get("swept_level", {}) or {}
            t_level_type = t_level.get("level_type", "")
            t_direction = t.direction
            t_sweep_ext = t_reason.get("sweep_extreme", 0)

            # 규칙 1: 같은 level_type 금지
            if new_level_type and new_level_type == t_level_type:
                return False

            # 규칙 2: 같은 방향이고 sweep이 같은 흐름(되돌림 없음) → 차단
            # 판단: 같은 방향이면, 새 sweep_extreme이 기존 대비
            #   long끼리: 새 sweep가 기존보다 더 아래(= 계속 하락 중 sweep) → 차단
            #   short끼리: 새 sweep가 기존보다 더 위(= 계속 상승 중 sweep) → 차단
            # 반대로 되돌림이 있었으면 새 sweep가 반대 방향에 있으므로 통과
            if new_direction == t_direction and new_sweep_ext and t_sweep_ext:
                if new_direction == "long":
                    # long: 둘 다 저점 sweep — 새 sweep가 기존보다 더 아래면 같은 흐름
                    if float(new_sweep_ext) <= float(t_sweep_ext):
                        return False
                else:
                    # short: 둘 다 고점 sweep — 새 sweep가 기존보다 더 위면 같은 흐름
                    if float(new_sweep_ext) >= float(t_sweep_ext):
                        return False

        return True

    def check_bar(self, bar: pd.Series,
                  pivots_5m: Optional[pd.DataFrame] = None,
                  liq_levels: Optional[list] = None,
                  structure_state_5m: str = "unknown",
                  current_bar_idx: int = 0,
                  atr_5m: float = 0.0) -> List[Trade]:
        """모든 포지션 체크, 청산된 것들 리스트로 반환."""
        closed_trades = []
        remaining = []

        for t in self.positions:
            result = self._check_one(t, bar, pivots_5m, liq_levels,
                                     structure_state_5m, current_bar_idx, atr_5m)
            if result is not None:
                closed_trades.append(result)
            else:
                remaining.append(t)

        self.positions = remaining
        return closed_trades

    def _check_shadow(self, t: Trade, bar: pd.Series, pivots_5m, liq_levels,
                      current_bar_idx: int):
        """
        Shadow 청산 체크 — 실제 포지션을 닫지 않고, '대체 trail 설정이었다면
        어디서 청산됐을지'를 기록만 한다.

        Primary가 TRAIL_MODE=off이면 shadow는 trail ON 로직 (2R/1.5R trail, 1R BE)을
        적용한다. 이미 shadow가 청산됐으면(shadow_exit_reason 설정됨) skip.

        한 번의 진입 → 두 청산을 동시 계산하므로 trade 시퀀스가 갈라지지 않음.
        """
        if not t.shadow_enabled or t.shadow_exit_reason is not None:
            return

        h, l = float(bar["high"]), float(bar["low"])

        # shadow_sl 초기화 (최초 1회)
        if t.shadow_sl == 0.0:
            t.shadow_sl = t.sl  # 초기 SL은 primary와 동일
        if t.shadow_peak_px == 0.0:
            t.shadow_peak_px = t.entry_price

        # peak 추적 (shadow 전용)
        if t.direction == "long":
            t.shadow_peak_px = max(t.shadow_peak_px, h)
        else:
            t.shadow_peak_px = min(t.shadow_peak_px, l) if t.shadow_peak_px > 0 else l

        # ── shadow TP/SL 판정 (TP는 primary와 동일 가격 사용)
        old_shadow_sl = t.shadow_sl
        _tp_hit = False
        _sl_hit = False
        if t.direction == "long":
            if t.tp is not None and h >= t.tp:
                _tp_hit = True
            if l <= old_shadow_sl:
                _sl_hit = True
        else:
            if t.tp is not None and l <= t.tp:
                _tp_hit = True
            if h >= old_shadow_sl:
                _sl_hit = True

        def _record_shadow(exit_px, reason):
            t.shadow_exit_price = float(exit_px)
            t.shadow_exit_time = pd.to_datetime(bar["time"])
            t.shadow_exit_reason = reason
            risk = abs(t.entry_price - t.initial_sl) if t.initial_sl else abs(t.entry_price - t.sl)
            if risk > 0:
                delta = (exit_px - t.entry_price) / t.entry_price
                if t.direction == "short":
                    delta = -delta
                t.shadow_pnl_r = (t.entry_price * delta) / risk
            else:
                t.shadow_pnl_r = 0.0

        if _tp_hit and _sl_hit:
            # 보수적: open에서 가까운 쪽
            bar_open = float(bar["open"])
            if abs(bar_open - float(t.tp)) <= abs(bar_open - old_shadow_sl):
                _record_shadow(float(t.tp), "tp")
            else:
                _record_shadow(old_shadow_sl,
                               "trail_sl" if t.shadow_trailing_activated else "sl")
            return
        elif _tp_hit:
            _record_shadow(float(t.tp), "tp")
            return
        elif _sl_hit:
            _record_shadow(old_shadow_sl,
                           "trail_sl" if t.shadow_trailing_activated else "sl")
            return

        # ── shadow trailing/BE (trail ON 설정: 2R/1.5R trail, 1R BE)
        cur_px = h if t.direction == "long" else l
        risk = abs(t.entry_price - t.shadow_sl)
        _is_group_a = (t.reason or {}).get("level_group", "B") == "A" if t.reason else False
        _shadow_trail_r = 1.5 if _is_group_a else 2.0  # trail ON 값
        _shadow_be_r = 1.0
        _swing_back = TRAIL_SWING_BACK_A if _is_group_a else TRAIL_SWING_BACK

        if risk > 0 and pivots_5m is not None:
            cur_delta = (cur_px - t.entry_price) / t.entry_price
            if t.direction == "short":
                cur_delta = -cur_delta
            cur_r = (t.entry_price * cur_delta) / risk if risk > 0 else 0

            # Breakeven (1R)
            if cur_r >= _shadow_be_r and not t.shadow_breakeven_set:
                t.shadow_breakeven_set = True
                buf = 0.001
                if t.direction == "long" and t.shadow_sl < t.entry_price:
                    t.shadow_sl = t.entry_price * (1 + buf)
                elif t.direction == "short" and t.shadow_sl > t.entry_price:
                    t.shadow_sl = t.entry_price * (1 - buf)

            # Trail activate (2R/1.5R)
            if cur_r >= _shadow_trail_r:
                if not t.shadow_trailing_activated:
                    t.shadow_trailing_activated = True
                    if t.direction == "long" and t.shadow_sl < t.entry_price:
                        t.shadow_sl = t.entry_price
                    elif t.direction == "short" and t.shadow_sl > t.entry_price:
                        t.shadow_sl = t.entry_price

                new_sl = _find_trail_sl(t.direction, pivots_5m, t.shadow_sl, cur_px,
                                        entry_bar_idx=t.entry_bar_idx,
                                        swing_back=_swing_back)
                if new_sl is not None:
                    if t.direction == "long" and new_sl > t.shadow_sl:
                        t.shadow_sl = new_sl
                    elif t.direction == "short" and new_sl < t.shadow_sl:
                        t.shadow_sl = new_sl
                else:
                    # fallback: peak 대비 0.8% 되돌림
                    fb = TRAIL_FALLBACK_PCT
                    if t.direction == "long":
                        fb_sl = t.shadow_peak_px * (1 - fb)
                        if fb_sl > t.shadow_sl:
                            t.shadow_sl = fb_sl
                    else:
                        fb_sl = t.shadow_peak_px * (1 + fb)
                        if fb_sl < t.shadow_sl:
                            t.shadow_sl = fb_sl

    def _check_one(self, t: Trade, bar: pd.Series,
                   pivots_5m, liq_levels, structure_state_5m,
                   current_bar_idx, atr_5m) -> Optional[Trade]:
        """단일 포지션 TP/SL/timeout/trailing 체크. 기존 check_bar 로직 그대로."""
        h, l = float(bar["high"]), float(bar["low"])

        # ── Shadow 청산 체크 (실제 포지션 안 닫고 대체 청산 기록만)
        #     primary 청산 판정 전에 먼저 shadow를 갱신
        if t.shadow_enabled and t.shadow_exit_reason is None:
            self._check_shadow(t, bar, pivots_5m, liq_levels, current_bar_idx)

        # ── ★ ML학습용 timeout: Displacement 기반 (원래 고정 60봉)
        ML_TIMEOUT_MIN_BARS = 24   # 최소 2시간
        ML_TIMEOUT_MAX_BARS = 144  # 최대 12시간
        bars_held = current_bar_idx - t.entry_bar_idx

        _reason = t.reason or {}
        _fd = _reason.get("feature_dict", {})
        _net_move_atr_ratio = float(_fd.get("net_move_atr_ratio", 1.0))
        _bars_since_sweep = float(_reason.get("bars_since_sweep", 10))

        if _SHADOW_TRAIL:
            # ── 방법 A: shadow 비교 시 primary/shadow가 동일한 긴 timeout 창을 공유
            timeout_bars = SHADOW_TIMEOUT_BARS
        elif _TIMEOUT_SWEEP:
            # ── timeout 스윕 모드: 무조건 144봉(12h)까지 살려서 각 시점 R 기록.
            #    사후에 4h/6h/8h/12h timeout을 재라벨링으로 비교.
            timeout_bars = 144
        else:
            timeout_bars = max(
                ML_TIMEOUT_MIN_BARS,
                min(
                    ML_TIMEOUT_MAX_BARS,
                    int((_net_move_atr_ratio * 8) + (_bars_since_sweep * 0.8))
                )
            )

        if bars_held > timeout_bars:
            return self._close(t, float(bar["close"]), bar["time"], "timeout_atr")

        # ── 최고 미실현 손익 갱신 + peak_px 추적
        best_px = h if t.direction == "long" else l
        if t.peak_px == 0.0:
            t.peak_px = t.entry_price
        if t.direction == "long":
            t.peak_px = max(t.peak_px, h)
        else:
            t.peak_px = min(t.peak_px, l) if t.peak_px > 0 else l

        # ── MFE clean + trail_path 누적 (러너 라벨 + 트레일 EV 사후재구성)
        #    risk 기준 = 초기 SL (트레일 이동 무관). entry_atr 아닌 실제 SL폭.
        _risk_mfe = abs(t.entry_price - (t.initial_sl if t.initial_sl else t.sl))
        if _risk_mfe > 0:
            if t.direction == "long":
                fav_r = (h - t.entry_price) / _risk_mfe
                adv_r = (t.entry_price - l) / _risk_mfe
                _sl_hit = (l <= (t.initial_sl if t.initial_sl else t.sl))
            else:
                fav_r = (t.entry_price - l) / _risk_mfe
                adv_r = (h - t.entry_price) / _risk_mfe
                _sl_hit = (h >= (t.initial_sl if t.initial_sl else t.sl))
            # clean MFE: 초기 SL 최초 터치 전까지만 best favorable 갱신
            if not t._sl_touched:
                if fav_r > t._mfe_best_r:
                    t._mfe_best_r = fav_r
                if _sl_hit:
                    t._sl_touched = True   # 이후 봉은 clean MFE 갱신 중단
            # trail_path: 봉별 (favR,advR) — stage6/4c replay용. horizon 상한.
            if len(t._trail_buf) < 600:
                t._trail_buf.append(f"{fav_r:.2f},{adv_r:.2f}")

        # ── Timeout 스윕 추적 (한 번 수집으로 여러 timeout 비교)
        _bars_held = current_bar_idx - t.entry_bar_idx
        _risk = abs(t.entry_price - (t.initial_sl if t.initial_sl else t.sl))
        if _risk > 0:
            # 이 봉에서 TP/SL 최초 도달 기록 (미도달이면 -1 유지)
            if t.bars_to_tp < 0 and t.tp is not None:
                if (t.direction == "long" and h >= t.tp) or \
                   (t.direction == "short" and l <= t.tp):
                    t.bars_to_tp = _bars_held
            if t.bars_to_sl < 0:
                _sl0 = t.initial_sl if t.initial_sl else t.sl
                if (t.direction == "long" and l <= _sl0) or \
                   (t.direction == "short" and h >= _sl0):
                    t.bars_to_sl = _bars_held
            # 각 timeout 지점의 R (종가 기준) — 도달한 시점에 기록
            _c = float(bar["close"])
            _move = (_c - t.entry_price) if t.direction == "long" else (t.entry_price - _c)
            _r_now = _move / _risk
            # ATR 정규화 (SL 길이 독립) — entry_atr이 0이면 risk로 fallback
            _atr = t.entry_atr if t.entry_atr and t.entry_atr > 0 else _risk
            _ratr_now = _move / _atr if _atr > 0 else 0.0
            if _bars_held == 48:
                t.r_at_48 = round(_r_now, 3); t.ratr_at_48 = round(_ratr_now, 3)
            elif _bars_held == 72:
                t.r_at_72 = round(_r_now, 3); t.ratr_at_72 = round(_ratr_now, 3)
            elif _bars_held == 96:
                t.r_at_96 = round(_r_now, 3); t.ratr_at_96 = round(_ratr_now, 3)
            elif _bars_held == 144:
                t.r_at_144 = round(_r_now, 3); t.ratr_at_144 = round(_ratr_now, 3)
        best_delta = (best_px - t.entry_price) / t.entry_price
        if t.direction == "short":
            best_delta = -best_delta
        best_pnl = t.size_usdt * best_delta - t.size_usdt * TAKER_FEE_RATE * 2
        if best_pnl > t.max_unrealized_pnl:
            t.max_unrealized_pnl = best_pnl

        # ── VP density 누적 (bar-by-bar)
        # 유리한 방향의 peak 가격 기반: 새로 돌파한 구간만 누적
        if hasattr(t, '_vp_profile') and t._vp_profile is not None:
            # _vp_prev_peak: 지금까지 유리한 방향으로 도달한 최대 가격
            if not hasattr(t, '_vp_prev_peak'):
                t._vp_prev_peak = t.entry_price

            if t.direction == "long":
                # long: high가 이전 peak를 넘어선 구간만 density 누적
                if h > t._vp_prev_peak:
                    t.vp_consumption += t._vp_profile.density_between(t._vp_prev_peak, h)
                    t._vp_prev_peak = h
            else:
                # short: low가 이전 peak(최저) 아래로 내려간 구간만 density 누적
                if l < t._vp_prev_peak:
                    t.vp_consumption += t._vp_profile.density_between(l, t._vp_prev_peak)
                    t._vp_prev_peak = l

        # ── max adverse price 추적 (adversity 계산용)
        if t.direction == "long":
            _adverse = l
            if t.max_adverse_px == 0.0 or _adverse < t.max_adverse_px:
                t.max_adverse_px = _adverse
        else:
            _adverse = h
            if t.max_adverse_px == 0.0 or _adverse > t.max_adverse_px:
                t.max_adverse_px = _adverse

        # ── disp_extreme 돌파 체크
        if t.disp_extreme > 0 and not t.disp_extreme_broken:
            if t.direction == "long" and h >= t.disp_extreme:
                t.disp_extreme_broken = True
            elif t.direction == "short" and l <= t.disp_extreme:
                t.disp_extreme_broken = True

        # ── TP/SL 판정
        old_sl = t.sl
        bar_open = float(bar["open"])
        _tp_hit = False
        _sl_hit = False
        if t.direction == "long":
            if t.tp is not None and h >= t.tp:
                _tp_hit = True
            if l <= old_sl:
                _sl_hit = True
        else:
            if t.tp is not None and l <= t.tp:
                _tp_hit = True
            if h >= old_sl:
                _sl_hit = True

        if _tp_hit and _sl_hit:
            _tp_px = float(t.tp)
            _dist_tp = abs(bar_open - _tp_px)
            _dist_sl = abs(bar_open - old_sl)
            if _dist_tp <= _dist_sl:
                return self._close(t, _tp_px, bar["time"], "tp")
            else:
                reason = "trail_sl" if t.trailing_activated else "sl"
                return self._close(t, old_sl, bar["time"], reason)
        elif _tp_hit:
            return self._close(t, float(t.tp), bar["time"], "tp")
        elif _sl_hit:
            reason = "trail_sl" if t.trailing_activated else "sl"
            return self._close(t, old_sl, bar["time"], reason)

        # ── swing_cb 트레일 (stage6 확정): close 돌파 BOS → 직전 swing low lock.
        #    별도 모드이므로 기존 R-게이트 트레일과 배타. SL만 갱신하고 봉 종료.
        if _TRAIL_MODE == "swing_cb":
            _update_swing_cb_trail(t, current_bar_idx, float(bar["close"]), pivots_5m)
            if t.sl != old_sl:
                t.trailing_activated = True
                t.sl_history.append((bar["time"], t.sl))
            return None  # 청산 없음, 다음 봉

        # ── 트레일링 스탑
        cur_px = h if t.direction == "long" else l
        risk = abs(t.entry_price - t.sl)
        _is_group_a = (t.reason or {}).get("level_group", "B") == "A" if t.reason else False
        _trail_activate_r = TRAIL_ACTIVATE_R_A if _is_group_a else TRAIL_ACTIVATE_R
        _trail_swing_back = TRAIL_SWING_BACK_A if _is_group_a else TRAIL_SWING_BACK
        _bar_hour = pd.to_datetime(bar["time"]).hour if "time" in bar.index else 0
        _late_session = (_bar_hour >= 21)

        if risk > 0 and pivots_5m is not None:
            cur_delta = (cur_px - t.entry_price) / t.entry_price
            if t.direction == "short":
                cur_delta = -cur_delta
            cur_r = (t.entry_price * cur_delta) / risk

            BREAKEVEN_R = _BREAKEVEN_R  # 모듈 상수 사용 (TRAIL_MODE에 따라 1.0 또는 1000.0)
            BREAKEVEN_BUFFER = 0.001  # 본절 + 0.1%
            if cur_r >= BREAKEVEN_R and not getattr(t, "_breakeven_set", False):
                t._breakeven_set = True
                if t.direction == "long" and t.sl < t.entry_price:
                    t.sl = t.entry_price * (1 + BREAKEVEN_BUFFER)
                    t.sl_history.append((bar["time"], t.sl, "breakeven_1r"))
                elif t.direction == "short" and t.sl > t.entry_price:
                    t.sl = t.entry_price * (1 - BREAKEVEN_BUFFER)
                    t.sl_history.append((bar["time"], t.sl, "breakeven_1r"))

            if cur_r >= _trail_activate_r:
                if not t.trailing_activated:
                    t.trailing_activated = True
                    if t.direction == "long" and t.sl < t.entry_price:
                        t.sl = t.entry_price
                        t.sl_history.append((bar["time"], t.sl, "breakeven"))
                    elif t.direction == "short" and t.sl > t.entry_price:
                        t.sl = t.entry_price
                        t.sl_history.append((bar["time"], t.sl, "breakeven"))
                    t.sl_history.append((bar["time"], t.sl, "trail_start"))

                new_sl = _find_trail_sl(t.direction, pivots_5m, t.sl, cur_px,
                                        entry_bar_idx=t.entry_bar_idx,
                                        swing_back=_trail_swing_back)
                if new_sl is not None and new_sl != t.sl:
                    if t.direction == "long" and new_sl > t.sl:
                        t.sl_history.append((bar["time"], new_sl, "trail_update"))
                        t.sl = new_sl
                    elif t.direction == "short" and new_sl < t.sl:
                        t.sl_history.append((bar["time"], new_sl, "trail_update"))
                        t.sl = new_sl

                _fb_pct = 0.004 if _late_session else TRAIL_FALLBACK_PCT
                if new_sl is None:
                    if t.direction == "long":
                        fb_sl = t.peak_px * (1 - _fb_pct)
                        if fb_sl > t.sl:
                            t.sl = fb_sl
                            t.sl_history.append((bar["time"], fb_sl, "trail_fallback"))
                    else:
                        fb_sl = t.peak_px * (1 + _fb_pct)
                        if fb_sl < t.sl:
                            t.sl = fb_sl
                            t.sl_history.append((bar["time"], fb_sl, "trail_fallback"))

                if liq_levels:
                    liq_sl = _find_liq_trail_sl(t.direction, liq_levels, t.sl, h, l)
                    if liq_sl is not None:
                        if t.direction == "long" and liq_sl > t.sl:
                            t.sl_history.append((bar["time"], liq_sl, "trail_liq"))
                            t.sl = liq_sl
                        elif t.direction == "short" and liq_sl < t.sl:
                            t.sl_history.append((bar["time"], liq_sl, "trail_liq"))
                            t.sl = liq_sl

        return None

    def force_close_all(self, bar: pd.Series) -> List[Trade]:
        closed = []
        for t in self.positions:
            closed.append(self._close(t, float(bar["close"]), bar["time"], "forced"))
        self.positions = []
        return closed

    def _close(self, t: Trade, exit_px: float, exit_time, reason: str) -> Trade:
        t.exit_price = exit_px
        t.exit_time = pd.to_datetime(exit_time)
        t.exit_reason = reason
        # ── MFE/trail 확정: 매봉 누적 결과를 출력 필드로 고정
        t.mfe_r_clean = round(float(t._mfe_best_r), 4)
        t.trail_path = ";".join(t._trail_buf) if t._trail_buf else ""
        delta = (exit_px - t.entry_price) / t.entry_price
        if t.direction == "short":
            delta = -delta
        t.pnl_usdt = t.size_usdt * delta - t.size_usdt * TAKER_FEE_RATE * 2

        # ── Shadow가 아직 청산 안 됐으면 primary와 동일하게 마무리
        #     (shadow의 trail이 primary보다 늦게/안 걸린 경우 → 같은 exit)
        if t.shadow_enabled and t.shadow_exit_reason is None:
            t.shadow_exit_price = float(exit_px)
            t.shadow_exit_time = pd.to_datetime(exit_time)
            t.shadow_exit_reason = reason
            risk = abs(t.entry_price - t.initial_sl) if t.initial_sl else abs(t.entry_price - t.sl)
            if risk > 0:
                t.shadow_pnl_r = (t.entry_price * delta) / risk
            else:
                t.shadow_pnl_r = 0.0
        return t


# ──────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────

def _print_trade(trade: Trade, equity: float, df_5m: "pd.DataFrame | None" = None):
    d   = "🟢 LONG " if trade.direction == "long" else "🔴 SHORT"
    res = "✅ WIN " if trade.pnl_usdt > 0 else "❌ LOSS"
    sgn = "+" if trade.pnl_usdt >= 0 else ""
    tp_str = f"{trade.tp:,.2f}" if trade.tp else "N/A"

    # ── 진입 근거 포맷
    reason_lines = ""
    r = trade.reason
    if r:
        trigger      = str(r.get("trigger", "?")).upper()
        sw           = r.get("sweep_extreme")
        choch_px     = r.get("choch_price")
        disp         = r.get("disp_extreme")
        ote_zone     = r.get("ote_zone")
        ob           = r.get("ob")
        fvg          = r.get("fvg")
        orders       = r.get("orders") or []
        size_mult    = r.get("size_mult", 1.0)
        score_pct    = r.get("score_pct")
        lv           = r.get("swept_level") or {}
        htf_state    = r.get("htf_state", "?")
        htf_aligned  = r.get("htf_aligned", False)
        htf_4h       = r.get("htf_4h", "?")
        level_group  = r.get("level_group", "?")
        sweep_vol_ok = r.get("sweep_vol_ok", True)
        has_disp_fvg = r.get("has_disp_fvg", False)
        sweep_ratio  = r.get("sweep_ratio", 0.0)
        sweep_good   = sweep_ratio >= 0.35

        lv_type = str(lv.get("level_type", "?"))
        lv_type_label = {
            "eqh":            "EQH (Equal Highs)",
            "eql":            "EQL (Equal Lows)",
            # london/ny 표시 매핑 제거 (v2.2: 레벨 자체 제거)
            "asia_high":      "Asia High",
            "asia_low":       "Asia Low",
            "pivot":          "Pivot",
        }.get(lv_type, lv_type)
        lv_zone = ""
        if lv.get("zone_low") and lv.get("zone_high"):
            lv_zone = f"  [{lv['zone_low']:,.2f} ~ {lv['zone_high']:,.2f}]"

        # 피봇 시각 출력 (idx_start_time/idx_end_time 키 사용)
        # 레벨 시각 출력 (ts_start/ts_end)
        lv_pivot_times = ""
        if lv:
            t_s = lv.get("ts_start")
            t_e = lv.get("ts_end")
            touches = lv.get("touches")
            if t_s:
                ts_str = pd.Timestamp(t_s).strftime("%m-%d %H:%M")
                te_str = pd.Timestamp(t_e).strftime("%m-%d %H:%M") if t_e else "ongoing"
                touches_str = f"  touches={int(touches)}" if touches is not None else ""
                lv_pivot_times = f"  [{ts_str} ~ {te_str}{touches_str}]"

        if ote_zone:
            ote_str = f"{ote_zone['low']:,.2f} ~ {ote_zone['high']:,.2f}  (ideal {ote_zone['ideal']:,.2f})"
        else:
            ote_str = "N/A (turtle soup)"

        ob_str = "없음"
        if ob:
            ob_str = f"{ob.get('type','ob')}  {ob.get('low',0):,.2f}~{ob.get('high',0):,.2f}"

        fvg_str = "없음"
        if fvg:
            fib = fvg.get("fib_pos") or fvg.get("ote_pos_mid", 0)
            fvg_str = (
                f"{fvg.get('fvg_type','fvg')}  "
                f"{fvg.get('zone_low',0):,.2f}~{fvg.get('zone_high',0):,.2f}  "
                f"fib={fib:.3f}"
            )

        order_str = "  ".join(
            f"{o.get('tag','?')}@{o.get('price',0):,.0f}({int(o.get('weight',0)*100)}%)"
            for o in orders
        )
        # 포지션비중 = 노출가치/잔고 (score_pct는 점수값이라 *100 하면 틀림)
        pos_ratio = float(r.get("_notional", 0)) / max(equity, 1) * 100 if r.get("_notional") else None
        score_str = f"{pos_ratio:.0f}%" if pos_ratio is not None else "?"
        pd_pos_val = r.get("pd_pos")
        pd_pos_str = f"{pd_pos_val:.2f}" if pd_pos_val is not None else "N/A"
        sw_str    = f"{sw:,.2f}" if sw else "?"
        cp_str    = f"{choch_px:,.2f}" if choch_px else "?"
        disp_str  = f"{disp:,.2f}" if disp else "?"
        htf_icon  = "✓" if htf_aligned else "✗"
        swq_icon  = f"{"✓" if sweep_good else "~"}({sweep_ratio:.2f})"
        dfvg_icon = "✓" if has_disp_fvg else "✗"

        trigger_flow = (
            f"sweep={sw_str} → disp={disp_str}"
            if level_group == "A"
            else f"sweep={sw_str} → CHOCH={cp_str} → disp={disp_str}"
        )

        reason_lines = (
            f"\n  ┌ 진입 근거 ──────────────────────────────────────────\n"
            f"  │ 스윕 레벨  : {lv_type_label}{lv_zone}{lv_pivot_times}\n"
            f"  │ Sweep      : wick={swq_icon}  HTF=1H:{htf_state} {htf_icon}  4H:{htf_4h}  vol={"✓" if sweep_vol_ok else "✗"}  group={level_group}\n"
            f"  │ 트리거     : {trigger}  {trigger_flow}  [grp={level_group}]\n"
            f"  │ OTE 구간   : {ote_str}\n"
            f"  │ OB         : {ob_str}  (SL=sweep_extreme if GroupA)\n"
            f"  │ FVG        : {fvg_str}\n"
            f"  │ 오더       : {order_str}\n"
            f"  │ 사이즈     : ×{size_mult:.2f}  포지션비중 {score_str}\n"
            f"  │ PD 위치    : {pd_pos_str}  (0=저점, 1=고점)\n"
            f"  └─────────────────────────────────────────────────────"
        )

    # SL/트레일링SL 청산이면 최고 미실현 손익 + 트레일링 이력 표시
    sl_extra = ""
    if trade.exit_reason in ("sl", "trail_sl") and trade.max_unrealized_pnl != 0:
        mu     = trade.max_unrealized_pnl
        sgn_mu = "+" if mu >= 0 else ""
        is_trail = trade.exit_reason == "trail_sl"
        if mu > 0:
            if is_trail:
                sl_extra = f"\n  🔒 트레일링SL 청산  최고 미실현: {sgn_mu}{mu:.2f} USDT  ✓ 구조 기반 보호"
            else:
                sl_extra = f"\n  ⚠️  SL 전 최고 미실현: {sgn_mu}{mu:.2f} USDT  (TP/SL 로직 점검)"
        else:
            sl_extra = f"\n  ⚠️  SL 전 최고 미실현: {sgn_mu}{mu:.2f} USDT  (진입 자체가 문제)"
        if trade.trailing_activated and trade.sl_history:
            moves = len([x for x in trade.sl_history if len(x) > 2 and x[2] == "trail_update"])
            if moves:
                final_sl = trade.sl_history[-1][1]
                sl_extra += f"\n  🔒 트레일링 SL 이동 {moves}회  (초기SL→최종SL: {trade.sl_history[0][1]:,.0f}→{final_sl:,.0f})"

    print(
        f"\n{'─'*65}\n"
        f"  [{trade.trade_id:>3}] {d}  {res}\n"
        f"  진입: {trade.entry_time.strftime('%Y-%m-%d %H:%M')}  @ {trade.entry_price:,.2f}\n"
        f"  청산: {trade.exit_time.strftime('%Y-%m-%d %H:%M')}  @ {trade.exit_price:,.2f}  ({trade.exit_reason})\n"
        f"  SL={trade.sl:,.2f}  TP={tp_str}\n"
        f"  노출가치={trade.size_usdt:,.1f} USDT   PnL={sgn}{trade.pnl_usdt:.2f} USDT\n"
        f"  잔고: {equity:,.2f} USDT"
        f"{sl_extra}"
        f"{reason_lines}\n"
        f"{'─'*65}"
    )


def _print_progress(
    i: int, total: int, t: pd.Timestamp, equity: float, n_trades: int,
    active_trade: Optional["Trade"] = None, current_price: float = 0.0,
):
    pct = i / total * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    base = (
        f"\r  [{bar}] {pct:5.1f}%  "
        f"{t.strftime('%Y-%m-%d %H:%M')}  "
        f"잔고={equity:,.1f}  거래={n_trades}건"
    )
    if active_trade is not None and current_price > 0:
        d = "🟢L" if active_trade.direction == "long" else "🔴S"
        tp_str = f"{active_trade.tp:,.0f}" if active_trade.tp else "N/A"
        sl_str = f"{active_trade.sl:,.0f}"
        # 미실현 손익
        delta = (current_price - active_trade.entry_price) / active_trade.entry_price
        if active_trade.direction == "short":
            delta = -delta
        unrealized = active_trade.size_usdt * delta
        sgn = "+" if unrealized >= 0 else ""
        base += (
            f"  │ {d} 진입={active_trade.entry_price:,.0f}  "
            f"현재={current_price:,.0f}  "
            f"SL={sl_str}  TP={tp_str}  "
            f"미실현={sgn}{unrealized:.1f}"
        )
    print(base + " " * 10, end="", flush=True)


def _print_liq_levels(
    t: pd.Timestamp,
    liq_all: list,
    current_price: float,
    strategy_state: str,
):
    """100봉마다 병합된 유동성 레벨 출력 (현재가 기준 위 3개 / 아래 3개)."""
    print(f"\n\n  {'─'*62}")
    print(f"  📊 유동성 레벨  [{t.strftime('%Y-%m-%d %H:%M')}]")
    print(f"     현재가: {current_price:,.2f}  |  전략상태: {strategy_state}")
    print(f"  {'─'*62}")

    if not liq_all:
        print("  레벨 없음")
        print(f"  {'─'*62}\n")
        return

    above = sorted(
        [lv for lv in liq_all if (lv["zone_low"] + lv["zone_high"]) / 2 > current_price],
        key=lambda x: (x["zone_low"] + x["zone_high"]) / 2
    )[:3]

    below = sorted(
        [lv for lv in liq_all if (lv["zone_low"] + lv["zone_high"]) / 2 < current_price],
        key=lambda x: (x["zone_low"] + x["zone_high"]) / 2,
        reverse=True
    )[:3]

    print("  위 저항 (sweep → short 기회)")
    if above:
        for lv in above:
            mid  = (lv["zone_low"] + lv["zone_high"]) / 2
            dist = (mid - current_price) / current_price * 100
            print(f"    ↑ {lv['zone_low']:>12,.2f} ~ {lv['zone_high']:>12,.2f}"
                  f"  (+{dist:.2f}%)  {lv['level_type']}  str={lv['strength']:.1f}")
    else:
        print("    없음")

    print("  아래 지지 (sweep → long 기회)")
    if below:
        for lv in below:
            mid  = (lv["zone_low"] + lv["zone_high"]) / 2
            dist = (current_price - mid) / current_price * 100
            print(f"    ↓ {lv['zone_low']:>12,.2f} ~ {lv['zone_high']:>12,.2f}"
                  f"  (-{dist:.2f}%)  {lv['level_type']}  str={lv['strength']:.1f}")
    else:
        print("    없음")

    print(f"  {'─'*62}\n")


def _print_final_report(result: BacktestResult, symbol: str, months: int):
    w, l = len(result.win_trades), len(result.lose_trades)
    sgn  = "+" if result.total_return_pct >= 0 else ""
    print(f"\n\n{'═'*65}")
    print(f"  📊 백테스트 최종 리포트")
    print(f"  심볼: {symbol}   기간: 최근 {months}개월")
    print(f"{'═'*65}")
    print(f"  초기 자산     : {result.initial_equity:>10,.2f} USDT")
    print(f"  최종 자산     : {result.final_equity:>10,.2f} USDT")
    print(f"  수익률        : {sgn}{result.total_return_pct:>9.2f} %")
    print(f"{'─'*65}")
    print(f"  총 거래 수    : {len(result.trades):>10}건")
    print(f"  승리          : {w:>10}건")
    print(f"  패배          : {l:>10}건")
    print(f"  승률          : {result.win_rate:>9.1f} %")
    if result.trades:
        print(f"{'─'*65}")
        print(f"  평균 수익     : +{result.avg_win:>8.2f} USDT")
        print(f"  평균 손실     :  {result.avg_loss:>8.2f} USDT")
        print(f"  손익비(PF)    : {result.profit_factor:>10.2f}")
        print(f"  최대 낙폭(DD) : {result.max_drawdown_pct:>9.2f} %")
        # 트레일링 통계
        trail_trades  = [t for t in result.trades if t.trailing_activated]
        trail_wins    = [t for t in trail_trades if t.pnl_usdt > 0]
        trail_sl_cnt  = [t for t in trail_trades if t.exit_reason == "trail_sl"]
        if trail_trades:
            print(f"{'─'*65}")
            print(f"  🔒 트레일링 활성화  : {len(trail_trades):>4}건  "
                  f"(수익={len(trail_wins)}건  트레일SL={len(trail_sl_cnt)}건)")
            saved = sum(t.max_unrealized_pnl - max(0, t.pnl_usdt)
                        for t in trail_sl_cnt if t.max_unrealized_pnl > 0)
            if saved > 0:
                print(f"  🔒 트레일링으로 보호된 수익 추정: +{saved:.2f} USDT")
    print(f"{'═'*65}\n")

    if result.trades:
        long_t  = [t for t in result.trades if t.direction == "long"]
        short_t = [t for t in result.trades if t.direction == "short"]
        long_w  = sum(1 for t in long_t  if t.pnl_usdt > 0)
        short_w = sum(1 for t in short_t if t.pnl_usdt > 0)
        print(f"{'─'*65}")
        print(f"  📈 LONG  : {len(long_t):>3}건  승{long_w}패{len(long_t)-long_w}  "
              f"PnL={sum(t.pnl_usdt for t in long_t):+.2f}")
        print(f"  📉 SHORT : {len(short_t):>3}건  승{short_w}패{len(short_t)-short_w}  "
              f"PnL={sum(t.pnl_usdt for t in short_t):+.2f}")

        # ── 유동성별 승패 기록
        lv_stats: dict = {}
        for t in result.trades:
            lv_type = "?"
            if t.reason:
                _sl = t.reason.get("swept_level") or {}
                lv_type = _sl.get("level_type", "?")
            if lv_type not in lv_stats:
                lv_stats[lv_type] = {"win": 0, "loss": 0, "pnl": 0.0}
            if t.pnl_usdt > 0:
                lv_stats[lv_type]["win"] += 1
            else:
                lv_stats[lv_type]["loss"] += 1
            lv_stats[lv_type]["pnl"] += t.pnl_usdt

        if lv_stats:
            print(f"\n{'─'*65}")
            print(f"  📊 유동성별 성과")
            print(f"  {'레벨':^15}  {'승':>3}  {'패':>3}  {'승률':>6}  {'PnL':>10}")
            print(f"  {'─'*50}")
            for lt, st in sorted(lv_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                total = st["win"] + st["loss"]
                wr = st["win"] / total * 100 if total > 0 else 0
                print(f"  {lt:^15}  {st['win']:>3}  {st['loss']:>3}  {wr:>5.1f}%  {st['pnl']:>+10.2f}")

    if result.trades:
        print("  📋 거래 내역 (최근 20건)")
        print(f"  {'#':>3}  {'방향':^5}  {'진입시각':^16}  {'진입가':>10}  {'청산가':>10}  {'PnL':>9}  결과")
        print(f"  {'─'*75}")
        for t in result.trades[-20:]:
            d   = "LONG " if t.direction == "long" else "SHORT"
            et  = t.entry_time.strftime("%m-%d %H:%M")
            sgn = "+" if t.pnl_usdt >= 0 else ""
            res = "WIN" if t.pnl_usdt > 0 else "LOSS"
            xp  = f"{t.exit_price:>10,.2f}" if t.exit_price else f"{'N/A':>10}"
            print(f"  {t.trade_id:>3}  {d}  {et:^16}  {t.entry_price:>10,.2f}  {xp}  {sgn}{t.pnl_usdt:>8.2f}  {res}")
        print()


def _calc_armed_timeout(df_struct: pd.DataFrame,
                         mult: float = ARMED_TIMEOUT_MULT,
                         min_bars: int = ARMED_TIMEOUT_MIN,
                         max_bars: int = ARMED_TIMEOUT_MAX) -> int:
    """
    워밍업 struct 데이터로 CHOCH/BOS 평균 발생 간격을 계산.
    평균 간격 * mult 를 ARMED timeout으로 사용 (min/max 클램프).
    """
    cols = [c for c in ["choch_bull","choch_bear","bos_bull","bos_bear"] if c in df_struct.columns]
    if not cols:
        return min_bars

    # 이벤트 발생 봉 인덱스 수집
    event_rows = df_struct[cols].any(axis=1)
    idxs = event_rows[event_rows].index.tolist()
    if len(idxs) < 2:
        return min_bars

    gaps = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
    avg_gap = sum(gaps) / len(gaps)
    timeout = int(avg_gap * mult)
    result = max(min_bars, min(max_bars, timeout))
    return result


def _check_limit_fill(bar: pd.Series, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    h, l, bar_open = float(bar["high"]), float(bar["low"]), float(bar["open"])
    bar_close = float(bar["close"])
    direction = plan["direction"]
    for o in plan["orders"]:
        # market 오더: 다음 봉 시가에 즉시 체결
        if o.get("type") == "market":
            return dict(o) | {"price": bar_open}
        px = float(o["price"])
        # 가격 유효성: 진입가가 현재 가격에서 2% 이상 벗어나면 무효
        mid_px = (h + l) / 2
        if mid_px > 0 and abs(px - mid_px) / mid_px > 0.02:
            continue
        if direction == "long" and l <= px:
            # 갭 오픈: 시가가 이미 주문가 이하면 시가에 체결 (더 유리하지만 현실적)
            fill_px = min(px, bar_open) if bar_open <= px else px
            return dict(o) | {"price": fill_px}
        if direction == "short" and h >= px:
            # 갭 오픈: 시가가 이미 주문가 이상이면 시가에 체결
            fill_px = max(px, bar_open) if bar_open >= px else px
            return dict(o) | {"price": fill_px}
    return None


# ──────────────────────────────────────────────
# 백테스터
# ──────────────────────────────────────────────

class Backtester:
    def __init__(
        self,
        exchange: ccxt.Exchange,
        symbol: str           = "BTC/USDT:USDT",
        months: int           = 1,
        initial_equity: float = 1000.0,
        swing_lookback: int   = 4,
        eq_band_pct: float    = 0.003,
        reclaim_seconds: int  = RECLAIM_SECONDS,
        orders_ttl_bars: int  = 15,
        sl_buffer_pct: float  = 0.001,
        print_every: int      = 100,
        end_date: str         = None,  # "2025-03-25" 형식, None이면 현재까지
        ml_model_path: str    = None,  # ML 필터 모델 경로 (None이면 미사용)
        df_5m: Optional[pd.DataFrame] = None,  # 사전 다운로드된 5m 선물 데이터
        df_spot: Optional[pd.DataFrame] = None,  # 사전 다운로드된 5m 현물 데이터
    ):
        self.exchange       = exchange
        self.symbol         = symbol
        self.months         = months
        self.initial_equity = initial_equity
        self.lb             = swing_lookback
        self.eq_band_pct    = eq_band_pct
        self.reclaim_secs   = reclaim_seconds
        self.orders_ttl     = orders_ttl_bars
        self.sl_buf         = sl_buffer_pct
        self.print_every    = print_every
        self.end_date       = end_date
        self._preloaded_df  = df_5m  # 병렬 실행용: 사전 로드된 선물 데이터
        self._preloaded_spot = df_spot  # 병렬 실행용: 사전 로드된 현물 데이터

        # ── Nasdaq Feature Provider 초기화 (end_date 오프셋 포함한 총 연수 계산)
        from datetime import datetime, timezone
        from nasdaq_features import NasdaqFeatureProvider
        
        offset_years = 0
        if self.end_date:
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            offset_days = (datetime.now(timezone.utc) - end_dt).days
            offset_years = max(0, offset_days / 365.0)
            
        total_years = max(1.5, (self.months / 12.0) + offset_years + 0.1)
        self.nasdaq_provider = NasdaqFeatureProvider(years=total_years, ticker="NQ=F")

        # ── 현물 5m 캔들 로드 (Spot-Futures 괴리 피처용)
        self._spot_sym = self.symbol.split("/")[0] + "/USDT"
        self._spot_exchange = None
        self.spot_df = None
        try:
            self._spot_exchange = ccxt.okx({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
        except Exception as e:
            print(f"  [Spot] ⚠️ exchange 생성 실패: {e}")

        # ── ML 필터 초기화 (Long/Short 분리 모델 자동 감지)
        self.ml_filter = None
        if ml_model_path and HAS_ML:
            try:
                import os
                _long_path = ml_model_path.replace(".json", "_long.json")
                _short_path = ml_model_path.replace(".json", "_short.json")
                # entry_model.json → entry_model_long.json / entry_model_short.json 확인
                if not _long_path.endswith("_long_long.json") and \
                   os.path.exists(_long_path) and os.path.exists(_short_path):
                    self.ml_filter = _EntryFilter(_long_path, _short_path)
                    print(f"[ML] ✅ EntryFilter SPLIT 모드 로드 완료")
                else:
                    self.ml_filter = _EntryFilter(ml_model_path)
                    print(f"[ML] ✅ EntryFilter 로드 완료: {ml_model_path}")
            except Exception as e:
                print(f"[ML] ⚠️ EntryFilter 로드 실패: {e}  (ML 필터 비활성화)")
        elif ml_model_path and not HAS_ML:
            print("[ML] ⚠️ ml_model 모듈 없음 — ML 필터 비활성화")

    def run(self) -> BacktestResult:
        # ── 1. 데이터 로드 (사전 로드된 데이터가 있으면 재사용)
        if self._preloaded_df is not None:
            df_all = self._preloaded_df
            print(f"[backtest] 사전 로드된 데이터 사용: {len(df_all):,}봉")
        else:
            df_all = fetch_historical_5m(self.exchange, self.symbol, self.months,
                                          end_date=self.end_date)
        if len(df_all) < WARMUP_BARS + 10:
            raise ValueError(f"데이터 부족: {len(df_all)}봉")

        # ── 1.5. 현물 5m 캔들 로드 (선물과 동일 방식)
        if self._preloaded_spot is not None:
            self.spot_df = self._preloaded_spot
            self.spot_df["time"] = pd.to_datetime(self.spot_df["time"], utc=True)
            self.spot_df = self.spot_df.set_index("time").resample("5min").ffill().reset_index()
            print(f"  [Spot] 사전 로드된 현물 데이터 사용: {len(self.spot_df):,}봉")
        elif self._spot_exchange is not None:
            try:
                import time as _time_mod
                _time_mod.sleep(1)
                print(f"  [Spot] {self._spot_sym} 현물 5m 캔들 로드 중...")
                self.spot_df = fetch_historical_5m(
                    self._spot_exchange, self._spot_sym,
                    self.months, end_date=self.end_date,
                )
                if self.spot_df is not None and len(self.spot_df) > 0:
                    self.spot_df["time"] = pd.to_datetime(self.spot_df["time"], utc=True)
                    self.spot_df = self.spot_df.set_index("time").resample("5min").ffill().reset_index()
                    print(f"  [Spot] ✅ {len(self.spot_df):,}봉 로드 완료")
                else:
                    self.spot_df = None
                    print(f"  [Spot] ⚠️ 현물 데이터 없음")
            except Exception as e:
                print(f"  [Spot] ⚠️ 로드 실패: {e}")
                self.spot_df = None

        # ── 1.6. ★ Spot-Futures 기간 정렬 (종료일 불일치 수정)
        #     spot이 futures보다 일찍 끝나면(예: spot 5/1, futures 5/4),
        #     resample().ffill()이 spot의 마지막 값을 futures 끝까지 끌고 가서
        #     실제로는 존재하지 않는 묵은 괴리값(stale basis)이 생긴다.
        #     → spot을 futures의 [시작, 종료] 범위로 자르고, futures 종료 이후로는
        #       ffill 연장하지 않도록 정리. spot_div 피처가 stale 값을 안 쓰게 함.
        if self.spot_df is not None and len(self.spot_df) > 0 and len(df_all) > 0:
            _fut_start = pd.to_datetime(df_all["time"].iloc[0], utc=True)
            _fut_end = pd.to_datetime(df_all["time"].iloc[-1], utc=True)
            _spot_start = self.spot_df["time"].iloc[0]
            _spot_end = self.spot_df["time"].iloc[-1]

            # 공통 구간으로 제한 (futures 범위 밖의 spot은 제거)
            _before = len(self.spot_df)
            self.spot_df = self.spot_df[
                (self.spot_df["time"] >= _fut_start) &
                (self.spot_df["time"] <= _fut_end)
            ].reset_index(drop=True)
            _after = len(self.spot_df)

            # 종료일 불일치 진단
            _gap_days = (_fut_end - _spot_end).total_seconds() / 86400
            if abs(_gap_days) > 0.5:
                print(f"  [Spot] ⚠️ 종료일 불일치: futures={_fut_end.date()}, "
                      f"spot={_spot_end.date()} ({_gap_days:+.1f}일)")
                print(f"  [Spot]    → spot이 끝난 이후 구간({_spot_end.date()}~{_fut_end.date()})은 "
                      f"spot_div 피처가 0으로 처리됨 (stale 값 방지)")
            if _before != _after:
                print(f"  [Spot]    범위 정렬: {_before:,}봉 → {_after:,}봉 "
                      f"(futures 기간으로 제한)")

            # spot_df의 실제 마지막 시각 기록 (피처 계산 시 이 이후는 spot 없음으로 간주)
            self._spot_last_time = _spot_end if _spot_end <= _fut_end else _fut_end
        else:
            self._spot_last_time = None

        # ── 1.7. Basis 피처 사전계산 (basis_z, basis_momentum, spot_lead)
        #     이미 로드된 spot_df + 선물 df_all 재활용 (재다운로드 없음).
        #     bisect backward lookup용 배열 구축 → 진입 시 미래참조 0.
        self._basis_ts = None
        self._basis_bz = None
        self._basis_bm = None
        self._basis_sl = None
        if self.spot_df is not None and len(self.spot_df) > 100 and len(df_all) > 100:
            try:
                import bisect as _bisect_mod
                self._bisect = _bisect_mod
                _sp = self.spot_df[["time", "close"]].copy()
                _sw = df_all[["time", "close", "volume"]].copy()
                _sp["time"] = pd.to_datetime(_sp["time"], utc=True)
                _sw["time"] = pd.to_datetime(_sw["time"], utc=True)
                _mg = pd.merge(_sw, _sp, on="time", how="inner",
                               suffixes=("_swap", "_spot"))
                if len(_mg) >= 100:
                    _bp = ((_mg["close_swap"] - _mg["close_spot"])
                           / (_mg["close_spot"] + 1e-10) * 100).clip(-1.0, 1.0)
                    _rm = _bp.rolling(360, min_periods=60).mean()
                    _rs = _bp.rolling(360, min_periods=60).std()
                    _bz = ((_bp - _rm) / (_rs + 1e-10)).clip(-4, 4).fillna(0.0)
                    _bm = _bp.diff(12).clip(-0.5, 0.5).fillna(0.0)
                    _spret = _mg["close_spot"].pct_change(12)
                    _swret = _mg["close_swap"].pct_change(12)
                    _sl = (_spret - _swret).clip(-0.05, 0.05).fillna(0.0)
                    # ★ ns로 명시적 통일 (datetime64[us]일 경우 .value(ns)와 단위 불일치 방지)
                    self._basis_ts = (
                        pd.to_datetime(_mg["time"], utc=True)
                        .astype("datetime64[ns, UTC]")
                        .astype(np.int64).tolist()
                    )
                    self._basis_bz = _bz.tolist()
                    self._basis_bm = _bm.tolist()
                    self._basis_sl = _sl.tolist()
                    print(f"  [Basis] ✅ {len(_mg):,}봉 사전계산 "
                          f"(basis_z=[{_bz.min():.2f},{_bz.max():.2f}])")
            except Exception as _e_basis:
                print(f"  [Basis] ⚠️ 사전계산 실패: {_e_basis}")

        # ── 2. 초기화
        equity        = self.initial_equity
        equity_curve  = [equity]
        trades: List[Trade] = []
        trade_id      = 0
        pending_plan: Optional[Dict[str, Any]] = None
        plan_bar_count = 0
        # ob_wait 카운터는 signals.py SweepReclaimOTE 내부에서 관리

        strategy = SweepReclaimOTE(
            reclaim_seconds=self.reclaim_secs,
            orders_ttl_bars=self.orders_ttl,
            sl_buffer_pct=self.sl_buf,
            ob_wait_bars=OB_WAIT_BARS,
        )
        pos_sim = _PositionSim()

        # ── 3. 워밍업: 초기 window로 IncrementalState 생성
        start_i = WARMUP_BARS
        total_i = len(df_all) - start_i
        warmup_window = df_all.iloc[:start_i].copy()

        print(f"\n[backtest] 워밍업 {WARMUP_BARS}봉 초기화 중 ...")
        incr = _IncrState(
            window_5m     = warmup_window,
            lb            = self.lb,
            eq_band_pct   = self.eq_band_pct,
        )
        # CHOCH/BOS 평균 간격으로 ARMED timeout 동적 계산
        armed_timeout = _calc_armed_timeout(incr.df_5m_struct)
        strategy.armed_timeout_bars = armed_timeout
        print(f"[backtest] 초기화 완료. 시뮬레이션 시작 (총 {total_i:,}봉)")
        print(f"[backtest] ARMED timeout = {armed_timeout}봉 "
              f"({armed_timeout*5//60}h {armed_timeout*5%60}m) "
              f"[mult={ARMED_TIMEOUT_MULT}×avg_gap]")

        # ── 초기 유동성 레벨 진단 출력
        _init_liq = incr.get_liq_all(bar_idx=0)
        _liq_by_type = {}
        for _lv in _init_liq:
            _t = _lv.get("level_type", "?")
            _liq_by_type[_t] = _liq_by_type.get(_t, 0) + 1
        print(f"[backtest] 초기 유동성 레벨: 총 {len(_init_liq)}개  {_liq_by_type}")
        # 피봇 수 진단
        print(f"[backtest] 피봇 수: 5m={len(incr.pivots_5m)}개  15m={len(incr.pivots_15m)}개  1h={len(incr.pivots_1h)}개")
        # 15m 저점 피봇 분포 진단 (EQL 생성 조건 확인)
        _pl = incr.pivots_15m[incr.pivots_15m["kind"] == "L"].copy()
        _pl = _pl.sort_values("idx").reset_index(drop=True)
        if len(_pl) >= 2:
            _prices = _pl["price"].to_numpy(dtype=float)
            _idxs   = _pl["idx"].to_numpy(dtype=float)
            _gaps_bars = [_idxs[i+1] - _idxs[i] for i in range(len(_idxs)-1)]
            _pct_diffs = [abs(_prices[i+1] - _prices[i]) / _prices[i] * 100 for i in range(len(_prices)-1)]
            _eq_pairs = sum(1 for pd_, gb in zip(_pct_diffs, _gaps_bars)
                            if pd_ <= 0.35 and gb >= 20)
            print(f"[backtest] 15m 저점 피봇 {len(_pl)}개  "
                  f"평균간격={sum(_gaps_bars)/len(_gaps_bars):.1f}봉  "
                  f"평균가격차={sum(_pct_diffs)/len(_pct_diffs):.2f}%  "
                  f"EQL조건충족(0.35%+5h) 인접쌍={_eq_pairs}개")
        if _init_liq:
            _c0 = float(df_all.iloc[start_i]["close"])
            _sorted = sorted(_init_liq, key=lambda lv: abs((lv["zone_low"]+lv["zone_high"])/2 - _c0))
            print(f"[backtest] 현재가={_c0:,.1f}  (가까운 순)")
            for _lv in _sorted[:8]:
                _zm = (_lv["zone_low"] + _lv["zone_high"]) / 2
                _gap = (_zm - _c0) / _c0 * 100
                _lt  = _lv.get("level_type", "?")
                _tc  = _lv.get("touches", "?")
                print(f"    {_lt:16s}  mid={_zm:>10,.1f}  gap={_gap:+.2f}%  touches={_tc}")
        else:
            print("[backtest] ⚠️  초기 유동성 레벨 없음 — 파라미터 확인 필요")
        print()

        # ── 진단 카운터
        _cnt = {"armed_timeout":0, "standby_timeout":0,
                "armed_long":0, "armed_short":0}

        for i in range(start_i, len(df_all)):
            bar = df_all.iloc[i]
            _orders_filled_this_bar = False

            if (i - start_i) % self.print_every == 0:
                _print_progress(
                    i - start_i, total_i, bar["time"], equity, len(trades),
                    active_trade=pos_sim.active,
                    current_price=float(bar["close"]),
                )
                # 유동성 레벨 현황 출력 (거래 없는 구간 진단용)
                # if pos_sim.active is None and strategy.state == "IDLE":
                #     c = float(bar["close"])
                #     liq_diag = sorted(incr.get_liq_all(bar_idx=i), key=lambda lv: abs((lv["zone_low"]+lv["zone_high"])/2 - c))
                #     print()
                #     print(f"  [LIQ] 현재가={c:,.1f}  감지 레벨 {len(liq_diag)}개  (가까운 순)")
                #     for lv in liq_diag[:6]:
                #         zm = (lv["zone_low"] + lv["zone_high"]) / 2
                #         gap = (zm - c) / c * 100
                #         lt  = lv.get("level_type","?")
                #         tc  = lv.get("touches","?")
                #         print(f"    {lt:16s}  mid={zm:>10,.1f}  gap={gap:+.2f}%  touches={tc}")

            # ── 포지션 SL/TP 체크 (멀티 포지션)
            _closed_this_bar = []
            if pos_sim.n_open > 0:
                _struct_5m = "unknown"
                if incr.df_5m_struct is not None and len(incr.df_5m_struct) > 0:
                    if "structure_state" in incr.df_5m_struct.columns:
                        _struct_5m = str(incr.df_5m_struct["structure_state"].iloc[-1])
                _atr_for_timeout = 0.0
                if incr.df_5m is not None and len(incr.df_5m) >= 15:
                    _df_t = incr.df_5m
                    _tail = _df_t.iloc[-15:]
                    _hi = _tail["high"].to_numpy(dtype=float)
                    _lo = _tail["low"].to_numpy(dtype=float)
                    _cl = _tail["close"].to_numpy(dtype=float)
                    _trs = [max(_hi[j]-_lo[j], abs(_hi[j]-_cl[j-1]), abs(_lo[j]-_cl[j-1]))
                            for j in range(1, len(_hi))]
                    _atr_for_timeout = sum(_trs) / len(_trs) if _trs else 0.0

                closed_list = pos_sim.check_bar(bar, pivots_5m=incr.pivots_5m,
                                                liq_levels=incr.get_liq_all(bar_idx=i),
                                                structure_state_5m=_struct_5m,
                                                current_bar_idx=i,
                                                atr_5m=_atr_for_timeout)
                for closed in closed_list:
                    closed.equity_after = equity + closed.pnl_usdt
                    equity = closed.equity_after
                    equity_curve.append(equity)
                    trades.append(closed)
                    _print_trade(closed, equity, df_5m=incr.df_5m)
                    _closed_this_bar.append(closed)

            # ── 펜딩 주문 체결 체크
            if pending_plan is not None:
                plan_bar_count += 1
                filled = _check_limit_fill(bar, pending_plan)
                if filled is not None:
                    trade_id += 1
                    _entry_atr = 0.0
                    if incr.df_5m is not None and len(incr.df_5m) >= 15:
                        _df_a = incr.df_5m.iloc[-15:]
                        _hi_a = _df_a["high"].to_numpy(dtype=float)
                        _lo_a = _df_a["low"].to_numpy(dtype=float)
                        _cl_a = _df_a["close"].to_numpy(dtype=float)
                        _trs_a = [max(_hi_a[j]-_lo_a[j], abs(_hi_a[j]-_cl_a[j-1]),
                                      abs(_lo_a[j]-_cl_a[j-1])) for j in range(1, len(_hi_a))]
                        _entry_atr = sum(_trs_a) / len(_trs_a) if _trs_a else 0.0
                    _initial_sl = float(pending_plan["sl"])
                    t = Trade(
                        trade_id=trade_id,
                        direction=pending_plan["direction"],
                        entry_price=float(filled["price"]),
                        sl=float(pending_plan["sl"]),
                        tp=pending_plan.get("tp"),
                        size_usdt=float(pending_plan["_notional"]),
                        entry_time=bar["time"],
                        reason=pending_plan.get("_reason"),
                        entry_bar_idx=i,
                        entry_atr=_entry_atr,
                        initial_sl=_initial_sl,
                        shadow_enabled=_SHADOW_TRAIL,
                    )

                    # ── VP scoring 초기화
                    try:
                        from volume_profile import VPDensityProfile
                        _vp_idx = len(incr.df_5m) - 1  # incr.df_5m은 tail이므로 마지막 행이 현재 봉
                        t._vp_profile = VPDensityProfile.build(
                            incr.df_5m, _vp_idx, lookback_bars=288, n_bins=50
                        )
                    except Exception as _vp_err:
                        t._vp_profile = None
                    t.vp_prev_close = float(filled["price"])

                    # ── disp_extreme 기록
                    _reason = pending_plan.get("_reason", {})
                    _de = _reason.get("disp_extreme", 0.0)
                    if _de:
                        t.disp_extreme = float(_de)

                    pos_sim.open(t)
                    pending_plan   = None
                    plan_bar_count = 0
                    _orders_filled_this_bar = True
                elif plan_bar_count >= self.orders_ttl:
                    pending_plan   = None
                    plan_bar_count = 0
                    strategy.reset()

            # ── incremental update
            incr.update(bar)

            # ── FVG (필요한 상태에서만)
            fvgs_5m = pd.DataFrame()
            if strategy.state in ("ARMED", "DISP_RUNNING", "FVG_WATCHING", "READY", "ORDERS_PLANNED"):
                fvgs_5m = detect_fvgs_lookback(
                    incr.df_5m_struct,
                    lookback_bars=FVG_LOOKBACK_BARS,
                    fill_mode="wick",
                    remove_filled=True,
                )

            in_position     = pos_sim.n_open > 0
            position_closed = len(_closed_this_bar) > 0
            _exit_reason    = _closed_this_bar[-1].exit_reason if _closed_this_bar else ""

            liq_now = incr.get_liq_all(bar_idx=i)
            result = strategy.step_on_5m_close(
                bar_5m=bar.to_dict(),
                liquidity_levels=liq_now,
                df_5m_struct=incr.df_5m_struct,
                fvgs_5m=fvgs_5m,
                obs_5m=incr.obs_5m_df,
                htf_state=incr.htf_state,
                tp_levels=liq_now,
                df_5m=incr.df_5m,
                df_1h=incr.df_1h,
                htf_4h=incr.htf_4h_state,
                in_position=in_position,
                position_closed=position_closed,
                orders_filled=_orders_filled_this_bar,
                pivots_15m=incr.pivots_15m,
                df_15m_struct=incr.df_15m_struct,
                exit_reason=_exit_reason,
            )

            # ── 상태 진단 로그 (최초 500봉마다)
            if i % 500 == 0 and i > 0:
                _state = result.get("state", "?")
                _action = result.get("action", "?")
                _cnt.setdefault("state_log", {})
                _cnt["state_log"][_state] = _cnt["state_log"].get(_state, 0) + 1
                # if _state not in ("STANDBY", "IDLE", "none"):
                #     print(f"  [{bar['time']}] [DIAG] state={_state} action={_action} "
                #           f"trades={len(trades)} df_5m={len(incr.df_5m) if incr.df_5m is not None else 0}봉")

            # ── 상태 전환 카운터 (최종 리포트용)
            _cur_state = result.get("state", "none")
            _cnt.setdefault("states_seen", set())
            if _cur_state not in _cnt["states_seen"]:
                _cnt["states_seen"].add(_cur_state)
                # if _cur_state not in ("STANDBY", "none"):
                #     print(f"  [{bar['time']}] [DIAG] 새 상태 감지: {_cur_state}")

            # ── enter_plan 생성 감지
            if result["action"] == "enter_plan":
                ep_diag = result.get("entry_plan", {})
                # print(f"  [{bar['time']}] [DIAG] ★ enter_plan 생성! "
                #       f"dir={ep_diag.get('direction','?')} "
                #       f"level={ep_diag.get('swept_level',{}).get('type','?')} "
                #       f"sl={ep_diag.get('sl','?')}")

            # ── notify 디버그 출력 (상태 변화만, SWEEP_HARD_BLOCK 요약)
            _notifs = result.get("notify", [])
            # 진단 카운터 집계
            for _m in _notifs:
                if "ARMED timeout"   in _m:   _cnt["armed_timeout"] += 1
                if "STANDBY timeout" in _m:   _cnt["standby_timeout"] += 1
                if "ARMED:" in _m:
                    _dir = result.get("direction","")
                    if _dir == "long":  _cnt["armed_long"] += 1
                    if _dir == "short": _cnt["armed_short"] += 1
            _shown = set()
            for msg in _notifs:
                if any(k in msg for k in ["STANDBY","ARMED","DISP_","ENTRY","reset",
                                           "SL_ERR","NO_TP","OB_WAIT","FVG_WAIT",
                                           "RE-ENTRY_NOTE","SWEEP_SHALLOW_NOTE",
                                           "FVG_HARD_LIMIT","FVG/OB_HARD_LIMIT"]):
                    if msg not in _shown:
                        # print(f"  [{bar['time']}] {msg}")
                        _shown.add(msg)

            # ── timeout된 레벨 유동성 풀에서 삭제
            for _dl in strategy.drain_pending_disables():
                removed = incr.liq_pool.disable_level(_dl["level_type"], _dl["price_center"])
                if removed:
                    _cnt.setdefault("liq_disabled", 0)
                    _cnt["liq_disabled"] += 1

            # ── enter_plan 처리 (ob_wait는 signals.py 내부에서 관리)
            if result["action"] == "enter_plan" and pending_plan is None:
                ep = result["entry_plan"]

                px       = float(bar["close"])
                ob_1h_k  = "bull_ob" if ep["direction"] == "long" else "bear_ob"
                ob_1h    = incr.get_obs_1h().get(ob_1h_k)
                ob_zone  = {"low": ob_1h["zone_low"], "high": ob_1h["zone_high"]} if ob_1h else None

                orders_  = ep["orders"]
                total_w  = sum(float(o["weight"]) for o in orders_)
                avg_e    = sum(float(o["price"]) * float(o["weight"]) for o in orders_) / total_w if total_w else px

                score = compute_total_score(
                    direction=ep["direction"], pd_pos=None,
                    htf_state=incr.htf_state, px=avg_e, ob_zone_1h=ob_zone,
                )

                # ── feature_dict 보강: signals.py에서 전달된 피처 + backtest에서 추가
                feature_dict = ep.get("feature_dict", {})

                # ── 4H FVG Lifecycle 맥락 feature (confluence layer)
                _dir = ep["direction"]
                _atr4h = incr._atr_4h()
                _fvg_ctx = incr.fvg_tracker_4h.context_at(
                    price=avg_e, direction=_dir, atr=_atr4h)
                feature_dict["in_4h_fvg"] = _fvg_ctx["in_4h_fvg"]
                feature_dict["in_4h_fvg_aligned"] = _fvg_ctx["in_4h_fvg_aligned"]
                feature_dict["nearest_4h_fvg_dist_atr"] = _fvg_ctx["nearest_4h_fvg_dist_atr"]
                feature_dict["active_4h_fvg_fill"] = _fvg_ctx["active_4h_fvg_fill"]
                feature_dict["n_active_4h_fvg"] = _fvg_ctx["n_active_4h_fvg"]
                # ── 4H FVG 진단 feature (약한 원인 분석: 터치후봉수/침투fib/상태/거리)
                #    sweep_extreme 기준 위치도 측정 (entry는 reclaim 후라 zone밖 착시 가능)
                _cur4h = len(incr.df_4h) - 1 if incr.df_4h is not None else -1
                _sweep_ext_diag = ep.get("reason", {}).get("sweep_extreme")
                if _sweep_ext_diag is None:
                    _sweep_ext_diag = ep.get("sweep_extreme")
                _diag = incr.fvg_tracker_4h.diagnose_at(
                    price=avg_e, direction=_dir, atr=_atr4h, cur_idx=_cur4h,
                    sweep_extreme=_sweep_ext_diag)
                for _k, _v in _diag.items():
                    feature_dict[_k] = _v
                # confluence: 진입가(=sweep된 liquidity)가 같은방향 4H FVG zone에 닿는가
                _conf = incr.fvg_tracker_4h.confluence_with(
                    liq_price=avg_e, liq_dir=_dir)
                feature_dict["is_4h_fvg_combined"] = 1.0 if _conf is not None else 0.0
                # alone: 4H FVG 안인데 결합은 아님 (단독 FVG 진입)
                feature_dict["is_4h_fvg_alone"] = (
                    1.0 if (_fvg_ctx["in_4h_fvg_aligned"] == 1.0 and _conf is None) else 0.0)
                # touch_to_entry: nearest FVG 최초접촉→진입 4H봉수. 기존 confluence
                #   (zone포함 0.13%)+ARMED후터치는 99.95% NaN → first_touch(상태무관) 대체.
                feature_dict["touch_to_entry_bars"] = _diag.get(
                    "diag_bars_since_first_touch", -1.0)
                if _conf is not None:
                    feature_dict["combined_4h_fvg_fill"] = _conf.fill_ratio
                    feature_dict["combined_4h_fvg_touches"] = float(_conf.touch_count)
                else:
                    feature_dict["combined_4h_fvg_fill"] = np.nan
                    feature_dict["combined_4h_fvg_touches"] = np.nan

                # ── 반대편 선행 sweep 정규화 (sweep 추적): signals가 원시 extreme 전달
                #    → atr 정규화 feature. opp_sweep_dist_atr = |extreme - 진입가| / 4h atr
                _opp_ext = feature_dict.get("opp_sweep_extreme", np.nan)
                if feature_dict.get("opp_sweep_present", 0.0) == 1.0 and \
                   np.isfinite(_opp_ext) and _atr4h and _atr4h > 0:
                    feature_dict["opp_sweep_dist_atr"] = round(abs(_opp_ext - avg_e) / _atr4h, 4)
                else:
                    feature_dict["opp_sweep_dist_atr"] = np.nan
                # 전달 레인지 높이 (반대 extreme ~ 셋업 extreme) / 4h atr
                _opp_rng = feature_dict.get("opp_sweep_range_raw", np.nan)
                if np.isfinite(_opp_rng) and _atr4h and _atr4h > 0:
                    feature_dict["opp_sweep_range_atr"] = round(_opp_rng / _atr4h, 4)
                else:
                    feature_dict["opp_sweep_range_atr"] = np.nan
                # is_last/unviolated는 signals에서 이미 0/1/NaN → 그대로 통과
                feature_dict.setdefault("opp_sweep_is_last", 0.0)
                feature_dict.setdefault("opp_sweep_unviolated", np.nan)
                # 원시 extreme/level_mid는 학습 feature 아님(가격 절대값 cross-symbol 불가)
                # 이나 진단/사후재구성용으로 보존 → ml_data가 diag_raw_* 로 emit.
                # 앞으로 sweep 추적 가설은 이 원시값으로 재수집 없이 사후분석 가능.
                feature_dict["opp_sweep_extreme_raw"] = feature_dict.pop("opp_sweep_extreme", np.nan)
                feature_dict.pop("opp_sweep_level_mid", None)
                feature_dict.pop("opp_sweep_range_raw", None)

                # ── 1H FVG Lifecycle 맥락 feature (1H confluence 검증)
                _atr1h = incr._atr_1h()
                _fvg_ctx_1h = incr.fvg_tracker_1h.context_at(
                    price=avg_e, direction=_dir, atr=_atr1h)
                feature_dict["in_1h_fvg"] = _fvg_ctx_1h["in_4h_fvg"]
                feature_dict["in_1h_fvg_aligned"] = _fvg_ctx_1h["in_4h_fvg_aligned"]
                feature_dict["nearest_1h_fvg_dist_atr"] = _fvg_ctx_1h["nearest_4h_fvg_dist_atr"]
                feature_dict["active_1h_fvg_fill"] = _fvg_ctx_1h["active_4h_fvg_fill"]
                feature_dict["n_active_1h_fvg"] = _fvg_ctx_1h["n_active_4h_fvg"]
                _conf1h = incr.fvg_tracker_1h.confluence_with(
                    liq_price=avg_e, liq_dir=_dir)
                feature_dict["is_1h_fvg_combined"] = 1.0 if _conf1h is not None else 0.0
                feature_dict["is_1h_fvg_alone"] = (
                    1.0 if (_fvg_ctx_1h["in_4h_fvg_aligned"] == 1.0 and _conf1h is None) else 0.0)

                # ── 4H + 1H 동시 결합 (두 시간대가 동의하는 강한 confluence)
                feature_dict["is_4h1h_fvg_combined"] = (
                    1.0 if (_conf is not None and _conf1h is not None) else 0.0)

                # ── 미회수 유동성 장부 (swing 기반 liquidity, draw-on-liquidity)
                _unswept = incr.liq_pool.unswept_counts(avg_e)
                feature_dict["unswept_above"] = _unswept["unswept_above"]
                feature_dict["unswept_below"] = _unswept["unswept_below"]
                feature_dict["unswept_imbalance"] = _unswept["unswept_imbalance"]
                # 진입 방향 정합: long인데 위에 미회수 많으면 순방향 draw
                if _dir == "long":
                    feature_dict["unswept_draw_aligned"] = _unswept["unswept_imbalance"]
                else:  # short: 아래 미회수 많을수록(-imbalance) 순방향
                    feature_dict["unswept_draw_aligned"] = -_unswept["unswept_imbalance"]

                # ── range_pos_30d / range_pos_7d: 5m 캔들 기반 고저점 위치
                _df = incr.df_5m
                if _df is not None and len(_df) > 0:
                    _close_arr = _df["close"].values.astype(float)
                    _high_arr = _df["high"].values.astype(float)
                    _low_arr = _df["low"].values.astype(float)

                    # 30일 = 8640봉 (5m × 288 × 30)
                    _n30 = min(8640, len(_df))
                    _h30 = float(_high_arr[-_n30:].max())
                    _l30 = float(_low_arr[-_n30:].min())
                    _r30 = _h30 - _l30
                    feature_dict["range_pos_30d"] = round(
                        (px - _l30) / _r30, 4) if _r30 > 0 else 0.5

                    # 7일 = 2016봉 (5m × 288 × 7)
                    _n7 = min(2016, len(_df))
                    _h7 = float(_high_arr[-_n7:].max())
                    _l7 = float(_low_arr[-_n7:].min())
                    _r7 = _h7 - _l7
                    feature_dict["range_pos_7d"] = round(
                        (px - _l7) / _r7, 4) if _r7 > 0 else 0.5
                else:
                    feature_dict["range_pos_30d"] = 0.5
                    feature_dict["range_pos_7d"] = 0.5

                # ── VP 피처 주입 (lvn_proximity — signals.py에서 접근 불가한 df_5m 필요)
                _df = incr.df_5m
                atr_now = 0.0
                if _df is not None and len(_df) >= 14:
                    c_prev = _df["close"].shift(1)
                    tr = np.maximum(_df["high"] - _df["low"],
                                    np.maximum((_df["high"] - c_prev).abs(),
                                               (_df["low"] - c_prev).abs()))
                    atr_now = float(tr.iloc[-14:].mean())
                    if np.isnan(atr_now): atr_now = 0.0

                if _df is not None and len(_df) > 50 and atr_now > 0:
                    from volume_profile import compute_vp_entry_features
                    _sel_fvg = ep.get("selected_fvg") or {}
                    _fvg_h = float(_sel_fvg.get("zone_high", avg_e))
                    _fvg_l = float(_sel_fvg.get("zone_low", avg_e))
                    _vp_feats = compute_vp_entry_features(
                        df_5m=_df,
                        current_idx=len(_df) - 1,
                        fvg_high=_fvg_h,
                        fvg_low=_fvg_l,
                        entry_price=avg_e,
                        atr=atr_now,
                    )
                    # v3: lvn_proximity는 NaN 가능 (XGBoost native handling)
                    feature_dict["lvn_proximity"] = _vp_feats.get("lvn_proximity", np.nan)
                    feature_dict["lvn_available"] = _vp_feats.get("lvn_available", 0.0)

                # ── Nasdaq 피처 주입
                if hasattr(self, 'nasdaq_provider') and self.nasdaq_provider.ready:
                    _nq_feats = self.nasdaq_provider.get_features(
                        t=pd.Timestamp(bar["time"]),
                        crypto_5m=_df if _df is not None else pd.DataFrame(),
                        atr_now=atr_now,
                        sweep_time=ep.get("trigger_time"),
                        symbol=self.symbol.split("/")[0],
                    )
                    feature_dict["nasdaq_session_divergence_pct"] = _nq_feats.get("nasdaq_session_divergence_pct", 0.0)

                # ── Market Regime 피처 주입 (adx_14, return_1h, return_4h)
                if _df is not None and len(_df) >= 50:
                    _close = _df["close"]
                    _high = _df["high"]
                    _low = _df["low"]

                    # ADX(14) — Wilder's smoothing 근사
                    _up = _high.diff()
                    _dn = -_low.diff()
                    _pos_dm = np.where((_up > _dn) & (_up > 0), _up, 0.0)
                    _neg_dm = np.where((_dn > _up) & (_dn > 0), _dn, 0.0)
                    _c_prev = _close.shift(1)
                    _tr_raw = np.maximum(_high - _low,
                                         np.maximum((_high - _c_prev).abs(),
                                                    (_low - _c_prev).abs()))
                    _alpha = 1.0 / 14
                    _pos_dm_s = pd.Series(_pos_dm, index=_high.index).ewm(alpha=_alpha, adjust=False).mean()
                    _neg_dm_s = pd.Series(_neg_dm, index=_high.index).ewm(alpha=_alpha, adjust=False).mean()
                    _tr_s = _tr_raw.ewm(alpha=_alpha, adjust=False).mean()
                    _pos_di = 100 * _pos_dm_s / (_tr_s + 1e-8)
                    _neg_di = 100 * _neg_dm_s / (_tr_s + 1e-8)
                    _dx = 100 * (_pos_di - _neg_di).abs() / (_pos_di + _neg_di + 1e-8)
                    _adx = _dx.ewm(alpha=_alpha, adjust=False).mean()
                    _adx_val = float(_adx.iloc[-1])
                    feature_dict["adx_14"] = round(_adx_val, 4) if not np.isnan(_adx_val) else 0.0

                    # return_1h (12봉), return_4h (48봉)
                    if len(_close) >= 13:
                        _r1h = float(_close.iloc[-1] / _close.iloc[-13] - 1.0)
                        feature_dict["return_1h"] = round(_r1h, 6) if not np.isnan(_r1h) else 0.0
                    else:
                        feature_dict["return_1h"] = 0.0

                    if len(_close) >= 49:
                        _r4h = float(_close.iloc[-1] / _close.iloc[-49] - 1.0)
                        feature_dict["return_4h"] = round(_r4h, 6) if not np.isnan(_r4h) else 0.0
                    else:
                        feature_dict["return_4h"] = 0.0

                # ── Spot-Futures signed 괴리 (stage7b/7c 검증판) + 방향 플래그
                feature_dict["spot_div_signed_z"] = 0.0
                feature_dict["is_long"] = 1.0 if ep["direction"] == "long" else 0.0

                # ── 청산 히트맵 피처 (Entry용, OHLCV 근사)
                #     sweep_liq_consumed: sweep이 먹은 청산 풀 강도 (셋업 진정성)
                #     liq_imbalance_hm  : 진입가 기준 위/아래 청산 풀 불균형
                feature_dict["sweep_liq_consumed"] = 0.0
                feature_dict["liq_imbalance_hm"] = 0.0
                try:
                    from liquidation_heatmap_features import compute_liq_features
                    _sweep_ext = ep.get("sweep_extreme")
                    _liq_feats = compute_liq_features(
                        df_5m=_df,
                        entry_bar_idx=i,
                        entry_price=px,
                        direction=ep["direction"],
                        sweep_extreme_price=float(_sweep_ext) if _sweep_ext else None,
                    )
                    feature_dict["sweep_liq_consumed"] = _liq_feats["sweep_liq_consumed"]
                    feature_dict["liq_imbalance_hm"] = _liq_feats["liq_imbalance_hm"]
                except Exception as _e_liq:
                    pass  # 실패 시 기본값 0 유지

                # ── Basis 피처 (basis_z, basis_momentum, spot_lead)
                #     사전계산된 시계열에서 bisect backward lookup (미래참조 차단)
                feature_dict["basis_z"] = 0.0
                feature_dict["basis_momentum"] = 0.0
                feature_dict["spot_lead"] = 0.0
                if getattr(self, "_basis_ts", None):
                    try:
                        _bt_ns = pd.Timestamp(bar["time"]).value
                        _bidx = self._bisect.bisect_right(self._basis_ts, _bt_ns) - 1
                        if 0 <= _bidx < len(self._basis_ts):
                            feature_dict["basis_z"] = round(float(self._basis_bz[_bidx]), 4)
                            feature_dict["basis_momentum"] = round(float(self._basis_bm[_bidx]), 6)
                            feature_dict["spot_lead"] = round(float(self._basis_sl[_bidx]), 6)
                    except Exception:
                        pass

                if self.spot_df is not None and _df is not None and len(_df) > 50:
                    _bar_time = pd.Timestamp(bar["time"])
                    if _bar_time.tzinfo is None:
                        _bar_time = _bar_time.tz_localize("UTC")

                    # ★ spot 종료일 이후면 spot_div 계산 스킵 (stale basis 방지)
                    #    inner join이 대부분 막지만, 의도를 명시하고 불필요 연산 제거
                    _spot_last = getattr(self, "_spot_last_time", None)
                    _spot_expired = (_spot_last is not None and _bar_time > _spot_last)

                    _sp_mask = self.spot_df["time"] <= _bar_time
                    _sp_avail = _sp_mask.sum()

                    if _sp_avail > 50 and not _spot_expired:
                        # tail 360: rolling(288) std 윈도가 sweep봉에서 꽉 차도록
                        _sp = self.spot_df[_sp_mask].tail(360)
                        _ft = _df.tail(360)

                        _sf = pd.merge(
                            _ft[["time", "close", "high", "low", "volume"]],
                            _sp[["time", "close", "high", "low", "volume"]],
                            on="time", how="inner", suffixes=("_fut", "_spot")
                        )

                        if len(_sf) > 60:
                            _direction = ep["direction"]
                            _bars_since = int(ep.get("bars_since_sweep", 10))
                            _sweep_idx = max(0, len(_sf) - 1 - _bars_since)

                            # ── signed 괴리 (bps): abs 금지 — 부호가 신호 (stage7b)
                            #    long : spot_low - fut_low   (+ = 선물 low가 더 깊음)
                            #    short: fut_high - spot_high (+ = 선물 high가 더 높음)
                            _mid_price = (_sf["close_fut"] + _sf["close_spot"]) / 2 + 1e-10
                            if _direction == "long":
                                _d_bps = (_sf["low_spot"] - _sf["low_fut"]) / _mid_price * 1e4
                            else:
                                _d_bps = (_sf["high_fut"] - _sf["high_spot"]) / _mid_price * 1e4

                            # 단일 정규화: signed 분포의 rolling std (z-of-z 금지, 7c 동일)
                            _sd_std = _d_bps.rolling(288, min_periods=60).std()
                            if _sweep_idx < len(_d_bps):
                                _dv = float(_d_bps.iloc[_sweep_idx])
                                _sv = float(_sd_std.iloc[_sweep_idx])
                                if np.isfinite(_dv) and np.isfinite(_sv) and _sv > 1e-10:
                                    feature_dict["spot_div_signed_z"] = round(
                                        max(-10.0, min(10.0, _dv / _sv)), 4)

                # ── Interaction 피처 재계산 (range_pos_7d, lvn_proximity 등이 backtest에서 갱신된 후)
                _sda = feature_dict.get("sweep_depth_atr", 0.0)
                _sds = feature_dict.get("strong_displacement_score", 0.0)
                _efs = feature_dict.get("effective_fvg_strength", 0.0)
                _bss = feature_dict.get("bars_since_sweep", 10)
                _sdn = feature_dict.get("sweep_duration_norm", 0.0)
                _lvn = feature_dict.get("lvn_proximity", np.nan)
                _rp7 = feature_dict.get("range_pos_7d", 0.5)

                feature_dict["sweep_x_disp"] = round(_sda * _sds, 4)
                feature_dict["disp_x_fvg"] = round(_sds * _efs, 4)
                feature_dict["sweep_velocity"] = round(_sda / (_bss + 1.0), 4)
                feature_dict["rangepos_x_sweep_dur"] = round(_rp7 * _sdn, 4)
                feature_dict["rangepos_x_sweep_depth"] = round(_rp7 * _sda, 4)
                # v3: lvn이 NaN이면 lvn_x_fvg도 NaN (XGBoost native)
                if isinstance(_lvn, float) and np.isnan(_lvn):
                    feature_dict["lvn_x_fvg"] = np.nan
                else:
                    feature_dict["lvn_x_fvg"] = round(_lvn * _efs, 4)
                feature_dict["killzone_x_disp"] = round(
                    feature_dict.get("is_killzone", 0.0) * _sds, 4)
                feature_dict["consistency_x_vol"] = round(
                    feature_dict.get("displacement_consistency", 0.0) * feature_dict.get("vol_ratio", 1.0), 4)

                # ── TP모델 → Entry 이주: 게이트2 검증 후 freshness만 채택.
                #    htf_4h_aligned/displacement/entry_path_vp 삭제 (코어 중복).
                #    level_freshness_score: 셋업 레벨 나이 = max(0, 1-나이h/168)
                feature_dict["level_freshness_score"] = 0.5
                _lv_fresh = ep.get("swept_level") or {}
                _ts_start = _lv_fresh.get("ts_start")
                if _ts_start is not None:
                    try:
                        _tsf = pd.Timestamp(_ts_start)
                        if _tsf.tzinfo is None:
                            _tsf = _tsf.tz_localize("UTC")
                        _ctf = pd.Timestamp(bar["time"])
                        if _ctf.tzinfo is None:
                            _ctf = _ctf.tz_localize("UTC")
                        _hours_old = (_ctf - _tsf).total_seconds() / 3600.0
                        feature_dict["level_freshness_score"] = round(
                            max(0.0, min(1.0, 1.0 - _hours_old / (24 * 7))), 4)
                    except Exception:
                        pass

                # ── ML 필터: feature_dict 기반 진입 판단
                _ml_size_adj = 1.0
                p_win = None
                _direction = ep["direction"]
                if self.ml_filter is not None:
                    p_win = self.ml_filter.predict(feature_dict, direction=_direction)
                    _ml_size_adj = self.ml_filter.get_size_adjustment(p_win)
                    if not self.ml_filter.should_enter(p_win, direction=_direction):
                        _th = self.ml_filter._thresholds.get(
                            _direction, self.ml_filter.threshold)
                        # print(f"  [{bar['time']}] [ML] ❌ 스킵 P(win)={p_win:.2f} < threshold={_th:.2f} ({_direction})")
                        strategy.reset()
                        continue
                    # print(f"  [{bar['time']}] [ML] ✅ 통과 P(win)={p_win:.2f}  size_adj={_ml_size_adj:.1f} ({_direction})")
                    _cnt.setdefault("ml_pass", 0)
                    _cnt["ml_pass"] += 1
                    _cnt.setdefault("ml_skip", 0)
                else:
                    # ML 없으면 모든 entry_plan 통과 (데이터 수집 모드)
                    _cnt.setdefault("no_ml_pass", 0)
                    _cnt["no_ml_pass"] += 1

                # ML size_adj 반영
                _final_size_mult = _ml_size_adj  # signals.py는 이제 항상 1.0을 보냄

                notional = calc_risk_capped_position_value(
                    equity_usdt=equity, position_pct=score.position_pct,
                    entry=avg_e, sl=float(ep["sl"]),
                    leverage=LEVERAGE, max_risk_frac=MAX_RISK_FRAC,
                    size_multiplier=_final_size_mult,
                    taker_fee_rate=TAKER_FEE_RATE,
                )

                ep["_notional"]  = notional
                ep["_avg_entry"] = avg_e

                # ── ★ ML학습용: TP가 없으면 1.5R 고정 TP 설정
                if ep.get("tp") is None:
                    _fallback_risk = abs(avg_e - float(ep["sl"]))
                    _fallback_tp_dist = _fallback_risk * 1.5
                    if ep["direction"] == "long":
                        ep["tp"] = avg_e + _fallback_tp_dist
                    else:
                        ep["tp"] = avg_e - _fallback_tp_dist

                ep["_reason"] = {
                    "trigger":       ep.get("reason_trigger", "?"),
                    "swept_level":   ep.get("swept_level"),
                    "sweep_extreme": ep.get("sweep_extreme"),
                    "choch_price":   ep.get("choch_price"),
                    "disp_extreme":  ep.get("disp_extreme"),
                    "ote_zone":      ep.get("ote_zone"),
                    "ob":            ep.get("selected_ob"),
                    "fvg":           ep.get("selected_fvg"),
                    "orders":        ep.get("orders"),
                    "size_mult":     _final_size_mult,
                    "score_pct":     score.position_pct,
                    "htf_state":     ep.get("htf_state", "?"),
                    "htf_aligned":   ep.get("htf_aligned", False),
                    "htf_4h":        ep.get("htf_4h", "?"),
                    "level_group":   ep.get("level_group", "?"),
                    "sweep_vol_ok":  ep.get("sweep_vol_ok", True),
                    "has_disp_fvg":  ep.get("has_disp_fvg", False),
                    "sweep_ratio":   ep.get("sweep_ratio", 0.0),
                    "_notional":     ep.get("_notional", 0.0),
                    "pd_pos":        None,  # pd_pos 삭제됨 → range_pos_30d/7d로 대체
                    "sweep_depth_score": feature_dict.get("sweep_depth_atr", 1.0),
                    "ml_p_win":      p_win,
                    "ml_size_adj":   _ml_size_adj,
                    "feature_dict":  feature_dict,
                    "bars_since_sweep": ep.get("bars_since_sweep", 10),  # ★ timeout 계산용
                }

                # ── 동시진입 필터: 같은 레벨 / 같은 흐름 차단
                if not pos_sim.can_open(ep):
                    _cnt.setdefault("multi_pos_block", 0)
                    _cnt["multi_pos_block"] += 1
                    strategy.reset()
                    continue

                pending_plan     = ep
                plan_bar_count   = 0

                # ★ intra-bar 체결 보정: mitigation 바에서 이미 진입가를 터치했는지 확인
                _filled_now = _check_limit_fill(bar, pending_plan)
                if _filled_now is not None:
                    trade_id += 1
                    
                    _entry_atr = 0.0
                    if incr.df_5m is not None and len(incr.df_5m) >= 15:
                        _df_a = incr.df_5m.iloc[-15:]
                        _hi_a = _df_a["high"].to_numpy(dtype=float)
                        _lo_a = _df_a["low"].to_numpy(dtype=float)
                        _cl_a = _df_a["close"].to_numpy(dtype=float)
                        _trs_a = [max(_hi_a[j]-_lo_a[j], abs(_hi_a[j]-_cl_a[j-1]),
                                      abs(_lo_a[j]-_cl_a[j-1])) for j in range(1, len(_hi_a))]
                        _entry_atr = sum(_trs_a) / len(_trs_a) if _trs_a else 0.0

                    t = Trade(
                        trade_id=trade_id,
                        direction=pending_plan["direction"],
                        entry_price=float(_filled_now["price"]),
                        sl=float(pending_plan["sl"]),
                        tp=pending_plan.get("tp"),
                        size_usdt=float(pending_plan["_notional"]),
                        entry_time=bar["time"],
                        reason=pending_plan.get("_reason"),
                        entry_bar_idx=i,
                        entry_atr=_entry_atr,
                        initial_sl=float(pending_plan["sl"]),
                        shadow_enabled=_SHADOW_TRAIL,
                    )

                    # ── VP scoring 초기화
                    try:
                        from volume_profile import VPDensityProfile
                        _vp_idx = len(incr.df_5m) - 1
                        t._vp_profile = VPDensityProfile.build(
                            incr.df_5m, _vp_idx, lookback_bars=288, n_bins=50
                        )
                    except Exception:
                        t._vp_profile = None
                    t.vp_prev_close = float(_filled_now["price"])

                    # ── disp_extreme 기록 (reason에서 추출)
                    _reason = pending_plan.get("_reason", {})
                    _de = _reason.get("disp_extreme", 0.0)
                    if _de:
                        t.disp_extreme = float(_de)

                    pos_sim.open(t)
                    pending_plan   = None
                    _orders_filled_this_bar = True
                    # signals.py의 상태를 업데이트하기 위해 step_on_5m_close 한 번 더 호출은 안 하지만, 다음 봉에서 해결됨.

        # ── 미결 포지션 강제 청산
        # ── 미결 포지션 강제 청산 (전체)
        if pos_sim.n_open > 0:
            for closed in pos_sim.force_close_all(df_all.iloc[-1]):
                closed.equity_after = equity + closed.pnl_usdt
                equity = closed.equity_after
                equity_curve.append(equity)
                trades.append(closed)
                _print_trade(closed, equity, df_5m=incr.df_5m)

        # ── 진단 카운터 출력
        print(f"\n{'─'*65}")
        print(f"  🔍 진단 카운터")
        print(f"  STANDBY timeout  : {_cnt['standby_timeout']:>6}회")
        print(f"  ARMED timeout    : {_cnt['armed_timeout']:>6}회")
        print(f"  ARMED long 진입  : {_cnt['armed_long']:>6}회")
        print(f"  ARMED short 진입 : {_cnt['armed_short']:>6}회")
        print(f"  LIQ 레벨 삭제    : {_cnt.get('liq_disabled', 0):>6}회  (STANDBY timeout)")

        # 상태 진단
        _states_seen = _cnt.get("states_seen", set())
        print(f"  감지된 상태      : {sorted(_states_seen)}")
        _state_log = _cnt.get("state_log", {})
        if _state_log:
            print(f"  상태 샘플링 (500봉마다):")
            for s, cnt in sorted(_state_log.items(), key=lambda x: -x[1]):
                print(f"    {s:20s}: {cnt}회")

        # ML 모드 카운터
        # ML 모드 카운터
        _total_entries = _cnt.get('ml_pass', 0) + _cnt.get('no_ml_pass', 0)
        print(f"  총 entry_plan    : {_total_entries:>6}회")
        if self.ml_filter is not None:
            _ml_skip = _cnt.get('ml_skip', 0)
            print(f"  ML 필터 통과     : {_cnt.get('ml_pass', 0):>6}회  (threshold={self.ml_filter.threshold:.2f})")
            print(f"  ML 필터 스킵     : {_ml_skip:>6}회")
        else:
            print(f"  ML 미사용 통과   : {_cnt.get('no_ml_pass', 0):>6}회  (데이터 수집 모드)")
        _multi_block = _cnt.get('multi_pos_block', 0)
        if _multi_block > 0:
            print(f"  동시진입 차단    : {_multi_block:>6}회")
        print()

        result_obj = BacktestResult(
            trades=trades, equity_curve=equity_curve,
            final_equity=equity, initial_equity=self.initial_equity,
        )
        _print_final_report(result_obj, self.symbol, self.months)

        # ── ML 학습 데이터 자동 저장
        try:
            from ml_data import extract_ml_dataset
            _sym = self.symbol.split("/")[0]  # "BTC/USDT:USDT" → "BTC"

            # v3: timeout label threshold 환경변수 (기본 +0.5R / -0.3R)
            #     실험: ML_TIMEOUT_WIN=0.3 / 0.5 / 0.7 으로 바꿔가며 비교
            _tw = float(os.environ.get("ML_TIMEOUT_WIN", "0.5"))
            _tl = float(os.environ.get("ML_TIMEOUT_LOSS", "-0.3"))
            _drop = os.environ.get("ML_DROP_DEAD_ZONE", "0") in ("1", "true", "True")

            ml_df = extract_ml_dataset(
                trades, symbol=_sym,
                timeout_win_threshold=_tw,
                timeout_loss_threshold=_tl,
                drop_dead_zone=_drop,
            )
            # ── 파일명 suffix 자동 결정
            # SHADOW_TRAIL=1 → _shadow (shadow 비교용, 기존 학습 데이터 보존)
            # TRAIL_MODE=on  → _traon  (trail on 단독 수집)
            # 둘 다 아니면   → 없음    (기존 호환)
            if _SHADOW_TRAIL:
                _suffix = "_shadow"
            elif _TRAIL_MODE == "on":
                _suffix = "_traon"
            else:
                _suffix = ""
            _ml_path = os.path.join(_collect_out_dir(), f"ml_train_{_sym.lower()}{_suffix}.csv")
            ml_df.to_csv(_ml_path, index=False)
            print(f"\n  📦 ML 학습 데이터 저장: {_ml_path}  ({len(ml_df)}건)  "
                  f"[shadow={_SHADOW_TRAIL}, trail_mode={_TRAIL_MODE}]")
        except Exception as e:
            print(f"\n  ⚠️ ML 데이터 추출 실패: {e}")

        # ── Runner 학습 데이터 자동 저장 (TP 폐기 → runner로 대체, 방향 A)
        #    run() 로컬 자원으로 15m swing 평가 + RSI/div 직접 생성.
        #    df_5m=df_all (entry_bar_idx·pivots_5m idx가 df_all 글로벌 positional),
        #    pivots_5m/pivots_15m/df_15m = incr.* (엔진 보유). Backtester엔 이 속성 없음.
        try:
            _sym = self.symbol.split("/")[0]
            runner_df = extract_runner_training_data(
                trades, df_all, incr.pivots_5m, incr.pivots_15m,
                incr.df_15m, _sym)
            if len(runner_df) > 0:
                if _SHADOW_TRAIL:
                    _suffix = "_shadow"
                elif _TRAIL_MODE == "on":
                    _suffix = "_traon"
                else:
                    _suffix = ""
                _runner_path = os.path.join(_collect_out_dir(), f"runner_train_{_sym.lower()}{_suffix}.csv")
                runner_df.to_csv(_runner_path, index=False)
                _n_tr = runner_df["trade_id"].nunique()
                print(f"  📦 Runner 학습 데이터 저장: {_runner_path}  "
                      f"({len(runner_df)}행, {_n_tr}건 트레이드, "
                      f"평균 {len(runner_df)/max(_n_tr,1):.1f}샘플)")
            else:
                # 무음 탈락 방지: 0행이면 원인 단서를 무조건 출력.
                _d5 = "None" if df_all is None else len(df_all)
                _d15 = "None" if incr.df_15m is None else len(incr.df_15m)
                _p15 = "None/empty" if (incr.pivots_15m is None or incr.pivots_15m.empty) \
                    else len(incr.pivots_15m)
                print(f"  ⚠️ Runner 데이터 0행 — 저장 안 함. "
                      f"df_all={_d5}, incr.df_15m={_d15}, incr.pivots_15m={_p15}, trades={len(trades)}")
        except Exception as e:
            import traceback
            print(f"\n  ⚠️ Runner 데이터 추출 실패: {e}")
            traceback.print_exc()

        return result_obj