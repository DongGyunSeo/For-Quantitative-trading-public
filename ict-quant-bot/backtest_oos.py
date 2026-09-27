# backtest.py ML_data
"""
backtest.py — ICT 전략 백테스터 (OOS 실전 세팅)

★ OOS 세팅 (실전 동일):
  - TRAIL_ACTIVATE_R:   2.0  (2R 도달 시 트레일링 시작)
  - TRAIL_ACTIVATE_R_A: 1.5  (Group A: 1.5R)
  - TRAIL_BREAKEVEN_R:  1.0  (1R 도달 시 본절 SL)
  - BREAKEVEN_R:        1.0  (본절 SL)
  - timeout: 고정 60봉 (5시간)
  - TP fallback: 없음 (TP 없으면 trail_only)

★ Swing 변경:
  - find_swings → find_swings_enhanced (TF별 dynamic lookback)
  - 5m/15m: lb_low=2, lb_high=4 (평균 lb≈3)
  - 1h/4h:  lb_low=3, lb_high=5 (평균 lb≈4)

★ Liquidity 변경:
  - NY H/L 삭제, IDM 추가
  - liq_pool.tick()에 df_15m_struct 전달

변경사항 (원본):
  - FVG_LOOKBACK_BARS 상수 누락 수정
  - incremental update 적용 (engine.py 동일 방식)
  - 거래 빈도 완화
"""

from __future__ import annotations

import logging
import os
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

    def __post_init__(self):
        if self.sl_history is None:
            self.sl_history = []


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

CACHE_DIR = "cache_5m_oos"  # OOS 백테스트 전용 캐시 (collect_ml_data와 분리)
                            # collect_ml_data.py는 backtest.py 사용 → cache_5m/
                            # oos_backtest.py는 backtest_oos.py 사용 → cache_5m_oos/
                            # 두 워크플로우 번갈아 실행해도 캐시 충돌/재다운로드 없음


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
            # London H/L: 완전 제거 확정 (생성 자체 차단). liquidity.py 동기화.
            # (London 워밍업 생성 삭제)
            # EQL/EQH 초기 계산
            self.liq_pool._rebuild_eq_levels(self.pivots_15m, today)

        self._last_15m_time = df_15m["time"].iloc[-1] if len(df_15m) else None
        self._last_1h_time  = df_1h["time"].iloc[-1]  if len(df_1h)  else None
        self._last_4h_time  = df_4h["time"].iloc[-1]  if len(df_4h)  else None

        # ── 4H FVG Lifecycle Tracker (confluence 맥락용, backtest.py와 동일)
        self.fvg_tracker_4h = FVGLifecycleTracker(
            tf="4h",
            n_depart=int(os.getenv("FVG4H_N_DEPART", "5")),
            n_depart_arm=int(os.getenv("FVG4H_N_DEPART_ARM", "4")),
            timeout_bars=int(os.getenv("FVG4H_TIMEOUT", "360")),
            fill_dead=float(os.getenv("FVG4H_FILL_DEAD", "0.5")),
            min_size_atr=float(os.getenv("FVG4H_MIN_SIZE_ATR", "0.3")),
            max_slots=int(os.getenv("FVG4H_MAX_SLOTS", "20")),
        )
        self._fvg4h_known_idx = 0
        self._prime_fvg_tracker_4h()

    def _atr_4h(self) -> float:
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

        # 4h — structure_state 업데이트
        if len(new_4h) > 0:
            latest_4h = new_4h["time"].iloc[-1]
            if not hasattr(self, "_last_4h_time") or self._last_4h_time is None or latest_4h > self._last_4h_time:
                self._last_4h_time = latest_4h
                self.df_4h = new_4h
                self._update_4h_state()
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
        df = self.df_4h
        if df is None or len(df) < 3:
            return
        cur_idx = len(df) - 1
        atr = self._atr_4h()
        if atr <= 0:
            return
        all_fvgs = detect_fvgs_lookback(df, lookback_bars=len(df),
                                        fill_mode="wick", remove_filled=False,
                                        min_size=0.0)
        if len(all_fvgs):
            new_fvgs = all_fvgs[all_fvgs["created_idx"] >= self._fvg4h_known_idx]
            if len(new_fvgs):
                self.fvg_tracker_4h.add_new_fvgs(new_fvgs, atr=atr, cur_idx=cur_idx)
        self._fvg4h_known_idx = len(df)
        row = df.iloc[-1]
        self.fvg_tracker_4h.update(cur_idx, bar_low=float(row["low"]),
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


TRAIL_ACTIVATE_R       = 2.0      # OOS: 트레일링 활성화 (2R 도달 시)
TRAIL_ACTIVATE_R_A     = 1.5      # OOS: Group A 트레일링 (1.5R 도달 시)
TRAIL_SWING_BACK_A     = 3     # Group A: 스윙 3개 기준
TRAIL_BREAKEVEN_R = 1.0            # OOS: 본절 SL 활성화 (1R 도달 시)
TRAIL_FALLBACK_PCT = 0.008 # 스윙 없을 때 최고점 대비 0.8% 되돌림
TRAIL_SWING_LB    = 3     # 스윙 감지 lookback (봉 수)


TRAIL_SWING_BACK = 4   # Group B: 4개 스윙 기준 (3→4, 더 여유)


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

    def _check_one(self, t: Trade, bar: pd.Series,
                   pivots_5m, liq_levels, structure_state_5m,
                   current_bar_idx, atr_5m) -> Optional[Trade]:
        """단일 포지션 TP/SL/timeout/trailing 체크. 기존 check_bar 로직 그대로."""
        h, l = float(bar["high"]), float(bar["low"])

        # ── OOS: 고정 timeout 60봉 (5시간)
        bars_held = current_bar_idx - t.entry_bar_idx
        timeout_bars = 60

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
        best_delta = (best_px - t.entry_price) / t.entry_price
        if t.direction == "short":
            best_delta = -best_delta
        best_pnl = t.size_usdt * best_delta - t.size_usdt * TAKER_FEE_RATE * 2
        if best_pnl > t.max_unrealized_pnl:
            t.max_unrealized_pnl = best_pnl

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

            BREAKEVEN_R = 1.0          # OOS: 본절 활성화 (1R 도달 시)
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
        delta = (exit_px - t.entry_price) / t.entry_price
        if t.direction == "short":
            delta = -delta
        t.pnl_usdt = t.size_usdt * delta - t.size_usdt * TAKER_FEE_RATE * 2
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
        tp_model_path: str    = None,  # TP 선택 모델 경로 (None이면 기존 로직)
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

        # ── TP 선택 모델 초기화
        self.tp_selector = None
        if tp_model_path:
            try:
                from tp_selector import TPSelector
                self.tp_selector = TPSelector(tp_model_path)
                print(f"[TP] ✅ TPSelector 로드 완료: {tp_model_path}")
            except Exception as e:
                print(f"[TP] ⚠️ TPSelector 로드 실패: {e}  (기존 TP 로직 사용)")
                import traceback
                traceback.print_exc()
        else:
            print(f"[TP] ℹ️  tp_model_path 미지정 — tp_selector 없이 진행")

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
                    )
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
                _cnt.setdefault("ep_entered_block", 0)
                _cnt["ep_entered_block"] += 1
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
                _conf = incr.fvg_tracker_4h.confluence_with(
                    liq_price=avg_e, liq_dir=_dir, atr=_atr4h, max_dist_atr=1.0)
                feature_dict["is_4h_fvg_combined"] = 1.0 if _conf is not None else 0.0
                feature_dict["is_4h_fvg_alone"] = (
                    1.0 if (_fvg_ctx["in_4h_fvg_aligned"] == 1.0 and _conf is None) else 0.0)
                if _conf is not None:
                    feature_dict["combined_4h_fvg_fill"] = _conf.fill_ratio
                    feature_dict["combined_4h_fvg_touches"] = float(_conf.touch_count)
                    _cur4h_idx = len(incr.df_4h) - 1 if incr.df_4h is not None else -1
                    if _conf.touched_idx >= 0 and _cur4h_idx >= 0:
                        feature_dict["touch_to_entry_bars"] = float(_cur4h_idx - _conf.touched_idx)
                    else:
                        feature_dict["touch_to_entry_bars"] = np.nan
                else:
                    feature_dict["combined_4h_fvg_fill"] = np.nan
                    feature_dict["combined_4h_fvg_touches"] = np.nan
                    feature_dict["touch_to_entry_bars"] = np.nan

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
                    # lvn_proximity만 업데이트 (vp_confluence 제거됨)
                    feature_dict["lvn_proximity"] = _vp_feats.get("lvn_proximity", 3.0)

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

                # ── Spot-Futures 괴리 피처 (v2: HTF 누적 괴리 + 기울기)
                feature_dict["return_div_1h"] = 0.0
                feature_dict["return_div_4h"] = 0.0
                feature_dict["return_div_1h_slope"] = 0.0
                feature_dict["return_div_4h_slope"] = 0.0
                feature_dict["basis_z"] = 0.0
                feature_dict["vol_div_1h"] = 0.0
                feature_dict["vol_div_1h_slope"] = 0.0

                if self.spot_df is not None and _df is not None and len(_df) > 50:
                    _bar_time = pd.Timestamp(bar["time"])
                    if _bar_time.tzinfo is None:
                        _bar_time = _bar_time.tz_localize("UTC")

                    _sp_mask = self.spot_df["time"] <= _bar_time
                    _sp_avail = _sp_mask.sum()

                    if _sp_avail > 50:
                        _sp = self.spot_df[_sp_mask].tail(800)
                        _ft = _df.tail(800)

                        _sf = pd.merge(
                            _ft[["time", "close", "high", "low", "volume"]],
                            _sp[["time", "close", "high", "low", "volume"]],
                            on="time", how="inner", suffixes=("_fut", "_spot")
                        )

                        if len(_sf) > 50:
                            # ── 1. Return Divergence 1h (12봉 누적 수익률 괴리)
                            _ret_f_1h = _sf["close_fut"].pct_change(12).fillna(0)
                            _ret_s_1h = _sf["close_spot"].pct_change(12).fillna(0)
                            _rd_1h = _ret_f_1h - _ret_s_1h
                            _v = float(_rd_1h.iloc[-1])
                            if np.isfinite(_v):
                                feature_dict["return_div_1h"] = round(max(-0.05, min(0.05, _v)), 6)

                            # 1h 기울기: 현재 - 12봉 전
                            if len(_rd_1h) >= 24:
                                _slope_1h = float(_rd_1h.iloc[-1] - _rd_1h.iloc[-13])
                                if np.isfinite(_slope_1h):
                                    feature_dict["return_div_1h_slope"] = round(max(-0.05, min(0.05, _slope_1h)), 6)

                            # ── 2. Return Divergence 4h (48봉 누적 수익률 괴리)
                            if len(_sf) >= 96:
                                _ret_f_4h = _sf["close_fut"].pct_change(48).fillna(0)
                                _ret_s_4h = _sf["close_spot"].pct_change(48).fillna(0)
                                _rd_4h = _ret_f_4h - _ret_s_4h
                                _v4 = float(_rd_4h.iloc[-1])
                                if np.isfinite(_v4):
                                    feature_dict["return_div_4h"] = round(max(-0.10, min(0.10, _v4)), 6)

                                # 4h 기울기: 현재 - 12봉 전
                                if len(_rd_4h) >= 60:
                                    _slope_4h = float(_rd_4h.iloc[-1] - _rd_4h.iloc[-13])
                                    if np.isfinite(_slope_4h):
                                        feature_dict["return_div_4h_slope"] = round(max(-0.10, min(0.10, _slope_4h)), 6)

                            # ── 3. Volume Divergence 1h (VWAP 기반)
                            if len(_sf) >= 50:
                                # 1h VWAP 괴리: 12봉 평균(거래량가중가격) 차이
                                _vwap_f = (_sf["close_fut"] * _sf["volume_fut"]).rolling(12, min_periods=5).sum() / \
                                          (_sf["volume_fut"].rolling(12, min_periods=5).sum() + 1e-10)
                                _vwap_s = (_sf["close_spot"] * _sf["volume_spot"]).rolling(12, min_periods=5).sum() / \
                                          (_sf["volume_spot"].rolling(12, min_periods=5).sum() + 1e-10)
                                _vwap_div = ((_vwap_f - _vwap_s) / (_vwap_s + 1e-10) * 100).fillna(0)
                                _vd_v = float(_vwap_div.iloc[-1])
                                if np.isfinite(_vd_v):
                                    feature_dict["vol_div_1h"] = round(max(-1.0, min(1.0, _vd_v)), 4)

                                # 기울기: 현재 - 12봉 전
                                if len(_vwap_div) >= 24:
                                    _vd_slope = float(_vwap_div.iloc[-1] - _vwap_div.iloc[-13])
                                    if np.isfinite(_vd_slope):
                                        feature_dict["vol_div_1h_slope"] = round(max(-1.0, min(1.0, _vd_slope)), 4)

                            # ── 4. Basis Z-Score (720봉 = 2.5일 윈도우)
                            _basis = (_sf["close_fut"] - _sf["close_spot"]) / (_sf["close_spot"] + 1e-10) * 100
                            _basis = _basis.clip(-1, 1)
                            _bz_window = min(720, len(_basis) // 2)
                            if _bz_window >= 50:
                                _bm = _basis.rolling(_bz_window, min_periods=50).mean()
                                _bs_std = _basis.rolling(_bz_window, min_periods=50).std()
                                _bz = (_basis - _bm) / (_bs_std + 1e-10)
                                _bz_val = float(_bz.iloc[-1])
                                if np.isfinite(_bz_val):
                                    feature_dict["basis_z"] = round(max(-4, min(4, _bz_val)), 4)

                # ── Interaction 피처 재계산 (range_pos_7d, lvn_proximity 등이 backtest에서 갱신된 후)
                _sda = feature_dict.get("sweep_depth_atr", 0.0)
                _sds = feature_dict.get("strong_displacement_score", 0.0)
                _efs = feature_dict.get("effective_fvg_strength", 0.0)
                _bss = feature_dict.get("bars_since_sweep", 10)
                _sdn = feature_dict.get("sweep_duration_norm", 0.0)
                _lvn = feature_dict.get("lvn_proximity", 3.0)
                _rp7 = feature_dict.get("range_pos_7d", 0.5)

                feature_dict["sweep_x_disp"] = round(_sda * _sds, 4)
                feature_dict["disp_x_fvg"] = round(_sds * _efs, 4)
                feature_dict["sweep_velocity"] = round(_sda / (_bss + 1.0), 4)
                feature_dict["rangepos_x_sweep_dur"] = round(_rp7 * _sdn, 4)
                feature_dict["rangepos_x_sweep_depth"] = round(_rp7 * _sda, 4)
                feature_dict["lvn_x_fvg"] = round(_lvn * _efs, 4)
                feature_dict["killzone_x_disp"] = round(
                    feature_dict.get("is_killzone", 0.0) * _sds, 4)
                feature_dict["consistency_x_vol"] = round(
                    feature_dict.get("displacement_consistency", 0.0) * feature_dict.get("vol_ratio", 1.0), 4)

                # ── TP 후보 수집 (tp_selector용) — ML 필터보다 먼저 실행해야 best_tp_score 확보
                from tp_selector import collect_tp_candidates
                _tp_cands = collect_tp_candidates(
                    entry_px=avg_e,
                    direction=ep["direction"],
                    sl=float(ep["sl"]),
                    liquidity_levels=liq_now,
                    fvgs_1h=incr.get_fvgs_1h(),
                    fvgs_4h=incr.get_fvgs_4h(),
                    obs_1h=incr.get_obs_1h(),
                    obs_4h=incr.get_obs_4h(),
                    atr=atr_now if atr_now > 0 else 1.0,
                )

                # ── TP 후보별 VP 피처 계산 (학습 데이터용 — 항상 실행)
                _vp_tp_list = []
                if _tp_cands and _df is not None and len(_df) > 50 and atr_now > 0:
                    from volume_profile import compute_vp_tp_features
                    _sds_for_tp = feature_dict.get("strong_displacement_score", 0.0)
                    for _tc in _tp_cands:
                        _vp_tp = compute_vp_tp_features(
                            df_5m=_df,
                            current_idx=len(_df) - 1,
                            candidate_price=float(_tc["price"]),
                            entry_price=avg_e,
                            atr=atr_now,
                            strong_displacement_score=_sds_for_tp,
                        )
                        _vp_tp_list.append(_vp_tp)

                # ── TPSelector override (모델이 있으면)
                # 🔧 best_tp_score의 의미: prob_reach (도달 확률, [0,1])
                #     prepare_entry_data.py가 만든 best_tp_score(=predict_proba)와 일치시킴.
                #     EV 값은 별도로 ep["tp"] 결정에만 사용 (min_ev=0.15 기준).
                _best_tp_score = 0.0
                if getattr(self, "tp_selector", None) is not None and _tp_cands:
                    _tp_ctx = {
                        "net_move_atr_ratio": feature_dict.get("net_move_atr_ratio", 0.0),
                        "nasdaq_divergence_sweep_pct_atr": 0.0,
                        "hour": pd.Timestamp(bar["time"]).hour,
                        "htf_4h_aligned": feature_dict.get("htf_4h_aligned", 0),
                        "all_liquidity_levels": liq_now,
                        "vp_tp_features": _vp_tp_list,
                    }
                    _scored = self.tp_selector.score_candidates(
                        avg_e, ep["direction"], float(ep["sl"]),
                        _tp_cands, atr_now, _tp_ctx,
                        current_time=pd.Timestamp(bar["time"]),
                    )
                    if _scored:
                        # ⚠️ 중요: best_tp_score = max(prob_reach_raw), NOT calibrated.
                        # prepare_entry_data.py가 predict_proba(=raw)로 best_tp_score를 만들기 때문에
                        # OOS도 raw로 측정해야 train CSV의 threshold(0.5)와 동일 의미.
                        # calibration은 EV 계산(TP 가격 결정)에만 적용됨.
                        _best_tp_score = max(s.get("prob_reach_raw", s.get("prob_reach", 0.0))
                                             for s in _scored)
                        _tp_best = _scored[0]  # EV 내림차순 정렬됨 (TP 가격 결정용)
                        if _tp_best.get("ev", 0) >= 0.15:
                            _tp_raw = float(_tp_best["price"])
                            _tp_buf = atr_now * 0.15 if atr_now > 0 else 0.0
                            _tp_risk = abs(avg_e - float(ep["sl"]))
                            _min_tp_dist = _tp_risk * 1.0
                            if ep["direction"] == "long":
                                _tp_final = _tp_raw - _tp_buf
                                _tp_floor = avg_e + _min_tp_dist
                                ep["tp"] = max(_tp_final, _tp_floor)
                            else:
                                _tp_final = _tp_raw + _tp_buf
                                _tp_ceil = avg_e - _min_tp_dist
                                ep["tp"] = min(_tp_final, _tp_ceil)
                elif _tp_cands:
                    # TPSelector 없어도 후보 수 기반으로 대략적 score 산출
                    _risk = abs(avg_e - float(ep["sl"]))
                    if _risk > 0:
                        _rrs = [abs(float(c["price"]) - avg_e) / _risk for c in _tp_cands]
                        _reachable = [rr for rr in _rrs if 1.5 <= rr <= 5.0]
                        _best_tp_score = min(len(_reachable) / 3.0, 1.0)

                feature_dict["best_tp_score"] = round(_best_tp_score, 4)

                # ── TP모델 → Entry 이주: freshness만 채택 (backtest.py 동기화).
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


                _cnt.setdefault("ep_reached_tp_block", 0)
                _cnt["ep_reached_tp_block"] += 1

                # ── TP prob threshold 필터 (환경변수 제어)
                # 환경변수 TP_PROB_THRESHOLD: best_tp_score(=max prob_reach, [0,1]) 기준 필터.
                # 0이면 미적용. 예: TP_PROB_THRESHOLD=0.5 → 도달 확률 >=0.5인 setup만 진입.
                #
                # 의미 통일:
                #   - prepare_entry_data.py가 만든 ml_train_with_tp.csv의 best_tp_score는
                #     predict_proba 결과(=확률, [0,1])이고,
                #   - backtest_oos.py:2092에서도 max(prob_reach)를 best_tp_score로 사용함.
                #   - 따라서 이 환경변수도 같은 확률 스케일로 해석됨.
                #
                # 하위 호환: TP_SCORE_THRESHOLD도 받지만 deprecation 경고
                _tp_thr_str = os.environ.get("TP_PROB_THRESHOLD")
                if _tp_thr_str is None:
                    _tp_thr_str = os.environ.get("TP_SCORE_THRESHOLD", "0.0")
                    if _tp_thr_str != "0.0":
                        # 한 번만 경고 (클래스 변수로 플래그)
                        if not getattr(self.__class__, "_tp_env_warned", False):
                            print(f"  ⚠️ TP_SCORE_THRESHOLD는 deprecated. TP_PROB_THRESHOLD 사용 권장 "
                                  f"(같은 의미: prob 기준 [0,1])")
                            self.__class__._tp_env_warned = True
                _tp_thr = float(_tp_thr_str)

                # TP_DEBUG=1이면 매 trade 시도마다 판단 결과 한 줄 출력
                _tp_debug = os.environ.get("TP_DEBUG", "0") in ("1", "true", "True")

                if _tp_thr > 0:
                    # 디버그 카운터: tp_selector 작동 여부 추적
                    _cnt.setdefault("tp_thr_pass", 0)
                    _cnt.setdefault("tp_thr_skip", 0)
                    _cnt.setdefault("tp_no_selector", 0)
                    _cnt.setdefault("tp_no_cands", 0)
                    _cnt.setdefault("tp_no_scored", 0)

                    # 진단: best_tp_score가 0인 경우 그 원인 분류
                    _reason_zero = ""
                    if _best_tp_score == 0.0:
                        if getattr(self, "tp_selector", None) is None:
                            _cnt["tp_no_selector"] += 1
                            _reason_zero = " [no_selector]"
                        elif not _tp_cands:
                            _cnt["tp_no_cands"] += 1
                            _reason_zero = " [no_cands]"
                        else:
                            _cnt["tp_no_scored"] += 1
                            _reason_zero = " [no_scored]"

                    _passed = _best_tp_score >= _tp_thr

                    if _tp_debug:
                        # 심볼 추출 (BTC/USDT:USDT → BTC)
                        try:
                            _sym_short = self.symbol.split("/")[0]
                            _t = bar.get("time", "?")
                            _t_str = str(_t)[:19] if _t != "?" else "?"
                            _dir_short = "L" if ep["direction"] == "long" else "S"
                            _mark = "PASS" if _passed else "SKIP"
                            _op = ">=" if _passed else "< "
                            _n_cands = len(_tp_cands) if _tp_cands else 0
                            _line = (f"[TP-DBG] {_sym_short:<5} {_t_str}  "
                                     f"{_dir_short}  thr={_tp_thr:.2f} {_op} "
                                     f"score={_best_tp_score:.3f}  "
                                     f"cands={_n_cands:>2}  {_mark}{_reason_zero}")
                            # 양쪽 채널로 출력 (Windows 워커 stdout 불안정 대비)
                            print("  " + _line, flush=True)
                            import sys as _sys
                            _sys.stderr.write(_line + "\n")
                            _sys.stderr.flush()
                        except Exception as _e:
                            import sys as _sys
                            _sys.stderr.write(f"[TP-DBG] print 실패: {_e}\n")
                            _sys.stderr.flush()

                    if not _passed:
                        _cnt["tp_thr_skip"] += 1
                        strategy.reset()
                        continue
                    _cnt["tp_thr_pass"] += 1

                # ── ML 필터: feature_dict 기반 진입 판단 (best_tp_score 확보 후)
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

                # OOS: TP fallback 없음 — TP 없으면 trail_only로 운영

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
                    "tp_candidates": _tp_cands,  # TP 학습 데이터용
                    "vp_tp_features": _vp_tp_list,
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
                    )
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

        # ── 핵심 진단: enter_plan → TP 블록 도달 흐름 (어디서 막히는지)
        _ep_block = _cnt.get('ep_entered_block', 0)
        _ep_reached_tp = _cnt.get('ep_reached_tp_block', 0)
        _gap = _ep_block - _ep_reached_tp
        print(f"  ─── ENTRY_PLAN 흐름 진단 ───")
        print(f"  enter_plan 블록 진입       : {_ep_block:>6}회  "
              f"(action==enter_plan & pending=None)")
        print(f"  TP 블록 도달               : {_ep_reached_tp:>6}회  "
              f"(feature_dict 빌드 완료)")
        if _gap > 0:
            print(f"  ⚠️  중간 손실                : {_gap:>6}회  "
                  f"(enter_plan 후 TP 블록 전에 빠짐)")
        elif _ep_block == 0:
            print(f"  ⚠️  enter_plan 자체가 0회 — signals.py 단에서 신호 없음")
        else:
            print(f"  ✅ enter_plan → TP 블록 100% 도달")

        # ── TP prob threshold 필터 카운터 (디버그용)
        _tp_thr_envstr = os.environ.get("TP_PROB_THRESHOLD") or os.environ.get("TP_SCORE_THRESHOLD", "0.0")
        _tp_thr_now = float(_tp_thr_envstr)
        if _tp_thr_now > 0:
            _tp_pass = _cnt.get('tp_thr_pass', 0)
            _tp_skip = _cnt.get('tp_thr_skip', 0)
            _tp_total = _tp_pass + _tp_skip
            _tp_pass_pct = 100 * _tp_pass / _tp_total if _tp_total > 0 else 0
            print(f"  ─── TP prob threshold = {_tp_thr_now:.3f} ───")
            print(f"  TP 필터 통과     : {_tp_pass:>6}회  ({_tp_pass_pct:.1f}%)")
            print(f"  TP 필터 스킵     : {_tp_skip:>6}회")

            # best_tp_score=0의 원인 분류
            _no_sel = _cnt.get('tp_no_selector', 0)
            _no_cand = _cnt.get('tp_no_cands', 0)
            _no_scored = _cnt.get('tp_no_scored', 0)
            if _no_sel + _no_cand + _no_scored > 0:
                print(f"  └ best_tp_score=0 원인 분류:")
                if _no_sel > 0:
                    print(f"      tp_selector 미로드  : {_no_sel:>6}회  ⚠️ "
                          f"TP 모델 파일 확인 필요")
                if _no_cand > 0:
                    print(f"      TP 후보 0개         : {_no_cand:>6}회  "
                          f"(collect_tp_candidates 실패)")
                if _no_scored > 0:
                    print(f"      score_candidates 빈결과: {_no_scored:>4}회  "
                          f"(피처 계산 또는 모델 추론 실패)")
        print()

        result_obj = BacktestResult(
            trades=trades, equity_curve=equity_curve,
            final_equity=equity, initial_equity=self.initial_equity,
        )
        _print_final_report(result_obj, self.symbol, self.months)

        # ── ML 학습 데이터 자동 저장
        try:
            from ml_data import extract_ml_dataset
            import os as _os
            _sym = self.symbol.split("/")[0]  # "BTC/USDT:USDT" → "BTC"

            # v3: timeout label threshold 환경변수 (collect 단계와 동일하게 유지)
            _tw = float(_os.environ.get("ML_TIMEOUT_WIN", "0.5"))
            _tl = float(_os.environ.get("ML_TIMEOUT_LOSS", "-0.3"))
            _drop = _os.environ.get("ML_DROP_DEAD_ZONE", "0") in ("1", "true", "True")

            ml_df = extract_ml_dataset(
                trades, symbol=_sym,
                timeout_win_threshold=_tw,
                timeout_loss_threshold=_tl,
                drop_dead_zone=_drop,
            )
            # ── OOS 전용 디렉토리에 저장 (collect_ml_data와 충돌 방지)
            #     디렉토리: oos_results/{timestamp_session}/
            #     파일명: ml_train_{symbol}.csv (디렉토리로 충돌 방지)
            _oos_dir = os.environ.get("OOS_OUTPUT_DIR")
            if not _oos_dir:
                _oos_dir = os.path.join("oos_results",
                                        datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(_oos_dir, exist_ok=True)
            _ml_path = os.path.join(_oos_dir, f"ml_train_{_sym.lower()}.csv")
            ml_df.to_csv(_ml_path, index=False)
            print(f"\n  📦 ML 학습 데이터 저장: {_ml_path}  ({len(ml_df)}건)")
        except Exception as e:
            print(f"\n  ⚠️ ML 데이터 추출 실패: {e}")

        # ── TP 학습 데이터: 폐기 (runner로 대체). oos는 라이브 검증 전용이므로
        #    runner 데이터는 in-sample backtest.py에서 생성. 여기선 미생성.

        return result_obj