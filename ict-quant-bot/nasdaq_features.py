"""
nasdaq_features.py — 나스닥-크립토 괴리 피처 계산 모듈
═══════════════════════════════════════════════════════
BTC, ETH, SOL 전용. 나머지 심볼은 전부 0으로 마스킹.

피처 목록:
  nasdaq_session_divergence_pct  : 나스닥 전일 종가 → 현재 % 괴리
  nasdaq_divergence_sweep_pct_atr: sweep→현재 나스닥 % 괴리 / ATR
  nasdaq_outperformance_flag     : 나스닥이 코인보다 강하게 움직인 정도 (연속값)
  btc_nasdaq_corr_24h            : 최근 24h 피어슨 상관계수 (5m 수익률)
  is_ny_am_session               : NY AM 세션 (09:30~12:00 EST)
  is_ny_pm_session               : NY PM 세션 (13:30~16:00 EST)
  nasdaq_volatility_spike        : 현재 변동성 / 1주일 평균 변동성

사용법:
  nq = NasdaqFeatureProvider(years=1.5)   # 초기화 시 다운로드
  feats = nq.get_features(
      t=pd.Timestamp("2025-01-15 14:30", tz="UTC"),
      crypto_5m=df_5m,       # 코인 5분봉 (time, close 필요)
      atr_now=500.0,         # 현재 코인 ATR
      sweep_time=sweep_ts,   # sweep 발생 시각 (Optional)
  )
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings

# 나스닥 피처 대상 심볼
NASDAQ_ENABLED_SYMBOLS = {"BTC", "ETH", "SOL"}

# 나스닥 피처 키 목록 (ml_data.py, signals.py에서 참조)
NASDAQ_FEATURE_KEYS = [
    "nasdaq_session_divergence_pct",
    "nasdaq_divergence_sweep_pct_atr",
    "nasdaq_outperformance_flag",
    "btc_nasdaq_corr_24h",
    "is_ny_am_session",
    "is_ny_pm_session",
    "nasdaq_volatility_spike",
]


def _zero_features() -> dict:
    """마스킹용 전부-0 피처."""
    return {k: 0.0 for k in NASDAQ_FEATURE_KEYS}


def download_nasdaq_5m(ticker: str = "NQ=F", years: float = 1.5) -> pd.DataFrame:
    """
    yfinance에서 나스닥 선물 5분봉을 60일 단위로 나눠서 다운로드.
    yfinance 5분봉은 최대 60일만 제공됨 (Yahoo Finance API 제한).
    반환: columns=[time, open, high, low, close, volume], tz=UTC
    """
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    
    # yfinance 5분봉 제공 한도: 최대 60일
    limit_start = end - timedelta(days=59)
    if start < limit_start:
        start = limit_start
    all_data = []
    current_end = end

    while current_end > start:
        current_start = max(current_end - timedelta(days=59), start)
        try:
            df = yf.download(
                tickers=ticker,
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval="5m",
                progress=False,
            )
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            warnings.warn(f"[nasdaq] 5m 다운로드 실패 {current_start}~{current_end}: {e}")
        current_end = current_start - timedelta(days=1)

    if not all_data:
        warnings.warn("[nasdaq] 5분봉 데이터 없음 — 빈 DataFrame 반환")
        return pd.DataFrame()

    full = pd.concat(all_data).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    # MultiIndex columns 처리 (yfinance 최신 버전 대응)
    if isinstance(full.columns, pd.MultiIndex):
        full.columns = full.columns.get_level_values(0)

    # 정규화
    full = full.reset_index()
    rename_map = {}
    for c in full.columns:
        cl = str(c).lower().strip()
        if cl in ("datetime", "date", "index"):
            rename_map[c] = "time"
        elif cl == "open":
            rename_map[c] = "open"
        elif cl == "high":
            rename_map[c] = "high"
        elif cl == "low":
            rename_map[c] = "low"
        elif cl == "close":
            rename_map[c] = "close"
        elif cl == "volume":
            rename_map[c] = "volume"
    full = full.rename(columns=rename_map)

    if "time" not in full.columns:
        full = full.rename(columns={full.columns[0]: "time"})

    full["time"] = pd.to_datetime(full["time"], utc=True)
    full = full[["time", "open", "high", "low", "close", "volume"]].dropna()
    full = full.sort_values("time").reset_index(drop=True)
    return full


def download_nasdaq_daily(ticker: str = "NQ=F", years: float = 4.0) -> pd.DataFrame:
    """
    yfinance에서 나스닥 선물 일봉을 다운로드 (수년 치 가능).
    5분봉 60일 제한을 넘는 과거 데이터용 fallback.
    반환: columns=[time, open, high, low, close, volume], tz=UTC
    """
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))

    try:
        df = yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
        )
    except Exception as e:
        warnings.warn(f"[nasdaq] 일봉 다운로드 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("datetime", "date", "index"):
            rename_map[c] = "time"
        elif cl == "open":
            rename_map[c] = "open"
        elif cl == "high":
            rename_map[c] = "high"
        elif cl == "low":
            rename_map[c] = "low"
        elif cl == "close":
            rename_map[c] = "close"
        elif cl == "volume":
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _setup_ssl_env():
    """SSL 인증서 경로 자동 설정 (certifi 경로에 한글이 있어서 fail나는 curl_cffi 버그 회피)"""
    import os
    import shutil
    try:
        import certifi
        orig_cert = certifi.where()
        
        # 경로에 한글 등 비-ASCII 문자가 있으면 curl_cffi가 인식 못하는 문제 해결
        safe_cert_dir = r"C:\temp"
        safe_cert_path = r"C:\temp\cacert.pem"
        
        if not os.path.exists(safe_cert_dir):
            os.makedirs(safe_cert_dir, exist_ok=True)
            
        if not os.path.exists(safe_cert_path) or os.path.getsize(orig_cert) != os.path.getsize(safe_cert_path):
            shutil.copy2(orig_cert, safe_cert_path)
            
        os.environ['SSL_CERT_FILE'] = safe_cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = safe_cert_path
        os.environ['CURL_CA_BUNDLE'] = safe_cert_path
    except Exception as e:
        pass


class NasdaqFeatureProvider:
    """
    백테스트 시작 전에 한 번 초기화. get_features()로 봉마다 피처 반환.

    Parameters
    ----------
    years : float
        다운로드 기간 (년)
    ticker : str
        나스닥 선물 티커 (기본 NQ=F)
    nq_df : pd.DataFrame or None
        이미 다운받은 나스닥 5분봉. 제공 시 다운로드 스킵.
    """

    def __init__(self, years: float = 1.5, ticker: str = "NQ=F",
                 nq_df: Optional[pd.DataFrame] = None,
                 nq_daily_df: Optional[pd.DataFrame] = None):
        cache_5m_path  = f"nasdaq_{ticker.replace('=','').lower()}_5m.csv"
        cache_1d_path  = f"nasdaq_{ticker.replace('=','').lower()}_1d.csv"
        self._ready = False
        self._has_5m = False  # 5분봉 보유 여부

        # ── 1) 외부 제공 데이터
        if nq_df is not None and not nq_df.empty:
            self.nq = nq_df.copy()
            self._has_5m = True
        else:
            self.nq = pd.DataFrame()

        # ── 2) 5분봉 캐시 시도
        if self.nq.empty and os.path.exists(cache_5m_path):
            try:
                print(f"[nasdaq] 5m 캐시 로드 중: {cache_5m_path}")
                self.nq = pd.read_csv(cache_5m_path)
                self.nq["time"] = pd.to_datetime(self.nq["time"], utc=True)
                
                # ------ Data Trimming & Forward-Fill ------
                # 1) years 기반 대량 데이터 자르기 (여유분 +30일)
                cutoff_time = pd.Timestamp.utcnow() - pd.Timedelta(days=int(years * 365 + 30))
                self.nq = self.nq[self.nq["time"] >= cutoff_time].copy()
                
                # 2) 결측치 처리: forward-fill (미래 데이터 누수 방지)
                #    linear interpolation은 미래 값을 참조하므로 사용 금지
                self.nq = self.nq.set_index("time").resample("5min").asfreq()
                cols_to_fill = ["open", "high", "low", "close", "volume"]
                self.nq[cols_to_fill] = self.nq[cols_to_fill].ffill()
                self.nq = self.nq.dropna().reset_index()
                # ---------------------------------------------
                
                print(f"[nasdaq] 5m 캐시 로드 완료 (컷팅&보간 적용): {len(self.nq):,}봉")
                self._has_5m = True
            except Exception as e:
                print(f"[nasdaq] 5m 캐시 로드 실패: {e}")
                self.nq = pd.DataFrame()

        # ── 3) 5분봉 다운로드 시도 (최대 60일)
        if self.nq.empty:
            print(f"[nasdaq] 5분봉 다운로드 시도 (최대 60일)...")
            _setup_ssl_env()
            self.nq = download_nasdaq_5m(ticker=ticker, years=years)
            if not self.nq.empty:
                self._has_5m = True
                print(f"[nasdaq] 5분봉 {len(self.nq):,}봉 수집")
                try:
                    self.nq.to_csv(cache_5m_path, index=False)
                except Exception:
                    pass

        # ── 4) 일봉 데이터 (수년 치 커버 — 5분봉이 없는 과거 구간 fallback)
        self.nq_daily = pd.DataFrame()

        # 4-a) 외부 제공 일봉
        if nq_daily_df is not None and not nq_daily_df.empty:
            self.nq_daily = nq_daily_df.copy()
            if "time" in self.nq_daily.columns:
                self.nq_daily["time"] = pd.to_datetime(self.nq_daily["time"], utc=True)
            print(f"[nasdaq] 일봉 외부 주입: {len(self.nq_daily):,}봉")

        # 4-b) 캐시 시도
        if self.nq_daily.empty and os.path.exists(cache_1d_path):
            try:
                print(f"[nasdaq] 일봉 캐시 로드 중: {cache_1d_path}")
                self.nq_daily = pd.read_csv(cache_1d_path)
                self.nq_daily["time"] = pd.to_datetime(self.nq_daily["time"], utc=True)
                
                # ------ Data Trimming & Forward-Fill ------
                _dl_years = max(years, 4.0)
                cutoff_time_1d = pd.Timestamp.utcnow() - pd.Timedelta(days=int(_dl_years * 365 + 30))
                self.nq_daily = self.nq_daily[self.nq_daily["time"] >= cutoff_time_1d].copy()
                
                self.nq_daily = self.nq_daily.set_index("time").resample("1D").asfreq()
                cols_to_fill = ["open", "high", "low", "close", "volume"]
                self.nq_daily[cols_to_fill] = self.nq_daily[cols_to_fill].ffill()
                self.nq_daily = self.nq_daily.dropna().reset_index()
                # ---------------------------------------------
                
                print(f"[nasdaq] 일봉 캐시 로드 완료 (컷팅&보간 적용): {len(self.nq_daily):,}봉")
            except Exception:
                self.nq_daily = pd.DataFrame()

        if self.nq_daily.empty:
            _dl_years = max(years, 4.0)
            print(f"[nasdaq] 일봉 다운로드 ({_dl_years:.0f}년)...")
            _setup_ssl_env()
            self.nq_daily = download_nasdaq_daily(ticker=ticker, years=_dl_years)
            if not self.nq_daily.empty:
                print(f"[nasdaq] 일봉 {len(self.nq_daily):,}봉 수집")
                try:
                    self.nq_daily.to_csv(cache_1d_path, index=False)
                except Exception:
                    pass

        # ── 5분봉도 일봉도 없으면 포기
        if self.nq.empty and self.nq_daily.empty:
            print(f"  ⚠️ [nasdaq] {ticker} 데이터 수집 실패 — 피처 0으로 채움")
            return

        self._ready = True

        # ── 전처리: 5분봉
        if not self.nq.empty:
            self.nq = self.nq.sort_values("time").reset_index(drop=True)
            self.nq["ret_5m"] = self.nq["close"].pct_change()
            _vol_window = min(288 * 5, len(self.nq) - 1) if len(self.nq) > 20 else 20
            self.nq["vol_20bar"] = self.nq["ret_5m"].rolling(20).std()
            self.nq["vol_1w"]    = self.nq["ret_5m"].rolling(_vol_window).std()
            self.nq = self.nq.set_index("time", drop=False)
            self.nq.index.name = "idx_time"

        # ── 전처리: 일봉 → 전일 종가 dict
        self._daily_close = {}
        _daily_src = self.nq_daily if not self.nq_daily.empty else self.nq
        if not _daily_src.empty:
            _tmp = _daily_src.copy()
            _tmp["date_utc"] = _tmp["time"].dt.date
            self._daily_close = _tmp.groupby("date_utc")["close"].last().to_dict()

        _range_5m = f"{self.nq['time'].min()} ~ {self.nq['time'].max()}" if not self.nq.empty else "없음"
        _range_1d = f"{self.nq_daily['time'].min()} ~ {self.nq_daily['time'].max()}" if not self.nq_daily.empty else "없음"
        print(f"  📊 나스닥 5m={_range_5m}")
        print(f"  📊 나스닥 1d={_range_1d}  (daily_close {len(self._daily_close)}일)")

    @property
    def ready(self) -> bool:
        return self._ready

    def get_features(
        self,
        t: pd.Timestamp,
        crypto_5m: pd.DataFrame,
        atr_now: float,
        sweep_time: Optional[pd.Timestamp] = None,
        symbol: str = "BTC",
    ) -> dict:
        """
        특정 시점 t에서 나스닥-크립토 괴리 피처 반환.

        Parameters
        ----------
        t : pd.Timestamp (UTC)
            현재 봉 시각
        crypto_5m : pd.DataFrame
            코인 5분봉 (time, close 필수). 최근 300봉 이상 권장.
        atr_now : float
            현재 코인 ATR
        sweep_time : Optional[pd.Timestamp]
            sweep 발생 시각 (없으면 해당 피처=0)
        symbol : str
            심볼 이름 (BTC, ETH, SOL만 활성)
        """
        # 마스킹: 지원 심볼 아니면 0
        base = symbol.upper().replace("/USDT:USDT", "").replace("/USDT", "")
        if base not in NASDAQ_ENABLED_SYMBOLS or not self._ready:
            return _zero_features()

        t_utc = pd.Timestamp(t, tz="UTC") if t.tzinfo is None else t.tz_convert("UTC")

        # 나스닥에서 현재 시점 이전 가장 가까운 봉 찾기
        # 1순위: 5분봉, 2순위: 일봉 fallback
        _use_5m = False
        nq_close_now = 0.0

        if not self.nq.empty:
            nq_before = self.nq.loc[self.nq["time"] <= t_utc]
            if not nq_before.empty:
                nq_now = nq_before.iloc[-1]
                nq_close_now = float(nq_now["close"])
                _use_5m = True

        # 5분봉에 해당 시점 데이터 없으면 → 일봉 fallback
        if nq_close_now == 0.0 and not self.nq_daily.empty:
            _t_date = t_utc.date()
            # 당일 또는 직전 영업일 종가
            for _offset in range(0, 5):
                _d = (t_utc - timedelta(days=_offset)).date()
                _dc = self._daily_close.get(_d)
                if _dc and _dc > 0:
                    nq_close_now = float(_dc)
                    break

        if nq_close_now == 0.0:
            return _zero_features()

        # ─── 1) nasdaq_session_divergence_pct
        #     전일 종가 대비 현재 %
        prev_date = (t_utc - timedelta(days=1)).date()
        prev_close = self._daily_close.get(prev_date)
        if prev_close and prev_close > 0:
            nasdaq_session_div = (nq_close_now - prev_close) / prev_close * 100
        else:
            # 전전일 시도
            prev_date2 = (t_utc - timedelta(days=2)).date()
            prev_close2 = self._daily_close.get(prev_date2)
            nasdaq_session_div = (
                (nq_close_now - prev_close2) / prev_close2 * 100
                if prev_close2 and prev_close2 > 0 else 0.0
            )

        # ─── 2) nasdaq_divergence_sweep_pct_atr
        nq_div_sweep = 0.0
        if sweep_time is not None and atr_now > 0:
            st = pd.Timestamp(sweep_time, tz="UTC") if sweep_time.tzinfo is None else sweep_time.tz_convert("UTC")
            nq_sweep_px = 0.0
            # 5분봉에서 sweep 시점 가격
            if _use_5m:
                nq_at_sweep = self.nq.loc[self.nq["time"] <= st]
                if not nq_at_sweep.empty:
                    nq_sweep_px = float(nq_at_sweep.iloc[-1]["close"])
            # fallback: 일봉
            if nq_sweep_px == 0.0:
                for _off in range(0, 5):
                    _sd = (st - timedelta(days=_off)).date()
                    _sc = self._daily_close.get(_sd)
                    if _sc and _sc > 0:
                        nq_sweep_px = float(_sc)
                        break
            if nq_sweep_px > 0:
                nq_pct_move = (nq_close_now - nq_sweep_px) / nq_sweep_px * 100
                nq_div_sweep = nq_pct_move / atr_now

        # ─── 3) nasdaq_outperformance_flag
        #     최근 12봉(1시간) 나스닥 vs 코인 수익률 비교 (5분봉 필수)
        nq_outperf = 0.0
        if _use_5m:
            lookback_12 = 12
            nq_before = self.nq.loc[self.nq["time"] <= t_utc]
            if len(nq_before) >= lookback_12 and len(crypto_5m) >= lookback_12:
                nq_ret_1h = (nq_close_now / float(nq_before.iloc[-lookback_12]["close"])) - 1
                crypto_close_now = float(crypto_5m["close"].iloc[-1])
                crypto_close_1h  = float(crypto_5m["close"].iloc[-lookback_12])
                if crypto_close_1h > 0:
                    crypto_ret_1h = (crypto_close_now / crypto_close_1h) - 1
                    nq_outperf = nq_ret_1h - crypto_ret_1h

        # ─── 4) btc_nasdaq_corr_24h (5분봉 필수)
        corr_24h = 0.0
        if _use_5m:
            corr_window = 288
            nq_before = self.nq.loc[self.nq["time"] <= t_utc]
            if len(nq_before) >= corr_window and len(crypto_5m) >= corr_window:
                nq_rets = nq_before["ret_5m"].iloc[-corr_window:].values
                crypto_rets = crypto_5m["close"].pct_change().iloc[-corr_window:].values
                mask = ~(np.isnan(nq_rets) | np.isnan(crypto_rets))
                if mask.sum() > 30:
                    # std=0인 경우 RuntimeWarning이 발생하므로 사전 체크
                    nq_clean = nq_rets[mask]
                    crypto_clean = crypto_rets[mask]
                    if nq_clean.std() > 0 and crypto_clean.std() > 0:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            corr_24h = float(np.corrcoef(nq_clean, crypto_clean)[0, 1])
                        if np.isnan(corr_24h):
                            corr_24h = 0.0

        # ─── 5) is_ny_am_session, is_ny_pm_session
        hour_utc = t_utc.hour
        minute_utc = t_utc.minute
        utc_minutes = hour_utc * 60 + minute_utc
        est_minutes = utc_minutes - 300
        if est_minutes < 0:
            est_minutes += 1440

        is_ny_am = 1 if (9 * 60 + 30) <= est_minutes < (12 * 60) else 0
        is_ny_pm = 1 if (13 * 60 + 30) <= est_minutes < (16 * 60) else 0

        # ─── 6) nasdaq_volatility_spike (5분봉 필수)
        vol_spike = 0.0
        if _use_5m:
            nq_before = self.nq.loc[self.nq["time"] <= t_utc]
            if not nq_before.empty:
                _nq_last = nq_before.iloc[-1]
                vol_short = _nq_last.get("vol_20bar", np.nan)
                vol_long  = _nq_last.get("vol_1w", np.nan)
                if pd.notna(vol_short) and pd.notna(vol_long) and vol_long > 0:
                    vol_spike = float(vol_short / vol_long)

        return {
            "nasdaq_session_divergence_pct":   round(nasdaq_session_div, 4),
            "nasdaq_divergence_sweep_pct_atr": round(nq_div_sweep, 4),
            "nasdaq_outperformance_flag":       round(nq_outperf, 6),
            "btc_nasdaq_corr_24h":             round(corr_24h, 4),
            "is_ny_am_session":                is_ny_am,
            "is_ny_pm_session":                is_ny_pm,
            "nasdaq_volatility_spike":         round(vol_spike, 4),
        }