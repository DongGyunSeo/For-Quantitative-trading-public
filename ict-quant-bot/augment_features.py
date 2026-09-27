"""
augment_features.py — 기존 데이터셋에 시간 소급 기반 시장 레짐 피처 보강
══════════════════════════════════════════════════════════════════════════
- 6시간짜리 백테스트 재실행을 피하기 위해, 진입 시간(entry_time)을 기준으로 
  과거 5m 캔들을 직접 로드하여 ADX, 모멘텀, 변동성 등의 시장 레짐(Market Regime) 피처를 고속 계산합니다.
- 계산된 피처를 기존 ml_train_with_tp.csv에 LEFT JOIN (또는 asof_merge)하여 ml_train_augmented.csv로 저장합니다.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone, timedelta

def compute_atr(high, low, close, window=14):
    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs())
    )
    return tr.rolling(window=window, min_periods=window//2).mean()

def compute_adx(high, low, close, window=14):
    """ADX, +DI, -DI 계산 (Welles Wilder 방식 근사치, 심플 EMA/SMA 활용)"""
    up = high.diff()
    down = -low.diff()
    
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs())
    )
    
    # pandas EWM 활용 (Wilder's Smoothing 근사값 : alpha=1/window)
    pos_dm_roll = pd.Series(pos_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean()
    neg_dm_roll = pd.Series(neg_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean()
    tr_roll = tr.ewm(alpha=1/window, adjust=False).mean()
    
    pos_di = 100 * pos_dm_roll / (tr_roll + 1e-8)
    neg_di = 100 * neg_dm_roll / (tr_roll + 1e-8)
    
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8)
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    
    return adx, pos_di, neg_di

def compute_regime_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    """다운로드된 5분봉 전체에 대해 벡터 연산으로 피처 고속 계산"""
    df = df_5m.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    
    df['adx_14'], df['di_plus'], df['di_minus'] = compute_adx(high, low, close, window=14)
    
    atr_14 = compute_atr(high, low, close, window=14)
    atr_100 = compute_atr(high, low, close, window=100)
    df['atr_ratio'] = atr_14 / (atr_100 + 1e-8)  # 변동성 수축/확장 비율
    
    # 1H(12봉), 4H(48봉) 모멘텀 (변동 퍼센트)
    df['return_1h'] = (close / close.shift(12)) - 1.0
    df['return_4h'] = (close / close.shift(48)) - 1.0
    
    return df

from backtest import fetch_historical_5m

def main():
    csv_file = "ml_train_with_tp.csv"
    try:
        trades_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ {csv_file} 파일이 없습니다.")
        return
        
    trades_df['entry_time_dt'] = pd.to_datetime(trades_df['entry_time'])
    symbols_in_dataset = trades_df['symbol'].unique()
    
    # OKX 현물/선물 API 연동
    exchange = ccxt.okx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    
    augmented_dfs = []
    
    print("═" * 70)
    print("  🚀 시장 레짐 피처 (Market Regime Feature) 소급 보강 시작 (OKX API)")
    print("═" * 70)
    
    for sym_short in symbols_in_dataset:
        symbol_full = f"{sym_short}/USDT:USDT" if sym_short != "BTC" else "BTC/USDT:USDT"
        sub_df = trades_df[trades_df['symbol'] == sym_short].copy()
        
        min_dt = sub_df['entry_time_dt'].min() - timedelta(days=5)
        max_dt = sub_df['entry_time_dt'].max() + timedelta(days=1)
        
        print(f"\n  📥 {sym_short:5s} 데이터 로드 및 피처 계산...")
        print(f"     기간: {min_dt.strftime('%Y-%m-%d')} ~ {max_dt.strftime('%Y-%m-%d')} ({len(sub_df)} trades)")
        
        # 1. 시계열 데이터 다운로드 (OKX - backtest.py 함수 재사용)
        months_diff = (max_dt - min_dt).days // 30 + 1
        df_5m = fetch_historical_5m(
            exchange=exchange,
            symbol=symbol_full,
            months=months_diff,
            end_date=max_dt.strftime('%Y-%m-%d')
        )
        if df_5m.empty:
            print(f"     ⚠️ 데이터 수집 실패, 해당 코인은 백업합니다.")
            augmented_dfs.append(sub_df.drop(columns=['entry_time_dt']))
            continue
            
        print(f"     ✅ 5분봉 캔들 {len(df_5m):,}개 다운로드 완료")
        
        # 2. 피처 고속 계산
        df_5m = compute_regime_features(df_5m)
        
        # 3. Time 기반 Asof Merge (가장 가까운 캔들에 매핑)
        sub_df['entry_time_dt'] = pd.to_datetime(sub_df['entry_time_dt'], utc=True).dt.as_unit('ns')
        df_5m['time'] = pd.to_datetime(df_5m['time'], utc=True).dt.as_unit('ns')
        
        sub_df = sub_df.sort_values('entry_time_dt')
        df_5m = df_5m.sort_values('time')
        
        merged = pd.merge_asof(
            sub_df, 
            df_5m[['time', 'adx_14', 'atr_ratio', 'return_1h', 'return_4h']], 
            left_on='entry_time_dt', 
            right_on='time', 
            direction='backward',
            suffixes=('_old', ''),
        )
        
        # 기존 컬럼이 있으면 새 값으로 덮어쓰기
        for _col in ['adx_14', 'return_1h', 'return_4h']:
            old_col = f'{_col}_old'
            if old_col in merged.columns:
                merged[_col] = merged[_col].fillna(merged[old_col])
                merged = merged.drop(columns=[old_col])
        if 'atr_ratio_old' in merged.columns:
            merged = merged.drop(columns=['atr_ratio_old'])
        
        # ── range_pos_30d / range_pos_7d: 5m 캔들 고저점 위치 (pd_pos 대체)
        _high_arr = df_5m["high"].values.astype(float)
        _low_arr = df_5m["low"].values.astype(float)
        _close_arr = df_5m["close"].values.astype(float)
        _time_arr = df_5m["time"].values  # nanosecond

        rp30_vals = []
        rp7_vals = []
        for et in merged['entry_time_dt']:
            # entry_time 이전 캔들만 사용 (미래 참조 방지)
            mask = df_5m["time"] <= et
            n_avail = mask.sum()
            if n_avail < 100:
                rp30_vals.append(0.5)
                rp7_vals.append(0.5)
                continue

            _h = _high_arr[mask.values]
            _l = _low_arr[mask.values]
            _px = float(_close_arr[mask.values][-1])

            # 30일 = 8640봉
            n30 = min(8640, len(_h))
            h30 = float(_h[-n30:].max())
            l30 = float(_l[-n30:].min())
            r30 = h30 - l30
            rp30_vals.append(round((_px - l30) / r30, 4) if r30 > 0 else 0.5)

            # 7일 = 2016봉
            n7 = min(2016, len(_h))
            h7 = float(_h[-n7:].max())
            l7 = float(_l[-n7:].min())
            r7 = h7 - l7
            rp7_vals.append(round((_px - l7) / r7, 4) if r7 > 0 else 0.5)

        merged["range_pos_30d"] = rp30_vals
        merged["range_pos_7d"] = rp7_vals

        # ── Interaction 피처 계산 (기존 컬럼 조합)
        for _col_pair, _name in [
            (("sweep_depth_atr", "strong_displacement_score"), "sweep_x_disp"),
            (("strong_displacement_score", "effective_fvg_strength"), "disp_x_fvg"),
            (("lvn_proximity", "effective_fvg_strength"), "lvn_x_fvg"),
        ]:
            a, b = _col_pair
            if a in merged.columns and b in merged.columns:
                merged[_name] = (merged[a] * merged[b]).round(4)

        if "sweep_depth_atr" in merged.columns and "bars_since_sweep" in merged.columns:
            merged["sweep_velocity"] = (merged["sweep_depth_atr"] / (merged["bars_since_sweep"] + 1.0)).round(4)

        if "sweep_duration_norm" in merged.columns:
            merged["rangepos_x_sweep_dur"] = (merged["range_pos_7d"] * merged["sweep_duration_norm"]).round(4)

        if "sweep_depth_atr" in merged.columns:
            merged["rangepos_x_sweep_depth"] = (merged["range_pos_7d"] * merged["sweep_depth_atr"]).round(4)

        # ── 현물-선물 괴리 피처 (return_div, vol_div, basis_z)
        try:
            spot_symbol = f"{sym_short}/USDT"
            print(f"     📥 현물 5m 캔들 다운로드 중... ({spot_symbol})")

            exchange_spot = ccxt.okx({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })

            df_spot = fetch_historical_5m(
                exchange=exchange_spot,
                symbol=spot_symbol,
                months=months_diff,
                end_date=max_dt.strftime('%Y-%m-%d')
            )

            if not df_spot.empty and len(df_spot) > 300:
                print(f"     ✅ 현물 {len(df_spot):,}봉 다운로드 완료")

                # 시간 동기화
                df_fut = df_5m[['time', 'close', 'high', 'low', 'volume']].copy()
                df_sp  = df_spot[['time', 'close', 'high', 'low', 'volume']].copy()
                df_fut['time'] = pd.to_datetime(df_fut['time'], utc=True)
                df_sp['time']  = pd.to_datetime(df_sp['time'], utc=True)

                # ffill + strict inner join
                df_fut = df_fut.set_index('time').resample('5min').ffill().reset_index()
                df_sp  = df_sp.set_index('time').resample('5min').ffill().reset_index()

                sf = pd.merge(df_fut, df_sp, on='time', how='inner',
                              suffixes=('_fut', '_spot'))

                if len(sf) > 300:
                    # 1. Return Divergence: 선물 수익률 - 현물 수익률
                    sf['ret_fut']  = sf['close_fut'].pct_change().fillna(0)
                    sf['ret_spot'] = sf['close_spot'].pct_change().fillna(0)
                    sf['return_div'] = (sf['ret_fut'] - sf['ret_spot']).clip(-0.01, 0.01).fillna(0)

                    # 2. Volume Divergence: 거래량 스파이크 비율 차이
                    sf['vol_surge_fut']  = sf['volume_fut'] / (sf['volume_fut'].rolling(288, min_periods=50).mean() + 1e-10)
                    sf['vol_surge_spot'] = sf['volume_spot'] / (sf['volume_spot'].rolling(288, min_periods=50).mean() + 1e-10)
                    sf['vol_div'] = (sf['vol_surge_fut'] - sf['vol_surge_spot']).clip(-10, 10).fillna(0)

                    # 3. Basis Z-Score
                    sf['basis_pct'] = ((sf['close_fut'] - sf['close_spot']) / (sf['close_spot'] + 1e-10) * 100)
                    sf['basis_pct'] = sf['basis_pct'].clip(-1, 1)
                    _bm = sf['basis_pct'].rolling(288, min_periods=50).mean()
                    _bs = sf['basis_pct'].rolling(288, min_periods=50).std()
                    sf['basis_z'] = ((sf['basis_pct'] - _bm) / (_bs + 1e-10)).clip(-4, 4).fillna(0)

                    # merge_asof로 trade에 매핑 (backward only)
                    spot_cols = sf[['time', 'return_div', 'vol_div', 'basis_z']].copy()
                    spot_cols = spot_cols.sort_values('time')

                    merged = pd.merge_asof(
                        merged, spot_cols,
                        left_on='entry_time_dt', right_on='time',
                        direction='backward',
                        suffixes=('', '_sf'),
                    )
                    merged = merged.drop(columns=['time'], errors='ignore')

                    # 충돌 처리
                    for _col in ['return_div', 'vol_div', 'basis_z']:
                        sf_col = f'{_col}_sf'
                        if sf_col in merged.columns:
                            merged[_col] = merged[_col].fillna(merged[sf_col])
                            merged = merged.drop(columns=[sf_col])
                        if _col in merged.columns:
                            merged[_col] = merged[_col].fillna(0.0)

                    _nz = (merged['basis_z'] != 0).sum()
                    print(f"     ✅ Spot-Futures 피처 보강 완료 (매칭: {_nz}/{len(merged)})")
                else:
                    print(f"     ⚠️ 현물-선물 매칭 부족 ({len(sf)}봉)")
                    for _col in ['return_div', 'vol_div', 'basis_z']:
                        merged[_col] = 0.0
            else:
                print(f"     ⚠️ 현물 다운로드 실패")
                for _col in ['return_div', 'vol_div', 'basis_z']:
                    merged[_col] = 0.0
        except Exception as e:
            print(f"     ⚠️ Spot-Futures 보강 실패: {e}")
            for _col in ['return_div', 'vol_div', 'basis_z']:
                if _col not in merged.columns:
                    merged[_col] = 0.0

        # 사용한 임시 컬럼 정리
        merged = merged.drop(columns=['entry_time_dt', 'time'], errors='ignore')
        augmented_dfs.append(merged)
        print(f"     ✅ 피처 보강 결합 완료")

    final_df = pd.concat(augmented_dfs, ignore_index=True)
    out_file = "ml_train_with_tp.csv"
    final_df.to_csv(out_file, index=False)
    
    # 보강 결과 확인
    print(f"\n  {'─' * 60}")
    print(f"  📊 보강 결과:")
    for _col in ['adx_14', 'return_1h', 'return_4h', 'range_pos_30d', 'range_pos_7d',
                 'return_div', 'vol_div', 'basis_z']:
        if _col in final_df.columns:
            _v = final_df[_col]
            _nz = (_v != 0).sum()
            print(f"     {_col:20s}  non-zero={_nz:5d}/{len(final_df)}  "
                  f"mean={_v.mean():+.6f}  std={_v.std():.6f}  "
                  f"[{_v.min():.6f}, {_v.max():.6f}]")
    
    print("\n" + "═" * 70)
    print(f"  🎉 모든 작업 완료! ({len(final_df):,}건)")
    print(f"  💾 저장 위치: {out_file}")
    print(f"  👉 다음 실행: python train_entry.py --csv {out_file}")
    print("═" * 70)

if __name__ == "__main__":
    main()
