"""
prepare_nasdaq.py — 구매한 나스닥 선물 원본 데이터 → 백테스트용 CSV 정제
══════════════════════════════════════════════════════════════════════════
사용법:
    python prepare_nasdaq.py  nasdaq_raw.csv
    python prepare_nasdaq.py  nasdaq_raw.csv --tz America/Chicago

입력 포맷 (세미콜론 구분):
    DD/MM/YYYY;HH:MM;Open;High;Low;Close;Volume
    시간 기준: GMT-6 (기본값, --tz 로 변경 가능)

출력 파일 (자동 생성):
    nasdaq_nqf_5m.csv   — 5분봉 (UTC, 정규 그리드)
    nasdaq_nqf_1d.csv   — 일봉   (UTC 기준 집계)

정제 과정:
    1. 세미콜론 파싱 + datetime 변환
    2. GMT-6 → UTC 변환 (또는 DST 자동 적용)
    3. 5분 그리드 리샘플 (동일 슬롯 OHLCV 합산)
    4. 일봉 집계
    5. 데이터 품질 리포트
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import timezone, timedelta

import pandas as pd


# ──────────────────────────────────────────────────────────
# 1. 원본 파싱
# ──────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """
    세미콜론 구분 나스닥 원본 파일 로드.
    DD/MM/YYYY;HH:MM;Open;High;Low;Close;Volume
    """
    print(f"\n{'═'*65}")
    print(f"  📂 원본 파일 로드: {path}")
    print(f"{'═'*65}")

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={"date": str, "time": str},
    )

    print(f"  원본 행 수: {len(df):,}")

    # datetime 파싱: DD/MM/YYYY HH:MM
    df["datetime_str"] = df["date"] + " " + df["time"]
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%d/%m/%Y %H:%M")

    # 숫자 변환
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    # NaN 제거
    n_before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  ⚠️  NaN 제거: {n_dropped:,}행")

    print(f"  유효 행 수: {len(df):,}")
    print(f"  기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    return df


# ──────────────────────────────────────────────────────────
# 2. 시간대 변환
# ──────────────────────────────────────────────────────────

def convert_timezone(df: pd.DataFrame, tz_mode: str = "fixed") -> pd.DataFrame:
    """
    원본 시간(GMT-6) → UTC 변환.

    Parameters
    ----------
    tz_mode : str
        "fixed"           — 고정 UTC-6 (DST 미적용, 사용자 명시한 GMT-6)
        "America/Chicago" — DST 자동 적용 (CST/CDT 자동 전환)
    """
    print(f"\n  🕐 시간대 변환: {tz_mode}")

    if tz_mode == "fixed":
        # 고정 GMT-6 → UTC (+6시간)
        gmt_minus_6 = timezone(timedelta(hours=-6))
        df["time_utc"] = df["datetime"].dt.tz_localize(gmt_minus_6).dt.tz_convert("UTC")
        print(f"     고정 UTC-6 → UTC (+6h)")

    else:
        # America/Chicago: DST 자동 적용
        # CST(겨울) = UTC-6,  CDT(여름) = UTC-5
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        chicago = ZoneInfo(tz_mode)

        # ambiguous/nonexistent 시간 처리 (DST 전환 시점)
        df["time_utc"] = (
            df["datetime"]
            .dt.tz_localize(chicago, ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )

        # DST 전환 시점에서 NaT 발생 → 제거
        n_nat = df["time_utc"].isna().sum()
        if n_nat > 0:
            print(f"     ⚠️  DST 전환 시점 NaT {n_nat}건 제거")
            df = df.dropna(subset=["time_utc"]).reset_index(drop=True)

        print(f"     {tz_mode} (DST 자동) → UTC")

    # 변환 샘플 출력
    print(f"     변환 전: {df['datetime'].iloc[0]}  →  변환 후: {df['time_utc'].iloc[0]}")
    if len(df) > 100:
        mid = len(df) // 2
        print(f"     변환 전: {df['datetime'].iloc[mid]}  →  변환 후: {df['time_utc'].iloc[mid]}")

    return df


# ──────────────────────────────────────────────────────────
# 3. 5분봉 정규화
# ──────────────────────────────────────────────────────────

def resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    """
    비정규 간격 봉 → 5분 그리드 리샘플.

    동일 5분 슬롯에 여러 봉이 있으면:
      open  = 첫 번째 봉의 open
      high  = 모든 봉의 max(high)
      low   = 모든 봉의 min(low)
      close = 마지막 봉의 close
      vol   = 합산
    """
    print(f"\n  📊 5분봉 정규화...")

    # 5분 그리드로 floor
    df["slot_5m"] = df["time_utc"].dt.floor("5min")

    # 그룹별 OHLCV 집계
    agg = df.groupby("slot_5m").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    agg = agg.rename(columns={"slot_5m": "time"})
    agg = agg.sort_values("time").reset_index(drop=True)

    # 중복 제거
    agg = agg.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    # 간격 분포 분석
    diffs = agg["time"].diff().dt.total_seconds() / 60
    print(f"     정규화 후: {len(agg):,}봉")
    print(f"     간격 분포: 5min={int((diffs == 5).sum()):,}  "
          f"10min={int((diffs == 10).sum()):,}  "
          f"15min+={int((diffs >= 15).sum()):,}")

    # time을 문자열로 (ISO format with UTC)
    agg["time"] = agg["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    return agg


# ──────────────────────────────────────────────────────────
# 4. 일봉 집계
# ──────────────────────────────────────────────────────────

def aggregate_daily(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    5분봉 → 일봉 집계 (UTC 날짜 기준).
    """
    print(f"\n  📊 일봉 집계...")

    df = df_5m.copy()
    df["time_dt"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time_dt"].dt.date

    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    daily["time"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d 00:00:00+00:00")
    daily = daily[["time", "open", "high", "low", "close", "volume"]]
    daily = daily.sort_values("time").reset_index(drop=True)

    print(f"     일봉: {len(daily):,}일")
    print(f"     기간: {daily['time'].iloc[0][:10]} ~ {daily['time'].iloc[-1][:10]}")

    return daily


# ──────────────────────────────────────────────────────────
# 5. 데이터 품질 리포트
# ──────────────────────────────────────────────────────────

def quality_report(df_5m: pd.DataFrame, df_1d: pd.DataFrame,
                   df_raw: pd.DataFrame = None):
    """정제 결과 품질 리포트."""
    print(f"\n{'─'*65}")
    print(f"  📋 데이터 품질 리포트")
    print(f"{'─'*65}")

    # 5분봉
    df = df_5m.copy()
    df["time_dt"] = pd.to_datetime(df["time"], utc=True)
    print(f"  5분봉:")
    print(f"    총 봉 수    : {len(df):>12,}")
    print(f"    기간        : {df['time'].iloc[0][:10]} ~ {df['time'].iloc[-1][:10]}")
    print(f"    일수        : {(df['time_dt'].iloc[-1] - df['time_dt'].iloc[0]).days:,}일")

    # OHLC 무결성 체크
    bad_hl = (df["high"] < df["low"]).sum()
    bad_oh = (df["open"] > df["high"]).sum()
    bad_ol = (df["open"] < df["low"]).sum()
    bad_ch = (df["close"] > df["high"]).sum()
    bad_cl = (df["close"] < df["low"]).sum()
    print(f"    H < L       : {bad_hl:>12,}건")
    print(f"    O > H       : {bad_oh:>12,}건")
    print(f"    O < L       : {bad_ol:>12,}건")
    print(f"    C > H       : {bad_ch:>12,}건")
    print(f"    C < L       : {bad_cl:>12,}건")

    # 가격 범위
    print(f"    가격 범위   : {df['low'].min():,.2f} ~ {df['high'].max():,.2f}")
    print(f"    볼륨 범위   : {df['volume'].min():,} ~ {df['volume'].max():,}")

    # 연도별 봉 수
    df["year"] = df["time_dt"].dt.year
    print(f"\n    연도별 봉 수:")
    for yr, cnt in df.groupby("year").size().items():
        yr_days = df[df["year"] == yr]["time_dt"].dt.date.nunique()
        print(f"      {yr}: {cnt:>10,}봉  ({yr_days:>4}일)")

    # 일봉
    print(f"\n  일봉:")
    print(f"    총 일수     : {len(df_1d):>12,}")

    zero_vol = (df_1d["volume"] == 0).sum()
    if zero_vol > 0:
        print(f"    볼륨 0 일수 : {zero_vol:>12,}")

    # ── DST 검증: 여름/겨울 RTH 시작 시각 비교
    if df_raw is not None and len(df_raw) > 1000:
        print(f"\n  🕐 DST 검증 (원본 시각 기준):")
        df_raw_check = df_raw.copy()
        df_raw_check["month"] = df_raw_check["datetime"].dt.month
        df_raw_check["hhmm"] = df_raw_check["datetime"].dt.strftime("%H:%M")

        # 여름(6~8월)과 겨울(12~2월)에서 08:30 봉 존재 여부
        summer = df_raw_check[df_raw_check["month"].isin([6, 7, 8])]
        winter = df_raw_check[df_raw_check["month"].isin([12, 1, 2])]

        summer_0830 = (summer["hhmm"] == "08:30").sum()
        summer_0730 = (summer["hhmm"] == "07:30").sum()
        winter_0830 = (winter["hhmm"] == "08:30").sum()
        winter_0730 = (winter["hhmm"] == "07:30").sum()

        print(f"    겨울(12~2월) 08:30 봉: {winter_0830:,}건  |  07:30 봉: {winter_0730:,}건")
        print(f"    여름(6~8월)  08:30 봉: {summer_0830:,}건  |  07:30 봉: {summer_0730:,}건")

        if summer_0830 > summer_0730 * 5 and winter_0830 > winter_0730 * 5:
            print(f"    → ✅ 여름/겨울 모두 08:30 시작 → CT 기준 (America/Chicago 권장)")
            print(f"       python prepare_nasdaq.py {'{input}'} --tz America/Chicago")
        elif summer_0730 > summer_0830 * 5:
            print(f"    → ✅ 여름에 07:30 시작 → 고정 UTC-6 (현재 설정 적합)")
        else:
            print(f"    → ⚠️  판단 불확실 — 수동 확인 필요")

    print(f"{'─'*65}")


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="나스닥 선물 원본 데이터 → 백테스트용 CSV 정제"
    )
    parser.add_argument(
        "input_file",
        help="원본 데이터 파일 (세미콜론 구분, DD/MM/YYYY;HH:MM;O;H;L;C;V)"
    )
    parser.add_argument(
        "--tz", default="fixed",
        help="시간대 모드: 'fixed' (고정 GMT-6) 또는 'America/Chicago' (DST 자동). 기본값: fixed"
    )
    parser.add_argument(
        "--out-5m", default="nasdaq_nqf_5m.csv",
        help="5분봉 출력 파일명. 기본값: nasdaq_nqf_5m.csv"
    )
    parser.add_argument(
        "--out-1d", default="nasdaq_nqf_1d.csv",
        help="일봉 출력 파일명. 기본값: nasdaq_nqf_1d.csv"
    )

    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.input_file}")
        sys.exit(1)

    # ── 1. 로드
    df = load_raw(args.input_file)

    # ── 2. 시간대 변환
    df = convert_timezone(df, tz_mode=args.tz)

    # ── 3. 5분봉 리샘플
    df_5m = resample_5m(df)

    # ── 4. 일봉 집계
    df_1d = aggregate_daily(df_5m)

    # ── 5. 저장
    df_5m.to_csv(args.out_5m, index=False)
    df_1d.to_csv(args.out_1d, index=False)

    print(f"\n{'═'*65}")
    print(f"  ✅ 정제 완료!")
    print(f"{'═'*65}")
    print(f"  5분봉: {args.out_5m}  ({len(df_5m):,}봉)")
    print(f"  일봉 : {args.out_1d}  ({len(df_1d):,}일)")
    print(f"{'═'*65}")

    # ── 6. 품질 리포트
    quality_report(df_5m, df_1d, df_raw=df)

    # ── 사용 안내
    print(f"\n  📌 사용법:")
    print(f"     이 파일들을 프로젝트 폴더에 복사하세요.")
    print(f"     backtest.py가 자동으로 캐시 파일로 인식합니다.")
    print(f"     (파일명이 nasdaq_nqf_5m.csv / nasdaq_nqf_1d.csv 이면 자동 로드)")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
