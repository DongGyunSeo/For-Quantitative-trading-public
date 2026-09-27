#실행하기전
#C:/Users/서동균/AppData/Roaming/uv/python/cpython-3.14.3-windows-x86_64-none/python.exe -m pip install ccxt pandas --break-system-packages 
#터미널 입력

import os
import sys

# Windows 환경에서 터미널 출력 시 UnicodeEncodeError(이모지 등) 방지
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import ccxt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
import os as _os_cfg


def _collect_out_dir() -> str:
    """수집 결과 CSV 폴더 = 프로젝트 폴더 내 collect_result/ (backtest._collect_out_dir와 동일 경로).
    이 파일(__file__) 기준 고정. .gitignore 대상이라 git pull 시 삭제 방지."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_result")
    os.makedirs(d, exist_ok=True)
    return d


# ╔══════════════════════════════════════════════════════════════╗
# ║  ★★★ 여기서 직접 설정 (PowerShell 환경변수 불필요) ★★★        ║
# ╠══════════════════════════════════════════════════════════════╣

# ── shadow trail 비교 모드 ─────────────────────────────────────
#   True  : 각 trade에 "trail ON이었다면 어디서 청산됐을지"를 동시 기록
#           → ml_train_*_shadow.csv 생성 (shadow_pnl_r 컬럼 포함)
#           → 기존 ml_train_*.csv 보존
#   False : 일반 수집 (기존 동작)
SHADOW_TRAIL = False          # ← shadow 비교하려면 True

SHADOW_TIMEOUT_BARS = 144     # primary/shadow 공유 timeout (144봉 = 12시간)

# ── 테스트용 기간 일괄 변경 ────────────────────────────────────
#   0    : SYMBOLS의 개별 months 사용 (정상 수집)
#   1~   : 모든 심볼을 이 개월 수로 통일 (빠른 검증용, 1이면 1개월만)
MONTHS_OVERRIDE = 0            # ← shadow 검증은 1로, 본격 수집은 0으로

# ── primary trail 모드 ─────────────────────────────────────────
#   "off" : trail/BE 비활성 (학습 데이터 raw EV)
#   "on"  : trail/BE 활성 (실거래 환경)
#   shadow 비교 시엔 "off" 유지 (primary=off, shadow=on 비교)
TRAIL_MODE = "off"

# ── 수집 종료일 (고정값 — 캐시 커버리지 안정화) ────────────────
END_DATE = "2025-05-04"

# ╚══════════════════════════════════════════════════════════════╝
# (아래는 환경변수가 있으면 코드 변수보다 우선 — 보통은 무시해도 됨)
if _os_cfg.environ.get("ML_END_DATE"):
    END_DATE = _os_cfg.environ["ML_END_DATE"]
if _os_cfg.environ.get("ML_MONTHS_OVERRIDE"):
    MONTHS_OVERRIDE = int(_os_cfg.environ["ML_MONTHS_OVERRIDE"])
if _os_cfg.environ.get("SHADOW_TRAIL"):
    SHADOW_TRAIL = _os_cfg.environ["SHADOW_TRAIL"] in ("1", "true", "True")

# ★ 백테스트 모듈 로드 전 환경변수 동기화
if SHADOW_TRAIL:
    _os_cfg.environ["SHADOW_TRAIL"] = "1"
else:
    _os_cfg.environ["SHADOW_TRAIL"] = "0"
_os_cfg.environ["SHADOW_TIMEOUT_BARS"] = str(SHADOW_TIMEOUT_BARS)
_os_cfg.environ["TRAIL_MODE"] = TRAIL_MODE

# 초기 자산 (USDT) — ML 데이터 수집용이라 결과에 큰 영향 없음
INITIAL_EQUITY = 1000.0

from backtest import Backtester, fetch_historical_5m

# 수집할 심볼 목록 + 코인별 학습 기간 (개월)
# 코인별 상장 시기/데이터 가용성에 맞춰 개별 설정
SYMBOLS = [
    ("BTC/USDT:USDT",  50),   
    ("ETH/USDT:USDT",  50),   
    ("SOL/USDT:USDT",  50),   
    ("XRP/USDT:USDT",  48),   
    ("AVAX/USDT:USDT", 40),   
    ("DOGE/USDT:USDT", 48),   
    ("ADA/USDT:USDT",  48),   
    ("TRX/USDT:USDT",  48),   
    ("BNB/USDT:USDT",  24),   
    ("TON/USDT:USDT",  26),   
    ("LINK/USDT:USDT", 48),
]

if MONTHS_OVERRIDE > 0:
    SYMBOLS = [(sym, MONTHS_OVERRIDE) for sym, _ in SYMBOLS]
    print(f"[collect] ⚠️ MONTHS_OVERRIDE={MONTHS_OVERRIDE} → 모든 심볼 {MONTHS_OVERRIDE}개월로 통일 (테스트 모드)")

# 진행 출력 주기  (봉 수)
PRINT_EVERY = 400

# 이미 생성된 CSV 파일이 있으면 건너뛸지 여부 (False면 덮어씌움)
SKIP_EXISTING = False

# ── 병렬 실행 설정 ──
# True: 데이터 다운로드는 순차, 백테스트 계산만 병렬 실행
# False: 기존 순차 실행
PARALLEL = True

# 병렬 워커 수 (기본: CPU 코어 - 1, 최대 심볼 수로 제한)
# 메모리 주의: 워커당 약 200~500MB 사용 (나스닥 데이터 + OHLCV 데이터)
MAX_WORKERS = 11  # 10심볼 전부 병렬 (RAM 16GB+ 권장)

# ══════════════════════════════════════════════════════════════


def _download_all_data(exchange, symbols, end_date):
    """
    모든 심볼의 OHLCV 데이터를 순차적으로 다운로드.
    선물 + 현물 모두 순차 처리 (API Rate Limit 방지).

    Returns
    -------
    dict : {symbol_full: {"futures": pd.DataFrame, "spot": pd.DataFrame}}
    """
    print(f"\n  📥 데이터 다운로드 시작 ({len(symbols)}개 심볼, 선물+현물 순차 처리)")
    print(f"  {'─'*55}")

    # 현물 exchange 생성
    spot_exchange = None
    try:
        import ccxt as _ccxt
        spot_exchange = _ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
    except Exception as e:
        print(f"  ⚠️ 현물 exchange 생성 실패: {e}")

    data = {}
    for i, (sym, months) in enumerate(symbols, 1):
        sym_short = sym.split("/")[0]

        # 선물 다운로드
        print(f"\n  [{i}/{len(symbols)}] {sym_short} 선물 다운로드 중... ({months}개월)")
        t0 = time.time()
        df_fut = None
        try:
            df_fut = fetch_historical_5m(exchange, sym, months, end_date=end_date)
            print(f"  ✅ {sym_short} 선물: {len(df_fut):,}봉 ({time.time()-t0:.0f}초)")
        except Exception as e:
            print(f"  ❌ {sym_short} 선물 다운로드 실패: {e}")

        # 현물 다운로드
        df_spot = None
        if spot_exchange is not None:
            spot_sym = f"{sym_short}/USDT"
            print(f"  [{i}/{len(symbols)}] {sym_short} 현물 다운로드 중...")
            time.sleep(1)  # 선물 직후 rate limit 여유
            t1 = time.time()
            try:
                df_spot = fetch_historical_5m(spot_exchange, spot_sym, months, end_date=end_date)
                if df_spot is not None and len(df_spot) > 0:
                    print(f"  ✅ {sym_short} 현물: {len(df_spot):,}봉 ({time.time()-t1:.0f}초)")
                else:
                    df_spot = None
            except Exception as e:
                print(f"  ⚠️ {sym_short} 현물 다운로드 실패: {e}")

        data[sym] = {"futures": df_fut, "spot": df_spot}

        # 심볼 간 API 부하 방지
        if i < len(symbols):
            time.sleep(1)

    ok_count = sum(1 for v in data.values() if v.get("futures") is not None)
    spot_count = sum(1 for v in data.values() if v.get("spot") is not None)
    print(f"\n  📥 다운로드 완료: 선물 {ok_count}/{len(symbols)}개, 현물 {spot_count}/{len(symbols)}개")
    return data


def _run_one_symbol(symbol_full, months, end_date, initial_equity, print_every,
                    df_5m, df_spot=None, shadow_trail=False,
                    shadow_timeout_bars=144, trail_mode="off"):
    """
    단일 심볼 백테스트 워커 (멀티프로세싱용).

    데이터는 이미 다운로드된 상태로 전달받으므로 API 호출 없음.

    shadow_trail / shadow_timeout_bars / trail_mode:
        Windows spawn 워커는 부모의 환경변수($env:SHADOW_TRAIL 등)를
        상속하지 않으므로, 명시적 인자로 받아 backtest import 전에 os.environ에 설정.
    """
    # Windows spawn 방식에서 encoding 재설정
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # 프로젝트 디렉토리를 sys.path에 추가 (spawn 워커에서 모듈 import 보장)
    import os
    _project_dir = os.path.dirname(os.path.abspath(__file__))
    if _project_dir not in sys.path:
        sys.path.insert(0, _project_dir)

    # ── ★ backtest import 전에 환경변수 설정 (import 시점에 _SHADOW_TRAIL 결정됨)
    #     Windows spawn 워커는 부모 환경변수를 상속 못 하므로 여기서 명시적 설정
    if shadow_trail:
        os.environ["SHADOW_TRAIL"] = "1"
    os.environ["SHADOW_TIMEOUT_BARS"] = str(shadow_timeout_bars)
    os.environ["TRAIL_MODE"] = trail_mode

    # SSL 설정 (나스닥 CSV 로드용)
    try:
        from nasdaq_features import _setup_ssl_env
        _setup_ssl_env()
    except ImportError:
        pass

    sym_short = symbol_full.split("/")[0]
    # shadow 모드에선 파일명에 suffix (기존 학습 데이터 보존)
    _suffix = "_shadow" if shadow_trail else ""
    csv_path = os.path.join(_collect_out_dir(), f"ml_train_{sym_short.lower()}{_suffix}.csv")

    t0 = time.time()

    try:
        import backtest
        import importlib
        importlib.reload(backtest)
        from backtest import Backtester as _BT
        bt = _BT(
            exchange=None,           # API 호출 안 함 (데이터 사전 로드됨)
            symbol=symbol_full,
            months=months,
            initial_equity=initial_equity,
            end_date=end_date,
            ml_model_path=None,
            print_every=print_every,
            df_5m=df_5m,             # ← 사전 다운로드된 선물 데이터
            df_spot=df_spot,         # ← 사전 다운로드된 현물 데이터
        )

        result = bt.run()
        elapsed = time.time() - t0

        n_trades = len(result.trades)
        wr       = result.win_rate

        # 생성된 CSV 행 수 확인
        n_csv = 0
        if Path(csv_path).exists():
            n_csv = len(pd.read_csv(csv_path))

        return {
            "symbol": sym_short,
            "ok": True,
            "skipped": False,
            "trades": n_trades,
            "wr": wr,
            "csv_rows": n_csv,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "symbol": sym_short,
            "ok": False,
            "error": str(e),
            "elapsed": elapsed,
        }


def _run_sequential(exchange, symbols, results):
    """기존 순차 실행 모드."""
    for symbol_full, months in symbols:
        sym_short = symbol_full.split("/")[0]
        csv_path  = os.path.join(_collect_out_dir(), f"ml_train_{sym_short.lower()}.csv")

        print(f"\n{'─'*65}")
        print(f"  [{sym_short}] 백테스트 및 데이터 추출 시작... ({months}개월)")
        print(f"{'─'*65}")
        t0 = time.time()

        try:
            bt = Backtester(
                exchange=exchange,
                symbol=symbol_full,
                months=months,
                initial_equity=INITIAL_EQUITY,
                end_date=END_DATE,
                ml_model_path=None,
                print_every=PRINT_EVERY,
            )

            result = bt.run()
            elapsed = time.time() - t0

            n_trades = len(result.trades)
            wr       = result.win_rate

            n_csv = 0
            if Path(csv_path).exists():
                n_csv = len(pd.read_csv(csv_path))

            results[sym_short] = {
                "ok": True,
                "skipped": False,
                "trades": n_trades,
                "wr": wr,
                "csv_rows": n_csv,
                "elapsed": elapsed,
            }
            print(f"\n  ✅ {sym_short} 완료 ({elapsed:.0f}초)")
            print(f"     거래량: {n_trades}건  |  승률: {wr:.1f}%  |  학습데이터: {n_csv}행")

        except Exception as e:
            elapsed = time.time() - t0
            results[sym_short] = {"ok": False, "error": str(e), "elapsed": elapsed}
            print(f"\n  ❌ {sym_short} 시뮬레이션 중 오류 발생: {e}")

        # API 부하 방지용 짧은 대기
        if (symbol_full, months) != symbols[-1]:
            time.sleep(2)


def _run_parallel(exchange, symbols, results, max_workers):
    """
    병렬 실행 모드.

    1단계: 데이터 다운로드 (순차 — API Rate Limit 방지)
    2단계: 백테스트 계산 (병렬 — CPU 코어 활용)
    """
    n_workers = min(max_workers, len(symbols))
    print(f"\n  🔀 병렬 실행 모드: {len(symbols)}개 심볼 × {n_workers}개 워커 (CPU {os.cpu_count()}코어)")
    print(f"  ⚠️  워커당 메모리 약 200~500MB 사용 (총 {n_workers * 350:.0f}MB 예상)")

    # ── 1단계: 데이터 순차 다운로드 (API Rate Limit 방지)
    all_data = _download_all_data(exchange, symbols, END_DATE)

    # 다운로드 실패 심볼 처리
    symbols_ok = [(sym, m) for sym, m in symbols if all_data.get(sym, {}).get("futures") is not None]
    for sym, m in symbols:
        if all_data.get(sym, {}).get("futures") is None:
            sym_short = sym.split("/")[0]
            results[sym_short] = {"ok": False, "error": "데이터 다운로드 실패", "elapsed": 0}

    if not symbols_ok:
        print("  ❌ 다운로드된 데이터 없음 — 중단")
        return

    # ── 2단계: 백테스트만 병렬 실행 (CPU-intensive 부분)
    print(f"\n  🚀 백테스트 병렬 시작 ({len(symbols_ok)}개 심볼, {n_workers}개 워커)")
    print(f"  {'─'*55}")

    # 모듈 전역 설정 변수를 워커에 명시적 전달 (spawn 상속 안 되므로 인자로)
    _shadow_trail = SHADOW_TRAIL
    _shadow_timeout = SHADOW_TIMEOUT_BARS
    _trail_mode = TRAIL_MODE
    if _shadow_trail:
        print(f"  🔧 SHADOW_TRAIL=True → 워커에 전달  "
              f"(timeout={_shadow_timeout}봉, primary trail_mode={_trail_mode})")
        print(f"     출력: ml_train_*_shadow.csv (기존 학습 데이터 보존)")
    else:
        print(f"  ℹ️  SHADOW_TRAIL=False (일반 수집, shadow 컬럼 없음)")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for sym, months in symbols_ok:
            future = executor.submit(
                _run_one_symbol,
                sym, months, END_DATE, INITIAL_EQUITY, PRINT_EVERY,
                all_data[sym]["futures"],
                all_data[sym].get("spot"),
                _shadow_trail, _shadow_timeout, _trail_mode,
            )
            futures[future] = sym

        # 완료되는 순서대로 결과 수집
        completed = 0
        for future in as_completed(futures):
            completed += 1
            sym_full = futures[future]
            sym_short = sym_full.split("/")[0]

            try:
                r = future.result()
                results[r["symbol"]] = r
                if r["ok"]:
                    print(f"\n  ✅ [{completed}/{len(symbols_ok)}] {r['symbol']} 완료 "
                          f"({r['elapsed']:.0f}초) — {r['trades']}건, "
                          f"WR {r['wr']:.1f}%, CSV {r['csv_rows']}행")
                else:
                    print(f"\n  ❌ [{completed}/{len(symbols_ok)}] {r['symbol']} 실패: "
                          f"{r.get('error', '알 수 없음')}")
            except Exception as e:
                results[sym_short] = {"ok": False, "error": str(e), "elapsed": 0}
                print(f"\n  ❌ [{completed}/{len(symbols_ok)}] {sym_short} 워커 오류: {e}")


def main():
    # SSL/Certifi 환경 설정 (나스닥 데이터 수집용)
    try:
        from nasdaq_features import _setup_ssl_env
        _setup_ssl_env()
    except ImportError:
        pass

    # OKX 공개 API
    exchange = ccxt.okx({
        "enableRateLimit": True,
        "options":         {"defaultType": "swap"},
    })

    print()
    print("═" * 65)
    print("  🚀 ML 학습 데이터 수집 파이프라인")
    print("═" * 65)
    print(f"  대상 심볼 : {len(SYMBOLS)}개")
    for sym, m in SYMBOLS:
        print(f"    {sym.split('/')[0]:6s} : {m}개월")
    print(f"  수집 종료일: {END_DATE}")
    print(f"  중복 스킵 : {'ON' if SKIP_EXISTING else 'OFF'}")
    print(f"  ML 필터   : OFF (전체 신호 기록 모드)")
    print(f"  실행 모드 : {'🔀 병렬 (' + str(MAX_WORKERS) + '워커)' if PARALLEL else '순차'}")
    print(f"  ─────────────────────────────────")
    if SHADOW_TRAIL:
        print(f"  🔧 SHADOW_TRAIL : ✅ ON  → ml_train_*_shadow.csv (shadow_pnl_r 포함)")
        print(f"     공유 timeout : {SHADOW_TIMEOUT_BARS}봉")
        print(f"     primary trail: {TRAIL_MODE}")
    else:
        print(f"  🔧 SHADOW_TRAIL : ❌ OFF (일반 수집)")
    if MONTHS_OVERRIDE > 0:
        print(f"  ⚠️  MONTHS_OVERRIDE={MONTHS_OVERRIDE} (테스트 모드, 모든 심볼 {MONTHS_OVERRIDE}개월)")
    print("═" * 65)
    print()

    results = {}
    total_start = time.time()

    # 스킵할 심볼 분리
    symbols_to_run = []
    for symbol_full, months in SYMBOLS:
        sym_short = symbol_full.split("/")[0]
        csv_path  = os.path.join(_collect_out_dir(), f"ml_train_{sym_short.lower()}.csv")

        if SKIP_EXISTING and Path(csv_path).exists():
            print(f"  ⏭️  [{sym_short}] 이미 데이터가 존재하여 건너뜁니다: {csv_path}")
            try:
                n_csv = len(pd.read_csv(csv_path))
                results[sym_short] = {"ok": True, "skipped": True, "csv_rows": n_csv}
            except:
                results[sym_short] = {"ok": False, "error": "CSV 읽기 실패"}
            continue
        symbols_to_run.append((symbol_full, months))

    # 실행
    if symbols_to_run:
        if PARALLEL and len(symbols_to_run) > 1:
            _run_parallel(exchange, symbols_to_run, results, MAX_WORKERS)
        else:
            if PARALLEL and len(symbols_to_run) == 1:
                print("  ℹ️  심볼 1개 → 순차 실행으로 전환")
            _run_sequential(exchange, symbols_to_run, results)

    # ── 최종 리포트
    total_elapsed = time.time() - total_start
    print()
    print("═" * 65)
    print(f"  📊 수집 작업 완료 리포트 (총 소요: {total_elapsed/60:.1f}분)")
    print("═" * 65)

    total_csv = 0
    for sym, r in results.items():
        if r["ok"]:
            status = "SKIPPED" if r.get("skipped") else "SUCCESS"
            trades = r.get("trades", 0)
            wr = r.get("wr", 0)
            elapsed_s = r.get("elapsed", 0)
            print(f"  [{status:7s}] {sym:6s} | {trades:4d} TRX | WR {wr:4.1f}% | "
                  f"CSV {r['csv_rows']:5d} rows | {elapsed_s:.0f}s")
            total_csv += r["csv_rows"]
        else:
            print(f"  [ERROR  ] {sym:6s} | 원인: {r['error'][:45]}")

    print(f"{'─'*65}")
    print(f"  📂 생성된 학습 데이터 총합: {total_csv:,} 건")

    if PARALLEL and len(symbols_to_run) > 1:
        # 병렬 효율 통계
        _elapsed_sum = sum(r.get("elapsed", 0) for r in results.values()
                          if not r.get("skipped"))
        if total_elapsed > 0 and _elapsed_sum > 0:
            speedup = _elapsed_sum / total_elapsed
            print(f"  ⚡ 병렬 가속: 총 {_elapsed_sum/60:.1f}분 작업을 "
                  f"{total_elapsed/60:.1f}분에 완료 (×{speedup:.1f} 속도 향상)")
    print(f"{'─'*65}")

    if total_csv > 0:
        print()
        print("  🚀 다음 단계:")
        print("     python notebook.py  (원클릭: TP학습 → 결합 → Entry학습)")
        print("     또는 수동:")
        print("       1) python prepare_runner_data_v2.py  # (선택) 또는 runner_train_*.csv 직접 사용")
        print("       2) python prepare_entry_data.py")
        print("       3) python train_entry.py --csv ml_train_with_tp.csv")
    else:
        print("\n  ⚠️ 유효한 학습 데이터가 수집되지 않았습니다. 심볼 설정을 확인해 주세요.")
    print("═" * 65)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n  🔥 치명적 오류 발생: {e}")
        sys.exit(1)