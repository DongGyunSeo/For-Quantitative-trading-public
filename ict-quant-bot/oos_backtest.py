"""
oos_backtest.py — Out-of-Sample 백테스트 (Long/Short 분리 모델)
═══════════════════════════════════════════════════════════════
ML 필터 적용 전/후를 비교하여 실전 성능을 검증합니다.

두 가지 모드로 실행:
  1) ML 필터 ON  — entry_model_long.json + entry_model_short.json
  2) ML 필터 OFF — 모든 신호 수용 (baseline)

사용법:
    python oos_backtest.py
"""
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import ccxt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest_oos import fetch_historical_5m

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
MONTHS = 12
END_DATE = "2025-05-04"

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "TRX/USDT:USDT",
    "TON/USDT:USDT",
]

INITIAL_EQUITY = 1000.0
PRINT_EVERY = 500
ENTRY_MODEL = "entry_model.json"   # Backtester가 _long/_short 자동 감지
TP_MODEL = "tp_model.json"
MAX_WORKERS = 6
# ══════════════════════════════════════════════════════════


def _calc_pnl_r(t):
    """단일 Trade → R-multiple 계산."""
    ep = float(t.entry_price) if t.entry_price else 0
    sl = float(t.initial_sl) if t.initial_sl else float(t.sl)
    risk = abs(ep - sl)
    if risk > 0 and ep > 0:
        xp = float(t.exit_price or ep)
        r = (xp - ep) / risk if t.direction == "long" else (ep - xp) / risk
        return r
    return 0.0


def _calc_peak_r(t):
    """단일 Trade → peak R-multiple 계산."""
    ep = float(t.entry_price) if t.entry_price else 0
    sl = float(t.initial_sl) if t.initial_sl else float(t.sl)
    peak = float(t.peak_px) if t.peak_px else ep
    risk = abs(ep - sl)
    if risk > 0 and ep > 0 and peak > 0:
        if t.direction == "long":
            return (peak - ep) / risk
        else:
            return (ep - peak) / risk
    return 0.0


def _analyze_trades(trades, months):
    """트레이드 리스트 → 상세 분석 dict."""
    n = len(trades)
    if n == 0:
        return {
            "trades": 0, "wr": 0, "total_pnl": 0, "avg_pnl": 0,
            "avg_win": 0, "avg_loss": 0, "avg_r": 0, "avg_peak_r": 0,
            "reasons": {}, "pf": 0, "max_dd": 0,
            "long_trades": 0, "long_wr": 0,
            "short_trades": 0, "short_wr": 0,
        }

    wins = [t for t in trades if t.pnl_usdt > 0]
    losses = [t for t in trades if t.pnl_usdt <= 0]
    total_pnl = sum(t.pnl_usdt for t in trades)

    wr = len(wins) / n * 100
    avg_win = sum(t.pnl_usdt for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_usdt for t in losses) / len(losses) if losses else 0

    pnl_rs = [_calc_pnl_r(t) for t in trades]
    peak_rs = [_calc_peak_r(t) for t in trades]
    avg_r = np.mean(pnl_rs) if pnl_rs else 0
    avg_peak_r = np.mean(peak_rs) if peak_rs else 0

    gross_win = sum(t.pnl_usdt for t in wins)
    gross_loss = abs(sum(t.pnl_usdt for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    reasons = {}
    for t in trades:
        k = t.exit_reason or "unknown"
        reasons[k] = reasons.get(k, 0) + 1

    # 방향별 분석
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    long_wins = sum(1 for t in long_trades if t.pnl_usdt > 0)
    short_wins = sum(1 for t in short_trades if t.pnl_usdt > 0)

    return {
        "trades": n, "wr": wr, "total_pnl": total_pnl,
        "avg_pnl": total_pnl / n,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_r": avg_r, "avg_peak_r": avg_peak_r,
        "pf": pf, "reasons": reasons,
        "long_trades": len(long_trades),
        "long_wr": long_wins / len(long_trades) * 100 if long_trades else 0,
        "long_pnl": sum(t.pnl_usdt for t in long_trades),
        "short_trades": len(short_trades),
        "short_wr": short_wins / len(short_trades) * 100 if short_trades else 0,
        "short_pnl": sum(t.pnl_usdt for t in short_trades),
    }


def _run_one(symbol_full, months, end_date, initial_equity, print_every,
             entry_model, tp_model, use_ml, df_5m, df_spot=None,
             tp_threshold=0.0, tp_debug=False, oos_output_dir=None):
    """
    단일 심볼 백테스트 워커.

    Parameters
    ----------
    tp_threshold : float
        best_tp_score >= 이 값인 setup만 진입. 0이면 미적용.
    tp_debug : bool
        True면 매 entry_plan 시점에 [TP-DBG] 한 줄 출력.
        ProcessPoolExecutor(spawn) 환경에서 환경변수 상속이 불완전할 수 있어
        명시적 인자로 전달.
    oos_output_dir : str or None
        ml_train_*.csv, tp_train_*.csv가 저장될 디렉토리.
        None이면 backtest_oos.py가 새 디렉토리 자동 생성.
    """
    import sys
    # ── stdout/stderr 양쪽으로 시작 로그 강제 출력 (어느 채널이 살아있는지 진단)
    _start_msg = (f"[_run_one] {symbol_full.split('/')[0]} 시작  "
                  f"tp_threshold={tp_threshold}  tp_debug={tp_debug}")
    try:
        sys.stderr.write(_start_msg + "  (via stderr)\n")
        sys.stderr.flush()
    except Exception:
        pass

    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # ── worker 프로세스의 stdout을 line buffered로 강제
    #    (TP_DEBUG=1 시 중간 진단 출력이 즉시 부모 콘솔에 보이도록)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        from nasdaq_features import _setup_ssl_env
        _setup_ssl_env()
    except ImportError:
        pass

    # ── 환경변수를 워커 프로세스에 명시적 설정
    #     (Windows spawn 방식에선 부모 환경변수 상속이 보장되지 않음)
    os.environ["TP_PROB_THRESHOLD"] = str(tp_threshold)
    if tp_debug:
        os.environ["TP_DEBUG"] = "1"
    if oos_output_dir:
        os.environ["OOS_OUTPUT_DIR"] = oos_output_dir

    # stdout 로그
    try:
        print(_start_msg + "  (via stdout)", flush=True)
    except Exception as _e:
        sys.stderr.write(f"[_run_one] stdout print 실패: {_e}\n")
        sys.stderr.flush()

    sym = symbol_full.split("/")[0]
    t0 = time.time()

    try:
        from backtest_oos import Backtester as _BT

        ml_path = entry_model if use_ml else None

        bt = _BT(
            exchange=None,
            symbol=symbol_full,
            months=months,
            initial_equity=initial_equity,
            end_date=end_date,
            ml_model_path=ml_path,
            tp_model_path=tp_model,
            print_every=print_every,
            df_5m=df_5m,
            df_spot=df_spot,
        )

        result = bt.run()
        elapsed = time.time() - t0
        trades = result.trades

        analysis = _analyze_trades(trades, months)
        analysis["symbol"] = sym
        analysis["ok"] = True
        analysis["elapsed"] = elapsed
        analysis["final_equity"] = result.final_equity
        analysis["return_pct"] = result.total_return_pct
        analysis["max_dd"] = result.max_drawdown_pct
        analysis["use_ml"] = use_ml
        analysis["tp_threshold"] = tp_threshold
        return analysis

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"symbol": sym, "ok": False, "error": str(e),
                "elapsed": time.time() - t0, "use_ml": use_ml,
                "tp_threshold": tp_threshold}


def _print_results_table(results, symbols, months, label):
    """결과 테이블 출력."""
    print(f"\n  {'심볼':>6} {'거래':>5} {'승률':>6} {'avg_R':>8} "
          f"{'peak_R':>8} {'총PnL':>12} {'수익률':>8} {'PF':>6} {'MDD':>6} "
          f"{'L건':>4} {'L승':>5} {'S건':>4} {'S승':>5}")
    print(f"  {'─'*100}")

    total_trades = 0
    total_pnl = 0.0
    total_wins = 0

    for sym in symbols:
        r = results.get(sym)
        if not r or not r.get("ok", False):
            err = r.get("error", "N/A") if r else "N/A"
            print(f"  {sym:>6}  ERROR: {err[:50]}")
            continue

        pf = r.get("pf", 0)
        pf_str = f"{pf:.2f}" if pf < 100 else "∞"

        print(f"  {sym:>6} {r['trades']:>5} {r['wr']:>5.1f}% {r['avg_r']:>+7.3f}R "
              f"{r['avg_peak_r']:>+7.3f}R "
              f"{r['total_pnl']:>+12.1f} {r['return_pct']:>+7.1f}% "
              f"{pf_str:>6} {r['max_dd']:>5.1f}% "
              f"{r['long_trades']:>4} {r['long_wr']:>4.0f}% "
              f"{r['short_trades']:>4} {r['short_wr']:>4.0f}%")

        total_trades += r["trades"]
        total_pnl += r["total_pnl"]
        if r["trades"] > 0:
            total_wins += int(r["wr"] / 100 * r["trades"])

    print(f"  {'─'*100}")
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    monthly = total_trades / months if months > 0 else 0
    print(f"  {'합계':>6} {total_trades:>5} {total_wr:>5.1f}% {'':>8} {'':>8} "
          f"{total_pnl:>+12.1f} {'':>8} {'':>6} {'':>6}")
    print(f"  월 거래: {monthly:.1f}건  월 PnL: {total_pnl/months:+,.1f}")

    return total_trades, total_pnl, total_wr


def main():
    try:
        from nasdaq_features import _setup_ssl_env
        _setup_ssl_env()
    except ImportError:
        pass

    exchange = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    sym_names = [s.split("/")[0] for s in SYMBOLS]

    print()
    print("═" * 100)
    print("  🧪 OOS 백테스트 — ML 필터 ON/OFF 비교 (Long/Short 분리 모델)")
    print("═" * 100)
    print(f"  기간       : 최근 {MONTHS}개월 (~{END_DATE})")
    print(f"  심볼       : {', '.join(sym_names)}")
    print(f"  Entry 모델 : {ENTRY_MODEL} (→ _long/_short 자동 감지)")
    print(f"  워커       : {MAX_WORKERS}개")
    print("═" * 100)

    # ── 1단계: 데이터 순차 다운로드 (선물 + 현물)
    print(f"\n  📥 데이터 다운로드 ({len(SYMBOLS)}심볼, 선물+현물 순차)")
    print(f"  {'─'*60}")

    # 현물 exchange 생성
    spot_exchange = None
    try:
        spot_exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
    except Exception as e:
        print(f"  ⚠️ 현물 exchange 생성 실패: {e}")

    all_data = {}      # {sym: df_futures}
    all_spot_data = {}  # {sym: df_spot}

    for i, sym in enumerate(SYMBOLS, 1):
        s = sym.split("/")[0]

        # 선물 다운로드
        print(f"  [{i}/{len(SYMBOLS)}] {s} 선물 다운로드 중...", end="", flush=True)
        t0 = time.time()
        try:
            df = fetch_historical_5m(exchange, sym, MONTHS, end_date=END_DATE)
            all_data[sym] = df
            print(f" ✅ {len(df):,}봉 ({time.time()-t0:.0f}초)")
        except Exception as e:
            print(f" ❌ {e}")
            all_data[sym] = None

        # 현물 다운로드
        if spot_exchange is not None:
            spot_sym = f"{s}/USDT"
            print(f"  [{i}/{len(SYMBOLS)}] {s} 현물 다운로드 중...", end="", flush=True)
            time.sleep(1)  # 선물 직후 rate limit 여유
            t1 = time.time()
            try:
                df_spot = fetch_historical_5m(spot_exchange, spot_sym, MONTHS, end_date=END_DATE)
                if df_spot is not None and len(df_spot) > 0:
                    all_spot_data[sym] = df_spot
                    print(f" ✅ {len(df_spot):,}봉 ({time.time()-t1:.0f}초)")
                else:
                    all_spot_data[sym] = None
                    print(f" ⚠️ 데이터 없음")
            except Exception as e:
                all_spot_data[sym] = None
                print(f" ⚠️ 실패: {e}")

        # 심볼 간 API 부하 방지
        if i < len(SYMBOLS):
            time.sleep(1)

    spot_count = sum(1 for v in all_spot_data.values() if v is not None)
    print(f"\n  📥 다운로드 완료: 선물 {sum(1 for v in all_data.values() if v is not None)}/{len(SYMBOLS)}개, "
          f"현물 {spot_count}/{len(SYMBOLS)}개")

    symbols_ok = [s for s in SYMBOLS if all_data.get(s) is not None]
    if not symbols_ok:
        print("  ❌ 다운로드된 데이터 없음")
        return

    # ── 2단계: 시나리오 백테스트
    #
    # 환경변수 OOS_SCENARIOS로 어느 시나리오를 돌릴지 선택 (콤마 구분).
    # 기본 = "baseline,ml,tp05"
    #
    # 시나리오 정의:
    #   baseline   ML OFF, TP threshold OFF       — 모든 setup 거래 (raw 전략 EV)
    #   ml         ML ON,  TP threshold OFF       — 기존 entry filter (비교 기준)
    #   tp03       ML OFF, TP threshold = 0.3     — 부드러운 TP 필터
    #   tp05       ML OFF, TP threshold = 0.5     — 핵심 검증 (CSV 분석에서 +0.184R)
    #   tp06       ML OFF, TP threshold = 0.6     — 보수적 필터
    #   tp07       ML OFF, TP threshold = 0.7     — 매우 보수적 (CSV에서 +0.320R)
    #   ml_tp05    ML ON + TP threshold = 0.5     — 둘 다 적용 (이중 필터)
    _scenario_str = os.environ.get("OOS_SCENARIOS", "baseline,ml,tp05")
    _scenario_keys = [s.strip().lower() for s in _scenario_str.split(",") if s.strip()]

    _all_scenarios = {
        "baseline": ("BASELINE (ML OFF, TP OFF)",       False, 0.0),
        "ml":       ("ML FILTER ON",                     True,  0.0),
        "tp03":     ("TP-ONLY (threshold=0.3)",          False, 0.3),
        "tp05":     ("TP-ONLY (threshold=0.5)",          False, 0.5),
        "tp06":     ("TP-ONLY (threshold=0.6)",          False, 0.6),
        "tp07":     ("TP-ONLY (threshold=0.7)",          False, 0.7),
        "ml_tp05":  ("ML + TP (둘 다, threshold=0.5)",   True,  0.5),
    }

    # 검증: 알 수 없는 시나리오 키
    # 'debug' 키워드는 환경변수 대신 디버그 활성화 (PowerShell 환경변수 누락 시 폴백)
    _scenario_debug = "debug" in _scenario_keys
    if _scenario_debug:
        _scenario_keys = [k for k in _scenario_keys if k != "debug"]
        print(f"\n  🐛 시나리오에 'debug' 포함 → TP_DEBUG 강제 활성화")

    unknown = [k for k in _scenario_keys if k not in _all_scenarios]
    if unknown:
        print(f"  ⚠️ 알 수 없는 시나리오 키: {unknown}")
        print(f"     사용 가능: {list(_all_scenarios.keys())} + 'debug'")
        _scenario_keys = [k for k in _scenario_keys if k in _all_scenarios]

    if not _scenario_keys:
        print(f"  ⚠️ 실행할 시나리오 없음 — 기본값 사용")
        _scenario_keys = ["baseline", "ml", "tp05"]

    scenarios = [(k, *_all_scenarios[k]) for k in _scenario_keys]
    print(f"\n  📋 실행 시나리오: {[s[1] for s in scenarios]}")

    # TP_DEBUG 활성 시 안내
    _tp_debug_raw = os.environ.get("TP_DEBUG", "<unset>")
    _tp_debug_on = _tp_debug_raw in ("1", "true", "True")
    print(f"\n  🔍 부모 환경변수 진단:")
    print(f"     TP_DEBUG       = {_tp_debug_raw!r}  (활성: {_tp_debug_on})")
    print(f"     OOS_SCENARIOS  = {os.environ.get('OOS_SCENARIOS', '<unset>')!r}")
    if _tp_debug_on:
        print(f"\n  🐛 TP_DEBUG=1 활성")
        print(f"     ℹ️  병렬 워커 출력이 뒤섞일 수 있음. "
              f"MAX_WORKERS={MAX_WORKERS} ← 1로 줄이거나 SYMBOLS=1개 권장")
        print(f"     ℹ️  매 entry_plan 시점에 [TP-DBG] 한 줄 출력")
    else:
        print(f"\n  ℹ️  TP_DEBUG 미활성 — TP-DBG 출력 없음")
        print(f"     ℹ️  활성화: PowerShell에서 다음을 같은 세션에 실행:")
        print(f"        $env:TP_DEBUG = \"1\"")
        print(f"        python oos_backtest.py")
        print(f"     ℹ️  확인 방법:  python -c \"import os; print(os.environ.get('TP_DEBUG'))\"")

    # ── OOS 세션 출력 디렉토리 생성 (모든 결과물의 부모 폴더)
    from datetime import datetime as _dt
    _session_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _session_dir = os.path.join("oos_results", _session_ts)
    os.makedirs(_session_dir, exist_ok=True)
    os.environ["OOS_OUTPUT_DIR"] = _session_dir  # 워커들이 사용 (각 워커가 spawn 시 상속)
    print(f"\n  📁 OOS 세션 디렉토리: {_session_dir}")
    print(f"     (ml_train_*.csv, tp_train_*.csv, 결과 JSON 등이 여기에 저장)")

    all_results = {}  # scenario_key → results dict

    for scenario_key, label, use_ml, tp_threshold in scenarios:
        print(f"\n\n{'▶'*30}")
        print(f"  📌 {label}")
        print(f"     use_ml={use_ml}  tp_threshold={tp_threshold}")
        print(f"{'▶'*30}")

        # 디버그 활성화: 환경변수 OR 시나리오 'debug' 키워드 (PowerShell env 미작동 폴백)
        _parent_tp_debug = _tp_debug_on or _scenario_debug

        results = {}
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for sym in symbols_ok:
                f = executor.submit(
                    _run_one, sym, MONTHS, END_DATE, INITIAL_EQUITY, PRINT_EVERY,
                    ENTRY_MODEL, TP_MODEL, use_ml,
                    all_data[sym], all_spot_data.get(sym),
                    tp_threshold, _parent_tp_debug, _session_dir,
                )
                futures[f] = sym

            done = 0
            for future in as_completed(futures):
                done += 1
                sym_full = futures[future]
                s = sym_full.split("/")[0]
                try:
                    r = future.result()
                    results[r["symbol"]] = r
                    if r["ok"]:
                        print(f"  ✅ [{done}/{len(symbols_ok)}] {r['symbol']} "
                              f"({r['elapsed']:.0f}초) — {r['trades']}건 "
                              f"WR {r['wr']:.1f}% PnL {r['total_pnl']:+,.1f}")
                    else:
                        print(f"  ❌ [{done}/{len(symbols_ok)}] {r['symbol']} 실패: {r['error']}")
                except Exception as e:
                    results[s] = {"ok": False, "error": str(e), "elapsed": 0,
                                  "use_ml": use_ml, "tp_threshold": tp_threshold}
                    print(f"  ❌ [{done}/{len(symbols_ok)}] {s} 오류: {e}")

        all_results[scenario_key] = (label, results)

    # ═══════════════════════════════════════════════════
    # 최종 비교 리포트 (N개 시나리오 일반화)
    # ═══════════════════════════════════════════════════

    print(f"\n\n{'═'*100}")
    print(f"  📊 OOS 결과 비교  |  최근 {MONTHS}개월 (~{END_DATE})")
    print(f"     시나리오: {len(all_results)}개")
    print(f"{'═'*100}")

    # ── 각 시나리오 상세 테이블
    scenario_summary = []  # [(key, label, trades, pnl, wr, results)]
    for scenario_key, (label, results) in all_results.items():
        print(f"\n\n  📋 {label}")
        n_trades, total_pnl, wr = _print_results_table(
            results, sym_names, MONTHS, label)
        scenario_summary.append((scenario_key, label, n_trades, total_pnl, wr, results))

    # ── 시나리오 비교 요약 테이블
    print(f"\n\n{'═'*100}")
    print(f"  🔍 시나리오 비교 요약")
    print(f"{'═'*100}")
    print(f"\n  {'시나리오':<35}  {'거래수':>7}  {'WR':>7}  {'총PnL':>12}  "
          f"{'건당PnL':>10}  {'필터율':>7}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*10}  {'-'*7}")

    # baseline을 비교 기준으로
    base_trades = 0
    if scenario_summary:
        base_key = scenario_summary[0][0]
        base_trades = scenario_summary[0][2]

    for key, label, n_trades, total_pnl, wr, _ in scenario_summary:
        per_trade = total_pnl / n_trades if n_trades > 0 else 0
        filter_rate = ((1 - n_trades / base_trades) * 100
                       if base_trades > 0 and key != scenario_summary[0][0] else 0)
        filter_str = f"{filter_rate:>+5.1f}%" if key != scenario_summary[0][0] else "  base"
        marker = "★" if (per_trade > 0 and n_trades > 50) else " "
        print(f"  {marker} {label:<33}  {n_trades:>7d}  {wr:>6.2f}%  "
              f"{total_pnl:>+12.1f}  {per_trade:>+10.2f}  {filter_str:>7}")

    # ── baseline 대비 각 시나리오의 효율성 분석
    if len(scenario_summary) >= 2 and base_trades > 0:
        print(f"\n  💡 baseline 대비 효율성:")
        base_pnl = scenario_summary[0][3]
        for key, label, n_trades, total_pnl, _, _ in scenario_summary[1:]:
            if n_trades == 0:
                continue
            filter_rate = (1 - n_trades / base_trades) * 100
            pnl_delta = total_pnl - base_pnl
            # 거래당 PnL 개선
            base_per = base_pnl / base_trades if base_trades > 0 else 0
            this_per = total_pnl / n_trades
            per_improve = this_per - base_per
            verdict = ("✅ 개선" if per_improve > 0 and total_pnl > 0
                       else ("🟡 거래만 줄음" if filter_rate > 30 else "❌ 악화"))
            print(f"     {label:<35}: 거래 -{filter_rate:.1f}%, "
                  f"건당PnL {base_per:+.2f}→{this_per:+.2f} ({per_improve:+.2f})  {verdict}")

    # ── 방향별 비교 (모든 시나리오)
    print(f"\n  📊 방향별 비교 (시나리오 × 방향):")
    print(f"  {'시나리오':<35}  {'LONG건':>6}  {'L_WR':>6}  {'L_PnL':>10}  "
          f"{'SHORT건':>7}  {'S_WR':>6}  {'S_PnL':>10}")
    for key, label, _, _, _, results in scenario_summary:
        long_t = sum(results.get(s, {}).get("long_trades", 0) for s in sym_names)
        long_wr_sum = sum(
            results.get(s, {}).get("long_wr", 0) * results.get(s, {}).get("long_trades", 0)
            for s in sym_names)
        long_wr = long_wr_sum / long_t if long_t > 0 else 0
        long_pnl = sum(results.get(s, {}).get("long_pnl", 0) for s in sym_names)

        short_t = sum(results.get(s, {}).get("short_trades", 0) for s in sym_names)
        short_wr_sum = sum(
            results.get(s, {}).get("short_wr", 0) * results.get(s, {}).get("short_trades", 0)
            for s in sym_names)
        short_wr = short_wr_sum / short_t if short_t > 0 else 0
        short_pnl = sum(results.get(s, {}).get("short_pnl", 0) for s in sym_names)

        print(f"  {label:<35}  {long_t:>6d}  {long_wr:>5.1f}%  {long_pnl:>+10.1f}  "
              f"{short_t:>7d}  {short_wr:>5.1f}%  {short_pnl:>+10.1f}")

    # ── 심볼별 시나리오 비교 (각 심볼이 어느 시나리오에서 가장 좋은가)
    if len(scenario_summary) >= 2:
        print(f"\n  📊 심볼별 최적 시나리오 (총 PnL 기준):")
        print(f"  {'심볼':>6}  {'최적 시나리오':<35}  {'PnL':>+10}  {'WR':>6}  {'건수':>5}")
        for sym in sym_names:
            best_key, best_label, best_pnl, best_wr, best_n = None, None, -float("inf"), 0, 0
            for key, label, _, _, _, results in scenario_summary:
                r = results.get(sym, {})
                if not r.get("ok"):
                    continue
                if r["total_pnl"] > best_pnl:
                    best_pnl = r["total_pnl"]
                    best_label = label
                    best_wr = r["wr"]
                    best_n = r["trades"]
            if best_label is not None:
                mark = "✅" if best_pnl > 0 else "❌"
                print(f"  {mark}{sym:>5}  {best_label:<35}  {best_pnl:>+10.1f}  "
                      f"{best_wr:>5.1f}%  {best_n:>5d}")

    print(f"\n{'═'*100}")

    # ═══════════════════════════════════════════════════
    # OOS 결과 영구 저장 (JSON + CSV)
    # ═══════════════════════════════════════════════════
    import json
    _summary_rows = []
    _full_results = {
        "session_ts": _session_ts,
        "months": MONTHS,
        "end_date": END_DATE,
        "symbols": [s.split("/")[0] for s in symbols_ok],
        "initial_equity": INITIAL_EQUITY,
        "entry_model": ENTRY_MODEL,
        "tp_model": TP_MODEL,
        "scenarios": {},
    }

    for scenario_key, (label, results) in all_results.items():
        # 요약 stats
        _trades = sum(r.get("trades", 0) for r in results.values() if r.get("ok"))
        _pnl = sum(r.get("total_pnl", 0) for r in results.values() if r.get("ok"))
        _wr_num = sum(r.get("wr", 0) * r.get("trades", 0)
                      for r in results.values() if r.get("ok"))
        _wr = _wr_num / _trades if _trades > 0 else 0
        _per = _pnl / _trades if _trades > 0 else 0

        _full_results["scenarios"][scenario_key] = {
            "label": label,
            "total_trades": _trades,
            "total_pnl": round(_pnl, 2),
            "avg_wr": round(_wr, 2),
            "pnl_per_trade": round(_per, 3),
            "by_symbol": {
                sym: {k: v for k, v in r.items()
                      if k not in ("symbol",)}
                for sym, r in results.items() if r.get("ok")
            },
        }

        # 심볼별 행
        for sym, r in results.items():
            if not r.get("ok"):
                continue
            _summary_rows.append({
                "scenario": scenario_key,
                "label": label,
                "symbol": sym,
                "trades": r.get("trades", 0),
                "wr": round(r.get("wr", 0), 2),
                "total_pnl": round(r.get("total_pnl", 0), 2),
                "long_trades": r.get("long_trades", 0),
                "long_wr": round(r.get("long_wr", 0), 2),
                "long_pnl": round(r.get("long_pnl", 0), 2),
                "short_trades": r.get("short_trades", 0),
                "short_wr": round(r.get("short_wr", 0), 2),
                "short_pnl": round(r.get("short_pnl", 0), 2),
                "final_equity": round(r.get("final_equity", 0), 2),
                "return_pct": round(r.get("return_pct", 0), 2),
                "max_dd": round(r.get("max_dd", 0), 2),
            })

    # JSON 저장 (전체 raw 결과)
    _json_path = os.path.join(_session_dir, "results.json")
    try:
        with open(_json_path, "w", encoding="utf-8") as fp:
            json.dump(_full_results, fp, indent=2, ensure_ascii=False, default=str)
        print(f"  💾 결과 JSON 저장: {_json_path}")
    except Exception as e:
        print(f"  ⚠️ JSON 저장 실패: {e}")

    # CSV 저장 (요약 테이블 — 엑셀로 보기 쉬움)
    _csv_path = os.path.join(_session_dir, "summary.csv")
    try:
        import pandas as _pd
        _df_summary = _pd.DataFrame(_summary_rows)
        _df_summary.to_csv(_csv_path, index=False, encoding="utf-8-sig")
        print(f"  💾 요약 CSV 저장: {_csv_path}")
    except Exception as e:
        print(f"  ⚠️ CSV 저장 실패: {e}")

    print(f"\n  📁 모든 결과: {_session_dir}/")
    print(f"     - results.json      (전체 raw 결과)")
    print(f"     - summary.csv       (심볼×시나리오 요약 테이블)")
    print(f"     - ml_train_*.csv    (ML 학습 데이터, 심볼별)")
    print(f"     - tp_train_*.csv    (TP 학습 데이터, 심볼별)")
    print(f"\n{'═'*100}")


if __name__ == "__main__":
    total_start = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n  🔥 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"\n  ⏱️ 총 소요: {time.time()-total_start:.0f}초")
