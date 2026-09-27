# ICT Quant Bot (아카이브)

> **상태: 중단 (2026-06 종료)** — 이 폴더의 내용이 프로젝트의 최종본입니다. 더 이상 수정하지 않습니다.
> 최종 시점은 git 태그 `ict-quant-bot-final` 로도 고정되어 있습니다.

ICT(Inner Circle Trader) 개념(유동성 스윕, FVG, 오더블록, BOS/CHoCH, Dealing Range)을 규칙 기반으로 구현하고,
그 위에 XGBoost 진입 필터를 얹어 코인 선물 5분봉에서 검증한 개인 연구 프로젝트입니다.
실험 과정은 [`iterations/`](iterations/) 에 001~048 로 기록되어 있습니다.

## 폴더 구성

| 구분 | 파일 |
|---|---|
| ICT 구조 감지 | `swings.py`, `pivots.py`, `struct_event.py`, `fvg.py`, `fvg_lifecycle.py`, `order_blocks.py`, `pd_zones.py`, `liquidity.py`, `volume_profile.py`, `timeframes.py` |
| 신호/사이징 | `signals.py`, `scoring.py` |
| 백테스트 | `backtest.py` (학습 데이터 수집용), `backtest_oos.py`, `oos_backtest.py` (OOS 검증) |
| 데이터 수집/가공 | `collect_ml_data.py`, `prepare_entry_data.py`, `prepare_entry_data_notp.py`, `augment_features.py`, `prepare_nasdaq.py`, `nasdaq_features.py`, `liquidation_heatmap_features.py` |
| ML | `ml_data.py`, `ml_preprocess.py`, `ml_model.py`, `train_entry.py`, `train_runner.py`, `ml_evaluate.py`, `ml_univariate_analysis.py` |
| 분석 스크립트 | `analyze_shadow.py`, `verify_and_stage0.py`, `stage6_swing_trail.py` |
| 분석 결과표 | `stage6_summary.csv`, `vp_topology_univariate.csv` |
| 연구 노트 | `iterations/` |

## 저장소에 없는 파일

용량 문제로 아래 파일은 git에 올리지 않습니다(`.gitignore`). 실행하려면 이 폴더 안에 직접 두거나 스크립트로 다시 만들어야 합니다.

- `cache_5m/`, `cache_5m_oos/` — 바이낸스 5분봉 캐시 (`collect_ml_data.py` 실행 시 생성)
- `backtest_result/`, `shadow/`, `ml_train_*.csv`, `tp_train_*.csv` — 학습 데이터셋
- `entry_model_*.json`, `*.pkl` — 학습된 모델 (`train_entry.py` 로 생성)
- `nasdaq_raw.csv`, `nasdaq_nqf_*.csv` — 나스닥 선물 데이터 (`prepare_nasdaq.py` 로 정제)

스크립트는 같은 폴더의 모듈을 import 하고 데이터도 현재 폴더 기준으로 읽으므로, 이 폴더에서 실행합니다.

```bash
cd ict-quant-bot
python collect_ml_data.py
```

## 프로젝트를 시작한 이유

2025-12-29: 1150 USDT
2026-02-06: 245 USDT (-88.7%)

주식에 관심을 갖기 시작한 지 약 2년이 되었다.
기업의 밸류에이션이나 장기적인 전망보다는, 주로 기술적 분석을 기반으로 한 스윙 트레이딩과 스캘핑 위주의 매매를 해왔다.

초기에는 대부분이 그렇듯, 명확한 기준 없이 ‘싸 보이면 사고, 비싸 보이면 파는’ 뇌동매매를 반복했다.
이후 약 6개월 차부터 RSI, MACD-OSC, 볼린저밴드 등 기본적인 기술적 지표를 공부하며 매매에 적용했고, 일정 수준의 성과를 거둘 수 있었다.

점차 VWAP, VRVP, 스토캐스틱 등 다양한 지표를 활용하면서 자신감이 생겼고, 그 결과 코인 선물 거래에도 진입하게 되었다.
현재는 거시경제 브리핑과 관점 공유를 제공하는 거래소 래퍼럴 커뮤니티를 활용하며 정보를 참고하고 있다.

그러나 코인 시장은 주식과는 차원이 다른 변동성을 가지고 있었다.
이상적인 진입 자리는 쉽게 주어지지 않았고, 진입 후 판단이 늦어질 때쯤이면 이미 반대 방향으로 스윕이 발생하는 경우가 반복되었다.
계좌의 지속적인 하락을 경험하면서, 단순한 지표 매매만으로는 한계가 있다는 점을 체감하게 되었다.

이 과정에서 시장의 구조와 유동성을 설명하는 ICT(Inner Circle Trader) 매매 기법을 접하게 되었고, 비교적 논리적인 접근 방식이라고 판단하여 본격적으로 공부하기 시작했다.
하지만 ICT를 학습했다고 해서 곧바로 성과가 개선되지는 않았고, 선물 계좌의 상당 부분을 잃으며 ‘기법을 가장한 감정 매매’의 문제를 다시 인식하게 되었다.

이러한 경험을 계기로, 감정을 배제한 시스템 기반 매매의 필요성을 느끼게 되었고, ICT 개념을 기반으로 한 퀀트 트레이딩 알고리즘 개발을 목표로 삼게 되었다.

현재는 자료구조와 기본적인 프로그래밍 역량을 정리하는 단계에 있으며, 장기적인 관점에서 알고리즘을 구축해 나갈 계획이다.
트레이딩과 복기를 병행하며, 실제로 수익이 발생했던 구간과 ICT 구조를 분석하고 이를 점진적으로 코드에 반영해 나갈 예정이다.

본 프로젝트는 이러한 과정을 기록하고, 감정에 의존하지 않는 매매 시스템을 구축하기 위한 개인 연구용 저장소이다.
