---
idx: 1
title: "백테스트 코어 설계 메모 (v0.1.0)"
created: "2026-08-19T10:04:28"
category: "설계/인프라"
kind: "설계"
verdict: "해당 없음"
key_numbers: "합성 1H·3심볼·1년: 수수료가 1R의 47%(fee_drag_r=0.47), 인트라바 정책 민감도 PF 0.344→0.422, 모호체결 3.6%(15% 초과 시 신뢰 불가)"
one_line: "진입 로직은 플러그인으로 분리하고 TP/SL 관리 레이어만 확정했으며, 합성 데이터에서 수수료가 1R의 47%를 먹는 것으로 나와 진입 로직을 정하기 전에 SL 폭·4H 상향·지정가 비중·좁은 봉 스킵을 먼저 결정해야 한다고 정리했다."
original_path: "claude/backtest-core-설계.md"
---

# 백테스트 코어 설계 메모 (v0.1.0)

> 다음 세션이 이어받을 수 있게 남기는 문서. 코드 전체는 `qbot-v0.1.0.zip` 으로 전달됨.

## 프로젝트 전제

- ICT 이론 기반 퀀트봇. 사용자는 파이썬 고급자, 퀀트 도구는 익숙하지 않음.
- OHLCV = OKX API, 캐시는 5분봉 CSV 9개 심볼 (`BTC_5m_futures.csv` 형식,
  컬럼 `time,open,high,low,close,volume`, time 은 UTC 오픈 타임).
- **핵심 아이디어**: HTF 봉(1H/4H)이 확정되면 그 봉을 피보나치로 분할하고,
  그 좌표계 안에서 TP/SL 을 관리한다.

## 세션에서 확정한 결정

| 항목 | 결정 |
|---|---|
| 첫 산출물 | 백테스트 엔진 코어 |
| 진입 로직 | **미정 — 플러그인 인터페이스로 분리**. TP/SL 관리 레이어만 확정 |
| 데이터 | 합성 데이터로 먼저 개발, 실데이터는 나중에 연결 |

## 아키텍처

```
qbot/core/fib.py        FibGrid  — HTF 봉 → 피보 좌표계 (핵심)
qbot/core/types.py      Bar / HTFCandle / EntryIntent / Position / Trade
qbot/data/loader.py     CSV 로더 + DataQuality 무결성 리포트
qbot/data/resample.py   5m→HTF, build_ref_map (룩어헤드 차단 지점)
qbot/data/synth.py      동일 스키마 합성 5m 생성기
qbot/strategy/base.py   Strategy ABC + Context (전략이 보는 시점 뷰)
qbot/strategy/builtin/  htf_fib · fib_limit · ict_sweep
qbot/execution/simulator.py  인트라바 정책 / 갭 / 수수료 / 사이징
qbot/engine/backtest.py 이벤트 루프
qbot/report/metrics.py  R 기준 지표
```

## 절대 규약 3개

1. **타임스탬프는 오픈 타임.** 1H 봉 `[10:00,11:00)` 은 `11:00` 확정 →
   `ts=11:00` 짜리 5m 봉부터 참조 가능. `build_ref_map()` 이 이걸 강제.
2. **전략은 가격만, 엔진은 돈만.** 전략은 `EntryIntent(side, sl, tp)` 만 반환.
3. **모호한 체결은 세어서 보고.** 한 5분봉에서 TP·SL 이 둘 다 닿으면 순서를 알 수 없음.
   정책: `SL_FIRST`(기본) / `TP_FIRST` / `OHLC_PATH` / `RANDOM`.
   `ambiguous_bars` 가 거래 대비 15% 초과면 백테스트 신뢰 불가.

## 피보 좌표계 규약

- 앵커: `RANGE`(low..high, 기본) / `BODY` / `UPPER_WICK` / `LOWER_WICK`
- 방향: `SPATIAL`(0=저가, 1=고가, 기본) / `DIRECTIONAL`(0=봉 출발점)
- 레벨은 `[0,1]` 밖도 허용 (확장 -0.618 ~ 1.618)
- `FibGrid.price(level)` / `.level(price)` / `.zones()` / `.equilibrium` /
  `.is_premium()` / `.ote(side)` / `.levels_to_prices(side, sl_lv, tp_lv)`

## 백테스트에서 나온 진단 (합성 데이터, 1H, 3심볼 1년)

**수수료가 1R 의 47%를 먹는다** (`fee_drag_r=0.47`).
원인: 1H 봉 폭이 좁아 SL 거리(=1R)가 작은데, 리스크 사이징 + 레버리지 상한 5배가
맞물려 명목가치는 크고 1R 금액은 작아짐. 왕복 taker 0.1% 가 그대로 R 을 깎음.

→ 진입 로직을 정하기 **전에** 결정해야 할 것:
- SL 을 HTF 봉 밖(예: 레벨 -0.3 이하)으로 넓힐지, 아니면 4H 로 올려 1R 을 키울지
- 지정가 진입(maker) 비중을 올릴지
- `FibConfig.min_range_pct` 로 좁은 봉을 아예 스킵할지

인트라바 정책 민감도는 낮았음 (PF 0.344 → 0.422, 모호체결 3.6%). 구조는 견고.

## 다음 단계 후보

- 실데이터 연결: OKX 수집기 + 9심볼 캐시 정합성 검증
- 진입 로직 확정 (FVG / OB / sweep / CISD 중 무엇을 1순위로)
- 포트폴리오 레벨 자본 공유 (지금은 심볼별 독립 자본)
- 펀딩비, 청산가, 부분 익절, 피라미딩
- 워크포워드 / 파라미터 안정성 스캔
- 라이브 실행 어댑터 (`Strategy` / `FibGrid` 는 재사용 가능하게 분리해 둠)

## 회귀 테스트에서 가장 중요한 것

`tests/test_engine.py::test_future_bars_cannot_change_past_decisions`
— 데이터 후반부 가격을 3배로 왜곡해도 그 이전 구간 거래가 한 건도 안 바뀌어야 통과.
룩어헤드가 생기면 이게 먼저 깨진다. 엔진 수정 시 반드시 확인할 것.
