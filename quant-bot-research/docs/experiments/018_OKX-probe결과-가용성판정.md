---
idx: 18
title: "OKX API 실측 probe 결과 — 무엇이 실제로 되는가 (2026-08-20)"
created: "2026-08-20T04:52:04"
category: "데이터"
kind: "데이터 감사"
verdict: "해당 없음"
key_numbers: "펀딩 이력 ~94일 (282행 × 10심볼 = 2,820 정산) · OI 1페이지 8h15m · 롱숏비율 575행 ≈ 2일 · 테이커 404 / 청산 400"
one_line: "펀딩·OI·롱숏비율·캔들류는 동작했지만 테이커 볼륨은 404, 청산은 400 이었고, 펀딩 이력은 ~94일뿐이라 전 구간 소급은 불가하며, OI 가 8시간에서 멈춘 것은 페이지네이션 문제로 의심돼 v0.16.0 에 자동 판별을 넣었다."
original_path: "claude/OKX-probe결과-가용성판정.md"
---

# OKX API 실측 probe 결과 — 무엇이 실제로 되는가 (2026-08-20)

`examples/okx_fetch.py --probe` 실사용 결과. **추정이 아니라 실측이다.**

## 가용성 판정

| 데이터셋 | 결과 | 실제 경로 | 이력 깊이 (실측) |
|---|---|---|---|
| **funding_rate_history** | ✓ | `/api/v5/public/funding-rate-history` | **~94일** (2026-05-18~) |
| **open_interest_history** | ✓ | `/api/v5/rubik/stat/contracts/open-interest-history` | **1페이지 = 8h15m 만** ⚠ |
| long_short_ratio | ✓ | `/api/v5/rubik/stat/contracts/long-short-account-ratio` | 575행 ≈ **2일** (5m) |
| mark_candles | ✓ | `/api/v5/market/mark-price-candles` | 정상 |
| index_candles | ✓ | `/api/v5/market/index-candles` | 정상 |
| history_candles | ✓ | `/api/v5/market/history-candles` | 정상 |
| contract_taker_volume | ✗ | 후보 2개 모두 **404** | — |
| taker_volume | ✗ | 후보 2개 모두 **404** | — |
| liquidation_orders | ✗ | **400 Bad Request** | — |

## 진단 — 404 와 400 은 의미가 다르다

**liquidation_orders 는 400.** 경로가 없으면 404 가 난다. 400 은
**경로는 살아있고 파라미터가 틀렸다** 는 뜻이다. SWAP 은 `instType` +
`uly`(또는 `instFamily`) + `state` 가 필수다. → 되살릴 가능성이 높다.

**taker volume 은 404.** 경로 자체가 없다. 다만 `rubik/stat` 네임스페이스는
살아있으므로(OI·롱숏비율 동작) 다른 이름일 것이다.

**OI 가 1페이지에서 멈춘 건 보존 한계가 아니라 페이지네이션 문제로 의심된다.**
근거: 같은 rubik 네임스페이스의 `long_short_ratio` 는 `limit=100` 을 줬는데도
**575행** 을 돌려줬다. 즉 이 계열은 `limit`/`after` 를 우리가 가정한 대로
해석하지 않는다. rubik 통계는 `after` 커서가 아니라 **`begin`/`end` 창** 을
쓰는 것으로 보인다.

## 적용한 수정 (v0.16.0)

1. **페이지네이션 자동 판별** — 1페이지를 받은 뒤 `after` 와 `begin/end` 를
   실제로 시도해 되는 쪽을 고른다. 둘 다 빈 응답이면 그건 페이지네이션 실패가
   아니라 **보존 한계** 이며, 그렇게 `fetch_meta` 에 구분해 기록한다.
   → OI 가 8시간에서 멈춘 게 어느 쪽인지 다음 실행에서 판명된다.
2. **경로 × 파라미터 매트릭스 탐침** — 404/400 이 파라미터 때문인 경우를 잡는다.
   - liquidation: `{instType:SWAP, uly:BTC-USDT, state:filled}` 등 3조합
   - taker volume: `instType` CONTRACTS/SPAT 조합 + 경로 후보 4개로 확대
3. `fetch_meta` 진단 — 경로·방식·페이지수·중단이유·가장 오래된 레코드 기록.
4. `Timestamp.utcnow` 폐기 경고 수정.

테스트 6개 추가 (총 145 passed): begin/end 폴백, after 우선, 보존한계 구분,
파라미터 매트릭스 성공/실패 경로.

## 펀딩 데이터 — 이미 쓸 수 있다

**282행 × 10심볼 = 2,820 정산, 2026-05-18 ~ 2026-08-20.**

우리 OHLCV 캐시는 2023-11-18 ~ 2026-07-05 이므로
**겹치는 구간 = 2026-05-18 ~ 2026-07-05 (약 7주)**. 짧지만 조건부 분석은 가능하다.

### 왜 이게 중요한가 — 부호를 반대로 가정했을 수 있다

```
fundingRate > 0  →  롱이 숏에게 지급  →  숏은 수취
```

살아남은 셋업은 **76%가 숏**. 그런데 지금까지 "항상 지급(불리)"으로 계산했다.
평상시 펀딩은 대체로 양수이므로, **우리 추정이 과도하게 보수적이었을 가능성**이 있다.

다만 반대 위험도 있다: 폭락 직후엔 펀딩이 음수로 기울어 숏이 지급한다.
우리 셋업은 하필 **캐스케이드 직후에 진입** 한다. 전체 평균으로는 안 보이고
**조건부로만** 보이는 문제다. 그래서 `--cascade` 모드를 만들었다.

손익분기 참조 (검증 문서 기준):

| 셋업 | BE 펀딩 (bps/정산) |
|---|---|
| fib lb3 상위1% TP2R | 13.69 |
| fib lb2 상위1% TP3R | 10.97 |
| vol 상위1% TP2R | 9.36 |
| choch 상위1% TP2R | 7.16 |
| fib lb3 상위3% TP2R | 5.28 |

통상 펀딩이 1bp 수준이면 전부 여유가 크다. 실측으로 확인만 하면 된다.

## 다음 실행

```bash
# 1) 수정된 탐침 — taker/liquidation 이 살아나는지, OI 가 더 깊이 가는지
python examples/okx_fetch.py --probe

# 2) 펀딩 분석 (이미 받은 데이터로 즉시 가능)
python examples/okx_funding_analysis.py --cascade

# 3) OI 재시도 (페이지네이션 자동 판별 적용)
python examples/okx_fetch.py --oi --days 30 --period 5m
```

## 확정된 한계

- **펀딩 이력 ~94일.** 2023~2026 전 구간 소급 적용은 불가능.
  7주 겹침으로 "펀딩 수준이 손익분기 대비 어디쯤인가" 는 답할 수 있지만,
  전 구간 재계산은 유료 데이터(Tardis 등) 없이는 안 된다.
- **롱숏비율 5m 은 2일.** 소급 백테스트용으로는 무용, 전향 기록만 가능.
- 청산 이력은 400 을 고쳐도 깊지 않을 가능성이 높다. 히트맵 근본 검증은
  여전히 전향 기록이 현실적인 경로.
