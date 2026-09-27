## 🧪 Iteration 001: Pure Hard Filter (Rule-based ICT)

### 세팅
- BTCUSDT.P (OKX) 5m bars / 30x
- 상태 머신:
  IDLE → STANDBY → ARMED(Sweep) → DISP_RUNNING → ORDERS_PLANNED → ENTRY

### 이유
- ICT 구조를 최대한 원형 그대로 하드코딩
- “이 패턴이면 진입”이라는 deterministic 전략 구현

### 구현 과정
- liquidity.py:
  Asia H/L, PDH/PDL, PWH/PWL, EQH/EQL, Trendline H/L 계산
- signals.py:
  - 유동성 터치 → STANDBY
  - reclaim → ARMED
  - CHOCH/BOS 발생 → DISP_RUNNING
- Entry:
  - FVG CE 진입
  - FVG 없으면 OB fallback
- TP: Next liquidity
- SL: OB zone 또는 sweep extreme 기반

### 결과
- 1년 수익률: -91%

### 인사이트
- 하드필터는 극단적 binary decision
- parameter 민감도 매우 높음
- 시장 noise 반영 불가
