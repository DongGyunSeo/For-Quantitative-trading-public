## 🧪 Iteration 041: Multi-TF FVG Lifecycle 트래커 설계 및 1차 통합
------------------------------------------------------------
### 문제
- 단일 TF(5m) 단면 feature가 0.54 천장에 막힘 → timeframe 축 확장 가설
- "이 5m sweep이 4H FVG 근처에서 일어나는가" = 맥락 레이어
### 해결
- FVGLifecycleTracker: PENDING→ARMED→TOUCHED→SOFT_DEAD 상태머신
- backtest에 4H tracker 통합: _prime(워밍업) + _advance(매 4H봉 닫힘)
- incremental detect 방식 → ARMED 전이 1/9 붕괴 발견(5252 FVG 추가 vs 760) → 전체 detect로 교체
### 결과(BTC 거친검증)
- in_4h_fvg_aligned +4.9pp, is_4h_fvg_combined +3.1pp (빈도 30%)
- **전체 11심볼 collect: is_4h_fvg_combined AUC 0.500 = 완전 무신호**
### 인사이트
- BTC in-sample 신호는 단일 심볼 행운. 전체서 소멸 패턴 반복
- timeframe FVG confluence (4H) 축 닫음
