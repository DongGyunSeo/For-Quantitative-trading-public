## 🧪 Iteration 029: 초기 파이프라인 + 하드필터 edge 측정
------------------------------------------------------------
### 문제
- ICT sniper 봇의 하드필터 단독 edge가 얼마인가
- base WR vs breakeven 갭이 ML이 메울 수 없는 수준인지 확인
### 결과
- base WR 26.6% vs breakeven 36.1% → 갭 9.4pp
- ML AUC 0.54짜리 필터로는 WR ~39%까지만 가능 → 메울 수 없음
### 인사이트
- "ML filtering cannot compensate for a fundamentally negative-EV strategy"
- 하드필터 자체의 edge는 중요한 문제이다
