## 🧪 Iteration 036: FVG Composite Score 개선 (23K 거래 empirical curve)
------------------------------------------------------------
### 문제
- FVG 진입 시 어느 FVG를 고를지 max(size) 단순 선택이 최적인가
### 해결
- 23K 거래에서 fib_wr = f(fib_position) empirical curve 추출
- np.interp로 실시간 WR 조회, LVN 밀도 × FVG size 가중치 곱셈 결합
- fvg_composite_score: (fib_wr × LVN × size) 가중(4:3:2)
### 결과
- **WR 22.1% → 27.8% (+5.7pp)**
- SOL 50개월 누적 수익률 539%
- 0.45~0.50 fib 구간이 peak(26.4% WR)
### 인사이트
- 진입 대상(FVG)의 질 개선이 의미 있는 첫 성과
- 고정 임계값보다 empirical curve가 훨씬 강함 (레짐 의존 해소)
- 과적합 가능성 있음
