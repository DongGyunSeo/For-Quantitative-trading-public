## 🧪 Iteration 037: VP Density + vp_score Regression Target
------------------------------------------------------------
### 문제
- TP 후보 선택의 기준이 없음 → TP 이후 가격 경로가 vp_score를 결정
### 해결
- Bar-by-bar VP 누적 (peak 기반, close-to-close는 non-zero 1%만)
- vp_score = vp_consumption / (1 + vp_adversity) × (1 + disp_extreme_broken)
- non-zero 비율 1% → 84.6% 개선
### 결과
- TP Selector v3: AUC 0.720
- path_vp_integral #1 importance (0.126), density_at_candidate #3 (0.079)
### 인사이트
- close-to-close VP 방식은 사실상 무용 → peak-based로만 의미 있음
- disp_extreme_broken은 forward-looking이라 training feature로 불가, 사후 보너스만
