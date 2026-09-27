## 🧪 Iteration 040: TP Selector 거리 편향 feature 제거
------------------------------------------------------------
### 문제
- dist_rank, rr_ratio, relative_rr 등이 importance 34.6% 독점
- 거리가 가까운 TP는 "맞히기 쉬운" leakage — 진짜 달성 능력과 무관
### 해결
- 3개 distance-biased feature 제거 → 15 feature로 축소
### 결과
- **AUC 0.720 유지, importance 분포 균일화**
- Corr 0.546 → 0.104 (group leakage 수정 전후)
### 인사이트
- 거리 편향 feature는 overfitting 주범. 제거가 오히려 일반화 개선
- reach_ratio(per-candidate)로 leakage 해소
