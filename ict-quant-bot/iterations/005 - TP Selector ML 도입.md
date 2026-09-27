## 🧪 Iteration 005: TP Selector ML 도입

### 문제
- 가까운 TP만 선택 → R:R 붕괴

### 원인
1. reach_ratio label → 거리 편향
2. distance feature dominance (43%)
3. EV 미반영 selection logic

### 결과
- 높은 WR but 낮은 R
- EV < 0

### 인사이트
- "확률만 최적화"는 의미 없음
- 반드시 EV 기반 설계 필요
