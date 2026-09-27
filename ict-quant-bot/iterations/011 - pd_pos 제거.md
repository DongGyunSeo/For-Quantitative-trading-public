## 🧪 Iteration 011: pd_pos 제거 및 대체
------------------------------------------------------------

### 문제
- 55% 값이 0.5 (default)
- AUC 0.497
- ATH / ATL 갱신시 로직 처리의 어려움

### 수정
- range_pos_7d / 30d로 대체

### 결과
- 안정적인 분포 확보

### 인사이트
- default-heavy feature는 noise

