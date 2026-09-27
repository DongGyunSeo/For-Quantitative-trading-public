## 🧪 Iteration 008: DEAD Feature 탐지 & 보강
------------------------------------------------------------

### 발견
- adx_14 / return_1h / return_4h → 전부 0

### 원인
- backtest.py에 계산 코드 없음 (silent failure)

### 조치
- feature 계산 로직 추가
- augment_features.py로 과거 데이터 소급 보강

### 결과
- AUC: 0.556 → 0.592 (+0.036)

### 인사이트
- zero feature는 AUC 0.500이지만 눈에 안 띔
- 반드시 non-zero ratio 체크 필요
