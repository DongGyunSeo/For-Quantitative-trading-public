## 🧪 Iteration 003: ML Entry Filter 도입

### Summary
ICT 구조 유지 + ML 확률 필터로 진입 결정

### 세팅
- 구조: Sweep → Displacement → FVG 유지
- 단순히 liqudiity touch -> reclaim -> FVG 생성 -> 5m 2봉 기다림 -> New FVG none -> edge 진입
- Entry decision: XGBoost P(win)
- 심볼: BTC, ETH, XRP, SOL, BNB

### 이유
- 좋은 패턴 vs 나쁜 패턴 구분 필요
- 하드 필터 → 확률 기반으로 전환

### 구현 과정

#### Phase 1: Feature Pipeline
- feature_dict 46개 구성
- Trade → CSV 변환
- preprocess 파이프라인 구축

#### Phase 2: Feature Engineering
- 46 → 41 → 45개 정리
- 제거:
  - weekend, ob_entry, pwhl, sparse feature 등
- 추가:
  - CHOCH depth / body strength
  - displacement score
  - interaction feature

#### Phase 3: ML Training Infra
- Focal Loss (gamma=1.6)
  ❌ 초기 gradient 오류 → AUC 0.5
  ✅ 수정 후 정상 작동
- PurgedKFold 적용 (leakage 방지)
- Isotonic Calibration 적용
- Threshold search: 0.15~0.45

#### Phase 4: Nasdaq Feature Pipeline
- 외부 데이터 사용 (Nq1! 5m 데이터 구매)
- DST 자동 처리
- feature coverage 0% → 100%

#### Phase 5: XGBoost 튜닝
- depth 증가
- lr 감소
- estimator 증가
- regularization 완화

### 결과
- 데이터: 2,218건
- 초기 AUC: 0.500 (버그)
- 수정 후: 재학습 필요

### 문제
- Focal Loss 구현 리스크 큼
- feature pipeline silent bug 다수
- 데이터 부족

### 인사이트
- custom loss는 반드시 수치 검증 필요
- feature overwrite는 매우 치명적
- 금융 ML은 calibration 필수
- 시계열 CV는 purge 필수
