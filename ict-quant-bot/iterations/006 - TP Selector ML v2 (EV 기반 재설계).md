## 🧪 Iteration 006: TP Selector ML v2 (EV 기반 재설계)
------------------------------------------------------------

### 문제
- 이전 모델(v1)은 확률 최적화에만 집중 → EV < 0 구조 고착
- dist feature dominance(43%)로 인해 가까운 TP 선택 편향 해소 필요

---

### 원인
1. 타겟 재정의:
   - reach_ratio (연속값) → reached (binary 0/1)
   - 순수 도달 확률 P를 예측하는 분류 모델로 전환

2. 거리 편향 피처 제거:
   - dist_to_candidate_atr
   - dist_rank
   - n_candidates_closer
   - 거리 기반 interaction 4개
   → 제거

3. 피처 수 축소:
   - 20개 → 15개

---

### 구현 과정
- EV 기반 선택 로직 도입:
  - max(score) → max(EV = P × RR - (1 - P) × 1.0)

- EV 임계값 도입:
  - min_ev = 0.15
  - 임계값 미만 → None (trail-only 전환)

- min_rr 상향:
  - 1.0 → 1.5

- CV 전략:
  - _PurgedGroupCV (trade_id 기준, purge=2, embargo=1)

- 클래스 불균형 보정:
  - scale_pos_weight = n_neg / n_pos (동적 계산)

- 평가 메트릭 추가:
  - AUC-ROC
  - Brier Score
  - LogLoss
  - CV 단계에서 EV>0 비율 직접 측정

---

### 결과

항목
v1 / v2 비교

- 타겟:
  - v1: reach_ratio (회귀)
  - v2: reached (이진 분류)

- 선택 기준:
  - v1: max(P)
  - v2: max(EV = P×RR − (1-P))

- 거리 편향 피처:
  - v1: 존재 (43%)
  - v2: 제거

- 피처 수:
  - v1: 20개
  - v2: 15개

- EV 임계값:
  - v1: 없음
  - v2: min_ev = 0.15

- 평가 지표:
  - v1: AUC only
  - v2: AUC + Brier + LogLoss + EV>0%

---

### 인사이트
- EV 임계값은 필터가 아니라 안전망:
  - min_ev < 0.15이면 TP를 배정하지 않고 베팅 자체를 거부

- 거리 제거만으로는 부족:
  - dist_gap_to_stronger 같은 거리 독립적 상대 피처 필요

- scale_pos_weight 동적화 필수:
  - 도달 샘플 30~40% → 고정값 사용 시 calibration 왜곡

- EV>0 비율을 CV에서 직접 측정:
  - AUC 높아도 EV>0 비율 낮으면 실전 기대값 음수 가능
  - AUC 단독 신뢰 불가
