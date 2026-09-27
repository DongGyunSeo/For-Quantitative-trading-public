## 🧪 Iteration 031: Timeout label ATR 정규화 설계 (dead zone + 분위수)
------------------------------------------------------------
### 문제
- timeout pnl_r 상위 n% 컷이 "미래 결과를 라벨로 쓰는 것" → label leakage 위험
- SL이 타이트하면 같은 움직임도 RR 크게 뻥튀기 → outlier
- U자형 AUC 곡선: 0% 및 100% 극단에서 AUC 0.64, 중간 50%에서 AUC 0.53
### 해결
- 양극단만 라벨, 중간은 dead zone(NaN, 학습 제외): timeout을 청산 부호로 가름
  → TP쪽(양수) 상위 tp_pct%, SL쪽(음수 절댓값) 상위 sl_pct% 각각 독립 지정
- ATR vs RR 두 버전 생성 후 비교 (ATR: SL 길이 독립, RR: 실제 손익 일관성)
- sl=8 고정 시 SHORT Δ −0.009 → −0.0001로 개선 확인
### 인사이트
- U자형 AUC는 "모델이 신호를 찾은 것"이 아니라 class imbalance로 극단 분류가 쉬워진 것
- 진짜 의미 있는 구간은 "±0 근처의 모호한 timeout을 제거한 뒤 중간 AUC"
