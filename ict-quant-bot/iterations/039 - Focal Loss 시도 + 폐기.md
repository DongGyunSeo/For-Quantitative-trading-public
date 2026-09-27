## 🧪 Iteration 039: Focal Loss 시도 + 폐기
------------------------------------------------------------
### 문제
- 클래스 불균형 완화 위해 Focal Loss(gamma=2.0, 1.0) 시도
### 결과
- 예측 범위가 [0.28, 0.45]로 압축 → Precision/Recall 붕괴
- scale_pos_weight 방식 대비 성능 열화
### 인사이트
- Focal Loss는 이 데이터셋에서 신뢰 불가. binary:logistic + scale_pos_weight 유지
