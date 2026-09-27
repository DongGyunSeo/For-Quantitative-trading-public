## 🧪 Iteration 028: Timeout 포함, 미포함 AUC 비교 인프라
------------------------------------------------------------
### 문제
- timeout label 추가가 신호를 키우는지 죽이는지 검증 필요
### 해결
- relabel_timeout.py에 두 label 동시 생성:
  - A: tp/sl + timeout(ATR 상위 n% win) 포함
  - B: tp/sl만 (timeout 제외, NaN으로 drop)
- 같은 피처로 각각 binary XGBoost(binary:logistic + scale_pos_weight) 학습 → 4-fold TimeSeriesSplit CV AUC 비교 (전체/long/short)
- A > B → timeout 유용, A < B → 노이즈
### 인사이트
- 한 번 수집으로 4가지 실험: binary vs regression / timeout 4길이 / timeout 포함 여부 / 상위 n%
- 재수집 없이 사후 실험 반복 가능
- sweep_depth vs TP 모델: 직접 중복 아님 (TP는 candidate 중심, Entry sweep_depth는 셋업 품질), "이동 크기" 공통 축만 존재
