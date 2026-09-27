## 🧪 Iteration 038: DBSCAN 유동성 클러스터링 (EQH, EQL 재설계)
------------------------------------------------------------
### 문제
- EQH/EQL의 ±0.3% 고정 밴드는 자기실현적 — noise를 유동성으로 인식
### 해결
- ATR-adaptive epsilon(EQ_EPS_ATR_RATIO=0.15)으로 DBSCAN 스타일 클러스터링
- chain-effect 차단, median 기반 center
- IDM(inducement) 레벨 추가 (15m 구조 기반)
### 결과
- EQ non-zero 클러스터 생성 품질 향상 (단순 밴드 대비 false positive 감소)
### 인사이트
- 고정 밴드는 "쉽게 충족되는 조건" = self-fulfilling. ATR 적응이 필수
