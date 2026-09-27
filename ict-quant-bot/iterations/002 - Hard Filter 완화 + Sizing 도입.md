## 🧪 Iteration 002: Hard Filter 완화 + Sizing 도입

### 세팅
- 기존 구조 유지, 필터 완화
- 일부 조건 → sizing factor로 전환

### 이유
- hard filter의 threshold cliff 문제 해결
- soft decision으로 전환 시도

### 구현 과정
- CHOCH/BOS 없어도 ENTRY 허용
- OTE 영역 FVG → sizing 증가
- HTF bias (1H, 4H), PD → sizing 반영
- sweep_ratio (꼬리비율) → sizing 반영
- Trailing SL:
  - 1R 도달 시 swing 기반 이동
- Trendline 제거 (pivot sensitivity 문제)

### 결과
- 1년 수익률: -80%

### 인사이트
- ICT는 구조적으로 “연속적 판단”이 필요한 시스템
- sweep_ratio 에서 Time-Windowing Artifact로 인해 wick 비율을 잘 해석하지 못하는 경향
- rule-based로는 한계 명확
- ML 필요성 확정
