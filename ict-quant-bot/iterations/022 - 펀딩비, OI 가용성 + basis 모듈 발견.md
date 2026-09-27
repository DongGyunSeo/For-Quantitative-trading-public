## 🧪 Iteration 022: 펀딩비, OI 가용성 + basis 모듈 발견
------------------------------------------------------------
### 문제
- 다른 정보 축(현물-선물, 펀딩, OI) 피처 추가 검토
- spot_div가 5h 단기만 봄 → 더 긴 시장 필요
### 해결
- 기존 미연결 모듈 2개 발견:
  - basis_features.py: basis_z(30h), basis_momentum(1h), spot_lead — OHLCV만 사용, 50개월 커버 가능
  - derivatives_features.py: funding_rate_z, oi_change_1h, taker_extreme — 거래소 API 기반
### 인사이트
- **derivatives 사용 불가**: 거래소 펀딩비/OI 히스토리가 collect 기간(최대 50개월)을 커버 못 함
- **펀딩비는 spot-futures 베이시스로 간접 추정 가능** (펀딩비 공식 자체가 베이시스 기반이라 proxy로 정확)
- **OI는 OHLCV로 추정 근본 불가** → 포기
- basis_features는 OHLCV만 쓰므로 긴 기간 가능 → 사용 가능
