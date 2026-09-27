## 🧪 Iteration 042: Confluence "닿음" 재정의 + 1H FVG + 4H+1H 동시결합
------------------------------------------------------------
### 문제
- 기존 confluence 정의: "liquidity가 FVG edge에서 1 ATR 이내" → noise 포함
- 1H FVG(시간 해상도 6배)와 동시 결합 신호 시도
### 해결
- confluence_with: liq_price가 FVG zone에 실제 포함(닿음)만 인정, 1 ATR 거리 허용 삭제
- 1H tracker 병렬 추가(timeout 720봉=30일)
- is_4h1h_fvg_combined feature 신설
### 결과(BTC 거친검증, 닿음 기준)
- 4H +4.9pp(4.4%), 1H +5.4pp(3.4%), **4H+1H동시 +10.9pp(0.8%, n=26)**
- **전체 AUC: is_1h_fvg_combined 0.5009, is_4h1h_fvg_combined 0.5007 = 무신호**
### 인사이트
- 4H+1H동시 +10.9pp는 26건 통계 → BTC 행운 반복
- timeframe FVG confluence 축 전체 닫음 (4H, 1H, 동시 모두)
