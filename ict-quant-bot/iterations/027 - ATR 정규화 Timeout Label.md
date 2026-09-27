## 🧪 Iteration 027: ATR 정규화 Timeout Label
------------------------------------------------------------
### 문제
- timeout trade의 pnl_r은 risk(SL 거리)로 나눔 → SL 타이트하면 작은 절대 이동도 큰 R = outlier
- timeout을 label에 포함하고 싶은데 신호 오염 우려
### 해결
- 각 시점의 (close-entry)/ATR 기록: ratr_at_48/72/96/144 (entry_atr 0이면 risk fallback)
- SL 길이에 독립적인 "움직임 품질" 측정
- relabel_timeout.py: timeout trade의 ATR 수익 상위 n%를 win, 나머지 loss
- n%는 --pct 인자로 조정 (재수집 없이 반복 실험)
### 인사이트
- ATR 정규화로 SL 타이트 outlier 방지
- 상위 n% 컷은 사후 조정 가능 → 재수집 없이 timeout label 품질 실험 반복
- 검증: timeout 1179건 중 ATR 상위 30%(임계값 +0.812 ATR) 354건이 win
