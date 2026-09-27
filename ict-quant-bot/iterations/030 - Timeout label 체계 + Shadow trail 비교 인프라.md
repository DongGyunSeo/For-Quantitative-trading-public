## 🧪 Iteration 030: Timeout label 체계 + Shadow trail 비교 인프라
------------------------------------------------------------
### 문제
- timeout trade(40%+)를 label에서 제외하면 학습 데이터 낭비
- timeout을 label에 포함하되 노이즈 최소화 설계 필요
### 해결
- TIMEOUT_SWEEP=1: 144봉 고정으로 한 번 수집, bars_to_tp/sl + r_at_48/72/96/144 기록
- analyze_timeout_sweep.py: 사후 4h/6h/8h/12h timeout WR/EV 비교 (재수집 불필요)
- Shadow trail 동시 비교 인프라 (primary vs trail-off 수익 비교)
### 인사이트
- 한 번 수집으로 4가지 timeout 길이 비교 가능 (shadow 방식의 timeout 버전)
- timeout 포함/미포함 AUC 비교로 timeout이 신호를 키우는지 죽이는지 진단
