## 🧪 Iteration 026: Timeout 길이 의심 + Binary 전환 인프라
------------------------------------------------------------
### 문제
- regression(vp_score)이 binary보다 약화 가설 (AUC 하락의 계기)
- timeout이 너무 짧을 가능성 (현재 dynamic, max 144봉)
- timeout 4h~12h 비교하고 싶으나 4번 collect는 무거움
### 해결
- **Timeout 스윕 인프라**: 12h(144봉)로 한 번 수집하면서 각 trade에 시점별 상태 기록 → 사후 재라벨링으로 4h/6h/8h/12h 비교 (재수집 불필요)
- Trade에 bars_to_tp, bars_to_sl, r_at_48/72/96/144 추가
- TIMEOUT_SWEEP=1 모드: 144봉 고정 + 매 bar _check_one에서 추적
- analyze_timeout_sweep.py: 한 CSV에서 4가지 timeout WR/EV 비교 + TP 도달 시점 분포
- train_entry.py: ENTRY_BINARY로 regression/binary 토글
### 인사이트
- 한 번 수집으로 4가지 timeout 비교 가능 (shadow 방식의 timeout 버전)
- TP 도달 시점 분포가 timeout 적정선 직접 진단
