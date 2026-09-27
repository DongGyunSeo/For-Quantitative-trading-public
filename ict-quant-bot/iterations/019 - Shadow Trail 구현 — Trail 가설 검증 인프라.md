## 🧪 Iteration 019: Shadow Trail 구현 — Trail 가설 검증 인프라
------------------------------------------------------------
### 문제
- collect는 trail OFF, OOS는 trail ON → 환경 불일치가 OOS 음수 EV의 원인인지 의심
- trail이 큰 winner를 잘라 edge를 죽이는가?
- trail OFF/ON 두 번 collect는 청산 시각이 달라져 trade 시퀀스가 갈라짐 (multi-position 제약상 치명적)
### 해결
- backtest.py에 _check_shadow() 추가: primary는 실제 청산, shadow는 "trail ON이었다면"의 청산을 가상으로 동시 기록 (포지션은 안 닫음)
- 방법 A 확정: SHADOW_TRAIL=1이면 primary timeout을 SHADOW_TIMEOUT_BARS(144봉=12h) 고정 → primary/shadow가 동일 timeout 창 공유 (공정 비교)
- ml_data.py에 shadow_pnl_r, shadow_exit_reason 컬럼 추가
- analyze_shadow.py: 같은 row의 pnl_r(off) vs shadow_pnl_r(on) 직접 비교
### 인사이트
- 한 진입 → 두 청산이므로 trade 시퀀스 안 갈라짐
- 청산 후 가격도 백테스트가 끝까지 추적하므로 정확
