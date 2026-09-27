## 🧪 Iteration 048: 4H FVG "왜 약한가" 진단 — ICT 본질 복원
------------------------------------------------------------
### 문제
- 4H FVG가 ICT의 핵심 confluence인데 단변량 AUC 0.500.
  active_4h_fvg_fill NaN 29,074건(99.9%), in_4h_fvg=1 비율 0.13%.
  신호 자체가 없는 건지, 구현 미스매치인지 구분 필요.
### 결과
- diagnose_at() 진단 메서드를 fvg_lifecycle.py에 추가. 진입 시점에 7개
  diag_* 컬럼 자동 기록(dist_atr/state_code/bars_since_touch/fill_ratio/
  zone_pos/size_atr/n_same_dir_alive). collect 정상 작동 확인.
  ETH 12건 샘플 분석:
  - diag_state_code: SOFT_DEAD(4) 절반 — 진입 시 FVG가 이미 50%+ 침투당해 사라짐.
  - nearest_4h_fvg_dist_atr 평균 9.8 ATR — FVG가 진입가에서 너무 멀리 있음.
  - diag_zone_pos 음수 다수 — 진입가가 FVG zone 밖.
  - diag_bars_since_touch 10~77봉(4H) = 터치 후 수일 지나 진입.
### 인사이트
- 4H FVG 약함은 "신호 없음"이 아니라 구현 미스매치 3종:
  (1) 순간 zone 체크(in_4h_fvg)가 너무 엄격 → 0.13% 희소.
  (2) SOFT_DEAD 과다(fill_dead=0.5 임계가 4H 크기 대비 너무 낮음).
  (3) 터치-진입 시간차가 커서 "터치 직후 반응" 맥락을 못 잡음.
  → 해결: "FVG 터치 후 N봉 이내 = FVG_ON" 윈도우 feature.
  현재 0.13% 희소 → 윈도우 적용 시 커버리지 대폭 상승, 머신 친화적 재설계.
