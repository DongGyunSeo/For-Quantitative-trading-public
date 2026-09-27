## 🧪 Iteration 043: "신기루 Liquidity" 통찰 → Swing 기반 재설계
------------------------------------------------------------
### 문제
- asia/london/PD/PW high·low는 touches=1 단순 극값 → 스탑 축적 검증 없는 신기루
- "London 직전 우연한 low"와 "장중 여러 번 지지받은 low"가 동일 취급됨
- 7종 유동성 신기루 위험 스펙트럼: 세션H/L(최상) > PD/PW > IDM > EQ(최하)
### 해결
- _extract_swing_liquidity(): 구간 내 swing pivot 극값 추출
- 침범 사후체크: 마지막 pivot 이후~구간끝 가격이 극값 침범하면 소비된 신기루로 제외 (서동균 설계)
- TF 해상도 매칭: asia→5m swing, PD→15m swing, PW→1H swing
- swept=True 마킹, unswept_counts() 메서드 (위/아래 미회수 개수 + imbalance)
- feature: unswept_above/below/imbalance/draw_aligned 4개 추가 (총 50개)
### 결과(검증)
- Asia 90일: 신기루 차단 9% 감소 (기존 180개 → 163개)
- 2023-02-06 PD에서 pdl 누락 = 침범 체크 실제 작동 증거
- pivot_neighbors: PD 6~12개, PW 11~13개 부착 확인
### 인사이트
- 진입 대상(유동성) 자체를 손보는 가장 근본적 시도
