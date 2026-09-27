# quant-bot-research — ICT 에서 시계열 추세까지

> **투자 권유가 아닙니다.** 이 저장소는 개인 연구 기록입니다. 모든 성과 숫자는 과거 백테스트(표본 안)이며,
> 최종 전략의 샤프 t 값은 **1.46 으로 통계적으로 유의하지 않습니다.** 봉인 창(2026-07-05 00:05 ~ 09-24 UTC)은 아직 열지 않았습니다.

**English summary.** A 5-week research log (2026-08-19 → 09-26, 83 documents) that tried to turn ICT (Inner Circle Trader) concepts —
FVG, CHoCH, CISD, liquidation cascades, weekend gaps, Fibonacci, volume profile — into rules that stay profitable after fees on
OKX USDT perpetual 5-minute data. None of the ICT short-horizon setups survived costs and out-of-sample tests. The only survivor is a
weekly time-series momentum (TSMOM) rule taken from the literature (47 coins, long if above the price 28 days ago, short otherwise),
with Sharpe 0.64 (t 1.46 — not significant). Every rejected experiment is kept, with pre-registrations paired to their results.

## 한 장 요약

**무엇을**: ICT(Inner Circle Trader) 개념 — FVG · CHoCH · CISD · 유동성(청산) · 주말 갭 · 피보나치 · 거래량 프로파일 — 을 코드로 옮겨,
**OKX USDT 무기한 선물 5분봉**에서 **수수료를 내고도 돈이 남는 규칙**을 찾았다. Python, 5주(2026-08-19 ~ 09-26), 문서 83개.

**결과 한 줄**: ICT 단기 셋업은 수수료 전에도 엣지가 0 이거나 작았고(대개 ±0.1R 안), 비용과 표본 밖 검정을 넘은 것은 **하나도 없다.**
끝까지 남은 것은 ICT 가 아니라 문헌에서 가져온 **주간 시계열 추세(TSMOM)** — 47개 코인 각각 "28일 전보다 높으면 롱, 낮으면 숏", 매주 월 00:00 UTC 리밸런스.

**최종 운용안 = 시계열 추세 + α** (α = 역변동성 비중 · 총노출 G 0.75 · 매월 복리 갱신 · 교차 증거금):

| | 값 (269주, 2021-05-03 ~ 2026-06-22 시작, 봉인 전) |
|---|---|
| 역사 5년 | **3.05배 · 연복리 24.1%** · 최대낙폭(장중 포함) −35.5% · 샤프 0.73 |
| 섞은 5년 경로 10,000개 (4주 블록 부트스트랩) | 연복리 중앙 **20.9%** (하위 5% −7%) · 최대낙폭 중앙 −48% · 95% −70% · 5년 손실 확률 11% |
| 통계적 확실성 | 기준 전략(동일 명목 · G 1) 270주 샤프 **0.64, t 1.46 — 유의하지 않다** |

![연구 문서 83개 타임라인](figures/00_timeline.png)

*가로 = 문서 순서(작성일), 세로 = 분야, 색 = 판정. 주황 = 기각(33), 파랑 = 통과 · 부분 통과(18), 회색 = 사전등록 · 설계 · 서술 · 데이터(32).*

**숫자로 본 여정**

| | |
|---|---|
| 문서 | 83개 — 결과 53 · 사전등록 12 · 서술 8 · 데이터 감사 4 · 설계 2 · 기록 2 · 운용 문서 2 |
| 판정 | 기각 33 · 부분 통과 14 · 통과 4 · 서술 9 · 해당 없음 23 |
| 분야 | 시계열 추세 16 · 청산 · 유동성 12 · FVG 11 · painR3 7 · ICT 신호 6 · 데이터 6 · 거래량 프로파일 6 · 비용 · 레짐 5 · 주말갭 4 · 기타 10 |
| 데이터 | 10심볼 5분봉(초기) → 47심볼 5분봉 2021-04-02 ~ 2026-09-24, 심볼당 약 576,500행, 공백 · 중복 · OHLC 위반 0 |
| 코드 | `qbot` 패키지 19개 모듈 · 예제 스크립트 48개 · 테스트 파일 21개(테스트 332개) — 2026-09-18 코드 유실 뒤 재구축 |

## 30분 안에 읽는 순서

1. 이 README — 결론 · 규칙 · 한계
2. [연구 여정 (9단계)](docs/TIMELINE.md) — 무엇을 어떤 순서로 시도했나
3. [교훈](docs/LESSONS.md) — 무엇이 안 됐고 왜
4. [최종 전략](docs/STRATEGY_TSMOM.md) → 원문 문서 [83](docs/experiments/083_시계열추세-트레이딩방법-상세.md)
5. 필요할 때: [실험 카탈로그 83건](docs/INDEX.md) · [연구 규약](docs/METHODOLOGY.md) · [데이터](docs/DATA.md) · [용어집](docs/GLOSSARY.md)

## 최종 전략 — 시계열 추세 + α

원문: 문서 [83](docs/experiments/083_시계열추세-트레이딩방법-상세.md)(운용), [78](docs/experiments/078_시계열추세-사이징-레버리지-결과.md)(사이징 · 레버리지), [80](docs/experiments/080_시계열추세-복리주기-결과.md)(복리 주기), [82](docs/experiments/082_시계열추세-매월갱신-최적G-결과.md)(최적 G). 자세히: [docs/STRATEGY_TSMOM.md](docs/STRATEGY_TSMOM.md).

### 규칙

| 항목 | 값 | 근거 · 판정 |
|---|---|---|
| 유니버스 | OKX USDT 무기한 **47종** 고정([데이터](docs/DATA.md) 목록) | 문서 [61](docs/experiments/061_OKX-47심볼-데이터감사-및-봉인규칙.md), [63](docs/experiments/063_예전신호16종-수익곡선-변곡점.md) |
| 가격 | **UTC 일 종가** (00:00 UTC 직전 마지막 가격, 5분봉에서 만든다) | 문서 [83](docs/experiments/083_시계열추세-트레이딩방법-상세.md) |
| 신호 | 매주 **월 00:00 UTC**(한국 월 09:00): `sign(오늘 종가 / 28일 전 종가 − 1)` → +1 롱 / −1 숏, 플랫 없음 | 문서 [40](docs/experiments/040_시계열추세-주간리밸런스-사전등록.md) (28일은 문헌값 — 데이터에서 고르지 않았다) |
| 비중 (+α ①) | **역변동성**: 직전 60일 일간 로그수익 표준편차 σ_i (최소 20일), a_i ∝ 1/σ_i, Σa = 1 | 문서 [78](docs/experiments/078_시계열추세-사이징-레버리지-결과.md) — 성과는 동일 명목과 같아 **채택 기준은 못 넘었다(사용자 선택)** |
| 총노출 (+α ②) | **G = 0.75** (자산 대비 총명목) | 문서 [78](docs/experiments/078_시계열추세-사이징-레버리지-결과.md), [82](docs/experiments/082_시계열추세-매월갱신-최적G-결과.md) — 사용자 선택. −50% 낙폭 한도 권장(0.40)보다 크고, 절반 켈리 근처 |
| 복리 (+α ③) | **B = 달력 월 첫 월요일 00:00 UTC 의 계좌 총자산**, 한 달 고정. 목표 명목_i = sign_i × a_i × G × B | 문서 [80](docs/experiments/080_시계열추세-복리주기-결과.md) — **사전등록 판정 채택** |
| 증거금 | **교차(cross).** 청산은 G 와 유지증거금률만으로 정해진다(G = 레버리지 × (1 − 잉여 비율)) | 문서 [78](docs/experiments/078_시계열추세-사이징-레버리지-결과.md) |
| 체결 | 신호 확정 직후 **시장가**(6bps 가정), 줄이는 주문 먼저 | 문서 [76](docs/experiments/076_시계열추세-금신호-월시가-주말갭지정가-결과-기각.md), [83](docs/experiments/083_시계열추세-트레이딩방법-상세.md) — 늦추는 변형은 전부 기각 |
| 보유 | 1주. 손절 · 익절 · 주중 개입 **없음** | 문서 [83](docs/experiments/083_시계열추세-트레이딩방법-상세.md) |

### 성과 (269주, 2021-05-03 ~ 2026-06-22 시작, 봉인 전, 되맞춤 수수료 6bps · 펀딩 포함)

| 구성 | 5년 | 연복리 | 최대낙폭(장중) | 샤프 | 실효 G | 섞은 연복리 중앙 | 섞은 낙폭 중앙 · 95% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 역변동성 · G 0.75 · 매주 갱신 | 2.72배 | +21.3% | −36.5% | 0.68 | 0.75 고정 | +18.9% | −49% · −72% |
| **역변동성 · G 0.75 · 매월 갱신 (최종)** | **3.05배** | **+24.1%** | **−35.5%** | **0.73** | 0.57 ~ 0.93 | **+20.9%** | **−48% · −70%** |
| 역변동성 · G 1.0 · 매월 | 3.90배 | +30.1% | −45% | | 최대 1.34 | +25.3% | −59% · −82% |
| 역변동성 · G 1.45 · 매월 (복리 최대) | 5.19배 | +37.5% | −59% | | 최대 2.29 | +29.2% | −76% · −94% |

비교 기준(문서 [83](docs/experiments/083_시계열추세-트레이딩방법-상세.md)): 동일 명목 · G 1 · 270주 — 샤프 0.64 (t 1.46), 연복리 22.5%, 최대낙폭 −46.1%.

![갱신 주기별 누적 자산 (로그 축) — 역변동성 · G 0.75](figures/tsmom_compound_01.png)

*매월 갱신 채택(문서 [80](docs/experiments/080_시계열추세-복리주기-결과.md)) — 갱신 주기별 누적 자산 (로그 축).*

![G 별 연복리](figures/tsmom_optg_01.png)

*매월 갱신의 G 별 연복리(문서 [82](docs/experiments/082_시계열추세-매월갱신-최적G-결과.md)) — 꼭대기 G ≈ 1.45, 사용자 값 0.75 는 절반 켈리.*

### 확정되지 않은 것

1. **통계적으로 유의하지 않다.** 샤프 0.64 (t 1.46), 표준오차 약 0.45 — 진짜 샤프는 0 근처일 수도, 1 을 넘을 수도 있다. t 2 로 확인하려면 약 10년이 필요하다.
2. **lookback 띠**: 26 ~ 36일에서만 샤프 0.42 ~ 0.65. 14일 0.00 · 56일 0.16 — 이 띠가 앞으로도 유지되는지가 가장 큰 위험.
3. **매월 갱신의 이점은 사후에 찾은 성질**(되돌림)에 기댄다.
4. **펀딩 이력이 없다** — 과거 5년에 실측 3개월 평균을 고정했다. 강세장 롱 구간에선 비용이 된다.
5. **생존 편향** — 47종은 2026 년에도 OKX 에 남은 코인이다.
6. **분산 효과가 작다** — 사실상 시장 추세 베팅 하나. 한 주 −22.3%(G 1), 67주 수면 아래 구간이 있었다.
7. **봉인 창(2026-07-05 00:05 ~ 09-24)은 열지 않았다** — 사용자 확인 뒤 한 번만, 큰 실패를 거르는 용도.

## 봉인 창

- **분석 창**: 2021-04-02 ~ 2026-07-05 00:00 UTC. 모든 성과 숫자는 이 창에서 나왔다.
- **봉인 창**: 2026-07-05 00:05 ~ 2026-09-24 (81일). **아직 열지 않았다.** 규칙을 확정한 뒤 사용자 확인을 받고 **한 번만** 연다.
- 81일로는 작은 효과를 확인할 수 없다 — 큰 실패만 거를 수 있다. 자세히: [데이터 · 봉인 규칙](docs/DATA.md#2-봉인-규칙-2026-09-25-부터-문서-61).

## 저장소 구조

```
quant-bot-research/
├── README.md               ← 이 문서
├── docs/
│   ├── INDEX.md            83건 카탈로그 · 사전등록 ↔ 결과 · 정정 관계
│   ├── TIMELINE.md         9단계 연구 여정
│   ├── METHODOLOGY.md      연구 규약 (비용 · 대조군 · 사전등록 · 부트스트랩)
│   ├── DATA.md             데이터 · 수집 · 감사 · 봉인 규칙
│   ├── LESSONS.md          무엇이 안 됐고 왜
│   ├── STRATEGY_TSMOM.md   최종 전략
│   ├── GLOSSARY.md         용어집
│   └── experiments/        원문 83개 (NNN_<이름>.md, 앞머리 메타데이터)
├── figures/                PNG 41장 + index.json
├── out/                    HTML 보고서 16개 (브라우저로 연다) + 보고서용 표
├── qbot/                   패키지 — data/ (로더 · 리샘플) · research/ (연구별 순수 함수)
├── examples/               실험 스크립트 48개 — *_build → *_run → *_report
├── tests/                  단위 테스트 332개 (합성 데이터)
├── tools/                  수집기 · 감사 · 소스 백업 · 문서 빌드
├── runs/  results_*.csv    작은 산출물 (보고서 입력)
└── _meta/                  catalog.json · doclist.json · codemap.json
```

HTML 보고서(`out/*.html`)는 외부 스크립트 없이 자체 완결이다. 저장소를 받은 뒤 브라우저로 열면 된다.

## 재현

**환경**: Python 3.11 · pandas 3.0.2 · numpy 2.4.4 · pytest · matplotlib 3.10(`examples/legacy_*.py` 두 개만) · requests(`tools/okx_collect.py` 만).

```bash
cd quant-bot-research
pip install -r requirements.txt
pytest -q                     # → 332 passed (원자료 없이, 약 35초)
```

- 스크립트는 **이 폴더(`quant-bot-research/`)에서** 실행한다(상대 경로 `cache_okx/` · `runs/` · `out/`).
- 보고서 스크립트는 이 저장소의 `runs/` · `results_*.csv` 만으로 다시 돌아간다 → HTML 11개(`tsmom_*` 8개 · `weekly_tgif` · `wgap` · `fvg_universe47`)가 원본과 바이트 단위로 같다.
  `costgate.html` · `legacy_curves.html` 은 큰 산출물(`runs/legacy/*.pkl`, 약 160MB)이 있어야 다시 만들어진다.
  `fvg_gate.html` · `fvg_levers.html` · `fvg_sweep.html` 은 **만든 스크립트가 저장소에 없다**(대화 중 일회성으로 만들었다) — 원본 그대로 보존한다.

**원자료부터 다시 (선택)** — 원자료 수집 · 감사는 [docs/DATA.md](docs/DATA.md#3-원자료-받는-법). 그다음 시계열 추세 사슬(47심볼 · 봉인 전, 앞 단계 산출물을 뒤 단계가 읽는다):

```bash
python examples/legacy_run.py                # 예전 신호 16종 + 일 종가 → runs/legacy/ (daily_close.pkl · tsmom.pkl …) — 문서 63
python examples/tsmom_describe.py            # 기준 전략 기술통계 → runs/tsmom_describe.json — 문서 83
python examples/tsmom_anchor_build.py        # 매 정시 가격 → runs/tsmom_anchor/hourly_close.pkl
python examples/tsmom_gap_build.py           # 코인·주 표 → runs/tsmom_gap/weeks.pkl
python examples/tsmom_gap_run.py             # 금 신호 · 월 시가 · 주말갭 → runs/tsmom_gap.json · tsmom_gap_weekly.csv — 문서 76
python examples/tsmom_size_build.py          # 5분봉 패널 → runs/tsmom_size/{close,high,low}.npy (약 300MB)
python examples/tsmom_size_run.py            # 사이징 판정 → part1.json · weights.pkl — 문서 78 1부
python examples/tsmom_lev_run.py             # 교차 증거금 레버리지 → part2.json — 문서 78 2부
python examples/tsmom_compound_run.py        # 복리 주기 → runs/tsmom_compound.json — 문서 80
python examples/tsmom_optg_run.py            # 매월 갱신 최적 G → runs/tsmom_optg.json — 문서 82
python examples/tsmom_size_report.py         # → out/tsmom_sizing.html (다른 *_report.py 도 같은 방식)
```

## 코드 지도

- `qbot/data/` — 로더(`loader.py`: CSV · CSV.GZ, 무결성 리포트) · 리샘플(`resample.py`: 5m → 상위 TF, **룩어헤드가 막히는 유일한 지점**)
- `qbot/research/` — 연구별 순수 함수. 시계열 추세의 핵심은 `legacy.py::tsmom_weekly`(기준 전략) · `sizing.py`(비중 · 장부 · 복리 주기 · 장중 경로 · 교차 증거금 청산)
- `examples/` — 한 실험 = `*_build.py`(데이터 패널) → `*_run.py`(사전등록 그대로 실행 → `runs/*.json`) → `*_report.py`(HTML → `out/`)
- `tests/` — 합성 5분봉(`conftest.py::synth_5m`)으로 도는 단위 테스트 332개. 원자료 없이 통과한다.
- `tools/` — `okx_collect.py`(수집) · `data_audit.py`(감사) · `backup_md.py`(소스를 마크다운 한 장으로) · `docs_build.py`(실험 문서 front matter · 링크 · 그림 절)

<details>
<summary>모듈 · 스크립트별 설명 (docstring 첫 줄)</summary>

| 파일 | 설명 |
|---|---|
| [`qbot/data/loader.py`](qbot/data/loader.py) | OHLCV CSV 로더 + 무결성 리포트. |
| [`qbot/data/resample.py`](qbot/data/resample.py) | 5m -> HTF 리샘플과 참조 맵 — 룩어헤드가 막히는 유일한 지점. |
| [`qbot/research/barrier.py`](qbot/research/barrier.py) | 범용 배리어 레이서 + 치환 대조군. |
| [`qbot/research/changepoint.py`](qbot/research/changepoint.py) | 월별 손익의 평균 변화점 — 수익 곡선의 기울기가 꺾인 곳. |
| [`qbot/research/choch.py`](qbot/research/choch.py) | 스윙(프랙탈) 탐지 — 구조 분석의 최소 원자. |
| [`qbot/research/costgate.py`](qbot/research/costgate.py) | 비용 상태 스위치 — `claude/비용상태스위치-사전등록.md` 의 규칙 그대로. |
| [`qbot/research/execution.py`](qbot/research/execution.py) | 체결 모델 — 지정가 진입 · 지정가 익절 · 스톱 시장가 손절. |
| [`qbot/research/fvg.py`](qbot/research/fvg.py) | FVG (3봉 임밸런스) 되돌림 진입. |
| [`qbot/research/fvg_sweep.py`](qbot/research/fvg_sweep.py) | FVG 관통 이후 — 반등은 어디서 일어나는가. |
| [`qbot/research/legacy.py`](qbot/research/legacy.py) | 예전 신호군 재구축 — 2026-09-18 코드 유실로 사라진 것들을 문서 정의대로 다시 짠다. |
| [`qbot/research/pain_decomp.py`](qbot/research/pain_decomp.py) | painR3 분해 — 1인당 기여를 한 번만 계산하고 임의로 재조합한다. |
| [`qbot/research/pain_proxy.py`](qbot/research/pain_proxy.py) | 무상태 프록시 — 트레이더 시뮬레이션 없이 "손절 임박 불균형" 을 재기. |
| [`qbot/research/sizing.py`](qbot/research/sizing.py) | 시계열 추세 — 코인별 사이징 · 교차 증거금 레버리지. |
| [`qbot/research/trader_pop.py`](qbot/research/trader_pop.py) | 합성 트레이더 집단 — 지표마다 자기 논리의 손절/청산을 갖는 가상 참가자들. |
| [`qbot/research/tsmom_gap.py`](qbot/research/tsmom_gap.py) | 시계열 추세 — 금 종가 신호 · 월 장 시가 체결 · 주말갭 지정가 진입. |
| [`qbot/research/tsmom_rules.py`](qbot/research/tsmom_rules.py) | 시계열 추세에 얹는 규칙 — `claude/시계열추세-뒤집기보류-ATR-사전등록.md`. |
| [`qbot/research/weekly.py`](qbot/research/weekly.py) | 주봉 캔들 · 요일 일봉 — `claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`. |
| [`qbot/research/wgap.py`](qbot/research/wgap.py) | 주말 갭 × 4H 본장정렬 × 15m CISD — `claude/주말갭-4H정렬-CISD-사전등록.md` 그대로. |
| [`qbot/research/yardstick.py`](qbot/research/yardstick.py) | 공통 잣대 — 모든 상태량 연구가 같은 자로 재도록. |
| [`tools/backup_md.py`](tools/backup_md.py) | 소스 전체를 마크다운 한 장으로 — 프로젝트 문서 `claude/code/qbot-소스-백업.md` 용. |
| [`tools/data_audit.py`](tools/data_audit.py) | 수집 데이터 감사 — **표시만 하고 제거하지 않는다.** |
| [`tools/okx_collect.py`](tools/okx_collect.py) | OKX 5분봉 수집기 — 한도를 지키면서 끝까지 간다. |
| [`examples/costgate_report.py`](examples/costgate_report.py) | 비용 상태 스위치 결과 보고서 (HTML) — 판정표 + 'bps 엣지 vs 비용' 점그림. |
| [`examples/costgate_run.py`](examples/costgate_run.py) | 비용 상태 스위치 — 사전등록(`claude/비용상태스위치-사전등록.md`) 그대로 실행. |
| [`examples/fvg_baseline_audit.py`](examples/fvg_baseline_audit.py) | FVG 기준선 — 치환 대조군 대비, TF 6종, 측정해상도 규칙 on/off. |
| [`examples/fvg_era_control.py`](examples/fvg_era_control.py) | 시기별 치환 대조군 — 거래마다 "같은 시기 안의 무작위 시점" 기대 R. |
| [`examples/fvg_excursion.py`](examples/fvg_excursion.py) | 1단계 — 반등은 어디서 일어나는가. 기술통계, 판정 없음. |
| [`examples/fvg_gate_stability.py`](examples/fvg_gate_stability.py) | 1R 게이트 안정성 — 최적 컷오프를 찾지 않는다. |
| [`examples/fvg_gate_stability_report.py`](examples/fvg_gate_stability_report.py) | 게이트 안정성 보고 — 부호가 유지되는가만 본다. |
| [`examples/fvg_gate_universe_report.py`](examples/fvg_gate_universe_report.py) | 47심볼 게이트 안정성 보고 — 10심볼 결과가 표본 탓이었는지 가른다. |
| [`examples/fvg_liquidity_split.py`](examples/fvg_liquidity_split.py) | 보조 절단 검사 — 유동성 3분위가 연도·홀드아웃에서도 버티는가. |
| [`examples/fvg_r_levers.py`](examples/fvg_r_levers.py) | FVG — TF 를 올리지 않고 1R 을 키우는 세 레버. |
| [`examples/fvg_r_levers_report.py`](examples/fvg_r_levers_report.py) | 1R 레버 보고 — 주 지표 판정 + 풍경 기술. |
| [`examples/fvg_regime_split.py`](examples/fvg_regime_split.py) | 2024 전/후 분할 — 무엇이 뒤집혔고, 뒤집힌 채로 안정적인가. |
| [`examples/fvg_sweep_choch.py`](examples/fvg_sweep_choch.py) | 2단계 — 갭이 뚫린 뒤 CHoCH 에 진입한다. |
| [`examples/fvg_sweep_choch_report.py`](examples/fvg_sweep_choch_report.py) | 스윕+CHoCH 보고 — 주 지표 판정 + 대조군 B 비교. |
| [`examples/fvg_universe_html.py`](examples/fvg_universe_html.py) | 47심볼 재검사 한 장 요약 (HTML). 숫자는 전부 runs/u47_*.csv 에서 다시 계산한다. |
| [`examples/legacy_changepoint.py`](examples/legacy_changepoint.py) | 예전 신호 전부 — 수익 곡선과 변곡점. |
| [`examples/legacy_report.py`](examples/legacy_report.py) | 예전 신호 16종 보고서 — 변곡점 원인 분해 + HTML + 문서용 표. |
| [`examples/legacy_run.py`](examples/legacy_run.py) | 예전 신호군 전부 — 47심볼 · 봉인 전(2026-07-05) · 신호별 거래 목록. |
| [`examples/legacy_validate.py`](examples/legacy_validate.py) | 재구축 검수 — 문서에 적힌 숫자가 다시 나오는가 (10심볼 · 문서와 같은 구간). |
| [`examples/tgif.py`](examples/tgif.py) | B. TGIF 요일 패턴 실존 검정 — 사전등록 그대로 (`claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`). |
| [`examples/tsmom_anchor_build.py`](examples/tsmom_anchor_build.py) | 시간 종가 패널 (47심볼 · 봉인 전) — 체결 시각 전수 스윕용. |
| [`examples/tsmom_anchor_report.py`](examples/tsmom_anchor_report.py) | 체결 시각 지도 — HTML. 입력 runs/tsmom_anchor.json → out/tsmom_anchor.html |
| [`examples/tsmom_anchor_sweep.py`](examples/tsmom_anchor_sweep.py) | 시계열 추세 — 체결 시각 전수(주 168시간) 서술. 사전등록 아님: '금 장마감 체결이 더 낫나' 에 답하기 위한 잡음 지도. |
| [`examples/tsmom_coins.py`](examples/tsmom_coins.py) | 시계열 추세 — 코인별 수익과 변동성 (서술, 판정 아님). |
| [`examples/tsmom_coins_report.py`](examples/tsmom_coins_report.py) | 시계열 추세 — 코인별 수익과 변동성 HTML 보고서 (examples/tsmom_coins.py 산출을 그린다). |
| [`examples/tsmom_compound_report.py`](examples/tsmom_compound_report.py) | 복리 주기 — HTML. 입력 runs/tsmom_compound.json · runs/tsmom_compound_weekly.csv → out/tsmom_compound.html |
| [`examples/tsmom_compound_run.py`](examples/tsmom_compound_run.py) | 복리 주기 — 사이징 기준 자산 갱신 주기 비교 (사전등록 `claude/시계열추세-복리주기-사전등록.md` 그대로). |
| [`examples/tsmom_decomp.py`](examples/tsmom_decomp.py) | 시계열 추세 수익 분해 — '다 같이 움직이는 부분' 과 '코인별 부분' (서술, 판정 아님). |
| [`examples/tsmom_decomp_report.py`](examples/tsmom_decomp_report.py) | 시계열 추세 수익 분해 — HTML 보고서 (examples/tsmom_decomp.py 산출을 그린다). |
| [`examples/tsmom_describe.py`](examples/tsmom_describe.py) | 시계열 추세 — 운용 설명서용 기술통계 (47심볼 · 봉인 전). 판정이 아니라 서술이다. |
| [`examples/tsmom_gap_build.py`](examples/tsmom_gap_build.py) | 금 신호 · 월 시가 · 주말갭 지정가 — 코인·주 표 만들기 (47심볼 · 봉인 전). |
| [`examples/tsmom_gap_report.py`](examples/tsmom_gap_report.py) | 금 신호 · 월 시가 · 주말갭 지정가 — HTML 보고서. 입력 runs/tsmom_gap.json · runs/tsmom_gap_weekly.csv → out/tsmom_gapentry.html |
| [`examples/tsmom_gap_run.py`](examples/tsmom_gap_run.py) | 금 신호 · 월 시가 · 주말갭 지정가 — 판정 + 서술. 사전등록 그대로. |
| [`examples/tsmom_gap_verify.py`](examples/tsmom_gap_verify.py) | 독립 재계산 — tsmom_gap 모듈 없이. |
| [`examples/tsmom_hold.py`](examples/tsmom_hold.py) | 시계열 추세 + 뒤집기 보류(주봉 ATR) — 사전등록 그대로 (`claude/시계열추세-뒤집기보류-ATR-사전등록.md`). |
| [`examples/tsmom_hold_report.py`](examples/tsmom_hold_report.py) | 시계열 추세 + 뒤집기 보류 — HTML 보고서. 입력 runs/tsmom_hold*.json · csv → out/tsmom_hold.html |
| [`examples/tsmom_lev_run.py`](examples/tsmom_lev_run.py) | 2부 — 교차 증거금 레버리지 (사전등록 `claude/시계열추세-사이징-레버리지-사전등록.md` 그대로). |
| [`examples/tsmom_optg_report.py`](examples/tsmom_optg_report.py) | 매월 갱신 최적 G — HTML. 입력 runs/tsmom_optg.json → out/tsmom_optg.html |
| [`examples/tsmom_optg_run.py`](examples/tsmom_optg_run.py) | 매월 갱신에서 복리가 가장 큰 G (사전등록 `claude/시계열추세-매월갱신-최적G-사전등록.md` 그대로). |
| [`examples/tsmom_size_build.py`](examples/tsmom_size_build.py) | 사이징 · 레버리지용 5분봉 패널 (47심볼 · 봉인 전). |
| [`examples/tsmom_size_report.py`](examples/tsmom_size_report.py) | 사이징 · 교차 증거금 레버리지 — HTML. 입력 runs/tsmom_size/{part1,part2}.json · weights.pkl → out/tsmom_sizing.html |
| [`examples/tsmom_size_run.py`](examples/tsmom_size_run.py) | 1부 — 사이징 판정 (사전등록 `claude/시계열추세-사이징-레버리지-사전등록.md` 그대로). |
| [`examples/weekly_build.py`](examples/weekly_build.py) | 47심볼 주봉(UTC) · 일봉(뉴욕 · UTC) — 봉인 전. `claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`. |
| [`examples/weekly_candles.py`](examples/weekly_candles.py) | A. 주봉 캔들(몸통·꼬리) → 다음 주 — 사전등록 그대로 (`claude/주봉캔들-꼬리몸통-및-TGIF-사전등록.md`). |
| [`examples/weekly_report.py`](examples/weekly_report.py) | 주봉 캔들 + TGIF — HTML 보고서. |
| [`examples/wgap_report.py`](examples/wgap_report.py) | 주말 갭 × 4H 본장정렬 × 15m CISD — HTML 보고서. |
| [`examples/wgap_run.py`](examples/wgap_run.py) | 주말 갭 × 4H 본장정렬 × 15m CISD — 47심볼 · 봉인 전(2026-07-05). |
| [`examples/wgap_stats.py`](examples/wgap_stats.py) | 주말 갭 × 4H 본장정렬 × 15m CISD — 판정과 진단 집계 (사전등록 기준 그대로). |

</details>

## 저자 · 라이선스

- 저자: [DongGyunSeo](https://github.com/DongGyunSeo)
- 라이선스: [MIT](LICENSE)
