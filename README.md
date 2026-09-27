# For-Quantitative-trading

퀀트 트레이딩 개인 연구 저장소입니다. 프로젝트마다 폴더를 따로 둡니다.

## 프로젝트

| 프로젝트 | 상태 | 설명 |
|---|---|---|
| [`ict-quant-bot/`](ict-quant-bot/) | 중단 (아카이브) | ICT 개념 기반 규칙 전략 + XGBoost 진입 필터, 코인 선물 5분봉 |
| [`quant-bot-research/`](quant-bot-research/) | 연구 완료 · 봉인 창 미개봉 | ICT 셋업 83건 검정 → 전부 비용·표본 밖에서 기각, 주간 시계열 추세(TSMOM)만 남음. OKX 47심볼 5분봉 |

## 브랜치 운영 규칙

- **`main`** — 완성되었거나 정리된 내용만 둡니다. 직접 작업하지 않습니다.
- **`project/<이름>`** — 새 프로젝트는 이 형태의 브랜치에서 `<이름>/` 폴더를 만들어 작업합니다.
  예: `project/orderflow-bot` 브랜치 → `orderflow-bot/` 폴더
- 작업이 한 단계 마무리되면 Pull Request로 `main`에 합칩니다.
- 프로젝트를 끝내거나 중단하면 `main`에 합친 뒤 `<이름>-final` 태그를 붙이고, 이 표의 상태를 바꿉니다.

데이터·캐시·학습된 모델 같은 대용량 파일은 올리지 않습니다(`.gitignore` 참고). GitHub는 100MB가 넘는 파일을 거부합니다.
