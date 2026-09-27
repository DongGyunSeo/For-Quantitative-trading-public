"""소스 전체를 마크다운 한 장으로 — 프로젝트 문서 `claude/code/qbot-소스-백업.md` 용.

2026-09-18 컨테이너 회수로 코드가 통째로 사라진 뒤 생겼다. 복원은 `=== FILE: 경로 ===`
마커로 잘라 저장하면 된다(`restore` 서브커맨드).

    python tools/backup_md.py dump    --out out/qbot-소스-백업.md
    python tools/backup_md.py restore --src qbot-소스-백업.md --dst .
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("examples", "qbot", "tests", "tools")

HEADER = """# qbot 소스 백업 (자동 생성)

> **이 문서는 코드 백업이다.** 2026-09-18 클라우드 컨테이너가 회수되면서
> `qbot` 코드베이스 전체가 사라졌다. 같은 일이 반복되지 않도록 소스를 여기 둔다.
>
> **복원 방법**: 아래 `=== FILE: 경로 ===` 마커를 기준으로 잘라 그대로 저장한다.
> `tools/backup_md.py restore --src <이 문서> --dst .` 가 그 일을 한다.
>
> 검수 기준(재구축 시 이 숫자가 나와야 한다):
> - FVG 규칙 없음 1h: 63,032건 · 1R 12.4bps · 실측 +0.176 · 모호 13.8%
> - FVG 4h · mid · 게이트120: 1,288건 · 1R 173.6bps · gross +0.1180 · 순 +0.0810
> - FVG 4h 관통률 91.9%(7일) · 뚫린 뒤 회복 90.3%
> - painR3 (STRUCT 15명, raw): +0.3045 t 5.79 10/10  (문서값 +0.3030 t 5.76)
> - `pivot_break@5m` 가동률 94.7% · 중앙보유 20봉 · 손절율 26.8%
> - 게이트 안정성 10심볼(`--cache cache_all --until none`): 9,264건, 4h 게이트150 876건 순 +0.0993
> - 게이트 안정성 47심볼(봉인 2026-07-05): 47,075건, 4h 게이트150 6,334건 순 +0.0252
> - 2024 분할(시기 내 대조, 거래당 20회): 4h 게이트150 엣지 2021-23 +0.108 / 2024+ −0.017
> - 워크포워드 L=12, 2024Q1~2026Q2: 거래 1,239건 거래당 −0.026 · 부호 지속 70/170
> - t 는 거래 가중 평균의 월 군집 강건 t(`yardstick.cluster_t`). 예전 `t(월)` 은 월평균의 t 였다
> - 예전 신호 16종(`examples/legacy_run.py`): FVG 4h 변곡 2025-09 (전 +7.76 → 후 −54.57 R/월, p≈0.018) ·
>   청산 반전 거래당 gross 2022-08 전후 +0.0465 → +0.0448 · 시계열 추세 47심볼 샤프 0.641
> - 비용 상태 스위치(`examples/costgate_run.py`, 표본 밖 2022-04~2026-06, ρ≥2): 15개 전부 기각 ·
>   FVG 4h 5,464건 −0.0044 (t −0.13) · 청산 반전 10,601건 −0.0362 (t −1.33) · 주말 갭 1,949건 −0.0213
> - 주말 갭 × 4H 본장정렬 × 15m CISD(`examples/wgap_run.py`): 기본형 8,976건 +0.032 (MTM) · 0R 규약 +0.028 ·
>   정렬 A+ 4,137건 +0.009 · CISD 진입 7,462건 −0.078 · 메운 뒤 연속 먼저 41.9% / 반전 먼저 37.2%
> - 시계열 추세 뒤집기 보류(`examples/tsmom_hold.py`): k 0.5 · 1 · 2 ATR 샤프 0.04 · −0.46 · −0.46 (기준 0.641)
> - 시계열 추세 금 신호 · 월 시가 · 주말갭 지정가(`examples/tsmom_gap_run.py`, 공통 269주): V0 0.677 · V2 0.297 ·
>   V3m 0.419 · V3e 0.246 · V3m − V2 +14.0bps/주 (t 1.55) · 갭 중간 체결률 85.8% · 주기 1·2·3·7·14일 0.38·0.47·0.54·0.70·0.42
> - 체결 시각 168시간(`examples/tsmom_anchor_sweep.py`, 시간 가격표 46,080×47): ET 평균 0.70 · 금 16:00 ET 0.775 (= V1) ·
>   월 00:00 UTC 0.677 (= V0) · 7개 요일 나눔 0.74 / MDD −36%
> - 사이징 · 레버리지(`examples/tsmom_size_run.py` · `tsmom_lev_run.py`, 실제 거래량 수수료): S0 0.672 · S1 0.676 · S2 0.727 ·
>   S0+VT 0.723 · 교차 청산 한계 G 2.61 (장중 2배 1.35) · 부트스트랩 권장 G 0.20 (−30%) · 0.40 (−50%) · G 1 MDD p95 84.2%
> - 복리 주기(`examples/tsmom_compound_run.py`, S1 · G 0.75): 매주 2.716배 · 매월 3.054배 · 갱신 안 함 2.428배 ·
>   부트스트랩 매월이 매주를 이긴 비율 87% · 분산비 13주 0.518
> - 매월 갱신 최적 G(`examples/tsmom_optg_run.py`): 부트스트랩 중앙 최적 1.45 (29.2%) · 역사 1.90 (39.9%) · 매주 1.15 ·
>   샤프 0.5 → 1.1 · 0.3 → 0.6 · 역사 G 1.45 연 37.46%
> - 검정 {ntests}개 통과
>
> 데이터: 기존 10심볼 캐시(`cache_all`, 프로젝트 첨부 CSV) + OKX 47심볼(`cache_okx`, 사용자 PC
> `C:\\okx_data\\raw` 의 `*_5m_futures.csv.gz`). 2026-07-05 00:05 이후 81일은 **봉인**이다
> (`claude/OKX-47심볼-데이터감사-및-봉인규칙.md`).
>
> `tools/okx_collect.py` 는 **로컬에서 돌리는 OKX 수집기**다 (사용법은
> `claude/OKX-5분봉-수집기-사용법.md`). history-candles 의 limit 상한은 문서의 100 이 아니라 **300**.
"""


def files() -> list[Path]:
    out = []
    for d in DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return sorted(out, key=lambda p: p.relative_to(ROOT).as_posix())


def dump(a) -> None:
    parts = [HEADER.format(ntests=a.ntests)]
    for p in files():
        rel = p.relative_to(ROOT).as_posix()
        body = p.read_text(encoding="utf-8").rstrip("\n")
        parts.append(f"=== FILE: {rel} ===\n```python\n{body}\n```\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"{len(parts) - 1}개 파일 -> {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} KB)")


def parse(txt: str) -> dict[str, str]:
    """마커 **줄**(줄 머리에서 시작)로만 자른다. 머리말 안의 `=== FILE: 경로 ===` 설명이나
    본문 안의 코드 펜스에 걸리지 않게 — 처음엔 정규식 하나로 짰다가 머리말에 걸렸다."""
    fence = "`" * 3
    marks = list(re.finditer(r"(?m)^=== FILE: ([^\n`]+) ===$", txt))
    out = {}
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(txt)
        chunk = txt[m.end() + 1:end].rstrip()
        if not (chunk.startswith(fence + "python\n") and chunk.endswith(fence)):
            raise ValueError(f"형식이 깨진 블록: {m.group(1)}")
        out[m.group(1).strip()] = chunk[len(fence) + 7:-len(fence)].rstrip("\n")
    return out


def restore(a) -> None:
    got = parse(Path(a.src).read_text(encoding="utf-8"))
    for rel, body in got.items():
        dst = Path(a.dst) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(body + "\n" if body else "", encoding="utf-8")   # 빈 __init__.py 는 빈 채로
    print(f"{len(got)}개 파일 복원 -> {a.dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--out", default="out/qbot-소스-백업.md")
    p.add_argument("--ntests", default="?")
    p = sub.add_parser("restore")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", default=".")
    a = ap.parse_args()
    {"dump": dump, "restore": restore}[a.cmd](a)


if __name__ == "__main__":
    main()
