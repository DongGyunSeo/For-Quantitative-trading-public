"""실험 문서 83개에 front matter · 링크 · 그림 절을 붙인다 — 저장소 문서화용, 한 번만 돌린다.

- front matter 값은 `_meta/catalog.json` 그대로. 문서 사이 관계(사전등록 ↔ 결과, 정정)는 `related:`.
- 본문 속 `claude/<이름>.md` · `<이름>.md` 참조를 같은 폴더의 `NNN_<이름>.md` 링크로 바꾼다(코드 블록 안은 건드리지 않는다).
- `figures/index.json` 의 그림을 해당 문서 끝 `## 그림` 절에 덧붙인다.

본문은 위 링크 재작성 말고는 바꾸지 않는다. 이미 처리된 파일(front matter 가 있는)은 건너뛴다.
실행: 저장소 루트(quant-bot-research/)에서 `python tools/docs_build.py`
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "docs" / "experiments"

# 사전등록 → 결과 (핸드오프 §7.1)
PREREG = {25: 26, 27: 32, 37: 38, 40: 41, 64: 65, 66: 67, 70: 71, 72: 73, 74: 76, 77: 78, 79: 80, 81: 82}

# 앞 문서 → (뒤 문서들, 무엇이 바뀌었나) (핸드오프 §7.2)
SUPERSEDE = [
    ([11, 12, 13, 14], [23], "OOS 에서 3종 전부 음수 — 기각"),
    ([15], [16, 24], "시간순서 버그 → 실측 검정에서 거울이 125% 재현, 기각"),
    ([19], [22], "펀딩 잠식률 계산 오류(23% → 0.23%) — 제외 근거 없음, 10심볼 유지"),
    ([27], [32, 66, 67], "OOS 기각, 필터 5종으로 다시 봐도 기각"),
    ([28, 29], [30, 34, 35], "표본 2배에서 4H 결과가 잡음 → 거울 · 레인지 기하 효과로 기각"),
    ([33], [34, 35, 36, 38], "1위 기각, 2위는 살아남았지만 비용을 못 넘음"),
    ([41], [63], "47심볼 재측정 샤프 0.641 → 심화 연구 · 운용안 (10심볼 판정은 그대로 기각으로 남는다)"),
    ([46], [47], "진입봉 오염 — 기각"),
    ([45, 50], [51, 52], "비용 8bps 를 못 넘어 최종 기각"),
    ([48, 49], [56, 57, 58, 60, 62], "1R 레버 · 관통 · 게이트 · 47심볼 · 체제 분할 전부 기각"),
    ([78], [80, 82], "사용자가 G 0.75 를 선택 — 권장은 유효, 운용값은 사용자 판단"),
]

# 보고서 → 문서 번호 (핸드오프 §9)
REPORT_DOC = {
    "fvg_levers": 56, "fvg_sweep": 57, "fvg_gate": 58, "fvg_universe47": 60, "legacy_curves": 63,
    "costgate": 65, "wgap": 67, "tsmom_decomp": 68, "tsmom_coins": 69, "weekly_tgif": 71,
    "tsmom_hold": 73, "tsmom_anchor": 75, "tsmom_gapentry": 76, "tsmom_sizing": 78,
    "tsmom_compound": 80, "tsmom_optg": 82,
}


def q(s) -> str:
    return json.dumps(s, ensure_ascii=False)


def main() -> None:
    cat = json.loads((ROOT / "_meta" / "catalog.json").read_text(encoding="utf-8"))
    by_idx = {c["idx"]: c for c in cat}
    fname = {i: Path(c["file"]).name for i, c in by_idx.items()}
    # 원래 이름 → 새 파일
    name_to_file = {Path(c["path"]).name: fname[c["idx"]] for c in cat}

    related: dict[int, dict[str, list]] = {i: {} for i in by_idx}
    for pre, res in PREREG.items():
        related[pre].setdefault("result", []).append(fname[res])
        related[res].setdefault("preregistration", []).append(fname[pre])
    notes: dict[int, str] = {}
    for olds, news, note in SUPERSEDE:
        for o in olds:
            related[o].setdefault("superseded_by", []).extend(fname[n] for n in news)
            notes[o] = note
        for n in news:
            related[n].setdefault("supersedes", []).extend(fname[o] for o in olds)

    figs: dict[int, list] = {}
    for f in json.loads((ROOT / "figures" / "index.json").read_text(encoding="utf-8")):
        doc = REPORT_DOC[Path(f["report"]).stem]
        figs.setdefault(doc, []).append(f)

    ref = re.compile(r"`(?:claude/)?([^`/\s]+\.md)`")

    def relink(line: str) -> str:
        def sub(m: re.Match) -> str:
            target = name_to_file.get(m.group(1))
            return f"[`{m.group(1)}`]({target})" if target else m.group(0)
        return ref.sub(sub, line)

    for i, c in sorted(by_idx.items()):
        path = EXP / fname[i]
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            continue

        out, fence = [], False
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                fence = not fence
            out.append(line if fence else relink(line))
        body = "\n".join(out)

        fm = ["---", f"idx: {i}", f"title: {q(c['title'])}", f"created: {q(c['created'])}",
              f"category: {q(c['category'])}", f"kind: {q(c['kind'])}", f"verdict: {q(c['verdict'])}",
              f"key_numbers: {q(c['key_numbers'])}", f"one_line: {q(c['one_line'])}",
              f"original_path: {q(c['path'])}"]
        rel = related[i]
        if rel:
            fm.append("related:")
            for k in ("preregistration", "result", "supersedes", "superseded_by"):
                if k in rel:
                    fm.append(f"  {k}: [{', '.join(q(x) for x in rel[k])}]")
        if i in notes:
            fm.append(f"superseded_note: {q(notes[i])}")
        fm.append("---")

        if i in figs:
            rep = figs[i][0]
            add = ["", "", "## 그림", "",
                   f"보고서: [{rep['report_title']}](../../{rep['report']}) (`{rep['report']}`)", ""]
            for f in sorted(figs[i], key=lambda f: f["file"]):
                alt = f["caption"] or f["section"]
                add += [f"![{alt}](../../{f['file']})", ""]
                if f["section"]:
                    add += [f"*절: {f['section']}*", ""]
            body = body.rstrip("\n") + "\n".join(add).rstrip("\n") + "\n"

        path.write_text("\n".join(fm) + "\n\n" + body, encoding="utf-8")


if __name__ == "__main__":
    main()
