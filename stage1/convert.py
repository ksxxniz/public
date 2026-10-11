"""2단계 데이터(metadata_*.jsonl + 문서 파일)를 학습 형식(datagen/evasion.py 형식)으로 바꾼다.

    uv run python -m stage1.convert datasets/v0.1 --out data/v0.1.jsonl

입력 (v0.1 형식)
  metadata_ko.jsonl / metadata_en.jsonl 한 줄 = 문서 하나
  {"id": "ko_0001", "context": "…", "label": 1, "language": "ko", "file_path": "ko_0001.txt"}

출력 (학습 형식, stage1/dataset.py 가 읽는 형식)
  {"id": "ko_0001", "text": "…", "label": "injection", "lang": "ko", "fmt": "txt",
   "source_group": "ko_0001", "source_file": "ko/ko_0001.txt"}

바꾸는 규칙
  text          file_path 의 파일 내용 (기본). 실제 RAG 가 읽는 것과 같게 하려고
                md/html 태그까지 그대로 둔다. --text-source context 면 context 칸을 쓴다
  label         1 → "injection", 0 → "benign" (이미 문자열이면 그대로)
  lang          language 칸 (없으면 lang 칸)
  fmt           파일 확장자 (txt / md / html)
  source_group  있으면 그대로, 없으면 id. 번역본처럼 같은 원본에서 나온 문서는
                데이터 담당이 같은 값을 넣어줘야 train/val 이 섞이지 않는다
  answer        있으면 그대로 둔다. text 안에 글자 그대로 있어야 한다 (없으면 에러)
  나머지 칸은 그대로 옮긴다 (context, language, file_path 는 위에서 바꿨으므로 뺀다)

아직 안 하는 것: 긴 문서를 384 토큰 조각으로 자르기. answer 가 채워지면 추가한다.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from preprocess.normalize import resolve_format

LABEL_MAP = {0: "benign", 1: "injection", "0": "benign", "1": "injection"}
VALID_LABELS = ("benign", "injection")
DROP_KEYS = ("context", "language", "file_path")


class ConvertError(ValueError):
    """변환 실패. 어느 파일 몇 번째 줄인지 메시지에 담는다."""


def find_metadata(inputs: list[Path]) -> list[Path]:
    """폴더를 주면 그 아래의 metadata_*.jsonl 을 모두 찾는다. 파일을 주면 그대로 쓴다."""
    found = []
    for p in inputs:
        if p.is_dir():
            found.extend(sorted(p.rglob("metadata_*.jsonl")))
        elif p.is_file():
            found.append(p)
        else:
            raise ConvertError(f"없는 경로입니다: {p}")
    if not found:
        raise ConvertError(f"metadata_*.jsonl 을 찾지 못했습니다: {', '.join(map(str, inputs))}")
    return found


def convert_file(
    meta_path: Path, text_source: str = "file", root: Path | None = None
) -> list[dict]:
    """metadata 파일 하나를 변환한다. root 는 source_file 을 상대 경로로 적을 기준 폴더."""
    rows = []
    with meta_path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                rows.append(
                    convert_row(line, meta_path, f"{meta_path.name}:{n}", text_source, root)
                )
    return rows


def convert_row(
    line: str, meta_path: Path, where: str, text_source: str, root: Path | None
) -> dict:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as e:
        raise ConvertError(f"{where}: JSON 이 아닙니다 ({e.msg})") from e

    if "id" not in rec:
        raise ConvertError(f"{where}: id 칸이 없습니다")
    sid = str(rec["id"])

    label = rec.get("label")
    label = LABEL_MAP.get(label, label)
    if label not in VALID_LABELS:
        raise ConvertError(f"{where}: label 은 0/1 또는 benign/injection 이어야 합니다: {label!r}")

    file_path = rec.get("file_path")
    doc = meta_path.parent / file_path if file_path else None
    if text_source == "file":
        if doc is None or not doc.is_file():
            raise ConvertError(f"{where}: 문서 파일이 없습니다: {doc}")
        text = doc.read_text(encoding="utf-8-sig")
    else:
        text = rec.get("context")
        if not isinstance(text, str):
            raise ConvertError(f"{where}: context 칸(문자열)이 없습니다")

    try:
        fmt = resolve_format(text, Path(file_path).suffix) if file_path else "txt"
    except ValueError as e:
        raise ConvertError(f"{where}: {e}") from e

    answer = rec.get("answer")
    if answer and answer not in text:
        raise ConvertError(f"{where}: answer 가 text 안에 그대로 없습니다 (띄어쓰기·마침표 확인)")

    out = {k: v for k, v in rec.items() if k not in DROP_KEYS}
    out.update(
        id=sid,
        text=text,
        label=label,
        lang=rec.get("language") or rec.get("lang"),
        fmt=fmt,
        source_group=str(rec.get("source_group") or sid),
    )
    if doc is not None:
        base = root or meta_path.parent
        try:
            out["source_file"] = doc.relative_to(base).as_posix()
        except ValueError:
            out["source_file"] = doc.as_posix()
    return out


def summarize(rows: list[dict]) -> str:
    by_lang = Counter((r.get("lang") or "?", r["label"]) for r in rows)
    fmts = Counter(r["fmt"] for r in rows)
    lines = [f"총 {len(rows)}개"]
    for lang in sorted({k[0] for k in by_lang}):
        n_ben, n_inj = by_lang[(lang, "benign")], by_lang[(lang, "injection")]
        lines.append(f"  {lang}: benign {n_ben} / injection {n_inj}")
    lines.append("  fmt: " + ", ".join(f"{k} {v}" for k, v in sorted(fmts.items())))
    n_answer = sum(1 for r in rows if r.get("answer"))
    n_inj = sum(1 for r in rows if r["label"] == "injection")
    lines.append(f"  answer 있는 공격 문서: {n_answer} / {n_inj}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="2단계 데이터 → 학습 형식(evasion.py 형식) 변환")
    p.add_argument("inputs", nargs="+", type=Path, help="데이터 폴더 또는 metadata_*.jsonl")
    p.add_argument("--out", type=Path, required=True, help="저장할 JSONL (data/ 아래 권장)")
    p.add_argument(
        "--text-source",
        choices=["file", "context"],
        default="file",
        help="text 에 넣을 내용. file = 문서 파일 내용(기본), context = metadata 의 context",
    )
    args = p.parse_args(argv)

    try:
        metas = find_metadata(args.inputs)
        root = args.inputs[0] if len(args.inputs) == 1 and args.inputs[0].is_dir() else None
        rows = [r for m in metas for r in convert_file(m, args.text_source, root)]
    except ConvertError as e:
        p.exit(1, f"오류: {e}\n")

    dup = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
    if dup:
        p.exit(1, f"오류: id 가 중복됩니다: {', '.join(dup[:5])}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(summarize(rows))
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
