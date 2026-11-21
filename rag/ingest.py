"""문서 넣기 명령 (5단계 ①·③ '저장할 때' 검문소).

  uv run python -m rag.ingest --dir rag/sample_docs          # 폴더를 통째로 넣기
  uv run python -m rag.ingest --dir 문서폴더 --prune            # 지운 문서는 색인에서도 빼기
  uv run python -m rag.ingest --file 공지.html                 # 파일 하나씩
  uv run python -m rag.ingest --list                           # 넣은 문서 목록
  uv run python -m rag.ingest --remove sample_docs/hr/휴가_규정.md

문서마다 전처리 → 룰 → 1차 → 2차 검사를 하고, 판정과 청크를 저장한다. 공격 판정 문서도
색인에 남긴다 (질문할 때 검문소가 막고, 화면에 "여기에 공격이 있었다"를 보여 주려고).
바뀌지 않은 문서는 건너뛴다. 원문 해시와 구성(pipeline_ver)이 같으면 같은 문서로 본다.
"""

import argparse
import sys
from pathlib import Path

from common.console import safe_stdio
from rag.cli import add_rag_args, build_service

_TAGS = {
    "indexed": "[넣음]  ",
    "unchanged": "[그대로]",
    "blocked": "[차단]  ",
    "skipped": "[건너뜀]",
    "removed": "[지움]  ",
}


def describe(r) -> str:
    line = f"{_TAGS[r.status]} {r.doc_id}"
    if r.status == "indexed":
        detail = f"조각 {r.n_chunks}개"
        if r.n_attack_chunks:
            detail += f", 공격 판정 {r.n_attack_chunks}개 (질문할 때 정책대로 처리)"
        else:
            detail += ", 정상"
        if r.encoding and r.encoding != "utf-8":
            detail += f", 인코딩 {r.encoding}"
        if r.n_fallback_chunks:
            detail += f", 장애로 판정 못 한 조각 {r.n_fallback_chunks}개 — 다음 넣기 때 다시 검사"
        line += f" — {detail}"
    elif r.status in ("blocked", "skipped", "removed"):
        line += f" — {r.reason}"
    for w in r.warnings:
        line += f"\n           경고: {w}"
    return line


def main(argv: list[str] | None = None) -> None:
    safe_stdio()
    parser = argparse.ArgumentParser(description="RAG 에 문서 넣기 (넣을 때 검사하고 판정 저장)")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--dir", help="이 폴더 안의 txt·md·html 을 모두 넣는다 (하위 폴더 포함)")
    what.add_argument("--file", nargs="+", help="파일을 하나씩 넣는다")
    what.add_argument("--list", action="store_true", help="넣은 문서 목록")
    what.add_argument("--remove", nargs="+", metavar="문서ID", help="색인에서 뺀다")
    parser.add_argument("--prune", action="store_true", help="--dir 에서 사라진 문서를 뺀다")
    parser.add_argument("--force", action="store_true", help="바뀌지 않은 문서도 다시 검사")
    parser.add_argument("-v", "--verbose", action="store_true", help="건너뛴 파일도 보여 준다")
    add_rag_args(parser)
    args = parser.parse_args(argv)
    # 잘못 적은 경로는 DB 를 만들기 전에 알린다. 없는 파일을 '차단된 문서'로 기록하지 않는다
    if args.dir and not Path(args.dir).is_dir():
        parser.exit(2, f"오류: 폴더가 아닙니다: {args.dir}\n")
    missing = [p for p in args.file or [] if not Path(p).is_file()]
    if missing:
        parser.exit(2, f"오류: 파일을 찾지 못했습니다: {', '.join(missing)}\n")
    svc = build_service(args)

    if args.list:
        rows = svc.docs()
        for d in rows:
            flags = " (구성이 바뀜 — 질문할 때 다시 검사)" if d["stale"] else ""
            print(f"[{d['status_label']}] {d['doc_id']} — 조각 {d['n_chunks']}개{flags}")
        print(f"모두 {len(rows)}개 문서")
        return
    if args.remove:
        for doc_id in args.remove:
            print(f"{'[지움]  ' if svc.remove(doc_id) else '[없음]  '} {doc_id}")
        return

    counts: dict[str, int] = {}

    def show(r) -> None:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status != "skipped" or args.verbose:
            print(describe(r), flush=True)

    if args.dir:
        svc.ingest_dir(args.dir, prune=args.prune, force=args.force, progress=show)
    else:
        for path in args.file:
            show(svc.ingest_file(path, force=args.force))

    summary = " · ".join(
        f"{label} {counts.get(key, 0)}"
        for key, label in (
            ("indexed", "넣음"),
            ("unchanged", "그대로"),
            ("blocked", "차단"),
            ("skipped", "건너뜀"),
            ("removed", "지움"),
        )
    )
    print(f"\n요약: {summary}")
    if counts.get("skipped") and not args.verbose:
        print("  (건너뛴 파일은 -v 로 볼 수 있습니다. 지원 형식: txt·md·html)")
    print(f"구성: {svc.orch.pipeline_ver}")
    print('질문하기: uv run python -m rag.ask "연차는 며칠이야?"   화면: uv run python -m web')
    sys.stdout.flush()


if __name__ == "__main__":
    main()
