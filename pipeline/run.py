"""파이프라인 진입점 (5단계 ④).

  python -m pipeline.run --file 문서.md
  python -m pipeline.run --file 문서.html --stage1 stub --stage2 stub --policy mask --json

문서를 전처리 → 룰 → 1차 → 2차 → 판정 합치기 → 정책(마스킹) → 로그로 검사한다.
부품은 준비되는 대로 바꾼다 (⑧): 룰은 진짜, 1차·2차는 기본이 스텁이고
GPU·ollama 가 있는 곳에서 --stage1 real / --stage2 real 로 돌린다.
"""

import argparse
import json
import sys

from common import config, logdb
from common.console import safe_stdio
from pipeline.components import ComponentUnavailable, load_stage1, load_stage2
from pipeline.orchestrator import Orchestrator


def build(args) -> Orchestrator:
    try:
        stage1 = load_stage1(args.stage1, args.stage1_model_dir)
    except ComponentUnavailable as e:
        print(f"경고: {e} — 1차 없이 합의한 fail-safe 로 진행 (⑥)", file=sys.stderr)
        stage1 = None
    try:
        stage2 = load_stage2(args.stage2, args.model)
    except ComponentUnavailable as e:
        print(f"경고: {e} — 2차 없이 진행, 2차 필요 조각은 fail-safe (⑥)", file=sys.stderr)
        stage2 = None
    conn = None if args.no_db else logdb.connect(args.db)
    return Orchestrator(
        stage1,
        stage2,
        conn,
        policy=args.policy,
        use_cache=not args.no_cache,
    )


def print_report(report, *, as_json: bool = False) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "request_id": report.request_id,
                    "doc": report.doc.model_dump(),
                    "masked_texts": report.masked_texts,
                    "routes": report.routes,
                    "timings": vars(report.timings),
                    "warnings": report.warnings,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return
    doc = report.doc
    if report.blocked_reason:
        print(f"[{doc.doc_id}] 문서 전체 차단: {report.blocked_reason}")
        return
    print(f"[{doc.doc_id}] 최종 {doc.final_label} / 조치 {doc.action}  ({doc.pipeline_ver})")
    for v in doc.verdicts:
        mark = {"allow": "·", "warn": "!", "mask": "▒", "block": "█"}[v.action]
        extras = []
        if v.cache_hit:
            extras.append("재사용")
        if v.fallback:
            extras.append("fail-safe")
        note = f" [{', '.join(extras)}]" if extras else ""
        print(
            f"  {mark} {v.chunk_id}: {v.final_label} {v.final_score:.2f} "
            f"({v.meta.get('merge', '')}) 경로 {'→'.join(v.path)}{note}"
        )
    t = report.timings
    print(
        f"  2차 경로 {report.routes} / 시간(ms) 전처리 {t.preprocess_ms:.0f}"
        f" 룰 {t.rule_ms:.0f} 1차 {t.stage1_ms:.0f} 2차 {t.stage2_ms:.0f}"
        f" 전체 {t.total_ms:.0f}"
    )
    for w in report.warnings:
        print(f"  경고: {w}")


def add_component_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage1", choices=["stub", "real", "off"], default="stub")
    parser.add_argument("--stage1-model-dir", default=None)
    parser.add_argument(
        "--stage2", choices=["stub", "benign", "broken_json", "real", "off"], default="stub"
    )
    parser.add_argument("--model", default="exaone", help="2차 real 일 때 MODELS 키")
    parser.add_argument(
        "--policy", choices=["mask", "block", "warn"], default=config.DEFAULT_POLICY
    )
    parser.add_argument("--db", default=str(config.DB_PATH))
    parser.add_argument("--no-db", action="store_true", help="로그·판정 저장 없이 돌린다")
    parser.add_argument("--no-cache", action="store_true", help="저장된 판정을 재사용하지 않는다")


def main(argv: list[str] | None = None) -> None:
    from pathlib import Path

    from preprocess.chunk import DocumentTooLongError
    from preprocess.load import UnsupportedDocumentError, read_document

    safe_stdio()
    parser = argparse.ArgumentParser(description="prompt-injection-protection 파이프라인")
    parser.add_argument("--file", required=True, nargs="+", help="검사할 문서 경로")
    parser.add_argument("--json", action="store_true")
    add_component_args(parser)
    args = parser.parse_args(argv)
    orch = build(args)
    for path in args.file:
        # 조작자 실수(없는 파일 등)는 traceback 없이 종료 코드 2.
        # 서비스 경로(inspect_file)에서는 같은 오류가 "문서 전체 차단"이 된다 (⑥)
        try:
            doc = read_document(path)
        except (UnsupportedDocumentError, DocumentTooLongError) as e:
            parser.exit(2, f"오류: {e}\n")
        except FileNotFoundError:
            parser.exit(2, f"오류: 파일이 없습니다: {path}\n")
        except OSError as e:  # 폴더를 넘겼거나 권한이 없는 경우 등
            parser.exit(2, f"오류: 파일을 읽을 수 없습니다: {path} ({e.strerror or e})\n")
        report = orch.inspect_text(doc.raw, doc_id=Path(path).stem, fmt=doc.fmt)
        if report.blocked_reason:
            parser.exit(2, f"오류: {report.blocked_reason}\n")
        print_report(report, as_json=args.json)


if __name__ == "__main__":
    main()
