"""일괄 재검사 (5단계 ⑦).

재학습·규칙 수정·임계값 변경 뒤에는 pipeline_ver 가 바뀌므로, 색인된 문서를
새 버전으로 다시 검사해 저장된 판정을 갈아 끼운다. 8주차 레드팀 대응 때 쓴다.

  python -m pipeline.recheck --file docs/*.md
"""

import argparse

from pipeline import store
from pipeline.run import add_component_args, build, print_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="색인 문서 일괄 재검사 (⑦)")
    parser.add_argument("--file", required=True, nargs="+", help="다시 검사할 문서 경로")
    parser.add_argument("--json", action="store_true")
    add_component_args(parser)
    args = parser.parse_args(argv)
    args.no_cache = True  # 재검사는 저장된 판정을 쓰지 않고 새로 만든다
    orch = build(args)
    for path in args.file:
        report = orch.inspect_file(path)
        if orch.conn is not None:
            # 옛 버전 판정을 지우고 새 판정만 남긴다 (inspect 가 새 버전으로 저장했다)
            old = store.drop_doc(orch.conn, report.doc.doc_id)
            for v in report.doc.verdicts:
                from pipeline.version import raw_hash

                chunk = next(c for c in report.chunks if c.chunk_id == v.chunk_id)
                store.put(orch.conn, report.doc.doc_id, raw_hash(chunk.raw_text), v)
            print(f"[{report.doc.doc_id}] 옛 판정 {old}건 삭제, {len(report.doc.verdicts)}건 저장")
        print_report(report, as_json=args.json)


if __name__ == "__main__":
    main()
