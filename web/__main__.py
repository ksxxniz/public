"""시연 화면 서버.

  uv run python -m web                                   # http://127.0.0.1:8000
  uv run python -m web --ingest-dir rag/sample_docs      # 시작할 때 폴더를 넣고 켜기
  uv run python -m web --answer real --stage2 real --model exaone   # ollama 로

문서는 python -m rag.ingest 로 같은 DB 에 넣는다. 서버가 켜져 있는 동안 다른 창에서
넣어도 다음 질문부터 반영된다.
"""

import argparse
from pathlib import Path

from common.console import safe_stdio
from rag.cli import add_answer_args, add_rag_args, build_service
from rag.ingest import describe


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from web.app import create_app

    safe_stdio()
    parser = argparse.ArgumentParser(description="RAG 프롬프트 인젝션 탐지 시스템 화면")
    parser.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 이면 같은 망의 다른 PC 에서도 접속"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--ingest-dir", action="append", default=[], metavar="폴더", help="시작할 때 넣을 폴더"
    )
    add_answer_args(parser)
    add_rag_args(parser)
    args = parser.parse_args(argv)
    bad = [folder for folder in args.ingest_dir if not Path(folder).is_dir()]
    if bad:  # 서버를 켜기 전에 알린다 (켠 뒤에 실패하면 긴 오류 메시지만 남는다)
        parser.error(f"폴더가 아닙니다: {', '.join(bad)}")

    def factory():
        svc = build_service(args)
        for folder in args.ingest_dir:
            print(f"문서 넣는 중: {folder}", flush=True)
            for r in svc.ingest_dir(folder):
                if r.status != "skipped":
                    print(describe(r), flush=True)
        health = svc.health()
        print(
            f"문서 {health['docs']}개 · 조각 {health['chunks']}개 · 구성 {health['pipeline_ver']}",
            flush=True,
        )
        if not health["docs"]:
            print(
                "색인된 문서가 없습니다: uv run python -m rag.ingest --dir rag/sample_docs",
                flush=True,
            )
        shown = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"화면: http://{shown}:{args.port}  (끄려면 Ctrl+C)", flush=True)
        return svc

    uvicorn.run(create_app(factory), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
