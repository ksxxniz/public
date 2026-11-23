"""질문 명령 (5단계 ③ '검색 직후' 검문소). 화면 없이 확인할 때 쓴다.

  uv run python -m rag.ask "연차는 며칠이야?"
  uv run python -m rag.ask "연차는 며칠이야?" --policy block
  uv run python -m rag.ask "연차는 며칠이야?" --answer real --stage2 real --model exaone

같은 조각·같은 프롬프트로 방어 OFF 와 ON 을 둘 다 답해서 비교한다.
먼저 python -m rag.ingest 로 같은 DB 에 문서를 넣어야 한다.
"""

import argparse
import json

from common.console import safe_stdio
from rag.cli import add_answer_args, add_rag_args, build_service

POLICY_NAMES = {"mask": "마스킹", "block": "차단", "warn": "경고만"}


def main(argv: list[str] | None = None) -> None:
    safe_stdio()
    parser = argparse.ArgumentParser(description="RAG 에 질문하고 방어 OFF/ON 답을 비교")
    parser.add_argument("question", help="질문")
    parser.add_argument("--policy", choices=list(POLICY_NAMES), default="mask")
    parser.add_argument("--k", type=int, default=4, help="검색할 조각 수")
    parser.add_argument("--defense", choices=["both", "on", "off"], default="both")
    parser.add_argument("--json", action="store_true", help="API 와 같은 JSON 으로 출력")
    add_answer_args(parser)
    add_rag_args(parser)
    args = parser.parse_args(argv)
    svc = build_service(args)
    try:
        res = svc.ask(args.question, k=args.k, policy=args.policy, defense=args.defense)
    except ValueError as e:  # 빈 질문, k 가 0 이하
        parser.error(str(e))

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    policy, model = POLICY_NAMES[res["policy"]], res["answer_model"]
    print(f"질문: {res['question']}   (정책: {policy}, 답변: {model})")
    if res["notice"]:
        print(res["notice"])
    for mode, name in (("off", "방어 OFF"), ("on", "방어 ON ")):
        if mode in res:
            print(f"{name} 답: {res[mode]['answer']}")
    print("검색된 조각:")
    for c in res["chunks"]:
        why = f" — {c['merge_text']}" if c["status"] != "pass" else ""
        out = " (답변 LLM 에 안 들어감)" if c["dropped"] else ""
        print(f"  {c['rank']}. [{c['status_label']}] {c['doc_id']}  유사도 {c['score']}{why}{out}")
    if res["reinspected"]:
        print(f"질문할 때 다시 검사한 문서: {', '.join(res['reinspected'])}")
    t = res["timings"]
    print(
        f"시간(ms): 검색 {t['retrieve_ms']} · 탐지 {t['guard_ms']} · "
        f"답변 ON {t['answer_on_ms']} · OFF {t['answer_off_ms']} · 전체 {t['total_ms']}"
    )


if __name__ == "__main__":
    main()
