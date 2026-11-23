"""rag 명령줄(rag.ingest, rag.ask)과 화면 서버(python -m web)가 같이 쓰는 인자와 서비스 만들기.

넣기와 질문은 같은 DB 를 써야 한다. 넣을 때 내린 판정을 질문할 때 꺼내 쓰기 때문이다.
"""

import argparse
import sys

from common import config, logdb
from pipeline.components import ComponentUnavailable, load_stage1, load_stage2
from pipeline.orchestrator import Orchestrator
from rag.service import RagService


def add_rag_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("탐지 구성 (넣기와 질문이 같아야 저장된 판정을 그대로 씀)")
    g.add_argument("--stage1", choices=["stub", "real", "off"], default="stub")
    g.add_argument("--stage1-model-dir", default=None)
    g.add_argument(
        "--stage2", choices=["stub", "benign", "broken_json", "real", "off"], default="stub"
    )
    g.add_argument("--model", default="exaone", help="real 로 쓸 2차·답변 LLM (MODELS 키)")
    g.add_argument(
        "--db",
        default=str(config.DB_PATH),
        help=f"판정·문서를 저장할 DB (기본 {config.DB_PATH})",
    )


def add_answer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--answer",
        choices=["stub", "real"],
        default="stub",
        help="답변 LLM. stub=지시에 복종하는 흉내(시험용), real=ollama",
    )


def build_service(args) -> RagService:
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
    orch = Orchestrator(stage1, stage2, logdb.connect(args.db), policy="mask")
    if getattr(args, "answer", "stub") == "real":
        from rag.demo import OllamaAnswerer

        answerer = OllamaAnswerer(args.model)
        name = f"ollama:{answerer.profile['name']}"
    else:
        from rag.demo import stub_answer

        answerer, name = stub_answer, "stub"
    return RagService(orch, answerer=answerer, answer_name=name)
