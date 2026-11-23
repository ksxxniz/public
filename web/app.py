"""백엔드 API (5단계 ④ 'API 모양으로 만들기', 8주차 FastAPI 연결).

입력(문서 또는 질문) + 정책 → 판정, 하이라이트, 단계별 시간. 화면(web/static/index.html)도
같은 서버가 보낸다.

  POST /api/ask       {question, policy, k, defense} → 방어 OFF/ON 답, 검색된 조각별 판정·하이라이트
  POST /api/inspect   {text, fmt, policy}            → 문서 하나 검사 (색인하지 않음)
  GET  /api/docs                                     → 넣은 문서 목록
  GET  /api/docs/{doc_id}                            → 문서 원문과 공격·숨긴 글·인코딩 구간
  GET  /api/health                                   → 구성, 2차 회로 상태, 문서 수

하이라이트는 글자 위치 대신 잘라 놓은 조각(segments)으로 보낸다. 파이썬 위치는 코드포인트,
브라우저는 UTF-16 이라 태그 문자·이모지에서 어긋나기 때문이다 (decisions.md ⑥).

SQLite 연결은 만든 스레드에서만 쓸 수 있어서, 검사와 DB 작업은 전용 스레드 하나에서 돈다.
요청을 한 번에 하나씩 처리하는 셈이라 시연용으로 충분하다. 2차 호출은 오케스트레이터가
따로 병렬로 보낸다.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag.service import RagService

STATIC = Path(__file__).parent / "static"
Policy = Literal["mask", "block", "warn"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    policy: Policy = "mask"
    k: int = Field(default=4, ge=1, le=20)
    defense: Literal["both", "on", "off"] = "both"


class InspectRequest(BaseModel):
    text: str = Field(min_length=1)
    fmt: Literal["txt", "md", "html"] | None = None  # None 이면 내용으로 판별
    policy: Policy = "mask"


def create_app(service_factory: Callable[[], RagService]) -> FastAPI:
    """service_factory 는 전용 스레드에서 한 번 불린다 (DB 연결을 그 스레드에서 만들려고)"""
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-guard")
    holder: dict[str, RagService] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        holder["service"] = await asyncio.get_running_loop().run_in_executor(
            worker, service_factory
        )
        yield
        worker.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="RAG 프롬프트 인젝션 탐지 시스템", lifespan=lifespan)

    async def run(fn: Callable, *args, **kwargs):
        def call():
            return fn(holder["service"], *args, **kwargs)

        try:
            return await asyncio.get_running_loop().run_in_executor(worker, call)
        except ValueError as e:  # 정책·형식·빈 질문 같은 입력 오류
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    async def health() -> dict:
        return await run(RagService.health)

    @app.post("/api/ask")
    async def ask(req: AskRequest) -> dict:
        return await run(
            RagService.ask, req.question, k=req.k, policy=req.policy, defense=req.defense
        )

    @app.post("/api/inspect")
    async def inspect(req: InspectRequest) -> dict:
        return await run(RagService.inspect, req.text, fmt=req.fmt, policy=req.policy)

    @app.get("/api/docs")
    async def docs() -> list[dict]:
        return await run(RagService.docs)

    @app.get("/api/docs/{doc_id:path}")
    async def doc(doc_id: str) -> dict:
        found = await run(RagService.doc_view, doc_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"문서가 없습니다: {doc_id}")
        return found

    return app
