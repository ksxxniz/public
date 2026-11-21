"""검색 저장소 (5단계 ①).

색인 단위 = 검사 단위: split_document 가 돌려준 청크를 그대로 넣고,
메타데이터에 chunk_id·doc_id·원문 span 을 같이 넣는다. 임베딩은 정리본(text)
기준이다 — 제로폭 문자나 HTML 태그가 섞인 원문은 검색 품질을 떨어뜨린다.
복원 청크는 색인하지 않는다 (검색되면 원문에 없던 텍스트가 답변에 섞인다).

구현 두 가지 (같은 모양):
- SimpleIndex: 글자 3-그램 코사인. 의존성 없음 — CI·통합 테스트용.
- ChromaIndex: Chroma 영속 색인. `pip install chromadb` 한 환경(시연 PC)에서 쓴다.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from common.schema import Chunk

_WS = re.compile(r"\s+")


def _grams(text: str, n: int = 3) -> Counter:
    t = _WS.sub(" ", text.strip().casefold())
    if len(t) < n:
        return Counter([t] if t else [])
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


@dataclass
class Entry:
    chunk_id: str
    doc_id: str
    text: str  # 정리본
    span_start: int
    span_end: int


@dataclass
class SimpleIndex:
    entries: list[Entry] = field(default_factory=list)
    _vectors: list[Counter] = field(default_factory=list)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.entries.append(Entry(c.chunk_id, c.doc_id, c.text, c.span.start, c.span.end))
            self._vectors.append(_grams(c.text))

    def query(self, question: str, k: int = 4) -> list[Entry]:
        return [e for _, e in self.query_scored(question, k)]

    def query_scored(self, question: str, k: int = 4) -> list[tuple[float, Entry]]:
        """(유사도, 조각) 목록. 유사도는 0~1 코사인. 화면에 검색 순위 근거로 보여 준다"""
        q = _grams(question)
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored = []
        for entry, vec in zip(self.entries, self._vectors, strict=True):
            dot = sum(cnt * q[g] for g, cnt in vec.items() if g in q)
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            scored.append((dot / (norm * qn), entry))
        scored.sort(key=lambda t: -t[0])
        return scored[:k]


class ChromaIndex:
    """Chroma 어댑터. 시연 환경(chromadb 설치)에서 SimpleIndex 대신 쓴다"""

    def __init__(self, path: str = "data/chroma", collection: str = "rag_guard"):
        import chromadb  # 선택 의존성

        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(collection)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],  # 임베딩은 정리본 기준
            metadatas=[
                {"doc_id": c.doc_id, "span_start": c.span.start, "span_end": c.span.end}
                for c in chunks
            ],
        )

    def query(self, question: str, k: int = 4) -> list[Entry]:
        res = self._col.query(query_texts=[question], n_results=k)
        out = []
        for cid, doc, meta in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], strict=True
        ):
            out.append(Entry(cid, meta["doc_id"], doc, meta["span_start"], meta["span_end"]))
        return out
