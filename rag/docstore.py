"""RAG 문서 저장소 (5단계 ①·③).

넣은 문서와 그 청크를 SQLite 에 보관한다. 판정 저장(pipeline/store.py)과 같은
DB 파일을 쓴다. 넣을 때 내린 판정을 질문할 때 꺼내 쓰려면 둘이 같은 곳에 있어야 한다.

- rag_docs: 문서 하나에 한 줄. 원문, 형식, 넣을 때의 구성(pipeline_ver), 문서 단위 결과
- rag_chunks: 색인 단위 = 검사 단위. split_document 가 돌려준 Chunk 를 그대로 JSON 으로 둔다.
  마스킹(offset_map)과 하이라이트(원문 좌표)에 Chunk 전체가 필요하다.
- rag_meta: generation. 문서가 바뀔 때마다 1 올린다. 화면 서버는 이 값이 바뀌면
  색인을 다시 만든다 (다른 창에서 rag.ingest 를 돌려도 반영되게).

복원 청크는 따로 없다. 복원 결과는 부모 청크의 decoded_segments 안에 있고 색인하지 않는다.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from common.schema import Chunk, Span

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_docs (
  doc_id          TEXT PRIMARY KEY,
  source          TEXT NOT NULL,          -- 넣은 파일 경로 (또는 '직접 입력')
  fmt             TEXT,                   -- txt / md / html
  encoding        TEXT,                   -- utf-8, cp949 등
  doc_hash        TEXT NOT NULL,          -- 원문 해시. 바뀌었는지 비교
  raw             TEXT,                   -- 원문. 질문 때 다시 검사하고 하이라이트에 씀
  status          TEXT NOT NULL,          -- indexed | blocked
  blocked_reason  TEXT,                   -- 문서 전체 차단 사유 (읽기·전처리 실패)
  final_label     TEXT,
  action          TEXT,                   -- 넣을 때 기본 정책(마스킹)으로 낸 문서 단위 조치
  attack_spans    TEXT,                   -- JSON [[start, end], ...] 문서 원문 좌표
  n_chunks        INTEGER DEFAULT 0,
  pipeline_ver    TEXT,                   -- 넣을 때 쓴 구성 (⑦)
  fallback        INTEGER DEFAULT 0,      -- 장애 중에 넣었나. 다음 넣기 때 다시 검사
  ingested_at     TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rag_chunks (
  chunk_id    TEXT PRIMARY KEY,
  doc_id      TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  chunk_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_chunks_doc ON rag_chunks (doc_id);
CREATE TABLE IF NOT EXISTS rag_meta (
  key    TEXT PRIMARY KEY,
  value  TEXT NOT NULL
);
"""


def doc_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class DocRecord:
    doc_id: str
    source: str
    fmt: str | None
    encoding: str | None
    doc_hash: str
    raw: str | None
    status: str
    blocked_reason: str | None
    final_label: str | None
    action: str | None
    attack_spans: list[Span]
    n_chunks: int
    pipeline_ver: str | None
    fallback: bool
    ingested_at: str | None


def ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def generation(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM rag_meta WHERE key = 'generation'").fetchone()
    return int(row[0]) if row else 0


def _bump(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO rag_meta (key, value) VALUES ('generation', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
    )


def _record(row) -> DocRecord:
    spans = [Span(start=s, end=e) for s, e in json.loads(row["attack_spans"] or "[]")]
    return DocRecord(
        doc_id=row["doc_id"],
        source=row["source"],
        fmt=row["fmt"],
        encoding=row["encoding"],
        doc_hash=row["doc_hash"],
        raw=row["raw"],
        status=row["status"],
        blocked_reason=row["blocked_reason"],
        final_label=row["final_label"],
        action=row["action"],
        attack_spans=spans,
        n_chunks=row["n_chunks"] or 0,
        pipeline_ver=row["pipeline_ver"],
        fallback=bool(row["fallback"]),
        ingested_at=row["ingested_at"],
    )


def get_doc(conn: sqlite3.Connection, doc_id: str) -> DocRecord | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM rag_docs WHERE doc_id = ?", (doc_id,)).fetchone()
    return _record(row) if row else None


def list_docs(conn: sqlite3.Connection) -> list[DocRecord]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM rag_docs ORDER BY doc_id").fetchall()
    return [_record(r) for r in rows]


def put_doc(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    source: str,
    fmt: str | None,
    encoding: str | None,
    raw: str | None,
    status: str,
    blocked_reason: str | None = None,
    final_label: str | None = None,
    action: str | None = None,
    attack_spans: list[Span] | None = None,
    chunks: list[Chunk] | None = None,
    pipeline_ver: str | None = None,
    fallback: bool = False,
) -> None:
    """문서 한 줄과 그 청크를 한꺼번에 바꾼다 (옛 청크는 지운다)."""
    chunks = chunks or []
    spans = json.dumps([[s.start, s.end] for s in attack_spans or []])
    with conn:
        conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
        conn.execute(
            "INSERT OR REPLACE INTO rag_docs (doc_id, source, fmt, encoding, doc_hash, raw, "
            "status, blocked_reason, final_label, action, attack_spans, n_chunks, pipeline_ver, "
            "fallback, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (
                doc_id,
                source,
                fmt,
                encoding,
                doc_hash(raw) if raw is not None else "",
                raw,
                status,
                blocked_reason,
                final_label,
                action,
                spans,
                len(chunks),
                pipeline_ver,
                int(fallback),
            ),
        )
        conn.executemany(
            "INSERT INTO rag_chunks (chunk_id, doc_id, seq, chunk_json) VALUES (?, ?, ?, ?)",
            [(c.chunk_id, doc_id, k, c.model_dump_json()) for k, c in enumerate(chunks)],
        )
        _bump(conn)


def remove_doc(conn: sqlite3.Connection, doc_id: str) -> bool:
    with conn:
        conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
        cur = conn.execute("DELETE FROM rag_docs WHERE doc_id = ?", (doc_id,))
        if cur.rowcount:
            _bump(conn)
    return bool(cur.rowcount)


def load_chunks(conn: sqlite3.Connection, doc_id: str | None = None) -> list[Chunk]:
    """색인할 청크 전부(또는 문서 하나의 청크). 차단 문서는 청크가 없다"""
    if doc_id is None:
        rows = conn.execute("SELECT chunk_json FROM rag_chunks ORDER BY doc_id, seq").fetchall()
    else:
        rows = conn.execute(
            "SELECT chunk_json FROM rag_chunks WHERE doc_id = ? ORDER BY seq", (doc_id,)
        ).fetchall()
    return [Chunk.model_validate_json(r[0]) for r in rows]
