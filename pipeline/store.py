"""판정 저장·재사용 (5단계 ③·⑦).

문서를 넣을 때 내린 판정을 chunk_id 로 저장하고, 질문할 때 꺼내 쓴다.
재사용 조건: 원문 조각 해시와 pipeline_ver 가 '둘 다' 같을 때만.
해시를 같이 보는 이유는 같은 문서ID 로 내용이 바뀐 문서가 다시 들어와도
옛 판정이 쓰이지 않게 하려는 것이다.

장애 처리(fallback)로 내린 판정은 저장하지 않는다 (⑥). 2차가 살아난 뒤에도
정상 문서가 계속 막히는 것을 막기 위해서다.
"""

import sqlite3

from common.schema import Verdict

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS verdict_cache (
  chunk_id     TEXT NOT NULL,
  doc_id       TEXT NOT NULL,
  raw_hash     TEXT NOT NULL,     -- 조각 원문 해시 (pipeline.version.raw_hash)
  pipeline_ver TEXT NOT NULL,
  verdict_json TEXT NOT NULL,
  created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id, raw_hash, pipeline_ver)
);
"""


def ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def get(
    conn: sqlite3.Connection, chunk_id: str, raw_hash: str, pipeline_ver: str
) -> Verdict | None:
    row = conn.execute(
        "SELECT verdict_json FROM verdict_cache "
        "WHERE chunk_id = ? AND raw_hash = ? AND pipeline_ver = ?",
        (chunk_id, raw_hash, pipeline_ver),
    ).fetchone()
    if row is None:
        return None
    return Verdict.model_validate_json(row[0])


def put(conn: sqlite3.Connection, doc_id: str, raw_hash: str, verdict: Verdict) -> bool:
    """저장했으면 True. 장애 중 판정(fallback)은 저장하지 않는다 (⑥)"""
    if verdict.fallback:
        return False
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO verdict_cache "
            "(chunk_id, doc_id, raw_hash, pipeline_ver, verdict_json) VALUES (?, ?, ?, ?, ?)",
            (verdict.chunk_id, doc_id, raw_hash, verdict.pipeline_ver, verdict.model_dump_json()),
        )
    return True


def drop_doc(conn: sqlite3.Connection, doc_id: str) -> int:
    """문서의 저장된 판정을 지운다 (recheck 때 새 버전으로 다시 쓰기 전에)"""
    with conn:
        cur = conn.execute("DELETE FROM verdict_cache WHERE doc_id = ?", (doc_id,))
    return cur.rowcount
