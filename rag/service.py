"""RAG 서비스: 문서 넣기와 질문 (5단계 ①·③).

검문소 위치 (③)
- 저장할 때: 문서를 넣으면 바로 검사하고 판정을 chunk_id 로 저장한다 (오케스트레이터가
  원문 조각 해시와 pipeline_ver 를 같이 적는다). 공격 판정 문서도 색인에 남긴다.
- 검색 직후: 검색된 조각의 판정을 꺼낸다. 해시와 pipeline_ver 가 맞으면 재사용하고,
  없거나 다르면 그 자리에서 문서를 다시 검사한다.
- 정책은 질문할 때 적용한다 (rag/explain.py). 넣을 때는 항상 기본 정책(mask)으로 검사한다.

명령줄은 rag/ingest.py(넣기)와 rag/ask.py(질문), 화면은 web/ 이 이 클래스를 감싼다.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from common import config
from common.schema import Chunk, Span, Verdict
from pipeline import store
from pipeline.orchestrator import Orchestrator
from pipeline.version import raw_hash
from rag import docstore
from rag import explain as X
from rag.index import SimpleIndex

SOURCE_TEXT = "(직접 입력)"
FORMATS = ("txt", "md", "html")


@dataclass
class IngestResult:
    doc_id: str
    source: str
    status: (
        str  # indexed(넣음) | unchanged(그대로) | blocked(차단) | skipped(건너뜀) | removed(지움)
    )
    reason: str = ""
    final_label: str | None = None
    action: str | None = None
    n_chunks: int = 0
    n_attack_chunks: int = 0  # 공격 판정 조각 수
    n_fallback_chunks: int = 0  # 장애로 판정 못 한 조각 수. 다음 넣기 때 다시 검사
    encoding: str | None = None
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class RagService:
    def __init__(self, orch: Orchestrator, *, answerer=None, answer_name: str = "stub"):
        if orch.conn is None:
            raise ValueError("RAG 는 판정과 문서를 저장할 DB 가 필요합니다")
        if orch.policy != "mask":
            raise ValueError("넣을 때는 기본 정책(mask)으로 검사합니다. 정책은 질문할 때 고릅니다")
        from rag.demo import stub_answer

        self.orch = orch
        self.conn = orch.conn
        self.answerer = answerer or stub_answer
        self.answer_name = answer_name
        docstore.ensure(self.conn)
        self._gen = -1
        self._index = SimpleIndex()
        self._chunks: dict[str, Chunk] = {}
        self._doc_chunks: dict[str, list[Chunk]] = {}
        self._docs: dict[str, docstore.DocRecord] = {}

    # ── 넣기 ──

    def ingest_dir(
        self, root: str | Path, *, prune: bool = False, force: bool = False, progress=None
    ) -> list[IngestResult]:
        """폴더를 통째로 넣는다. 문서ID 는 '폴더 이름/폴더 안 경로' (예: sample_docs/hr/휴가.md).

        prune: 이 폴더에서 넣었던 문서 중 지금은 없는 것을 색인에서 뺀다.
        force: 바뀌지 않은 문서도 다시 검사한다."""
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"폴더가 아닙니다: {root}")
        prefix = root.resolve().name
        results: list[IngestResult] = []
        seen: set[str] = set()
        for path in _walk(root):
            doc_id = f"{prefix}/{path.relative_to(root).as_posix()}"
            if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                kind = path.suffix or "확장자 없음"
                r = IngestResult(
                    doc_id, str(path), "skipped", reason=f"지원하지 않는 형식 ({kind})"
                )
            else:
                seen.add(doc_id)
                r = self.ingest_file(path, doc_id=doc_id, force=force)
            results.append(r)
            if progress:
                progress(r)
        if prune:
            for rec in docstore.list_docs(self.conn):
                if rec.doc_id.startswith(prefix + "/") and rec.doc_id not in seen:
                    self.remove(rec.doc_id)
                    r = IngestResult(rec.doc_id, rec.source, "removed", reason="폴더에 없는 문서")
                    results.append(r)
                    if progress:
                        progress(r)
        return results

    def ingest_file(
        self, path: str | Path, *, doc_id: str | None = None, force: bool = False
    ) -> IngestResult:
        from preprocess.load import read_document

        path = Path(path)
        doc_id = doc_id or path.name
        t0 = time.perf_counter()
        try:
            loaded = read_document(path)
        except Exception as e:  # 읽을 수 없음·너무 큼 → 문서 전체 차단 (⑥)
            reason = f"load_error: {e}"
            docstore.put_doc(
                self.conn,
                doc_id=doc_id,
                source=str(path),
                fmt=None,
                encoding=None,
                raw=None,
                status="blocked",
                blocked_reason=reason,
                final_label="injection",
                action="block",
                pipeline_ver=self.orch.pipeline_ver,
            )
            store.drop_doc(self.conn, doc_id)
            return IngestResult(doc_id, str(path), "blocked", reason=reason, elapsed_ms=_ms(t0))
        return self.ingest_text(
            loaded.raw,
            doc_id=doc_id,
            fmt=loaded.fmt,
            source=str(path),
            encoding=loaded.encoding,
            force=force,
            started=t0,
        )

    def ingest_text(
        self,
        raw: str,
        *,
        doc_id: str,
        fmt: str | None,
        source: str = SOURCE_TEXT,
        encoding: str | None = None,
        force: bool = False,
        started: float | None = None,
    ) -> IngestResult:
        t0 = started or time.perf_counter()
        old = docstore.get_doc(self.conn, doc_id)
        warnings = []
        if old and old.source != source:
            warnings.append(f"같은 문서ID 를 다른 곳에서 넣었습니다: {old.source} → {source}")
        # 차단된 문서는 바뀌지 않았어도 다시 검사한다. 토크나이저를 못 불러오는 것처럼
        # 환경 탓에 막혔으면, 고친 뒤 다시 넣을 때 --force 없이도 들어가야 한다
        unchanged = (
            old is not None
            and not force
            and old.status == "indexed"
            and old.doc_hash == docstore.doc_hash(raw)
            and old.pipeline_ver == self.orch.pipeline_ver
            and not old.fallback  # 장애 중에 넣은 문서는 다시 검사 (⑥)
        )
        if unchanged:
            return IngestResult(
                doc_id,
                source,
                "unchanged",
                final_label=old.final_label,
                action=old.action,
                n_chunks=old.n_chunks,
                encoding=old.encoding,
                elapsed_ms=_ms(t0),
                warnings=warnings,
            )
        report = self.orch.inspect_text(raw, doc_id=doc_id, fmt=fmt)
        if report.blocked_reason:  # 전처리 실패·너무 긴 문서 → 문서 전체 차단 (⑥)
            docstore.put_doc(
                self.conn,
                doc_id=doc_id,
                source=source,
                fmt=fmt,
                encoding=encoding,
                raw=raw,
                status="blocked",
                blocked_reason=report.blocked_reason,
                final_label="injection",
                action="block",
                pipeline_ver=self.orch.pipeline_ver,
            )
            store.drop_doc(self.conn, doc_id)
            return IngestResult(
                doc_id,
                source,
                "blocked",
                reason=report.blocked_reason,
                encoding=encoding,
                elapsed_ms=_ms(t0),
                warnings=warnings + report.warnings,
            )
        verdicts = report.doc.verdicts
        docstore.put_doc(
            self.conn,
            doc_id=doc_id,
            source=source,
            fmt=fmt,
            encoding=encoding,
            raw=raw,
            status="indexed",
            final_label=report.doc.final_label,
            action=report.doc.action,
            attack_spans=report.doc.attack_spans_raw,
            chunks=report.chunks,
            pipeline_ver=self.orch.pipeline_ver,
            fallback=report.doc.fallback,
        )
        return IngestResult(
            doc_id,
            source,
            "indexed",
            final_label=report.doc.final_label,
            action=report.doc.action,
            n_chunks=len(report.chunks),
            n_attack_chunks=sum(v.final_label == "injection" for v in verdicts),
            n_fallback_chunks=sum(v.fallback for v in verdicts),
            encoding=encoding,
            elapsed_ms=_ms(t0),
            warnings=warnings + report.warnings,
        )

    def remove(self, doc_id: str) -> bool:
        removed = docstore.remove_doc(self.conn, doc_id)
        store.drop_doc(self.conn, doc_id)
        return removed

    # ── 질문 ──

    def ask(self, question: str, *, k: int = 4, policy: str = "mask", defense: str = "both"):
        """검색 → 저장된 판정 조회(없으면 그 자리에서 검사) → 정책 적용 → 답변.

        defense: "on" | "off" | "both". both 면 같은 조각·같은 프롬프트로 두 번 답한다.
        답변 프롬프트가 ON·OFF 에서 같아야 차이를 검문소 효과로만 설명할 수 있다 (①)."""
        if policy not in X.POLICIES:
            raise ValueError(f"정책을 모름: {policy}")
        if defense not in ("on", "off", "both"):
            raise ValueError(f"defense 는 on/off/both 중 하나: {defense}")
        question = question.strip()
        if not question:
            raise ValueError("질문이 비어 있습니다")
        if k < 1:
            raise ValueError(f"검색할 조각 수(k)는 1 이상이어야 합니다: {k}")
        t0 = time.perf_counter()
        self.refresh()
        timings = {"retrieve_ms": 0.0, "guard_ms": 0.0, "answer_on_ms": 0.0, "answer_off_ms": 0.0}
        result = {
            "question": question,
            "policy": policy,
            "k": k,
            "pipeline_ver": self.orch.pipeline_ver,
            "answer_model": self.answer_name,
            "reinspected": [],
            "chunks": [],
            "notice": "",
            "timings": timings,
        }
        if not self._chunks:
            result["notice"] = (
                "색인된 문서가 없습니다. 먼저 python -m rag.ingest --dir 폴더 로 문서를 넣으세요"
            )
            hits: list = []
            verdicts: dict[str, Verdict] = {}
        else:
            hits, verdicts, reinspected, retrieve_ms, guard_ms = self._retrieve(question, k)
            result["reinspected"] = reinspected
            timings["retrieve_ms"], timings["guard_ms"] = retrieve_ms, guard_ms

        rows = []
        for rank, (score, entry) in enumerate(hits, 1):
            chunk = self._chunks[entry.chunk_id]
            v = verdicts.get(entry.chunk_id)
            decision = X.decide(v, policy) if v else X.Decision("unscannable_block")
            rows.append(self._chunk_row(chunk, v, decision, rank=rank, score=score))
        result["chunks"] = rows

        if defense in ("on", "both"):
            contexts = [r["llm_input_on"] for r in rows if r["llm_input_on"] is not None]
            result["on"], timings["answer_on_ms"] = self._answer(question, contexts)
            result["on"]["dropped"] = [r["chunk_id"] for r in rows if r["dropped"]]
        if defense in ("off", "both"):
            contexts = [r["llm_input_off"] for r in rows]
            result["off"], timings["answer_off_ms"] = self._answer(question, contexts)
        timings["total_ms"] = _ms(t0)
        for key in timings:
            timings[key] = round(timings[key], 1)
        return result

    def inspect(self, raw: str, *, fmt: str | None = None, policy: str = "mask") -> dict:
        """문서 하나를 검사만 한다 (색인하지 않음). 화면의 '문서 검사' 탭과 /api/inspect"""
        if policy not in X.POLICIES:
            raise ValueError(f"정책을 모름: {policy}")
        if fmt not in (None, *FORMATS):
            raise ValueError(f"형식을 모름: {fmt} (가능: {', '.join(FORMATS)})")
        doc_id = f"inspect/{docstore.doc_hash(raw)}"
        report = self.orch.inspect_text(raw, doc_id=doc_id, fmt=fmt)
        out = {
            "doc_id": doc_id,
            "policy": policy,
            "pipeline_ver": self.orch.pipeline_ver,
            "blocked_reason": report.blocked_reason,
            "final_label": report.doc.final_label,
            "routes": report.routes,
            "warnings": report.warnings,
            "timings": {k: round(v, 1) for k, v in vars(report.timings).items()},
            "chunks": [],
        }
        if report.blocked_reason:
            out["status"], out["status_label"] = "unscannable_block", "검사 불가로 차단"
            return out
        for chunk, v in zip(report.chunks, report.doc.verdicts, strict=True):
            out["chunks"].append(self._chunk_row(chunk, v, X.decide(v, policy), raw=raw))
        out["status"], out["status_label"] = _worst(out["chunks"])
        # 같은 글을 전에 검사했으면 저장된 판정을 다시 쓴다 (⑦). 그때는 단계 시간이 0 이다
        out["cache_hits"] = sum(v.cache_hit for v in report.doc.verdicts)
        return out

    # ── 조회 ──

    def docs(self) -> list[dict]:
        self.refresh()
        return [self._doc_row(rec) for rec in self._docs.values()]

    def doc_view(self, doc_id: str) -> dict | None:
        """문서 전체 원문에 공격 구간·숨긴 글·인코딩 구간을 표시한다"""
        self.refresh()
        rec = self._docs.get(doc_id) or docstore.get_doc(self.conn, doc_id)
        if rec is None:
            return None
        row = self._doc_row(rec)
        if rec.raw is None:
            row["segments"] = []
            return row
        hidden, encoded = [], []
        hidden_notes, encoded_notes = {}, {}
        for c in self._doc_chunks.get(doc_id, []):
            base = c.span.start
            for ts in c.transform_spans:
                if ts.kind == "hidden_text_found":
                    s = Span(start=base + ts.span.start, end=base + ts.span.end)
                    hidden.append(s)
                    hidden_notes[(s.start, s.end)] = X.hidden_note_text(ts.note)
            for seg in c.decoded_segments:
                s = Span(start=base + seg.span.start, end=base + seg.span.end)
                encoded.append(s)
                encoded_notes[(s.start, s.end)] = X.encoded_note_text(seg)
        layers = {"flagged": rec.attack_spans, "hidden": hidden, "encoded": encoded}
        notes = {
            "hidden": [(Span(start=a, end=b), t) for (a, b), t in hidden_notes.items()],
            "encoded": [(Span(start=a, end=b), t) for (a, b), t in encoded_notes.items()],
        }
        row["segments"] = X.segments(rec.raw, layers, notes)
        return row

    def health(self) -> dict:
        self.refresh()
        docs = list(self._docs.values())
        return {
            "pipeline_ver": self.orch.pipeline_ver,
            "stage1": self.orch.stage1.model_ver if self.orch.stage1 else "off",
            "stage2": self.orch.stage2.model_ver if self.orch.stage2 else "off",
            "answer_model": self.answer_name,
            "circuit": self.orch.circuit.state,
            "docs": sum(d.status == "indexed" for d in docs),
            "blocked_docs": sum(d.status == "blocked" for d in docs),
            "chunks": len(self._chunks),
            "stale_docs": sum(
                d.status == "indexed" and d.pipeline_ver != self.orch.pipeline_ver for d in docs
            ),
            "policies": list(X.POLICIES),
            "mask_marker": config.MASK_MARKER,  # 화면이 LLM 입력에서 삭제 표시를 강조할 때 쓴다
        }

    # ── 내부 ──

    def refresh(self) -> None:
        """문서가 바뀌었으면(generation) 색인을 다시 만든다. 다른 창의 rag.ingest 도 반영된다"""
        gen = docstore.generation(self.conn)
        if gen == self._gen:
            return
        chunks = docstore.load_chunks(self.conn)
        self._chunks = {c.chunk_id: c for c in chunks}
        self._doc_chunks = {}
        for c in chunks:
            self._doc_chunks.setdefault(c.doc_id, []).append(c)
        self._docs = {d.doc_id: d for d in docstore.list_docs(self.conn)}
        index = SimpleIndex()
        index.add_chunks(chunks)  # 색인 단위 = 검사 단위. 임베딩은 정리본(text) 기준 (①)
        self._index = index
        self._gen = gen

    def _search(self, question: str, k: int):
        # 질문과 겹치는 글자가 하나도 없는 조각(유사도 0)은 뺀다. 문서가 적으면
        # 관련 없는 조각이 상위 k 개에 끼어 다른 질문의 답을 흐린다
        return [(score, e) for score, e in self._index.query_scored(question, k) if score > 0]

    def _retrieve(self, question: str, k: int):
        t = time.perf_counter()
        hits = self._search(question, k)
        retrieve_ms = _ms(t)
        t = time.perf_counter()
        verdicts: dict[str, Verdict] = {}
        reinspected: list[str] = []
        looked_up: set[str] = set()
        # 전처리가 바뀌어 청크 경계가 달라진 문서가 있으면 색인을 다시 만들고 다시 검색한다.
        # 새로 검색된 문서가 없을 때까지 되풀이한다. 문서마다 한 번만 보므로 반드시 끝난다.
        # (횟수를 정해 두면 마지막 검색에서 처음 잡힌 조각이 판정 없이 '검사 불가'로 보인다)
        while need := sorted({e.doc_id for _, e in hits} - looked_up):
            changed = False
            for doc_id in need:
                looked_up.add(doc_id)
                got, did_inspect, boundaries_changed = self._doc_verdicts(doc_id)
                verdicts.update(got)
                if did_inspect:
                    reinspected.append(doc_id)
                changed = changed or boundaries_changed
            if not changed:
                break
            self.refresh()
            hits = self._search(question, k)
        verdicts = {cid: v for cid, v in verdicts.items() if cid in self._chunks}
        return hits, verdicts, reinspected, retrieve_ms, _ms(t)

    def _doc_verdicts(self, doc_id: str) -> tuple[dict[str, Verdict], bool, bool]:
        """(조각별 판정, 다시 검사했나, 청크 경계가 바뀌었나).
        저장된 판정은 해시와 pipeline_ver 가 둘 다 맞을 때만 쓴다 (⑦)"""
        chunks = self._doc_chunks.get(doc_id, [])
        got: dict[str, Verdict] = {}
        for c in chunks:
            v = store.get(self.conn, c.chunk_id, raw_hash(c.raw_text), self.orch.pipeline_ver)
            if v is None:
                break
            got[c.chunk_id] = v.model_copy(update={"cache_hit": True})
        if chunks and len(got) == len(chunks):
            return got, False, False
        rec = self._docs.get(doc_id) or docstore.get_doc(self.conn, doc_id)
        if rec is None or rec.raw is None:
            return {}, False, False
        # 판정이 없거나(장애 중 판정은 저장하지 않음) 구성이 바뀜 → 그 자리에서 검사 (③)
        report = self.orch.inspect_text(rec.raw, doc_id=doc_id, fmt=rec.fmt)
        if report.blocked_reason:
            docstore.put_doc(
                self.conn,
                doc_id=doc_id,
                source=rec.source,
                fmt=rec.fmt,
                encoding=rec.encoding,
                raw=rec.raw,
                status="blocked",
                blocked_reason=report.blocked_reason,
                final_label="injection",
                action="block",
                pipeline_ver=self.orch.pipeline_ver,
            )
            return {}, True, True
        # 전처리가 바뀌면 경계가 같아도 정리본·대응표가 달라질 수 있어 청크 전체를 비교한다
        changed = [c.model_dump_json() for c in chunks] != [
            c.model_dump_json() for c in report.chunks
        ]
        summary_changed = (
            rec.pipeline_ver != self.orch.pipeline_ver
            or rec.fallback != report.doc.fallback
            or rec.final_label != report.doc.final_label
            or rec.action != report.doc.action
        )
        if changed or summary_changed:
            before = docstore.generation(self.conn)
            docstore.put_doc(
                self.conn,
                doc_id=doc_id,
                source=rec.source,
                fmt=rec.fmt,
                encoding=rec.encoding,
                raw=rec.raw,
                status="indexed",
                final_label=report.doc.final_label,
                action=report.doc.action,
                attack_spans=report.doc.attack_spans_raw,
                chunks=report.chunks,
                pipeline_ver=self.orch.pipeline_ver,
                fallback=report.doc.fallback,
            )
            if not changed:  # 청크는 그대로라 색인은 다시 만들 필요 없다. 문서 정보만 맞춘다
                if self._gen == before:  # 그 사이 다른 창의 넣기가 없었을 때만 건너뛴다
                    self._gen = docstore.generation(self.conn)
                self._docs[doc_id] = docstore.get_doc(self.conn, doc_id)
        return {v.chunk_id: v for v in report.doc.verdicts}, True, changed

    def _chunk_row(
        self,
        chunk: Chunk,
        v: Verdict | None,
        decision: X.Decision,
        *,
        rank: int | None = None,
        score: float | None = None,
        raw: str | None = None,
    ) -> dict:
        rec = self._docs.get(chunk.doc_id)
        merge = v.meta.get("merge") if v else None
        raw = raw if raw is not None else (rec.raw if rec else None)
        return {
            "rank": rank,
            "score": round(score, 3) if score is not None else None,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source": rec.source if rec else SOURCE_TEXT,
            "span": [chunk.span.start, chunk.span.end],
            # 화면 원문 보기의 첫 줄 번호 (문서 전체 기준). 청크가 줄 중간에서 시작할 수 있다
            "line_start": raw.count("\n", 0, chunk.span.start) + 1 if raw else 1,
            "status": decision.status,
            "status_label": decision.label,
            "dropped": decision.dropped,
            "final_label": v.final_label if v else "injection",
            "stored_action": v.action if v else "block",
            "merge": merge,
            "merge_text": X.why_text(v, decision),
            "fallback": v.fallback if v else True,
            "cache_hit": v.cache_hit if v else False,
            "score_max": v.final_score if v else None,
            "stages": X.stage_rows(v) if v else [],
            "segments": X.chunk_segments(chunk, decision),
            "llm_input_on": X.llm_input(chunk, decision),
            "llm_input_off": chunk.text,  # 방어 OFF: 정리본 그대로 (숨긴 글 포함)
        }

    def _doc_row(self, rec: docstore.DocRecord) -> dict:
        if rec.status == "blocked":
            status, label = "unscannable_block", "검사 불가로 차단"
        elif rec.final_label == "injection":
            status, label = "attack", "공격 탐지"
        elif rec.action == "warn":
            status, label = "warn", "경고"
        else:
            status, label = "pass", "정상"
        return {
            "doc_id": rec.doc_id,
            "source": rec.source,
            "fmt": rec.fmt,
            "encoding": rec.encoding,
            "status": status,
            "status_label": label,
            "blocked_reason": rec.blocked_reason,
            "n_chunks": rec.n_chunks,
            "n_attack_spans": len(rec.attack_spans),
            "fallback": rec.fallback,
            "stale": rec.status == "indexed" and rec.pipeline_ver != self.orch.pipeline_ver,
            "ingested_at": rec.ingested_at,
        }

    def _answer(self, question: str, contexts: list[str]) -> tuple[dict, float]:
        t = time.perf_counter()
        try:
            answer, error = self.answerer(question, contexts), None
        except Exception as e:  # ollama 가 꺼져 있어도 화면이 멈추지 않게
            answer, error = f"(답변 LLM 오류: {e})", repr(e)
        return {"answer": answer, "error": error, "n_contexts": len(contexts)}, _ms(t)


# ── 도우미 ──

_SEVERITY = (
    "unscannable_block",
    "attack_block",
    "mask",
    "unscannable_warn",
    "attack_warn",
    "warn",
    "pass",
)


def _worst(rows: list[dict]) -> tuple[str, str]:
    for status in _SEVERITY:
        if any(r["status"] == status for r in rows):
            return status, X.STATUS_LABELS[status]
    return "pass", X.STATUS_LABELS["pass"]


def _walk(root: Path):
    """숨은 폴더·파일(.git 등)은 건너뛰고 이름 순으로"""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
