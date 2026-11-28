"""오케스트레이터 (5단계 ④). 문서 하나를 전처리 → 룰 → 1차 → 2차 → 판정
합치기·정책 엔진(⑤) → 로그로 통과시킨다.

검사 순서
1. 전처리: split_document 로 청크 목록(복원 결과 포함)을 만든다.
2. 룰: 원래 청크마다 check_chunk. 복원 청크에는 부르지 않는다 (복원 글은
   룰이 decoded_segments 로 이미 검사한다 — 두 번 세지 않기).
3. 1차: 모든 청크를 한 번에 배치로 (본문+복원 조각은 1차가 내부에서 처리).
4. 2차: 경로 표에 맞는 조각만, 동시 상한·타임아웃·회로 차단기를 걸고.
5. 판정 합치기(⑤) → 정책(마스킹·재검사) → 문서 단위 합집합 → 로그.

조기 종료 [바꿈]: 1차 점수가 θ_low 미만이고 '룰 점수가 2차 경계 이하'일 때만
2차를 건너뛴다. 룰이 의심하면 1차가 낮아도 2차로 간다.

장애 정책 (⑥)
- 공격자가 일으킬 수 있는 장애(전처리 예외·크기 초과, 룰 예외, 2차 형식 오류)는
  항상 차단. "고장 내면 통과"를 막는다.
- 인프라 장애(1차·2차 장애, 회로 열림)는 합의한 fail-safe (config.FAIL_SAFE_ACTION).
  fail-open 이어도 "정상"이 아니라 "경고"로 넘긴다. 룰 0.9 이상은 그대로 공격.
- 장애로 내린 판정은 저장하지 않는다 (verdict.fallback=True).
- 로그 DB 가 죽어도 판정은 계속하고, 예비 파일(JSONL)에 쓰고 경고한다.
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path

from common import config, logdb
from common.schema import Chunk, DocumentVerdict, Span, StageResult, Verdict
from pipeline import policy as P
from pipeline import store
from pipeline.circuit import CircuitBreaker
from pipeline.components import Stage1Handle, Stage2Handle
from pipeline.merge import Merged, merge
from pipeline.version import pipeline_ver, raw_hash

PREPROCESS_VER_PREFIX = "normalize-"


@dataclass
class Timings:
    preprocess_ms: float = 0.0
    rule_ms: float = 0.0
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class DocumentReport:
    doc: DocumentVerdict
    chunks: list[Chunk] = field(default_factory=list)
    masked_texts: dict[str, str] = field(default_factory=dict)  # chunk_id → 답변 LLM 입력
    routes: dict[str, int] = field(default_factory=dict)  # 2차 경로별 호출 수 (④)
    warnings: list[str] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    request_id: str = ""
    blocked_reason: str | None = None  # 문서 전체 차단 사유 (⑥)


class Orchestrator:
    def __init__(
        self,
        stage1: Stage1Handle | None,
        stage2: Stage2Handle | None,
        conn=None,
        *,
        policy: str = config.DEFAULT_POLICY,
        fail_safe: str = config.FAIL_SAFE_ACTION,
        use_cache: bool = True,
        stage2_timeout_s: float = config.STAGE2_TIMEOUT_S,
        stage2_max_concurrent: int = config.STAGE2_MAX_CONCURRENT,
        circuit: CircuitBreaker | None = None,
        log_fallback_path: str | Path | None = None,
    ):
        assert policy in ("mask", "block", "warn")
        assert fail_safe in ("warn", "block")
        self.stage1 = stage1
        self.stage2 = stage2
        self.conn = conn
        self.policy = policy
        self.fail_safe = fail_safe
        self.use_cache = use_cache
        self.stage2_timeout_s = stage2_timeout_s
        self.circuit = circuit or CircuitBreaker()
        self._pool = ThreadPoolExecutor(max_workers=stage2_max_concurrent)
        self.log_fallback_path = Path(
            log_fallback_path or config.DATA_DIR / "inspection_log_fallback.jsonl"
        )
        self.pipeline_ver = pipeline_ver(
            stage1.model_ver if stage1 else "off",
            stage2.model_ver if stage2 else "off",
        )
        if conn is not None:
            store.ensure(conn)

    # ── 진입점 ──

    def inspect_file(self, path: str | Path, **kw) -> DocumentReport:
        from preprocess.load import read_document

        doc_id = kw.pop("doc_id", Path(path).stem)
        try:
            doc = read_document(path)
        except Exception as e:  # 깨진 입력·크기 초과·읽기 실패 → 항상 문서 전체 차단 (⑥)
            return self._blocked_report(doc_id, f"load_error: {e}")
        return self.inspect_text(doc.raw, doc_id=doc_id, fmt=doc.fmt, **kw)

    def inspect_text(
        self,
        raw: str,
        *,
        doc_id: str,
        fmt: str | None = "txt",
        defense_enabled: bool = True,
        request_id: str | None = None,
    ) -> DocumentReport:
        """오케스트레이터 입구 (④ API 모양): 입력(문서) + 정책 → 판정·하이라이트·
        단계별 시간. 8주차 FastAPI 는 이 함수를 감싸기만 하면 된다"""
        from preprocess.chunk import split_document

        t0 = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex[:12]
        report = DocumentReport(
            doc=DocumentVerdict(
                doc_id=doc_id,
                final_label="benign",
                action="allow",
                pipeline_ver=self.pipeline_ver,
                total_latency_ms=0.0,
            ),
            request_id=request_id,
        )
        routes = {"s1_ambiguous": 0, "rule_suspect": 0, "locate": 0, "recheck": 0}

        try:
            chunks = split_document(doc_id, raw, fmt=fmt)
        except Exception as e:  # 너무 긴 문서·전처리 예외 → 항상 문서 전체 차단 (⑥)
            return self._blocked_report(doc_id, f"preprocess_error: {e}", request_id)
        report.timings.preprocess_ms = (time.perf_counter() - t0) * 1000
        report.chunks = chunks

        # ── 저장된 판정 재사용 (⑦): 해시와 pipeline_ver 둘 다 맞을 때만 ──
        hashes = {c.chunk_id: raw_hash(c.raw_text) for c in chunks}
        verdicts: dict[str, Verdict] = {}
        todo: list[Chunk] = []
        for c in chunks:
            cached = None
            if self.use_cache and self.conn is not None:
                cached = store.get(self.conn, c.chunk_id, hashes[c.chunk_id], self.pipeline_ver)
            if cached is not None:
                verdicts[c.chunk_id] = cached.model_copy(update={"cache_hit": True})
            else:
                todo.append(c)

        # ── 룰: 조각마다. 예외는 그 조각만 차단하고 나머지는 정상 진행 (⑥) ──
        rule_results: dict[str, StageResult | None] = {}
        rule_hits: dict[str, list] = {}  # 키워드 위치. 마스킹이 다시 계산하지 않게 넘긴다
        for c in todo:
            t = time.perf_counter()
            try:
                from rule.filter import check_chunk_with_hits

                rule_results[c.chunk_id], rule_hits[c.chunk_id] = check_chunk_with_hits(c)
            except Exception as e:
                rule_results[c.chunk_id] = self._rule_error_result(e, time.perf_counter() - t)
        report.timings.rule_ms = sum(r.latency_ms for r in rule_results.values() if r is not None)

        # ── 1차: 배치 한 번 (④). 장애면 fail-safe 경로 (⑥) ──
        s1_results: dict[str, StageResult | None] = dict.fromkeys(rule_results, None)
        s1_failed = self.stage1 is None
        if self.stage1 is not None and todo:
            try:
                for c, r in zip(todo, self.stage1.classify_chunks(todo), strict=True):
                    s1_results[c.chunk_id] = r
            except Exception as e:
                s1_failed = True
                report.warnings.append(f"1차 장애: {e!r} — 합의한 fail-safe 로 처리 (⑥)")
        elif self.stage1 is None:
            report.warnings.append("1차 모델 없음 — 시작 경고 (⑥). 요청은 합의한 fail-safe")
        report.timings.stage1_ms = sum(r.latency_ms for r in s1_results.values() if r is not None)

        # ── 2차 경로 (④ 표): 하나라도 맞으면 보낸다. 경로마다 따로 센다 ──
        chunk_routes: dict[str, list[str]] = {}
        for c in todo:
            rr, s1 = rule_results[c.chunk_id], s1_results[c.chunk_id]
            lanes = []
            if s1 is not None and config.THETA_LOW <= s1.risk_score < config.THETA_HIGH:
                lanes.append("s1_ambiguous")
            if (
                rr is not None
                and not rr.fallback
                and config.RULE_TO_STAGE2 < rr.risk_score < config.RULE_BLOCK
            ):
                lanes.append("rule_suspect")
            chunk_routes[c.chunk_id] = lanes
            for lane in lanes:
                routes[lane] += 1

        s2_results: dict[str, StageResult | None] = {}
        s2_failed: dict[str, bool] = {}
        pending = [c for c in todo if chunk_routes[c.chunk_id]]
        for c, (res, failed, err, ms) in zip(pending, self._stage2_batch(pending), strict=True):
            report.timings.stage2_ms += ms
            s2_results[c.chunk_id] = res
            s2_failed[c.chunk_id] = failed
            if failed and err:
                report.warnings.append(f"2차 장애({c.chunk_id}): {err}")

        # ── 합치기 + 정책 (⑤) ──
        for c in todo:
            verdicts[c.chunk_id] = self._finalize_chunk(
                c,
                rule_results[c.chunk_id],
                s1_results[c.chunk_id],
                s2_results.get(c.chunk_id),
                s2_needed=bool(chunk_routes[c.chunk_id]),
                s2_was_failed=s2_failed.get(c.chunk_id, False),
                s1_failed=s1_failed,
                routes=routes,
                report=report,
                rule_hits=rule_hits.get(c.chunk_id),
            )

        # ── 문서 단위 집계 (⑤): 공격 구간 합집합을 모든 조각에 적용 ──
        doc_spans = P.union(
            [
                s
                for c in chunks
                for s in P.to_doc_spans(c, verdicts[c.chunk_id].masked_spans)
                if verdicts[c.chunk_id].final_label == "injection"
            ]
        )
        for c in chunks:
            v = verdicts[c.chunk_id]
            applied = P.from_doc_spans(c, doc_spans)
            if v.action == "block":
                report.masked_texts[c.chunk_id] = config.MASK_MARKER
            else:
                if applied != v.masked_spans:
                    v = v.model_copy(update={"masked_spans": applied})
                    verdicts[c.chunk_id] = v
                report.masked_texts[c.chunk_id] = P.mask_text(c, applied)

        # ── 저장 (⑦, fallback 제외) + 로그 (⑥) ──
        for c in chunks:
            v = verdicts[c.chunk_id]
            if self.conn is not None and not v.cache_hit:
                store.put(self.conn, doc_id, hashes[c.chunk_id], v)
            self._log(v, request_id, doc_id, defense_enabled, report)

        # ── 문서 판정 ──
        ordered = [verdicts[c.chunk_id] for c in chunks]
        any_attack = any(v.final_label == "injection" for v in ordered)
        if any(v.action == "block" for v in ordered) and self.policy == "block":
            doc_action = "block"
        elif any(v.action in ("mask", "block") for v in ordered):
            doc_action = "mask"
        elif any(v.action == "warn" for v in ordered):
            doc_action = "warn"
        else:
            doc_action = "allow"
        report.timings.total_ms = (time.perf_counter() - t0) * 1000
        report.doc = DocumentVerdict(
            doc_id=doc_id,
            final_label="injection" if any_attack else "benign",
            action=doc_action,
            attack_spans_raw=doc_spans,
            verdicts=ordered,
            pipeline_ver=self.pipeline_ver,
            fallback=any(v.fallback for v in ordered),
            total_latency_ms=report.timings.total_ms,
        )
        report.routes = routes
        return report

    # ── 조각 하나 마무리: 합치기 → 정책 → Verdict ──

    def _finalize_chunk(
        self,
        chunk: Chunk,
        rule_res: StageResult | None,
        s1: StageResult | None,
        s2: StageResult | None,
        *,
        s2_needed: bool,
        s2_was_failed: bool,
        s1_failed: bool,
        routes: dict[str, int],
        report: DocumentReport,
        rule_hits: list | None = None,
    ) -> Verdict:
        results: list[StageResult] = [r for r in (rule_res, s1, s2) if r is not None]
        merged = merge(rule_res, s1, s2, s2_needed=s2_needed, s2_failed=s2_was_failed)

        # 1차 장애 중의 판정은 룰 0.9 이상 차단만 믿고, 나머지는 전부 fail-safe 표시 (⑥)
        if s1_failed and merged.why != f"rule>={config.RULE_BLOCK}":
            if merged.final == "benign":
                merged = Merged("fail_safe", merged.score, "stage1_unavailable")
            fallback = True
        else:
            fallback = merged.final == "fail_safe"

        action = "allow"
        masked_spans: list[Span] = []
        final_label = "benign"

        if merged.final == "injection":
            final_label = "injection"
            action, masked_spans, extra = self._apply_policy(
                chunk, rule_res, s1, s2, routes, report, rule_hits
            )
            results += extra
        elif merged.final == "warn":
            action = "warn"
        elif merged.final == "fail_safe":
            # fail-open 이어도 "경고"로 넘긴다. 재검사 대상 로그 (⑥)
            action = self.fail_safe
            final_label = "injection" if self.fail_safe == "block" else "benign"
        if rule_res is not None and rule_res.fallback:  # 룰 예외 → 그 조각 차단 (⑥)
            final_label, action, fallback = "injection", "block", True

        path = ["preprocess"]
        for r in results:
            if r.stage not in path:
                path.append(r.stage)
        return Verdict(
            chunk_id=chunk.chunk_id,
            final_label=final_label,
            final_score=round(merged.score, 4),
            action=action,
            path=path,
            stage_results=results,
            masked_spans=masked_spans,
            pipeline_ver=self.pipeline_ver,
            fallback=fallback,
            total_latency_ms=sum(r.latency_ms for r in results),
            meta={"merge": merged.why},
        )

    def _apply_policy(
        self,
        chunk: Chunk,
        rule_res: StageResult | None,
        s1: StageResult | None,
        s2: StageResult | None,
        routes: dict[str, int],
        report: DocumentReport,
        rule_hits: list | None = None,
    ) -> tuple[str, list[Span], list[StageResult]]:
        """공격 조각 처리 (⑤). (action, 지울 구간, 추가 단계 결과) 를 돌려준다"""
        whole = [Span(start=0, end=len(chunk.raw_text))]
        if self.policy == "warn":  # 테스트·모니터링용: 그대로 넘기고 표시만
            return "warn", [], []
        if self.policy == "block":
            return "block", whole, []
        # 2차 형식 오류는 항상 조각 차단 (⑥). 판정 JSON 을 흉내 낸 문장 대비
        if s2 is not None and s2.reason.startswith("format_error"):
            return "block", whole, []

        spans: list[Span] = []
        extra: list[StageResult] = []
        rule_spans, _body_only = ([], False)
        if rule_res is not None and not rule_res.fallback:
            rule_spans, _body_only = P.rule_mask_spans(chunk, rule_hits)
        spans += rule_spans
        spans += P.stage_mask_spans(chunk, s1)
        spans += P.stage_mask_spans(chunk, s2)

        # 위치 찾기 경로: 공격인데 지울 위치가 없으면 2차에 위치만 묻는다 (⑤·④ 표).
        # 1차에서 바로 차단된 평문 공격 조각이 통째로 빠지는 것을 막는다 [바꿈].
        if not spans and self.stage2 is not None:
            routes["locate"] += 1
            res, failed, err, ms = self._call_stage2(chunk)
            report.timings.stage2_ms += ms
            if err:
                report.warnings.append(f"2차 장애(locate, {chunk.chunk_id}): {err}")
            if not failed and res is not None:
                extra.append(res.model_copy(update={"reason": f"locate: {res.reason}"}))
                # 2차가 "정상"이라고 해도 판정을 뒤집지 않는다 (판정 합치기 둘째 줄)
                spans += P.stage_mask_spans(chunk, res)
        if not spans:
            return "block", whole, extra  # 위치가 없으니 조각 전체 차단

        spans = P.union(spans)
        # 재검사 (⑤): 마스킹한 조각이 다시 걸리면 마스킹 대신 통째로 막는다. 한 번만
        recheck = self._recheck(chunk, spans, routes, report)
        extra += recheck[1]
        if recheck[0]:
            return "block", whole, extra
        return "mask", spans, extra

    def _recheck(
        self,
        chunk: Chunk,
        spans: list[Span],
        routes: dict[str, int],
        report: DocumentReport,
    ) -> tuple[bool, list[StageResult]]:
        """마스킹본 재검사. (아직도 공격인가, 추가 단계 결과) 를 돌려준다"""
        from rule.filter import check_chunk

        extra: list[StageResult] = []
        try:
            masked = P.mask_chunk(chunk, spans)
        except Exception as e:  # 마스킹 실패는 차단으로 (⑤ 정책 고르기)
            report.warnings.append(f"마스킹 실패({chunk.chunk_id}): {e!r} — 조각 차단")
            return True, extra
        try:
            rule2 = check_chunk(masked)
        except Exception:
            return True, extra
        extra.append(rule2.model_copy(update={"reason": f"recheck: {rule2.reason}"}))
        s1_2 = None
        if self.stage1 is not None:
            try:
                s1_2 = self.stage1.classify_chunks([masked])[0]
                extra.append(s1_2.model_copy(update={"reason": f"recheck: {s1_2.reason}"}))
            except Exception:
                s1_2 = None
        if rule2.risk_score >= config.RULE_BLOCK or (
            s1_2 is not None and s1_2.risk_score >= config.THETA_HIGH
        ):
            return True, extra
        needs_s2 = (config.RULE_TO_STAGE2 < rule2.risk_score < config.RULE_BLOCK) or (
            s1_2 is not None and config.THETA_LOW <= s1_2.risk_score < config.THETA_HIGH
        )
        if needs_s2 and self.stage2 is not None:
            routes["recheck"] += 1
            res, failed, err, ms = self._call_stage2(masked)
            report.timings.stage2_ms += ms
            if err:
                report.warnings.append(f"2차 장애(recheck, {chunk.chunk_id}): {err}")
            if not failed and res is not None:
                extra.append(res.model_copy(update={"reason": f"recheck: {res.reason}"}))
                if res.label == "injection":
                    return True, extra
            elif failed:
                # 재검사가 장애면 조각 전체를 막는다 (⑤ 1차에서 바로 차단된 조각 [바꿈])
                return True, extra
        return False, extra

    # ── 2차 호출 (동시 상한·타임아웃·회로 차단기, ④) ──

    def _stage2_batch(self, chunks: list[Chunk]):
        futures = []
        for c in chunks:
            if self.stage2 is None:
                futures.append(None)
            elif not self.circuit.allow():
                futures.append("circuit_open")
            else:
                futures.append(self._pool.submit(self._timed_judge, c))
        out = []
        for fut in futures:
            if fut is None:
                out.append((None, True, "stage2_off", 0.0))
            elif fut == "circuit_open":
                out.append((None, True, "circuit_open", 0.0))
            else:
                out.append(self._collect(fut))
        return out

    def _call_stage2(self, chunk: Chunk):
        if self.stage2 is None:
            return None, True, "stage2_off", 0.0
        if not self.circuit.allow():
            return None, True, "circuit_open", 0.0
        return self._collect(self._pool.submit(self._timed_judge, chunk))

    def _timed_judge(self, chunk: Chunk):
        """2차 호출 하나와 걸린 시간(ms). 실패해도 시간은 남기려고 예외를 값으로 돌려준다"""
        started = time.perf_counter()
        try:
            return self.stage2.judge_chunk(chunk), None, (time.perf_counter() - started) * 1000
        except Exception as e:
            return None, e, (time.perf_counter() - started) * 1000

    def _collect(self, fut):
        """(결과, 실패 여부, 오류 설명, 걸린 시간 ms). 시간은 Timings.stage2_ms 에 더한다.
        시간 초과는 기다린 만큼(타임아웃 값)을 걸린 시간으로 센다"""
        try:
            res, exc, ms = fut.result(timeout=self.stage2_timeout_s)
        except FutureTimeout:
            fut.cancel()
            self.circuit.record_failure()
            timeout_ms = self.stage2_timeout_s * 1000
            return None, True, f"timeout>{self.stage2_timeout_s}s", timeout_ms
        if exc is not None:
            self.circuit.record_failure()
            return None, True, repr(exc), ms
        self.circuit.record_success()
        return res, False, None, ms

    # ── 장애·로그 보조 ──

    def _rule_error_result(self, e: Exception, elapsed_s: float) -> StageResult:
        """룰 예외 (⑥: 공격자가 일으킬 수 있음 → 그 조각 차단, 저장 금지)"""
        from rule.rules import RULE_VERSION

        return StageResult(
            stage="rule",
            label="injection",
            risk_score=1.0,
            confidence=0.0,
            reason=f"rule_error: {e!r}",
            fallback=True,
            latency_ms=elapsed_s * 1000,
            model_ver=RULE_VERSION,
        )

    def _blocked_report(
        self, doc_id: str, reason: str, request_id: str | None = None
    ) -> DocumentReport:
        return DocumentReport(
            doc=DocumentVerdict(
                doc_id=doc_id,
                final_label="injection",
                action="block",
                pipeline_ver=self.pipeline_ver,
                total_latency_ms=0.0,
                meta={"blocked_reason": reason},
            ),
            request_id=request_id or uuid.uuid4().hex[:12],
            blocked_reason=reason,
        )

    def _log(self, v: Verdict, request_id: str, doc_id: str, defense_enabled: bool, report) -> None:
        from preprocess.normalize import VERSION as PP_VER

        if self.conn is None:
            return
        try:
            logdb.log_verdict(
                self.conn,
                v,
                request_id=request_id,
                doc_id=doc_id,
                policy=self.policy,
                defense_enabled=defense_enabled,
                preprocess_latency_ms=report.timings.preprocess_ms / max(len(report.chunks), 1),
                preprocess_ver=PREPROCESS_VER_PREFIX + PP_VER,
            )
        except Exception as e:  # 로그 DB 실패 → 판정은 계속, 예비 파일에 쓰고 경고 (⑥)
            report.warnings.append(f"로그 DB 실패: {e!r} — 예비 파일에 기록")
            try:
                self.log_fallback_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_fallback_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"request_id": request_id, "doc_id": doc_id, **v.model_dump()},
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
            except OSError:
                pass
