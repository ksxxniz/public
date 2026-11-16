"""2차 스텁 (5단계 ⑧). LLM 서버 없이 계약만 지킨다.

계약
- judge_chunk(chunk) -> StageResult(stage="stage2"): label·attack_type·reason 이 있고,
  공격이면 근거 문장을 원문에서 찾아 evidence_span(과 evidence_spans)을 채운다.
  진짜 부품처럼 인용문을 preprocess.quote.find_quote 로 찾는다 — 스텁도 키워드가
  든 문장을 그대로 돌려줘야 스텁으로 돌린 통합 테스트가 의미 있다.
- mode="benign": 항상 정상 (판정 합치기 표의 "2차를 불렀고 정상" 줄 검증용)
- mode="broken_json": 진짜 부품의 fail-closed 와 같은 결과를 돌려준다
  (JSON 을 다시 물어도 못 읽음 → label="injection", risk 1.0, confidence 0.0, 결정 ⑦).
  reason 이 "format_error" 로 시작하면 정책 엔진은 마스킹 대신 조각을 통째로 막는다.
"""

import re
import time

from common.schema import Chunk, StageResult
from preprocess.quote import find_quote

MODEL_VER = "stage2-stub+v0.0"

_ATTACK = re.compile(
    r"이전\s*지[시침]|지[시침][를은]?\s*무시|무시하고|라고\s*답|답하라|답하세요|"
    r"ignore\s+(all|previous|above|prior)|disregard|respond\s+with|reveal",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?。…])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class StubJudge:
    def __init__(self, mode: str = "ok"):
        assert mode in ("ok", "benign", "broken_json")
        self.mode = mode
        self.model_ver = MODEL_VER if mode == "ok" else f"{MODEL_VER}+{mode}"

    def judge_chunk(self, chunk: Chunk) -> StageResult:
        started = time.perf_counter()
        if self.mode == "broken_json":
            return StageResult(
                stage="stage2",
                label="injection",
                risk_score=1.0,
                confidence=0.0,
                reason="format_error: 다시 물어도 JSON 을 읽지 못함 (fail-closed)",
                attack_type=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                model_ver=self.model_ver,
            )
        quotes = []
        if self.mode == "ok":
            from rule.filter import _quoted

            texts = [chunk.text] + [seg.decoded for seg in chunk.decoded_segments]
            for text in texts:
                for s in _sentences(text):
                    m = _ATTACK.search(s)
                    if m is None:
                        continue
                    # 인용·예시 안의 공격 문구는 "교육 자료의 인용"으로 보고 정상
                    # 처리한다 (진짜 2차 프롬프트 v0.3 의 인용 처리와 같은 취지)
                    base = text.find(s)
                    if base >= 0 and _quoted(text, base + m.start(), base + m.end()):
                        continue
                    quotes.append(s)
        if not quotes:
            return StageResult(
                stage="stage2",
                label="benign",
                risk_score=0.05,
                confidence=0.9,
                reason="공격 지시 문장을 찾지 못함",
                latency_ms=(time.perf_counter() - started) * 1000,
                model_ver=self.model_ver,
            )
        # 인용문을 원문 위치로 (여러 문장이면 전부 — common PR 2 evidence_spans)
        spans = []
        for q in quotes:
            match = find_quote(chunk, q)
            if match is not None:
                spans.append(match.span)
        return StageResult(
            stage="stage2",
            label="injection",
            risk_score=0.97,
            confidence=0.9,
            reason=f"지시 문장 발견: {quotes[0][:80]}",
            attack_type="instruction_override",
            evidence_span=spans[0] if spans else None,
            evidence_spans=spans,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_ver=self.model_ver,
        )
