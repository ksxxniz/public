"""1차 스텁 (5단계 ⑧). 모델 파일 없이 계약만 지킨다.

계약 (진짜 부품과 같은 모양)
- classify_chunks(chunks) -> list[StageResult]: 점수 0~1, 조각마다 결과 하나, 입력 순서 유지
- 진짜처럼 본문(text)과 복원 조각(decoded_inputs)을 각각 점수 매겨 최댓값을 쓴다
- 복원 조각이 최고점이면 evidence_span = 그 조각의 원문 인코딩 구간, 본문이면 None

점수는 키워드 기반이라 통합 테스트에서 결정적으로 돌아간다.
- 강한 지시 문구 → 0.95 (θ_high 위: 바로 차단 경로)
- 약한 의심 문구 → 0.60 (θ_low~θ_high: 2차로 가는 애매 경로)
- 그 외 → 0.03 (θ_low 아래: 조기 종료 경로)
"""

import re
import time
from dataclasses import dataclass

from common.schema import Chunk, Span, StageResult
from preprocess.decoded import decoded_inputs

MODEL_VER = "stage1-stub"
THRESHOLD = 0.5  # 진짜 부품과 같다: label 만 정하고, 넘기기·막기 기준값은 파이프라인 몫

_STRONG = re.compile(
    r"이전\s*지[시침]|지[시침][를은]?\s*무시|무시하고|"
    r"ignore\s+(all|previous|above|prior)|disregard\s+(the|all|previous)",
    re.IGNORECASE,
)
_WEAK = re.compile(
    r"시스템\s*프롬프트|system\s*prompt|비밀번호|password|라고\s*답|답하라|답하세요|"
    r"respond\s+with|reveal|새로운?\s*지시",
    re.IGNORECASE,
)


def _score(text: str) -> float:
    # 인용·예시 표시 안의 공격 문구는 점수를 낮춘다 (룰 필터의 인용문 절반 규칙,
    # 1차 학습 데이터의 hard negative 와 같은 취지). 정상 변형 7종의
    # quoted_attack 기준선을 스텁으로도 지키기 위해 필요하다 (⑧ 스텁도 계약을 지킨다)
    from pipeline.stubutil import looks_quoted

    for m in _STRONG.finditer(text):
        if not looks_quoted(text, m.start(), m.end()):
            return 0.95
        return 0.60  # 인용된 공격 문구 → 애매 (2차로)
    if _WEAK.search(text):
        return 0.60
    return 0.03


@dataclass(frozen=True)
class _Best:
    score: float
    source: str  # "body" 또는 "decoded:<method>"
    span: Span | None


class StubStage1:
    model_ver = MODEL_VER

    def classify_chunk(self, chunk: Chunk) -> StageResult:
        return self.classify_chunks([chunk])[0]

    def classify_chunks(self, chunks: list[Chunk]) -> list[StageResult]:
        started = time.perf_counter()
        results = []
        for chunk in chunks:
            best = _Best(_score(chunk.text), "body", None)
            for item in decoded_inputs(chunk):
                s = _score(item.text)
                if s > best.score:
                    best = _Best(s, f"decoded:{item.method}", item.span)
            per_chunk_ms = (time.perf_counter() - started) * 1000 / max(len(chunks), 1)
            results.append(
                StageResult(
                    stage="stage1",
                    label="injection" if best.score >= THRESHOLD else "benign",
                    risk_score=best.score,
                    confidence=max(best.score, 1 - best.score),
                    reason=best.source,
                    attack_type=None,
                    evidence_span=best.span,
                    evidence_spans=[best.span] if best.span else [],
                    latency_ms=per_chunk_ms,
                    model_ver=self.model_ver,
                )
            )
        return results


def score(chunks: list[Chunk]) -> list[StageResult]:
    """E(3단계 서빙)와 합의한 배치 입력 함수 모양 (5단계 ④ 부품 호출 방식)"""
    return StubStage1().classify_chunks(chunks)
