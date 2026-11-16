"""부품 경계 계약 테스트 (5단계 ⑧).

스텁으로도 돌고 진짜 부품으로도 도는 형식 검사다. 진짜 부품 테스트는
@pytest.mark.needs_model 로 분리되어 GPU·서버가 있는 곳에서 따로 돈다.

- 전처리 → 룰·1차: Chunk 규칙 (offset_map 길이·단조 증가, span 범위)
  → tests/test_schema_offsets.py 가 이미 검사한다.
- 룰 → 파이프라인: stage=="rule", 0~1 점수, evidence_span 이 raw_text 범위 안,
  model_ver == RULE_VERSION, evidence_spans(0.3.0)에 걸린 위치 전부.
- 1차 → 파이프라인: 점수 0~1, 조각마다 결과 하나, 입력 순서 유지.
- 2차 → 정책 엔진: label·attack_type·reason 이 있고 인용문이 정리본에서 찾아진다.
- 파이프라인 → 화면: 하이라이트(masked_spans) 좌표가 원문 범위 안.
"""

import pytest

from common.examples import CHUNK, RAW_DOC
from preprocess.chunk import split_document
from rule.rules import RULE_VERSION

BENIGN_TEXT = "연차는 1년에 15일입니다. 반차는 오전과 오후로 나눠 쓸 수 있습니다."


def _chunks(raw: str = RAW_DOC):
    return split_document("contract", raw, fmt="txt")


# ── 룰 → 파이프라인 ──


def test_rule_contract():
    from rule.filter import check_chunk

    for chunk in [CHUNK] + _chunks():
        r = check_chunk(chunk)
        assert r.stage == "rule"
        assert 0.0 <= r.risk_score <= 1.0
        assert r.model_ver == RULE_VERSION
        n = len(chunk.raw_text)
        if r.evidence_span is not None:
            assert 0 <= r.evidence_span.start < r.evidence_span.end <= n
        for s in r.evidence_spans:
            assert 0 <= s.start < s.end <= n
        # 대표 위치는 목록에도 있어야 한다
        if r.evidence_span is not None:
            assert any(
                s.start == r.evidence_span.start and s.end == r.evidence_span.end
                for s in r.evidence_spans
            )


# ── 1차 → 파이프라인 (스텁) ──


def _check_stage1(handle, chunks):
    results = handle.classify_chunks(chunks)
    assert len(results) == len(chunks)  # 조각마다 결과 하나, 입력 순서 유지
    for chunk, r in zip(chunks, results, strict=True):
        assert r.stage == "stage1"
        assert 0.0 <= r.risk_score <= 1.0
        if r.evidence_span is not None:  # 복원 조각 근거면 원문 인코딩 구간
            assert 0 <= r.evidence_span.start < r.evidence_span.end <= len(chunk.raw_text)


def test_stage1_stub_contract():
    from stage1.stub import StubStage1, score

    chunks = _chunks() + _chunks(BENIGN_TEXT)
    _check_stage1(StubStage1(), chunks)
    # E 와 합의한 배치 함수 모양: score(chunks) -> list[StageResult]
    assert [r.risk_score for r in score(chunks)] == [
        r.risk_score for r in StubStage1().classify_chunks(chunks)
    ]


@pytest.mark.needs_model
def test_stage1_real_contract():
    from stage1.infer import load_classifier

    _check_stage1(load_classifier(), _chunks() + _chunks(BENIGN_TEXT))


# ── 2차 → 정책 엔진 (스텁) ──


def test_stage2_stub_contract():
    from preprocess.quote import find_quote
    from stage2_llm.stub import StubJudge

    chunk = _chunks()[0]
    r = StubJudge().judge_chunk(chunk)
    assert r.stage == "stage2"
    assert r.label == "injection"
    assert r.attack_type is not None
    assert r.reason
    # 인용문(근거 위치)이 정리본에서 찾아져야 한다
    assert r.evidence_span is not None
    assert r.evidence_spans
    quote_text = chunk.raw_text[r.evidence_span.start : r.evidence_span.end]
    assert find_quote(chunk, quote_text) is not None

    benign = StubJudge("benign").judge_chunk(chunk)
    assert benign.label == "benign"

    broken = StubJudge("broken_json").judge_chunk(chunk)  # 결정 ⑦ fail-closed 모양
    assert (broken.label, broken.risk_score, broken.confidence) == ("injection", 1.0, 0.0)
    assert broken.reason.startswith("format_error")


@pytest.mark.needs_model
def test_stage2_real_contract():
    from stage2_llm.config import MODELS, PROMPT_VERSION
    from stage2_llm.judge import judge_chunk

    chunk = _chunks()[0]
    r, _info = judge_chunk(chunk, MODELS["exaone"], PROMPT_VERSION)
    assert r.stage == "stage2"
    assert r.label in ("benign", "injection")
    assert 0.0 <= r.risk_score <= 1.0


# ── 파이프라인 → 화면 ──


def test_pipeline_highlight_contract():
    from pipeline.components import load_stage1, load_stage2
    from pipeline.orchestrator import Orchestrator

    orch = Orchestrator(load_stage1("stub"), load_stage2("stub"), None)
    report = orch.inspect_text(RAW_DOC, doc_id="contract")
    doc_len = len(RAW_DOC)
    for s in report.doc.attack_spans_raw:  # 문서 원문 좌표
        assert 0 <= s.start < s.end <= doc_len
    for chunk, v in zip(report.chunks, report.doc.verdicts, strict=True):
        for s in v.masked_spans:  # 조각 raw_text 좌표
            assert 0 <= s.start < s.end <= len(chunk.raw_text)
        assert v.pipeline_ver == orch.pipeline_ver
