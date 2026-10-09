"""common/schema.py 의 위치 규칙 테스트 (raw_span 끝 계산, offset_map 감소 금지 등)."""

import unicodedata

import pytest
from pydantic import ValidationError

from common import examples
from common.schema import Chunk, DecodedSegment, Span, TransformLog, TransformSpan
from preprocess.chunk import split_document


def _chunk(raw: str, text: str, offset_map: list[int], **kw) -> Chunk:
    return Chunk(
        chunk_id="c",
        doc_id="d",
        raw_text=raw,
        text=text,
        span=Span(start=0, end=len(raw)),
        offset_map=offset_map,
        **kw,
    )


def test_raw_span_covers_merged_jamo():
    raw = unicodedata.normalize("NFD", "무시해")  # 자모 6개
    c = _chunk(raw, "무시해", [0, 2, 4])
    assert c.raw_span(0, 3) == Span(start=0, end=6)  # 예전 계산은 end=5
    assert c.raw_span(0, 2) == Span(start=0, end=4)


def test_raw_span_inside_expansion():
    c = _chunk("A㈜B", "A(주)B", [0, 1, 1, 1, 2])
    assert c.raw_span(1, 3) == Span(start=1, end=2)
    assert c.raw_span(0, 5) == Span(start=0, end=3)


def test_raw_span_includes_deleted_chars_right_after():
    c = _chunk("무시\u200b해", "무시해", [0, 1, 3])
    assert c.raw_span(0, 2) == Span(start=0, end=3)  # 문서화된 부작용


def test_example_evidence_span_unchanged():
    chunk = examples.CHUNK
    start = chunk.text.index(examples.ATTACK_PHRASE)
    span = chunk.raw_span(start, start + len(examples.ATTACK_PHRASE))
    assert span == examples.EVIDENCE_SPAN
    assert chunk.raw_text[span.start : span.end].replace(examples.ZWSP, "") == (
        examples.ATTACK_PHRASE
    )


def test_decreasing_offset_map_rejected():
    with pytest.raises(ValidationError, match="감소"):
        _chunk("abc", "abc", [2, 0, 1])


def test_transform_span_kind_must_be_log_field():
    bad = TransformSpan(kind="zero_width_remove", span=Span(start=0, end=1))
    with pytest.raises(ValidationError, match="transform_spans"):
        _chunk("ab", "ab", [0, 1], transform_spans=[bad])


def test_transform_span_must_be_inside_raw_text():
    bad = TransformSpan(kind="zero_width_removed", span=Span(start=0, end=5))
    with pytest.raises(ValidationError, match="transform_spans"):
        _chunk("ab", "ab", [0, 1], transform_spans=[bad])


def test_transform_span_accepted():
    ok = TransformSpan(kind="hidden_text_found", span=Span(start=0, end=1), note="display:none")
    assert _chunk("ab", "ab", [0, 1], transform_spans=[ok]).transform_spans == [ok]


@pytest.mark.parametrize("method", ["unicode_tag", "html_entity", "unicode_escape"])
def test_new_decoded_methods_accepted(method):
    seg = DecodedSegment(method=method, span=Span(start=0, end=1), decoded="x")
    assert seg.method == method


def test_transform_log_counts_nfkc_changes():
    assert TransformLog(nfkc_changed=3).nfkc_changed == 3


def test_raw_span_covers_whole_group_from_any_char():
    c = _chunk("\ufb01\u0301x", "f\u00edx", [0, 0, 2])
    assert c.raw_span(0, 1) == c.raw_span(1, 2) == c.raw_span(0, 2) == Span(start=0, end=2)
    assert c.raw_span(2, 3) == Span(start=2, end=3)


def test_raw_span_of_ligature_with_accent_after_normalization():
    (chunk,) = split_document("d", "\ufb01\u0301x")
    assert chunk.text == "f\u00edx"
    assert chunk.raw_span(0, 1) == Span(start=0, end=2)
