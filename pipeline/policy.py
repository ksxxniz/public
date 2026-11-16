"""정책 엔진: 걸린 조각 처리와 마스킹 (5단계 ⑤).

마스킹 위치 표 (근거 → 지울 구간, raw_text 기준)
- 룰 +hidden (숨김 구간 안의 공격 문구) → 그 숨김 구간 전체 (사람이 못 보던 글)
- 룰 +decoded / 1차 복원 조각 / 공격으로 나온 복원 결과 → 인코딩 구간 전체
- 룰 태그 문자(SIG-TAG) → 지울 것 없음 (정리본에서 이미 지워짐, 하이라이트만)
- 2차 인용문 → find_quote 로 찾은 문장. 못 찾으면 조각 전체 차단
- 룰의 본문 키워드·신호 규칙 위치 → 마스킹에 쓰지 않는다
  (정규식은 "이전 지시를 모두 무시" 같은 문구만 잡아서 그 뒤가 남는다)

마스킹 순서: 지울 구간을 모으고 → 표시 문자열로 바꾸고(그냥 지우면 앞뒤 문장이
붙어 뜻이 바뀔 수 있다) → mask_chunk 로 재검사용 청크를 만들고 → 지운 구간을
원문 좌표로 바꿔 화면 하이라이트와 로그에 남긴다.

문서 단위 집계: 공격 구간은 문서 좌표의 합집합으로 만들어 그 문서의 모든 조각에
적용한다. 조각이 50토큰씩 겹쳐 같은 공격 문장이 두 조각에 나오므로, 이 합집합이
곧 겹침 중복 제거다.
"""

from common.config import MASK_MARKER
from common.schema import Chunk, Span, StageResult
from preprocess.quote import find_quote
from rule import filter as rule_filter


def union(spans: list[Span]) -> list[Span]:
    """겹치거나 맞닿은 구간을 합친다"""
    out: list[Span] = []
    for s in sorted(spans, key=lambda x: (x.start, x.end)):
        if out and s.start <= out[-1].end:
            if s.end > out[-1].end:
                out[-1] = Span(start=out[-1].start, end=s.end)
        else:
            out.append(Span(start=s.start, end=s.end))
    return out


def _covering_hidden(chunk: Chunk, span: Span) -> Span:
    """숨김 구간 안에서 걸린 위치 → 그 숨김 구간 전체"""
    for ts in chunk.transform_spans:
        if ts.kind != "hidden_text_found":
            continue
        if span.start < ts.span.end and ts.span.start < span.end:
            return ts.span
    return span


def rule_mask_spans(chunk: Chunk) -> tuple[list[Span], bool]:
    """룰 근거의 마스킹 구간. (구간들, 본문에서만 걸린 키워드가 있는가)를 돌려준다.
    본문 키워드는 위치를 마스킹에 쓰지 않으므로 2차 인용문이 필요하다는 뜻이다"""
    spans: list[Span] = []
    body_only = False
    for h in rule_filter.keyword_hits(chunk):
        if h.rule.endswith("+decoded"):
            spans.append(h.span)  # 인코딩 구간 전체 (Base64 일부만 지우는 건 의미 없음)
        elif h.rule.endswith("+hidden"):
            spans.append(_covering_hidden(chunk, h.span))
        elif h.span is not None:
            body_only = True  # 본문·골격 키워드: 문장 단위가 아니라 마스킹에 못 쓴다
    return union(spans), body_only


def stage_mask_spans(chunk: Chunk, result: StageResult | None) -> list[Span]:
    """1차·2차 결과의 마스킹 구간.
    1차: 복원 조각이 근거면 그 인코딩 구간 (본문 근거면 위치 없음 — 2차가 채운다).
    2차: 인용문 위치 전부 (to_stage_result·스텁이 find_quote 로 채워 둔 값)"""
    if result is None or result.label != "injection":
        return []
    if result.evidence_spans:
        return list(result.evidence_spans)
    return [result.evidence_span] if result.evidence_span else []


def locate_quote(chunk: Chunk, quote: str) -> Span | None:
    match = find_quote(chunk, quote)
    return match.span if match else None


def mask_text(chunk: Chunk, raw_spans: list[Span], marker: str = MASK_MARKER) -> str:
    """정리본(text)에 마스킹을 적용한 답변 LLM 입력.
    raw_spans 는 raw_text 기준. offset_map 으로 정리본 글자를 고른다"""
    if not raw_spans:
        return chunk.text
    spans = union(raw_spans)
    out: list[str] = []
    masked_open = False
    for i, ch in enumerate(chunk.text):
        raw_i = chunk.offset_map[i]
        inside = any(s.start <= raw_i < s.end for s in spans)
        if inside:
            if not masked_open:
                out.append(marker)
                masked_open = True
        else:
            out.append(ch)
            masked_open = False
    return "".join(out)


def mask_chunk(chunk: Chunk, raw_spans: list[Span], marker: str = MASK_MARKER) -> Chunk:
    """재검사용 청크 (5단계 ⑤ 재검사용 청크).

    변환 기록이 마스킹해도 그대로면 원래 청크로 재검사할 때 룰 신호가 다시 걸린다
    (시연 3번: 제로폭 문자로 쪼갠 공격을 지웠는데도 조각 전체가 막히는 문제).
    그래서 지운 구간 밖의 transform_spans·decoded_segments 만 남긴 청크를 새로
    만들고, 룰 필터의 check_chunk 를 고치지 않고 재검사에 그대로 쓴다.

    raw_text 와 span 은 그대로 둔다 (좌표 체계 유지). text 는 마스킹본이고,
    표시 문자열의 offset_map 은 지운 구간의 시작 위치를 가리킨다.
    transform_log 는 남긴 transform_spans 로 다시 센다."""
    if not raw_spans:
        return chunk
    spans = union(raw_spans)

    def inside(raw_i: int) -> bool:
        return any(s.start <= raw_i < s.end for s in spans)

    def span_inside(s: Span) -> bool:
        return any(m.start < s.end and s.start < m.end for m in spans)

    text_parts: list[str] = []
    offsets: list[int] = []
    masked_open = False
    for i, ch in enumerate(chunk.text):
        raw_i = chunk.offset_map[i]
        if inside(raw_i):
            if not masked_open:
                anchor = next(s.start for s in spans if s.start <= raw_i < s.end)
                text_parts.append(marker)
                offsets.extend([anchor] * len(marker))
                masked_open = True
        else:
            text_parts.append(ch)
            offsets.append(raw_i)
            masked_open = False

    kept_spans = [ts for ts in chunk.transform_spans if not span_inside(ts.span)]
    kept_segments = [seg for seg in chunk.decoded_segments if not span_inside(seg.span)]
    # 변환 개수를 남긴 위치 기준으로 다시 센다 (신호 7종 + 복원 개수)
    log = chunk.transform_log.model_copy(
        update={
            **{
                kind: sum(1 for ts in kept_spans if ts.kind == kind)
                for kind in (
                    "zero_width_removed",
                    "hangul_filler_removed",
                    "variation_selector_removed",
                    "tag_chars_decoded",
                    "bidi_removed",
                    "jamo_assembled",
                    "homoglyph_replaced",
                    "hidden_text_found",
                )
            },
            "decoded_count": len(kept_segments),
        }
    )
    return Chunk(
        chunk_id=f"{chunk.chunk_id}/masked",
        doc_id=chunk.doc_id,
        raw_text=chunk.raw_text,
        text="".join(text_parts),
        span=chunk.span,
        offset_map=offsets,
        decoded_segments=kept_segments,
        transform_log=log,
        transform_spans=kept_spans,
        meta={**chunk.meta, "masked_from": chunk.chunk_id},
    )


def to_doc_spans(chunk: Chunk, raw_spans: list[Span]) -> list[Span]:
    """조각 raw 좌표 → 문서 원문 좌표 (chunk.span.start 를 더한다)"""
    base = chunk.span.start
    return [Span(start=base + s.start, end=base + s.end) for s in raw_spans]


def from_doc_spans(chunk: Chunk, doc_spans: list[Span]) -> list[Span]:
    """문서 좌표 합집합 → 이 조각에 겹치는 부분만 조각 raw 좌표로"""
    base, top = chunk.span.start, chunk.span.end
    out: list[Span] = []
    for s in doc_spans:
        lo, hi = max(s.start, base), min(s.end, top)
        if lo < hi:
            out.append(Span(start=lo - base, end=hi - base))
    return out
