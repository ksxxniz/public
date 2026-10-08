"""정규화된 문서를 common.schema.Chunk 목록으로 자른다.

순서: 문서 전체를 먼저 정규화(normalize_document) → 정리본 기준으로 자르기 → 각 청크의
원문 구간은 위치 대응표로 역산한다. 원문을 먼저 자르면 인코딩이 청크 경계에서 잘리고,
숨김 HTML 판정이 깨지고, 토큰 수 기준이 어긋나고, 조각별 NFKC 결과가 전체와 달라진다.
"""

from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence

from common.schema import Chunk, DecodedSegment, Span, TransformLog, TransformSpan
from preprocess.context import Event
from preprocess.normalize import NormalizedDoc, normalize_document
from preprocess.tokens import Tokenizer, token_windows


def split_document(
    doc_id: str, raw: str, fmt: str | None = None, tokenizer: Tokenizer | None = None
) -> list[Chunk]:
    """문서 하나를 Chunk 목록으로 자른다. fmt 가 None 이면 내용으로 판별한다."""
    doc = normalize_document(raw, fmt)
    return chunks_from_document(doc_id, doc, windows=text_windows(doc.text, tokenizer))


def text_windows(text: str, tokenizer: Tokenizer | None = None) -> list[tuple[int, int]]:
    """정리본을 자를 구간 [start, end) 목록."""
    return token_windows(text, tokenizer)


def chunks_from_document(
    doc_id: str, doc: NormalizedDoc, windows: Sequence[tuple[int, int]] | None = None
) -> list[Chunk]:
    """정규화된 문서를 청크로 만든다. windows 를 주면 그 구간대로 자른다 (테스트용)."""
    windows = text_windows(doc.text) if windows is None else list(windows)
    if not windows:
        # 정리본이 비어도(전부 보이지 않는 문자였던 문서 등) 청크 1개는 반드시 만든다.
        # 청크가 0개면 파이프라인이 판정 없이 통과시키고, 변환 기록이라는 증거도 사라진다.
        windows = [(0, 0)]
    events = sorted(doc.events, key=lambda ev: (ev.start, ev.end))
    starts = [ev.start for ev in events]
    return [_make_chunk(doc_id, doc, k, s, e, events, starts) for k, (s, e) in enumerate(windows)]


def raw_bounds(doc: NormalizedDoc, start: int, end: int) -> tuple[int, int]:
    """정리본 [start, end) 를 덮는 원문 구간 [r0, r1).

    - 첫 청크는 문서 맨 앞부터, 마지막 청크는 문서 끝까지 포함한다.
    - 청크 사이에서 지워진 글자(제로폭 등)는 앞 청크에 붙인다. 어느 청크에도 안 들어가면
      그 글자의 변환 기록이 빠지기 때문이다.
    - 1 → N 으로 늘어난 글자 중간에서 잘리면 양쪽 청크가 그 원문 글자를 함께 가진다.
    """
    n_text, n_raw = len(doc.text), len(doc.raw)
    if n_text == 0:
        if (start, end) != (0, 0):
            raise ValueError(f"빈 정리본에는 (0, 0) 구간만 쓸 수 있습니다: ({start}, {end})")
        return 0, n_raw
    if not 0 <= start < end <= n_text:
        raise ValueError(f"정리본 범위를 벗어난 구간입니다: ({start}, {end})")
    r0 = 0 if start == 0 else doc.starts[start]
    r1 = n_raw if end == n_text else max(doc.ends[end - 1], doc.starts[end])
    return r0, r1


def _make_chunk(
    doc_id: str,
    doc: NormalizedDoc,
    k: int,
    start: int,
    end: int,
    events: list[Event],
    starts: list[int],
) -> Chunk:
    r0, r1 = raw_bounds(doc, start, end)
    inside = events[bisect_left(starts, r0) : bisect_left(starts, r1)]
    segments = _segments_in(doc, r0, r1)
    return Chunk(
        chunk_id=f"{doc_id}-c{k}",
        doc_id=doc_id,
        raw_text=doc.raw[r0:r1],
        text=doc.text[start:end],
        span=Span(start=r0, end=r1),
        offset_map=[s - r0 for s in doc.starts[start:end]],
        decoded_segments=segments,
        transform_log=_log_in(inside, n_decoded=len(segments)),
        transform_spans=_spans_in(inside, r0, r1),
        meta={
            "normalize_version": doc.version,
            "unicode_version": doc.unicode_version,
            "fmt": doc.fmt,
        },
    )


def _segments_in(doc: NormalizedDoc, r0: int, r1: int) -> list[DecodedSegment]:
    """청크 원문 구간과 겹치는 복원 결과를 청크 기준 위치로 바꿔서 돌려준다."""
    out = []
    for seg in doc.decoded_segments:
        s, e = seg.span.start, seg.span.end
        if s < r1 and e > r0:
            span = Span(start=max(s, r0) - r0, end=min(e, r1) - r0)
            out.append(seg.model_copy(update={"span": span}))
    return out


def _log_in(events: list[Event], n_decoded: int) -> TransformLog:
    """청크 원문 구간 안에서 시작한 변환만 센다.

    decoded_count 는 이 청크에 들어간 복원 결과(decoded_segments) 개수로 채운다.
    복원 방법과 상관없이 세므로 개수와 목록이 어긋나지 않는다.
    """
    counts = Counter(ev.kind for ev in events)
    counts["decoded_count"] = n_decoded
    return TransformLog(**counts)


def _spans_in(events: list[Event], r0: int, r1: int) -> list[TransformSpan]:
    """청크 원문 구간 안에서 시작한 변환의 위치를 청크 기준으로 돌려준다.

    같은 종류의 변환이 바로 붙어 있으면 (제로폭 문자 5개 연속 등) 한 구간으로 합친다.
    개수는 TransformLog 에 따로 있으니 위치 목록은 짧게 유지한다.
    """
    out: list[TransformSpan] = []
    for ev in events:
        s, e = ev.start - r0, min(ev.end, r1) - r0
        last = out[-1] if out else None
        if last and last.kind == ev.kind and last.note == ev.note and last.span.end == s:
            out[-1] = last.model_copy(update={"span": Span(start=last.span.start, end=e)})
        else:
            out.append(TransformSpan(kind=ev.kind, span=Span(start=s, end=e), note=ev.note))
    return out
