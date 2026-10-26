"""정규화 단계들이 함께 쓰는 작업대(Context)와 변환 기록(Event).

normalize.py 와 각 단계 모듈(unicode_steps.py 등)이 서로를 import 하지 않도록
따로 떼어 둔 파일이다.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from common.schema import DecodedSegment, Span, TransformLog
from preprocess.tracked import RawSpan, TrackedText

MAX_DOCUMENT_CHARS = 2_000_000


class DocumentTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    """변환 1건. kind 는 TransformLog 의 칸 이름, 위치는 문서 원문 기준."""

    kind: str
    start: int
    end: int
    note: str = ""


@dataclass
class Context:
    """정규화 단계들이 함께 쓰는 작업대."""

    tt: TrackedText
    fmt: str
    events: list[Event] = field(default_factory=list)
    # span 은 문서 원문 기준. chunk.py 가 청크 기준으로 바꿔서 넣는다
    decoded_segments: list[DecodedSegment] = field(default_factory=list)
    renormalize: Callable[[str], str] | None = None

    def record(self, kind: str, spans: Iterable[RawSpan], note: str = "") -> None:
        """변환 위치를 남긴다. 청크마다 개수를 세고 위치를 표시하는 데 쓴다."""
        if kind not in TransformLog.model_fields:
            raise ValueError(f"TransformLog 에 없는 기록 종류입니다: {kind!r}")
        self.events.extend(Event(kind, s, e, note) for s, e in spans)

    def add_decoded(self, method: str, raw: RawSpan, decoded: str, depth: int = 1) -> None:
        """복원 결과를 남긴다. 본문(text)에는 섞지 않는다."""
        self.decoded_segments.append(
            DecodedSegment(
                method=method, span=Span(start=raw[0], end=raw[1]), decoded=decoded, depth=depth
            )
        )
