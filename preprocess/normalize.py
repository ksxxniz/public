"""1단계 정규화. 문서 단위로 한 번에 처리하고, 위치 대응표를 끝까지 유지한다.

처리 순서 (가이드 개정판 '처리 순서'). 순서가 틀리면 우회 구멍이 생긴다.
  ① 입력 디코딩 ......... load.py (바이트 → 문자열)
  ② 포맷 파싱 ........... TODO(3주차) HTML/MD 텍스트 추출, 엔티티 복원, 숨김 구간 표시
  ③ 유니코드 정규화 ..... unicode_steps.py  태그 문자 → 양방향 제어 → 보이지 않는 문자
                                            → NFKC → 자모 조립 → 공백 정리
  ④ 홈글리프 ............ homoglyph.py  문자 체계가 섞인 단어만
  ⑤ 인코딩 복원 ......... TODO(3주차) 결과는 decoded_segments 로 (본문에 섞지 않음)
  ⑦ 청킹 ................ chunk.py

새 단계 만드는 법: Context 를 받아 ctx.tt(TrackedText)를 고치고, 고친 원문 위치를
ctx.record() 로 남기는 함수를 만들어 STEPS 에 순서대로 넣는다.

    def remove_zero_width(ctx: Context) -> None:
        spans = ctx.tt.map_chars(lambda c: "" if c in ZERO_WIDTH else None)
        ctx.record("zero_width_removed", spans)
"""

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from common.schema import DecodedSegment
from preprocess.context import Context, Event
from preprocess.homoglyph import replace_homoglyphs
from preprocess.tracked import TrackedText
from preprocess.unicode_steps import (
    apply_nfkc,
    assemble_jamo,
    decode_tag_chars,
    remove_bidi,
    remove_invisible,
    tidy_whitespace,
)

__all__ = [
    "Context",
    "Event",
    "NormalizedDoc",
    "STEPS",
    "UNICODE_VERSION",
    "VERSION",
    "normalize_document",
    "normalize_text",
    "resolve_format",
]

# 정규화 동작이 바뀔 때마다 올린다. 0.1 = 2주차 말 고정(3단계 학습용), 1.0 = 3주차 말
VERSION = "0.0.6"

# 파이썬 버전마다 유니코드 데이터가 달라 NFKC 결과가 바뀔 수 있다 (3.11 = 14.0, 3.12 = 15.0).
# 학습과 서빙이 같은 정규화를 거쳤는지 확인할 수 있게 청크 meta 에 같이 남긴다.
UNICODE_VERSION = unicodedata.unidata_version

FORMATS = ("txt", "md", "html")
_FORMAT_ALIASES = {"htm": "html", "markdown": "md", "text": "txt"}
_HTML_HEAD = re.compile(r"\A\ufeff?\s*(?:<!doctype\s+html|<html[\s>])", re.IGNORECASE)


Step = Callable[[Context], None]

# 처리 순서대로 넣는다. 순서를 바꾸면 우회 구멍이 생긴다 (unicode_steps.py 맨 위 설명).
STEPS: list[Step] = [
    decode_tag_chars,
    remove_bidi,
    remove_invisible,
    apply_nfkc,
    replace_homoglyphs,
    assemble_jamo,
    tidy_whitespace,
]


@dataclass
class NormalizedDoc:
    """문서 단위 정규화 결과. text[i] 는 원문 raw[starts[i]:ends[i]] 에서 나왔다."""

    raw: str
    text: str
    starts: list[int]
    ends: list[int]
    fmt: str
    events: list[Event]
    decoded_segments: list[DecodedSegment]
    version: str = VERSION
    unicode_version: str = UNICODE_VERSION


def normalize_document(
    raw: str, fmt: str | None = None, steps: Iterable[Step] | None = None
) -> NormalizedDoc:
    """문서 전체를 정규화한다. steps 를 주면 STEPS 대신 그것만 돌린다 (테스트용)."""
    ctx = Context(tt=TrackedText(raw), fmt=resolve_format(raw, fmt))
    for step in STEPS if steps is None else steps:
        step(ctx)
    tt = ctx.tt
    return NormalizedDoc(
        raw=raw,
        text=tt.text,
        starts=list(tt.starts),
        ends=list(tt.ends),
        fmt=ctx.fmt,
        events=list(ctx.events),
        decoded_segments=list(ctx.decoded_segments),
    )


def normalize_text(raw: str, fmt: str | None = None) -> str:
    """정리본 문자열만 돌려준다.

    3단계 학습 코드는 이 함수를 import 해서 쓴다 (복사 금지). 학습과 서빙이 같은
    normalize_document() 를 거쳐야 교재와 시험지의 글씨체가 같아진다.
    """
    return normalize_document(raw, fmt).text


def resolve_format(raw: str, fmt: str | None) -> str:
    """fmt("html", ".htm", "MD" 등)를 txt/md/html 중 하나로 맞춘다.

    None 이면 내용으로 판별한다. HTML 은 문서 첫머리로 알아볼 수 있지만 MD 는 내용만으로
    확실히 알 수 없어서 txt 로 본다. 파일로 들어온 문서는 확장자를 넘겨주는 게 안전하다.
    """
    if fmt is not None:
        f = fmt.strip().lower().lstrip(".")
        f = _FORMAT_ALIASES.get(f, f)
        if f not in FORMATS:
            raise ValueError(f"지원하지 않는 형식입니다: {fmt!r} (지원: {', '.join(FORMATS)})")
        return f
    if _HTML_HEAD.match(raw):
        return "html"
    return "txt"
