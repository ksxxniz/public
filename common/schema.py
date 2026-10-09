"""팀 공용 데이터 양식. 이 파일을 고치는 PR 은 전원 리뷰 필수.

좌표 규칙
- Span 은 [start, end) 반열린 구간, 글자(파이썬 str 인덱스) 단위.
- Chunk.span                     : 원본 '문서' 안에서 이 청크의 위치
- Chunk.offset_map[i]            : text[i] 가 chunk.raw_text 의 몇 번째 글자였나
- DecodedSegment.span,
  StageResult.evidence_span      : chunk.raw_text 기준
  → 문서 기준 위치가 필요하면 chunk.span.start 를 더한다.
- 정리본(text) 에서 찾은 위치는 반드시 Chunk.raw_span() 으로 원문 위치로 바꿔서 내보낸다.
- raw_span() 의 끝은 '다음 글자의 원문 시작 위치'까지 잡는다. 여러 글자가 한 글자로
  합쳐진 경우(NFD 자모 → 음절)에도 원문을 빠짐없이 덮기 위해서다. 대신 구간 바로 뒤에서
  지워진 글자(제로폭 등)도 함께 포함된다. 원문 시작 위치가 같은 다음 글자들(한 원문
  묶음에서 함께 나온 글자, 예: ﬁ́ → fí)은 건너뛰고 위치가 바뀌는 글자까지 잡는다.
- offset_map 은 감소하지 않는다. 전처리는 글자 순서를 바꾸지 않는다.
- 위치 단위는 파이썬 코드포인트다. 웹(JS)은 UTF-16 단위라서 이모지·태그 문자처럼
  U+FFFF 를 넘는 글자가 2칸이 된다. web 에서 바꿔 쓴다.
"""

from bisect import bisect_right
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.2.3"

Label = Literal["benign", "injection"]
Stage = Literal["preprocess", "rule", "stage1", "stage2"]
Action = Literal["allow", "mask", "warn", "block"]


class _Model(BaseModel):
    # 모르는 필드는 에러 (오타 난 필드가 조용히 무시되지 않도록)
    model_config = ConfigDict(extra="forbid")


class Span(_Model):
    """원문(raw_text) 기준 글자 위치 [start, end)"""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self):
        if self.end < self.start:
            raise ValueError(f"end({self.end}) 가 start({self.start}) 보다 작습니다")
        return self


class DecodedSegment(_Model):
    """Base64 등을 복원한 결과. 본문에 섞지 않고 여기 따로 담는다"""

    method: Literal["base64", "hex", "url", "rot13", "unicode_tag", "html_entity", "unicode_escape"]
    span: Span  # 원문에서 인코딩된 문자열이 있던 위치
    decoded: str  # 복원된 내용
    depth: int = Field(default=1, ge=1)  # 몇 겹으로 감싸져 있었나


class TransformLog(_Model):
    """정규화 과정에서 무엇을 몇 개 처리했나. 개수만 담는다 (위치는 Chunk.transform_spans).
    이 숫자 자체가 1차 분류기·룰 필터의 단서가 된다. 청크마다 길이가 다르므로
    비교할 때는 raw_text 길이로 나눈 비율(예: 1000자당 개수)로 쓴다"""

    # 제로폭 문자 등 보이지 않는 글자와 제어 문자 (이모지 안 ZWJ·맨 앞 BOM 제외)
    zero_width_removed: int = Field(default=0, ge=0)
    emoji_zwj: int = Field(default=0, ge=0)  # 이모지 조합 안의 ZWJ. 정상 신호라 따로 센다
    hangul_filler_removed: int = Field(default=0, ge=0)  # U+3164 등. Cf 가 아니라 따로 센다
    variation_selector_removed: int = Field(default=0, ge=0)
    tag_chars_decoded: int = Field(default=0, ge=0)  # 태그 문자 U+E0000~E007F
    bidi_removed: int = Field(default=0, ge=0)  # 방향 제어문자
    nfkc_changed: int = Field(default=0, ge=0)  # NFKC 로 바뀐 곳 수 (0.1.0 에서는 참/거짓)
    jamo_assembled: int = Field(default=0, ge=0)  # 쪼갠 자모(ㄱㅗㅇ)를 조립해 만든 음절 수
    homoglyph_replaced: int = Field(default=0, ge=0)  # 닮은꼴 글자
    hidden_text_found: int = Field(default=0, ge=0)  # 숨김 텍스트. 지우지 않고 표시만 한다
    # 이 청크의 decoded_segments 개수. 복원 방법(Base64·태그 문자 등)과 상관없이 센다.
    # 종류는 decoded_segments[].method 로, 숨긴 태그 글자 수는 tag_chars_decoded 로 본다
    decoded_count: int = Field(default=0, ge=0)


class TransformSpan(_Model):
    """변환이 일어난 원문 위치. web 이 '여기에 보이지 않는 문자가 있었다'를 표시할 때 쓴다"""

    kind: str  # TransformLog 의 칸 이름
    span: Span  # chunk.raw_text 기준
    note: str = ""  # 예: 숨김 이유 "display:none"


class Chunk(_Model):
    chunk_id: str
    doc_id: str
    raw_text: str  # 손대지 않은 원문
    text: str  # 정규화 후
    span: Span  # 원본 문서에서 이 청크의 위치
    # text 의 i번째 글자가 raw_text 의 몇 번째 글자였나
    offset_map: list[int] = Field(default_factory=list)
    decoded_segments: list[DecodedSegment] = Field(default_factory=list)
    transform_log: TransformLog = Field(default_factory=TransformLog)
    transform_spans: list[TransformSpan] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_offsets(self):
        n_raw = len(self.raw_text)
        if self.span.end - self.span.start != n_raw:
            raise ValueError(
                f"span 길이({self.span.end - self.span.start})와 raw_text 길이({n_raw})가 다릅니다"
            )
        # offset_map 은 text 글자 수만큼 반드시 있어야 한다
        if len(self.offset_map) != len(self.text):
            raise ValueError(
                f"offset_map 길이({len(self.offset_map)})가 text 길이({len(self.text)})와 다릅니다"
            )
        if self.offset_map and (min(self.offset_map) < 0 or max(self.offset_map) >= n_raw):
            raise ValueError(f"offset_map 값은 0 이상 {n_raw} 미만이어야 합니다")
        for i in range(1, len(self.offset_map)):
            if self.offset_map[i] < self.offset_map[i - 1]:
                raise ValueError(f"offset_map 이 {i}번째 글자에서 감소합니다")
        for seg in self.decoded_segments:
            if seg.span.end > n_raw:
                raise ValueError(f"decoded_segments.span({seg.span.end})이 raw_text 밖입니다")
        for ts in self.transform_spans:
            if ts.kind not in TransformLog.model_fields:
                raise ValueError(f"transform_spans.kind({ts.kind!r})가 TransformLog 칸이 아닙니다")
            if ts.span.end > n_raw:
                raise ValueError(f"transform_spans.span({ts.span.end})이 raw_text 밖입니다")
        return self

    def raw_span(self, start: int, end: int) -> Span:
        """text 기준 [start, end) → raw_text 기준 Span.
        판정 근거를 원문에 표시할 때 반드시 이걸 거친다.

        끝 위치는 '다음 글자의 원문 시작 위치'까지 잡는다 (맨 위 좌표 규칙 참고).
        예전 계산(offset_map[end - 1] + 1)은 NFD 자모가 합쳐진 글자에서 끝을 잘랐다."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"text 범위를 벗어난 구간입니다: [{start}, {end})")
        last = self.offset_map[end - 1]
        k = bisect_right(self.offset_map, last, end)
        nxt = self.offset_map[k] if k < len(self.offset_map) else len(self.raw_text)
        return Span(start=self.offset_map[start], end=max(last + 1, nxt))


class StageResult(_Model):
    stage: Stage
    label: Label
    risk_score: float = Field(ge=0.0, le=1.0)  # 높을수록 위험
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    attack_type: str | None = None
    evidence_span: Span | None = None  # raw_text 기준
    latency_ms: float = Field(ge=0.0)
    model_ver: str


class Verdict(_Model):
    chunk_id: str
    final_label: Label
    final_score: float = Field(ge=0.0, le=1.0)
    action: Action
    path: list[Stage]  # 예: ["preprocess","stage1"] ← 2차 안 감
    stage_results: list[StageResult]
    total_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_path(self):
        missing = [r.stage for r in self.stage_results if r.stage not in self.path]
        if missing:
            raise ValueError(f"stage_results 의 단계 {missing} 가 path 에 없습니다")
        return self
