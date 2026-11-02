"""룰 필터. 전처리가 만든 Chunk 를 받아 StageResult(stage="rule") 를 돌려준다.

규칙은 두 가지다.
- 신호 규칙: 전처리가 남긴 개수(transform_log)·숨김 이유(transform_spans)로
  "일부러 숨긴" 흔적을 잡는다.
- 키워드 규칙: 정리본(text)·복원 결과(decoded_segments)·골격 보기에서
  전형적인 공격 문구를 찾는다.

점수는 risk = 1 - ∏(1 - w) 로 합친다. 같은 규칙 id 는 가장 높은 점수 하나만 쓴다.
룰 점수는 1차 분류기 입력에 넣지 않는다. 규칙 표는 rule/rules.py.
"""

import re
import time
from dataclasses import dataclass

from common.schema import Chunk, Span, StageResult
from rule import rules as R


@dataclass(frozen=True)
class Hit:
    rule: str  # 규칙 id. 찾은 곳에 따라 +hidden, +decoded, +skeleton 이 붙는다
    weight: float
    attack_type: str | None  # 신호 규칙은 None
    span: Span | None  # raw_text 기준


def check_chunk(chunk: Chunk) -> StageResult:
    started = time.perf_counter()
    best = strongest_per_rule(signal_hits(chunk) + keyword_hits(chunk))
    risk = combine(best.values())
    top = max(
        best.values(),
        key=lambda h: (h.weight, h.attack_type is not None, h.span is not None),
        default=None,
    )
    keyword = max(
        (h for h in best.values() if h.attack_type is not None),
        key=lambda h: h.weight,
        default=None,
    )
    return StageResult(
        stage="rule",
        label="injection" if risk >= R.INJECTION_AT else "benign",
        risk_score=round(risk, 4),
        confidence=top.weight if top else R.NO_HIT_CONFIDENCE,
        reason=_reason(best),
        attack_type=keyword.attack_type if keyword else None,
        evidence_span=top.span if top else None,
        latency_ms=(time.perf_counter() - started) * 1000,
        model_ver=R.RULE_VERSION,
    )


def strongest_per_rule(hits: list[Hit]) -> dict[str, Hit]:
    """같은 규칙 id 가 여러 번 걸려도 가장 높은 점수 하나만 남긴다.
    같은 문구를 여러 번 적어 점수를 부풀리는 것을 막는다."""
    best: dict[str, Hit] = {}
    for h in hits:
        if h.rule not in best or h.weight > best[h.rule].weight:
            best[h.rule] = h
    return best


def combine(hits) -> float:
    """하나라도 맞으면 위험: 1 - ∏(1 - w)"""
    safe = 1.0
    for h in hits:
        safe *= 1 - h.weight
    return 1 - safe


def _reason(best: dict[str, Hit]) -> str:
    out = ""
    for rule in sorted(best):
        nxt = f"{out}, {rule}" if out else rule
        if len(nxt) > R.REASON_MAX:
            break
        out = nxt
    return out


# ── 규칙 1: 전처리 신호 ──


def signal_hits(chunk: Chunk) -> list[Hit]:
    log = chunk.transform_log
    per_k = 1000 / max(len(chunk.raw_text), 1)
    out: list[Hit] = []

    def add(rule: tuple[str, float], span: Span | None) -> None:
        out.append(Hit(rule[0], rule[1], None, span))

    if log.tag_chars_decoded >= 1:
        add(R.SIG_TAG, _first_segment(chunk, "unicode_tag"))
    zw_rate = log.zero_width_removed * per_k
    if zw_rate >= R.ZW_PER_K:
        add(R.SIG_ZW, _first_span(chunk, "zero_width_removed"))
    elif zw_rate >= R.ZW_LOW_PER_K:
        add(R.SIG_ZW_LOW, _first_span(chunk, "zero_width_removed"))
    if log.hangul_filler_removed >= 1:
        add(R.SIG_FILLER, _first_span(chunk, "hangul_filler_removed"))
    if log.homoglyph_replaced >= R.HOMO_MIN:
        add(R.SIG_HOMO, _first_span(chunk, "homoglyph_replaced"))
    if log.jamo_assembled >= R.JAMO_MIN:
        add(R.SIG_JAMO, _first_span(chunk, "jamo_assembled"))
    if log.variation_selector_removed >= R.VS_MIN:
        add(R.SIG_VS, _first_span(chunk, "variation_selector_removed"))
    # 아랍어·히브리어 문서에는 방향 제어문자가 정상으로 있다
    if log.bidi_removed >= 1 and not R.RTL_SCRIPT.search(chunk.text):
        add(R.SIG_BIDI, _first_span(chunk, "bidi_removed"))
    for ts in chunk.transform_spans:
        if ts.kind != "hidden_text_found":
            continue
        if ts.note in R.DELIBERATE:
            add(R.SIG_HIDE, ts.span)
        elif ts.note in R.COMMENTS:
            add(R.SIG_COMMENT, ts.span)
    return out


def _first_span(chunk: Chunk, kind: str) -> Span | None:
    return next((t.span for t in chunk.transform_spans if t.kind == kind), None)


def _first_segment(chunk: Chunk, method: str) -> Span | None:
    return next((s.span for s in chunk.decoded_segments if s.method == method), None)


# ── 규칙 2: 키워드·패턴 ──


def keyword_hits(chunk: Chunk) -> list[Hit]:
    hidden = [
        ts.span
        for ts in chunk.transform_spans
        if ts.kind == "hidden_text_found" and ts.note in R.DELIBERATE | R.COMMENTS
    ]
    out: list[Hit] = []
    for rule in R.KEYWORD_RULES:
        for m in rule.pattern.finditer(chunk.text):
            span = chunk.raw_span(m.start(), m.end())
            if _overlaps(span, hidden):
                out.append(Hit(rule.id + "+hidden", R.STRONG, rule.attack_type, span))
            else:
                weight = rule.weight / 2 if _quoted(chunk.text, m.start(), m.end()) else rule.weight
                out.append(Hit(rule.id, weight, rule.attack_type, span))
        for seg in chunk.decoded_segments:
            m = rule.pattern.search(seg.decoded)
            # html_entity·unicode_escape 는 줄 전체를 풀어서 평범한 글도 들어 있다.
            # 본문에 그대로 있는 문구는 숨긴 게 아니므로 본문 규칙에 맡긴다
            if m and m.group() not in chunk.text:
                out.append(Hit(rule.id + "+decoded", R.STRONG, rule.attack_type, seg.span))
    out += skeleton_hits(chunk, {h.rule.split("+")[0] for h in out})
    return out


def _overlaps(span: Span, spans: list[Span]) -> bool:
    return any(span.start < s.end and s.start < span.end for s in spans)


def _quoted(text: str, start: int, end: int) -> bool:
    """공격 문구가 따옴표 안이거나 근처에 예시 표시가 있나 (인용한 정상 글이면 점수를 절반으로)"""
    before = text[max(0, start - R.QUOTE_WINDOW) : start]
    after = text[end : end + R.QUOTE_WINDOW]
    if R.EXAMPLE_MARK.search(before) or R.EXAMPLE_MARK.search(after):
        return True
    before = before.rsplit("\n", 1)[-1]
    after = after.split("\n", 1)[0]
    # 영어 축약형(don't)의 ' 를 따옴표로 보지 않도록 단어 경계에 있는 따옴표만 센다
    opened = any(
        c in R.QUOTE_OPEN and (k == 0 or not before[k - 1].isalnum()) for k, c in enumerate(before)
    )
    closed = any(
        c in R.QUOTE_CLOSE and (k + 1 == len(after) or not after[k + 1].isalnum())
        for k, c in enumerate(after)
    )
    return opened and closed


# ── 골격 보기 ──

_SKELETON_RUN = re.compile(r"[^\W_]+|[@$]+")


def skeleton(text: str) -> tuple[str, list[int]]:
    """casefold → 공백·구두점 제거 → 리트 치환. 글자마다 text 위치를 함께 돌려준다.
    모델 입력에는 절대 쓰지 않는다 (preprocess/quote.py 의 _compress 와 같은 방식)"""
    parts: list[str] = []
    index: list[int] = []
    for m in _SKELETON_RUN.finditer(text):
        run = m.group().casefold().translate(R.LEET)
        if len(run) == m.end() - m.start():
            parts.append(run)
            index.extend(range(m.start(), m.end()))
            continue
        # casefold 로 글자 수가 바뀌는 경우(ß → ss 등)만 한 글자씩
        for i in range(m.start(), m.end()):
            for f in text[i].casefold().translate(R.LEET):
                parts.append(f)
                index.append(i)
    return "".join(parts), index


def skeleton_hits(chunk: Chunk, found: set[str]) -> list[Hit]:
    """본문·복원 결과에서 이미 걸린 규칙은 건너뛴다 (골격 보기에서만 찾은 경우만 점수)"""
    rules = [r for r in R.SKELETON_RULES if r.id not in found]
    if not rules:
        return []
    skel, index = skeleton(chunk.text)
    out: list[Hit] = []
    for rule in rules:
        m = rule.pattern.search(skel)
        if m:
            span = chunk.raw_span(index[m.start()], index[m.end() - 1] + 1)
            out.append(Hit(rule.id + "+skeleton", rule.weight, rule.attack_type, span))
    return out
