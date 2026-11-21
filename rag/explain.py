"""판정을 사람이 읽는 말로 바꾸고, 질문할 때 정책을 적용한다 (5단계 ③·⑤·⑥).

정책은 질문할 때 적용 (③)
  저장하는 것은 판정(점수, 근거, 공격 구간)까지다. 넣을 때는 항상 기본 정책(마스킹)으로
  검사해서 공격 구간이 가장 자세히 남게 하고, 요청이 고른 정책은 여기서 적용한다.
  - mask : 저장된 판정 그대로. 공격 문장만 지우고, 위치를 못 찾은 조각은 통째로 뺀다
  - block: 공격 판정 조각은 통째로 뺀다
  - warn : 공격 판정 조각도 그대로 넘기고 표시만 한다 (테스트·모니터링용)

화면에서 구분 (⑥)
  "공격으로 차단", "검사 불가로 차단", "경고"를 다르게 표시한다. 검사 불가(장애, 룰 예외,
  2차 형식 오류)는 공격자가 일으킬 수 있으므로 정책과 상관없이 장애 정책을 따른다.
"""

from dataclasses import dataclass, field

from common.schema import Chunk, Span, Verdict
from pipeline import policy as P

POLICIES = ("mask", "block", "warn")

STATUS_LABELS = {  # 가이드 ⑥ 용어("공격으로 차단", "검사 불가로 차단", "경고" 등)는 그대로 쓴다
    "pass": "정상",
    "warn": "경고",
    "mask": "마스킹",
    "attack_block": "공격으로 차단",
    "attack_warn": "탐지 후 통과",
    "unscannable_block": "검사 불가로 차단",
    "unscannable_warn": "검사 불가 · 경고 후 통과",
}

# 답변 LLM 에 들어가지 않는 상태
DROPPED = frozenset({"attack_block", "unscannable_block"})

_MERGE = (  # 판정 합치기 표(pipeline/merge.py)의 줄 → 판정 근거. 앞에서부터 처음 맞는 것
    ("rule>=", "룰 점수 ≥ 차단 임계값"),
    ("stage1>=", "1차 점수 ≥ θ_high"),
    ("stage2=injection", "2차 판정: 공격"),
    ("stage2=benign,rule>=", "2차 판정 정상, 룰 점수 ≥ 경고 임계값"),
    ("stage2=benign", "2차 판정: 정상"),
    ("stage2_unavailable", "2차 판정 불가 (장애)"),
    ("stage1_unavailable", "1차 판정 불가 (장애)"),
    ("early_exit", "룰·1차 점수 미달, 2차 생략"),
)

_RULE_KIND = {
    "OVERRIDE": "지시 무력화",
    "LEAK": "시스템 프롬프트 유출",
    "ROLE": "역할 변경",
    "EXFIL": "데이터 유출",
    "TOOL": "도구 실행 유도",
    "OUTPUT": "응답 조작",
}
_SIGNALS = {
    "SIG-TAG": "유니코드 태그 문자",
    "SIG-ZW": "제로폭 문자 다수",
    "SIG-ZW-LOW": "제로폭 문자 소수",
    "SIG-FILLER": "한글 채움 문자",
    "SIG-HOMO": "동형 문자",
    "SIG-JAMO": "자모 분리 표기",
    "SIG-VS": "변형 선택자",
    "SIG-BIDI": "양방향 제어 문자",
    "SIG-HIDE": "숨김 텍스트",
    "SIG-COMMENT": "HTML·마크다운 주석",
}
_SCOPES = {"hidden": "숨김 영역", "decoded": "디코딩 결과"}
_HIDDEN_NOTES = {
    "display_none": "display:none",
    "visibility_hidden": "visibility:hidden",
    "opacity_zero": "opacity:0",
    "font_size_zero": "font-size:0",
    "transparent_color": "투명 글자색",
    "same_color": "배경과 같은 글자색",
    "offscreen": "화면 밖 배치",
    "zero_size_box": "크기 0 요소",
    "scale_zero": "scale(0)",
    "content_visibility_hidden": "content-visibility:hidden",
    "hidden_attr": "hidden 속성",
    "dialog_closed": "닫힌 dialog 요소",
    "svg_non_text": "SVG 내부 텍스트",
    "comment": "HTML 주석",
    "cdata": "CDATA 구간",
    "md_comment": "마크다운 주석",
    "attr_alt": "alt 속성",
    "attr_title": "title 속성",
    "attr_text": "속성 값 텍스트",
    "meta_content": "meta content 속성",
}


def merge_text(why: str | None) -> str:
    for prefix, text in _MERGE:
        if why and why.startswith(prefix):
            return text
    return why or ""


def rule_findings(reason: str) -> list[dict]:
    """룰 reason("KW-OVERRIDE-KO+hidden, SIG-HIDE")을 규칙별로 나눈다. 모르는 id 는 id 그대로.

    id: 규칙 id, scope: 걸린 곳(hidden: 숨김 영역, decoded: 디코딩 결과, "": 본문), text: 설명"""
    out = []
    for part in (p.strip() for p in reason.split(",")):
        if not part:
            continue
        rule, _, where = part.partition("+")
        if rule.startswith("KW-"):
            kind = rule.split("-")[1] if rule.count("-") >= 1 else rule
            text = _RULE_KIND.get(kind, rule)
        else:
            text = _SIGNALS.get(rule, rule)
        if where in _SCOPES:
            text += f" ({_SCOPES[where]})"
        out.append({"id": rule, "scope": where, "text": text})
    return out


def hidden_note_text(note: str) -> str:
    return _HIDDEN_NOTES.get(note, note or "숨김 텍스트")


def encoded_note_text(seg) -> str:
    """인코딩 구간 설명 (마우스를 올리면 보임): 방식과 디코딩 결과 앞부분"""
    return f"{seg.method} 디코딩: {seg.decoded[:300]}"


def _format_error(v: Verdict) -> bool:
    return any(r.stage == "stage2" and r.reason.startswith("format_error") for r in v.stage_results)


def is_unscannable(v: Verdict) -> bool:
    """검사 불가: 장애로 내린 판정(룰 예외 포함), 공격 판정이 2차 형식 오류에서만 나온 경우 (⑥).

    룰 0.9 이상이나 1차 θ_high 이상으로 이미 공격이 확정된 조각은 2차가 고장 났어도
    '공격으로 차단'이다. 판정 합치기 표에서 그 줄들이 2차보다 위에 있다."""
    if v.fallback:
        return True
    return (v.meta.get("merge") or "").startswith("stage2=injection") and _format_error(v)


def why_text(v: Verdict | None, decision: "Decision") -> str:
    """'판정 근거' 한 줄. 검사 불가는 무엇이 고장 났는지를 쓴다"""
    if v is None:
        return "재검사 실패, 청크 차단"
    if decision.status.startswith("unscannable"):
        if any(r.stage == "rule" and r.reason.startswith("rule_error") for r in v.stage_results):
            return "룰 필터 예외, 해당 청크만 차단"
        if _format_error(v):
            return "2차 응답 형식 오류, 우회 방지를 위해 차단"
        merge = v.meta.get("merge") or ""
        if merge.startswith("stage1_unavailable"):
            return "1차 장애, 장애 정책 적용"
        return "2차 장애, 장애 정책 적용"
    return merge_text(v.meta.get("merge"))


@dataclass
class Decision:
    status: str  # STATUS_LABELS 의 키
    removed: list[Span] = field(
        default_factory=list
    )  # 답변 LLM 입력에서 지운 구간 (조각 원문 좌표)
    flagged: list[Span] = field(default_factory=list)  # 공격 구간. 지우지 않았어도 화면에 표시

    @property
    def label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def dropped(self) -> bool:
        return self.status in DROPPED


def decide(v: Verdict, policy: str) -> Decision:
    """저장된 판정(기본 정책 마스킹으로 낸 것)에 요청 정책을 적용한다"""
    if policy not in POLICIES:
        raise ValueError(f"정책을 모름: {policy} (가능: {', '.join(POLICIES)})")
    spans = list(v.masked_spans)  # 이 조각의 공격 구간 + 겹치는 이웃 조각의 공격 구간
    keep = [] if policy == "warn" else spans
    if is_unscannable(v):  # 장애 정책은 요청 정책과 상관없이 그대로 (⑥)
        if v.action == "block":
            return Decision("unscannable_block", [], spans)
        return Decision("unscannable_warn", keep, spans)
    if v.final_label == "injection":
        if policy == "warn":
            return Decision("attack_warn", [], spans)
        if policy == "block" or v.action == "block":
            return Decision("attack_block", [], spans)
        return Decision("mask", spans, spans)
    return Decision("warn" if v.action == "warn" else "pass", keep, spans)


def llm_input(chunk: Chunk, decision: Decision) -> str | None:
    """방어 ON 에서 답변 LLM 이 받는 글. None 이면 이 조각은 빠진다"""
    if decision.dropped:
        return None
    return P.mask_text(chunk, decision.removed)


def segments(text: str, layers: dict[str, list[Span]], notes: dict[str, list] | None = None):
    """text 를 표시용 조각으로 나눈다. 화면이 글자 위치를 계산하지 않게 하려는 것이다.

    파이썬 위치는 코드포인트, 브라우저(JS)는 UTF-16 이라 이모지·태그 문자에서 어긋난다
    (decisions.md ⑥). 서버에서 잘라 보내면 화면은 이어 붙이기만 한다.
    layers: 이름 → 구간 목록. 조각마다 덮는 층 이름을 flags 로 단다.
    notes: 이름 → [(구간, 설명)]. 덮는 구간의 설명을 note 로 단다 (마우스를 올리면 보임)."""
    n = len(text)
    points = {0, n}
    for spans in layers.values():
        for s in spans:
            points.add(max(0, min(n, s.start)))
            points.add(max(0, min(n, s.end)))
    ordered = sorted(points)
    out: list[dict] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        if a >= b:
            continue
        flags = [
            name
            for name, spans in layers.items()
            if any(s.start <= a and b <= s.end for s in spans)
        ]
        note_texts = []
        for name, items in (notes or {}).items():
            if name in flags:
                note_texts += [t for s, t in items if s.start <= a and b <= s.end]
        # 겹치는 두 조각이 같은 숨긴 글을 각각 잡아도 설명은 한 번만 (그래야 한 덩어리로 합쳐진다)
        note = " / ".join(dict.fromkeys(note_texts))
        piece = {"text": text[a:b], "flags": flags, "note": note}
        last = out[-1] if out else None
        if last and last["flags"] == piece["flags"] and last["note"] == piece["note"]:
            last["text"] += piece["text"]
        else:
            out.append(piece)
    return out


def chunk_segments(chunk: Chunk, decision: Decision | None) -> list[dict]:
    """조각 원문을 표시용으로: 지운 곳, 공격 구간, 숨긴 글, 인코딩 구간"""
    hidden = [ts for ts in chunk.transform_spans if ts.kind == "hidden_text_found"]
    layers = {
        "removed": list(decision.removed) if decision else [],
        "flagged": list(decision.flagged) if decision else [],
        "hidden": [ts.span for ts in hidden],
        "encoded": [seg.span for seg in chunk.decoded_segments],
    }
    notes = {
        "hidden": [(ts.span, hidden_note_text(ts.note)) for ts in hidden],
        "encoded": [(seg.span, encoded_note_text(seg)) for seg in chunk.decoded_segments],
    }
    return segments(chunk.raw_text, layers, notes)


def stage_rows(v: Verdict) -> list[dict]:
    """단계별 결과. 재검사·위치 찾기 결과는 reason 앞에 'recheck:'·'locate:' 가 붙어 있다"""
    rows = []
    for r in v.stage_results:
        phase, _, reason = r.reason.partition(": ")
        if phase not in ("recheck", "locate"):
            phase, reason = "main", r.reason
        # 형식 오류·룰 예외는 판정이 아니라 고장이다. 화면에 '공격' 대신 '오류'로 보인다
        error = reason.startswith(("format_error", "rule_error"))
        rows.append(
            {
                "stage": r.stage,
                "phase": phase,
                "label": r.label,
                "error": error,
                "score": round(r.risk_score, 3),
                "reason": reason,
                "rules": rule_findings(reason) if r.stage == "rule" and not error else [],
                "attack_type": r.attack_type,
                "model_ver": r.model_ver,
                "latency_ms": round(r.latency_ms, 1),
                "fallback": r.fallback,
            }
        )
    return rows
