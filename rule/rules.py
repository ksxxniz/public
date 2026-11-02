"""룰 필터 규칙 표. 규칙을 고치면 RULE_VERSION 을 올린다.

기준값은 모두 잠정값이다. 정상 문서 기준선(rule.baseline)으로 다시 정하고
docs/decisions.md 에 남긴다.

정규식 작성 규칙
- 자유 간격은 항상 .{0,30}? 처럼 길이를 제한한다. .* 나 겹친 반복은 ReDoS 위험.
- 모듈을 불러올 때 한 번만 compile 한다.
- 정리본(text)은 이미 NFKC·홈글리프·자모 조립을 거쳤으므로 전각·키릴 문자는 따로 다루지 않는다.
"""

import re
from dataclasses import dataclass

RULE_VERSION = "rule-0.2"

INJECTION_AT = 0.7  # risk_score 가 이 값 이상이면 label="injection"
STRONG = 0.9  # 숨김 구간·복원 결과 안에서 찾은 공격 문구
SKELETON = 0.4  # 골격 보기에서만 찾은 공격 문구
NO_HIT_CONFIDENCE = 0.3  # 아무 규칙에도 안 걸렸을 때 confidence
REASON_MAX = 200

# ── 규칙 1: 전처리 신호 ──
# (규칙 id, 점수). 조건은 filter.signal_hits 에 있다.
SIG_TAG = ("SIG-TAG", 0.9)  # tag_chars_decoded >= 1
SIG_ZW = ("SIG-ZW", 0.6)  # zero_width_removed 1000자당 ZW_PER_K 이상
SIG_ZW_LOW = ("SIG-ZW-LOW", 0.2)  # 1000자당 ZW_LOW_PER_K 이상 ZW_PER_K 미만
SIG_FILLER = ("SIG-FILLER", 0.6)  # hangul_filler_removed >= 1
SIG_HOMO = ("SIG-HOMO", 0.6)  # homoglyph_replaced >= HOMO_MIN
SIG_JAMO = ("SIG-JAMO", 0.4)  # jamo_assembled >= JAMO_MIN
SIG_VS = ("SIG-VS", 0.4)  # variation_selector_removed >= VS_MIN
SIG_BIDI = ("SIG-BIDI", 0.5)  # bidi_removed >= 1 이고 text 에 아랍·히브리 글자가 없음
SIG_HIDE = ("SIG-HIDE", 0.5)  # 숨김 이유가 DELIBERATE
SIG_COMMENT = ("SIG-COMMENT", 0.3)  # 숨김 이유가 COMMENTS

ZW_PER_K = 3.0
ZW_LOW_PER_K = 1.0
HOMO_MIN = 2
JAMO_MIN = 2
VS_MIN = 3

# 아랍·히브리 문자 (있으면 방향 제어문자가 정상이므로 SIG-BIDI 를 끈다)
RTL_SCRIPT = re.compile(r"[֐-ࣿיִ-﷿ﹰ-ﻼ]")

# docs/preprocess_transform_log.md 4절의 note 묶음
DELIBERATE = frozenset(
    {
        "display_none",
        "visibility_hidden",
        "opacity_zero",
        "font_size_zero",
        "transparent_color",
        "same_color",
        "offscreen",
        "zero_size_box",
        "scale_zero",
        "content_visibility_hidden",
        "hidden_attr",
        "dialog_closed",
        "svg_non_text",
    }
)
COMMENTS = frozenset({"comment", "cdata", "md_comment"})


# ── 규칙 2: 키워드·패턴 ──
@dataclass(frozen=True)
class KeywordRule:
    id: str
    attack_type: str  # datagen/evasion.py 의 attack_type 과 같은 이름
    weight: float  # 본문(text)에서 찾았을 때 점수
    pattern: re.Pattern[str]


_I = re.IGNORECASE

KEYWORD_RULES: tuple[KeywordRule, ...] = (
    # instruction_override: 이전 지시 무시
    KeywordRule(
        "KW-OVERRIDE-KO",
        "instruction_override",
        0.6,
        re.compile(
            r"(?:이전|위|앞)\s*의?\s*(?:모든\s*)?(?:지시|명령|규칙|안내)\w{0,3}\s*(?:모두\s*|전부\s*)?"
            r"(?:무시|잊|따르지\s*(?:마|말))"
            r"|(?:이전|기존|위|앞)\s*의?\s*(?:모든\s*)?(?:지시|명령|규칙|안내)\w{0,3}\s*"
            r"(?:와|과)?\s*(?:관계|상관)\s*없이"
        ),
    ),
    KeywordRule(
        "KW-OVERRIDE-EN",
        "instruction_override",
        0.6,
        re.compile(
            r"\b(?:ignore|disregard|forget)\b.{0,30}?\b(?:previous|prior|above|earlier)\b"
            r".{0,20}?\b(?:instructions?|rules?|prompts?|directions?)\b"
            r"|\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+(?:above|before|previously)\b"
            r"|\b(?:regardless\s+of|despite|instead\s+of|overriding|override)\s+(?:any\s+|all\s+|the\s+|your\s+)?"
            r"(?:previous|prior|earlier|above|original|system)\s+(?:instructions?|rules|prompts?|guidelines)\b",
            _I,
        ),
    ),
    # system_prompt_leak: 시스템 프롬프트·위 지시문 유출
    # "시스템 프롬프트" 단어만으로는 LLM 개발 문서에 흔해서 출력·공개 동사가 함께 있을 때만 잡는다
    KeywordRule(
        "KW-LEAK-KO",
        "system_prompt_leak",
        0.6,
        re.compile(
            r"(?:시스템\s*프롬프트|(?:위|이전|앞)\s*의?\s*(?:지시문|지시\s*사항|프롬프트|명령문))"
            r"\w{0,3}\s*(?:그대로\s*|전부\s*|모두\s*)?(?:출력|알려|보여|공개|말해|반복|적어)"
            # 초기 지침·전체 구성을 달라는 요구. 매뉴얼 설명문(~합니다)은 빼고 명령형만
            r"|(?:전체|모든|원래|초기|처음\s*받은)\s*(?:구성|설정|지침|지시)\s*(?:텍스트|문|내용|사항)?"
            r"\w{0,3}\s*(?:그대로\s*|전부\s*|모두\s*)?"
            r"(?:(?:제공|출력|공개)\s*(?:해라|하라|하시오|해\s*줘|해\s*주세요|할\s*것)"
            r"|(?:알려|보여)\s*(?:줘|주세요|달라|라))"
        ),
    ),
    KeywordRule(
        "KW-LEAK-EN",
        "system_prompt_leak",
        0.6,
        re.compile(
            r"\b(?:reveal|print|show|repeat|output|display|leak)\b.{0,30}?"
            r"\b(?:system\s+prompt|(?:instructions|prompt)\s+(?:above|verbatim)|initial\s+instructions)\b"
            r"|\b(?:system\s+prompt|instructions\s+above)\b.{0,30}?\bverbatim\b"
            # AI 에게(your) 전체 구성·초기 지침을 달라는 요구
            r"|\b(?:provide|share|print|output|reveal|show|give|tell)\b(?:\s+me)?\s+your\s+"
            r"(?:(?:complete|full|entire)\s+(?:configuration\s+text|config\s+text|system\s+prompt)"
            r"|(?:initial|original|hidden|exact)\s+(?:instructions|system\s+prompt))\b",
            _I,
        ),
    ),
    # role_hijack: 역할 탈취. "개발자 모드"·"act as" 는 정상 글(휴대폰 설명서, 기술 문서)에 흔해서
    # 제한 없는·탈옥 같은 말이 붙을 때만 잡는다
    KeywordRule(
        "KW-ROLE-KO",
        "role_hijack",
        0.5,
        re.compile(
            r"(?:너|당신)\s*(?:은|는)\s*이제\s*.{0,20}?(?:AI|에이아이|봇|모델|어시스턴트|역할)"
            r"|(?:제한|검열|필터)\s*(?:이\s*|가\s*)?없는\s*(?:AI|모드|모델|어시스턴트|챗봇)"
            r"|탈옥\s*모드"
            r"|(?:답|대답|응답|행동|말)\w{0,3}\s*.{0,20}?(?:제한|검열|필터)\s*(?:없이|을\s*무시하고)"
            r"|(?:제한|검열|필터)\s*(?:없이|을\s*무시하고)\s*.{0,20}?(?:답|대답|응답|행동)",
            _I,
        ),
    ),
    KeywordRule(
        "KW-ROLE-EN",
        "role_hijack",
        0.5,
        re.compile(
            r"\byou\s+are\s+now\b.{0,30}?\b(?:AI|assistant|model|bot|DAN|unrestricted|jailbroken)\b"
            r"|\b(?:unrestricted|unfiltered|uncensored|jailbroken)\s+(?:AI|assistant|model|mode|chatbot)\b"
            r"|\bdeveloper\s+mode\s+(?:enabled|activated|on)\b|\bDAN\s+mode\b|\bdo\s+anything\s+now\b"
            r"|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken|evil)\b"
            r"|\b(?:respond|answer|act|behave|reply)\b.{0,40}?\b(?:without|no)\s+(?:any\s+)?"
            r"(?:restrictions|filters|limitations|censorship|guidelines)\b",
            _I,
        ),
    ),
    # data_exfiltration: 대화·비밀 정보를 외부로 보내기
    KeywordRule(
        "KW-EXFIL-KO",
        "data_exfiltration",
        0.6,
        re.compile(
            r"(?:대화\s*(?:내용|기록)|비밀\s*번호|패스워드|개인\s*정보|API\s*키|토큰|이전\s*메시지)"
            r".{0,20}?(?:URL|주소|링크|서버|https?://|이메일)\w{0,3}\s*.{0,10}?"
            r"(?:보내|전송|업로드|첨부)"
            # AI 가 접근할 수 있는 사용자 정보를 꺼내 보이라는 요구
            r"|(?:접근|열람|조회|볼)\s*(?:할\s*)?수\s*있는\s*(?:모든\s*)?(?:사용자|고객|회원|개인)\s*"
            r"(?:데이터|정보)\w{0,3}\s*.{0,10}?(?:나열|출력|알려|보여|공개|전송|보내)",
            _I,
        ),
    ),
    KeywordRule(
        "KW-EXFIL-EN",
        "data_exfiltration",
        0.6,
        re.compile(
            r"\b(?:send|post|forward|upload|transmit|exfiltrate|leak)\b.{0,30}?"
            r"\b(?:conversation|chat\s+history|messages|passwords?|credentials|api\s+keys?|secrets?"
            r"|personal\s+data)\b.{0,30}?\b(?:to|at|via)\b.{0,20}?"
            r"(?:https?://|\burl\b|\bendpoint\b|\bserver\b|\bwebhook\b|\be-?mail\b)"
            r"|\b(?:list|show|print|dump|reveal|output|send)\b.{0,20}?"
            r"\b(?:user|customer|personal)\s+(?:data|information|records)\b.{0,20}?"
            r"\byou\s+(?:have\s+access\s+to|can\s+(?:access|see))",
            _I,
        ),
    ),
    # 쿼리로 데이터를 실어 보내는 외부 이미지. README 배지(?style=…)는 걸리지 않게 이름을 좁힌다
    KeywordRule(
        "KW-EXFIL-IMG",
        "data_exfiltration",
        0.5,
        re.compile(
            r"!\[[^\]\n]{0,100}\]\(https?://[^)\s?]{1,200}\?(?:[^)\s]{0,100}&)?"
            r"(?:data|q|secret|token|key|msg|message|conversation|chat|history|leak|payload)="
        ),
    ),
    # tool_execution: 명령 실행 유도. IT 매뉴얼에 흔해서 낮게 둔다
    KeywordRule(
        "KW-TOOL-KO",
        "tool_execution",
        0.4,
        re.compile(
            r"(?:다음|아래|이)\s*(?:의\s*)?(?:명령어?|코드|스크립트)\w{0,3}\s*"
            r"(?:즉시\s*|바로\s*|반드시\s*)?(?:실행|수행)"
        ),
    ),
    KeywordRule(
        "KW-TOOL-EN",
        "tool_execution",
        0.4,
        re.compile(
            r"\b(?:run|execute)\s+(?:the\s+)?(?:following|this|below)\s+(?:command|code|script|shell)",
            _I,
        ),
    ),
    KeywordRule(
        "KW-TOOL-SHELL",
        "tool_execution",
        0.4,
        re.compile(
            r"\brm\s+-rf\s+[/~]|\b(?:curl|wget)\s[^\n|]{1,200}\|\s*(?:ba|z)?sh\b"
            r"|\b(?:curl|wget)\s+[^\n&;]{1,200}(?:&&|;)\s*(?:unzip|tar|chmod|sh|bash|python3?)\b"
        ),
    ),
    # output_manipulation: 답변 조작. 상품 소개 글에 흔해서 낮게 둔다
    KeywordRule(
        "KW-OUTPUT-KO",
        "output_manipulation",
        0.4,
        re.compile(
            r"(?:이|해당|우리)\s*(?:제품|서비스|회사|상품|사이트)\w{0,3}\s*.{0,15}?"
            r"(?:1위|최고|1등|가장\s*좋)\w{0,5}\s*.{0,10}?추천"
            r"|반드시\s*.{1,30}?(?:라고|이라고)\s*만?\s*(?:답|대답|응답|말)"
            r"|사용자\w{0,2}\s*(?:에게|한테)\s*.{0,20}?(?:알리지|말하지|공개하지|밝히지)\s*(?:마|말)"
        ),
    ),
    KeywordRule(
        "KW-OUTPUT-EN",
        "output_manipulation",
        0.4,
        re.compile(
            r"\brecommend\b.{0,30}?\b(?:as|is)\s+(?:the\s+)?(?:best|number\s+one|#1|top\s+choice)"
            r"|\balways\s+(?:answer|respond|reply|say)\s+(?:with|that)\b"
            r"|\b(?:do\s+not|don't|never)\s+(?:disclose|reveal|mention|tell)\b.{0,40}?\bto\s+(?:the\s+)?users?\b",
            _I,
        ),
    ),
)

# ── 골격 보기 (띄어쓰기·리트 우회 대비) ──
# 골격 보기 = casefold → 공백·구두점·기호 제거 → 리트 글자 치환. 소문자·공백 없는 글에 맞춘다.
# 리트로 l 을 1 로 쓰면 i 가 되므로 l 자리는 [li] 로 받는다.
LEET = str.maketrans(
    {"1": "i", "0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)

SKELETON_RULES: tuple[KeywordRule, ...] = (
    KeywordRule(
        "KW-OVERRIDE-KO",
        "instruction_override",
        SKELETON,
        re.compile(r"(?:이전|위|앞)의?(?:모든)?(?:지시|명령|규칙)\w{0,3}(?:모두|전부)?(?:무시|잊)"),
    ),
    KeywordRule(
        "KW-OVERRIDE-EN",
        "instruction_override",
        SKELETON,
        re.compile(
            r"(?:ignore|disregard|forget)(?:a[li][li]|any|every|the|your){0,2}"
            r"(?:previous|prior|above|earlier)(?:instructions?|ru[li]es|prompts?)"
            r"|(?:ignore|disregard|forget)a[li][li](?:the)?(?:instructions|ru[li]es)"
        ),
    ),
    KeywordRule(
        "KW-LEAK-KO",
        "system_prompt_leak",
        SKELETON,
        re.compile(r"시스템프롬프트\w{0,3}(?:그대로|전부|모두)?(?:출력|알려|보여|공개|말해)"),
    ),
    KeywordRule(
        "KW-LEAK-EN",
        "system_prompt_leak",
        SKELETON,
        re.compile(
            r"(?:reveal|print|show|repeat|output|display|leak)(?:me)?(?:your|the)?"
            r"(?:fu[li][li]|entire|origina[li]|hidden)?systemprompt"
        ),
    ),
)

# ── 인용문 점수 절반 ──
# 공격 문구가 따옴표 안이거나 가까이에 "예:"·"예시"·"example" 이 있으면 본문 점수를 절반으로.
# 보안 교육 글처럼 공격 문구를 인용한 정상 글을 위해서다. 나머지 판단은 2차 LLM 이 한다.
QUOTE_WINDOW = 40
QUOTE_OPEN = "\"'`“‘「『«"
QUOTE_CLOSE = "\"'`”’」』»"
EXAMPLE_MARK = re.compile(
    r"예\s*[:：)]|예시|예를\s*들|의\s*예|example|e\.g\.|for\s+instance|such\s+as", _I
)
