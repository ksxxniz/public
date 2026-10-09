"""③ 유니코드 정규화 단계 6개. normalize.STEPS 에 이 순서대로 들어간다.

  1. decode_tag_chars   태그 문자 해독        → tag_chars_decoded, decoded_segments
  2. remove_bidi        양방향 제어문자 제거   → bidi_removed
  3. remove_invisible   보이지 않는 문자 제거  → zero_width_removed, hangul_filler_removed,
                                                  variation_selector_removed, emoji_zwj
  4. apply_nfkc         NFKC (호환 자모는 건드리지 않음) → nfkc_changed
  5. assemble_jamo      쪼갠 자모 조립 (ㄱㅗㅇㄱㅕㄱ → 공격) → jamo_assembled
  6. tidy_whitespace    줄바꿈·공백 정리 (기록 안 함, 의심 신호가 아님)

순서가 중요하다. 태그 문자와 양방향 제어문자도 유니코드 분류가 Cf(형식 문자)라서,
3번의 "Cf 는 지운다" 규칙이 먼저 돌면 태그 문자는 해독 전에 사라지고 양방향 제어문자는
제로폭 문자로 잘못 세어진다. 자모 조립은 제로폭 문자를 지운 뒤에 해야
"ㅁ<제로폭>ㅜㅅㅣ" 처럼 섞어 쓴 우회도 한 덩어리로 보인다.
"""

import re
import unicodedata
from itertools import accumulate

from preprocess.context import Context
from preprocess.tracked import Edit

# ── 1. 태그 문자 ──

_TAG_FIRST, _TAG_LAST = 0xE0000, 0xE007F
_CANCEL_TAG = "\U000e007f"
_BLACK_FLAG = "\U0001f3f4"  # 🏴 + 태그 문자 = 잉글랜드·스코틀랜드 같은 지역 깃발 이모지
# 실제로 쓰이는(RGI) 지역 깃발은 이 셋뿐이다.
# 모양만 보고 판정하면 🏴 + 명령문 + 취소 태그로 숨길 수 있다
_RGI_SUBDIVISION_FLAGS = frozenset({"gbeng", "gbsct", "gbwls"})


def _is_tag(ch: str) -> bool:
    return _TAG_FIRST <= ord(ch) <= _TAG_LAST


def _is_skippable(ch: str) -> bool:
    """태그 문자 사이에 끼워 넣어도 화면에 안 보이는 글자 (태그 자체는 제외)."""
    return not _is_tag(ch) and (
        ch in _BIDI
        or ch in _HANGUL_FILLERS
        or _is_variation_selector(ch)
        or _is_extra_invisible(ch)
        or unicodedata.category(ch) == "Cf"
    )


def decode_tag_chars(ctx: Context) -> None:
    """태그 문자를 해독해서 복원 칸에 넣고 본문에서는 지운다.

    태그 문자는 화면에 안 보이지만 하나하나가 ASCII 한 글자에 대응해서, AI 가 숨긴 영어
    문장을 읽어낼 수 있다 (ASCII 스머글링). 지우기만 하면 증거가 사라지므로 먼저 해독한다.
    단, 🏴 뒤에 붙은 지역 깃발 이모지는 정상이라 지우기만 하고 세지 않는다.
    """
    tt = ctx.tt
    text = tt.text
    suspicious: list[int] = []
    i = 0
    while i < len(text):
        if not _is_tag(text[i]):
            i += 1
            continue
        # 태그 사이에 제로폭·변형 선택자 등을 끼워 해독 결과를 조각내는 우회를 막는다.
        # 끼어 있는 글자는 여기서 지우지 않는다 (뒤 단계가 지우고 따로 센다)
        j = last = i
        while j < len(text) and (_is_tag(text[j]) or _is_skippable(text[j])):
            if _is_tag(text[j]):
                last = j
            j += 1
        j = last + 1
        tag_pos = [k for k in range(i, j) if _is_tag(text[k])]
        codes = [ord(text[k]) - _TAG_FIRST for k in tag_pos]
        decoded = "".join(chr(c) for c in codes if 0x20 <= c <= 0x7E)
        is_flag = (
            i > 0
            and text[i - 1] == _BLACK_FLAG
            and text[j - 1] == _CANCEL_TAG
            and decoded in _RGI_SUBDIVISION_FLAGS
        )
        if not is_flag:
            suspicious.extend(tag_pos)
            if decoded:
                ctx.add_decoded("unicode_tag", tt.raw_span(i, j), decoded)
        i = j

    flagged = set(suspicious)
    edits = [Edit(k, k + 1, "") for k, ch in enumerate(text) if _is_tag(ch)]
    spans = tt.apply(edits)
    ctx.record(
        "tag_chars_decoded", [s for e, s in zip(edits, spans, strict=True) if e.start in flagged]
    )


# ── 2. 양방향 제어문자 ──

_BIDI = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c")


def remove_bidi(ctx: Context) -> None:
    """글자 표시 순서를 뒤집는 특수문자를 지운다.

    아랍어·히브리어 정상 문서에도 들어 있으니 개수는 신호로만 쓴다.
    """
    spans = ctx.tt.map_chars(lambda c: "" if c in _BIDI else None)
    ctx.record("bidi_removed", spans)


# ── 3. 보이지 않는 문자 ──

_ZWJ = "\u200d"
_VS15, _VS16 = "\ufe0e", "\ufe0f"
_KEYCAP = "\u20e3"
_HANGUL_FILLERS = frozenset("\u3164\u115f\u1160\uffa0")
# Cf 가 아니라서 분류 규칙으로 안 잡히는 보이지 않는 문자
_EXTRA_INVISIBLE = frozenset("\u034f\u17b4\u17b5\u2065\u2800\U0001d159")
_INVISIBLE_RANGES = ((0xFFF0, 0xFFF8), (0xE0080, 0xE00FF), (0xE01F0, 0xE0FFF))
_MONGOLIAN_FVS = frozenset("\u180b\u180c\u180d\u180f")

# 이모지 범위 (근사치). 파이썬 unicodedata 에는 Extended_Pictographic 속성이 없다
_PICTO_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x2B00, 0x2BFF),
    (0x2190, 0x21FF),
    (0x25FB, 0x25FE),
    (0x2934, 0x2935),
)
_PICTO_SINGLES = frozenset(
    "\u00a9\u00ae\u203c\u2049\u2122\u2139\u3030\u303d\u3297\u3299\u24c2\u25aa\u25ab\u25b6\u25c0"
)


def _is_picto(ch: str) -> bool:
    cp = ord(ch)
    return ch in _PICTO_SINGLES or any(a <= cp <= b for a, b in _PICTO_RANGES)


def _is_variation_selector(ch: str) -> bool:
    cp = ord(ch)
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF or ch in _MONGOLIAN_FVS


def _is_extra_invisible(ch: str) -> bool:
    cp = ord(ch)
    if cp < 0xA0:
        return (cp < 0x20 or cp >= 0x7F) and not ch.isspace()
    return ch in _EXTRA_INVISIBLE or (
        cp >= 0xFFF0 and any(a <= cp <= b for a, b in _INVISIBLE_RANGES)
    )


def _prev_picto(text: str, i: int) -> bool:
    """i 앞의 글자가 (변형 선택자를 건너뛰고) 이모지인가."""
    k = i - 1
    while k >= 0 and _is_variation_selector(text[k]):
        k -= 1
    return k >= 0 and _is_picto(text[k])


def remove_invisible(ctx: Context) -> None:
    """제로폭 문자·한글 채움 문자·변형 선택자를 지우고 종류별로 센다.

    정상적인 이모지를 이루는 문자는 지우되 의심 신호에서 뺀다.
    - 이모지 사이의 ZWJ (👨‍👩‍👧)  → emoji_zwj 로 따로 센다
    - 이모지 바로 뒤의 U+FE0F (❤️), 키캡 숫자 (1️⃣)  → 세지 않는다
    """
    text = ctx.tt.text
    edits: list[Edit] = []
    kinds: list[str | None] = []
    for i, ch in enumerate(text):
        if " " <= ch <= "~" or "\uac00" <= ch <= "\ud7a3":
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch == _ZWJ:
            kind = "emoji_zwj" if _prev_picto(text, i) and nxt and _is_picto(nxt) else None
            kind = kind or "zero_width_removed"
        elif ch in _HANGUL_FILLERS:
            kind = "hangul_filler_removed"
        elif _is_variation_selector(ch):
            prev = text[i - 1] if i else ""
            normal_emoji = ch in (_VS15, _VS16) and (
                (prev and _is_picto(prev)) or (prev in "0123456789#*" and prev and nxt == _KEYCAP)
            )
            kind = None if normal_emoji else "variation_selector_removed"
        elif _is_extra_invisible(ch) or unicodedata.category(ch) == "Cf":
            kind = "zero_width_removed"
        else:
            continue
        edits.append(Edit(i, i + 1, ""))
        kinds.append(kind)

    spans = ctx.tt.apply(edits)
    by_kind: dict[str, list] = {}
    for kind, span in zip(kinds, spans, strict=True):
        if kind is not None:
            by_kind.setdefault(kind, []).append(span)
    for kind, kind_spans in by_kind.items():
        ctx.record(kind, kind_spans)


# ── 4. NFKC ──


def _is_conjoining_jamo(ch: str) -> bool:
    cp = ord(ch)
    return 0x1100 <= cp <= 0x11FF or 0xA960 <= cp <= 0xA97F or 0xD7B0 <= cp <= 0xD7FF


_CONJOINING = re.compile("[\u1100-\u11ff\ua960-\ua97f\ud7b0-\ud7ff]")


def _is_jamo_letter(ch: str) -> bool:
    """호환 자모(ㄱ, ㅏ …) 또는 반각 자모(ﾡ …)."""
    cp = ord(ch)
    return 0x3131 <= cp <= 0x318E or 0xFFA1 <= cp <= 0xFFDC


def _build_compat_jamo() -> dict[str, str]:
    """조합형 자모 → 호환 자모 (NFKC 의 반대 방향). 예: U+110F(ᄏ) → U+314B(ㅋ)"""
    table: dict[str, str] = {}
    for cp in range(0x3131, 0x318F):
        d = unicodedata.normalize("NFKC", chr(cp))
        if len(d) == 1 and _is_conjoining_jamo(d):
            table.setdefault(d, chr(cp))
    return table


_TO_COMPAT_JAMO = _build_compat_jamo()
# 반각 자모 → 호환 자모. 예: U+FFA1(ﾡ) → U+3131(ㄱ)
_HALFWIDTH_TO_COMPAT = {
    chr(cp): _TO_COMPAT_JAMO[unicodedata.normalize("NFKC", chr(cp))]
    for cp in range(0xFFA1, 0xFFDD)
    if unicodedata.normalize("NFKC", chr(cp)) in _TO_COMPAT_JAMO
}


# 현대 한글 조합형 자모: 초성 U+1100~1112, 중성 U+1161~1175, 종성 U+11A8~11C2
_STRAY_JAMO = re.compile("[\u1100-\u1112\u1161-\u1175\u11a8-\u11c2]")


def _build_stray_to_compat() -> dict[str, str]:
    """홀로 남은 현대 조합형 자모 → 호환 자모. 종성은 이름으로 짝을 찾는다 (ᆨ → ㄱ)."""
    table = {c: v for c, v in _TO_COMPAT_JAMO.items() if _STRAY_JAMO.match(c)}
    for cp in range(0x11A8, 0x11C3):
        name = unicodedata.name(chr(cp)).replace("JONGSEONG", "LETTER")
        table.setdefault(chr(cp), unicodedata.lookup(name))
    return table


_STRAY_TO_COMPAT = _build_stray_to_compat()
_OLD_JAMO = re.compile(
    "[\u1113-\u115e\u1176-\u11a7\u11c3-\u11ff\ua960-\ua97c\ud7b0-\ud7c6\ud7cb-\ud7fb]"
)
_HANGUL_JOINS = frozenset(
    {
        ("L", "L"),
        ("L", "V"),
        ("L", "LV"),
        ("L", "LVT"),
        ("LV", "V"),
        ("LV", "T"),
        ("V", "V"),
        ("V", "T"),
        ("LVT", "T"),
        ("T", "T"),
    }
)


def _hangul_type(ch: str) -> str:
    cp = ord(ch)
    if 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C:
        return "L"
    if 0x1160 <= cp <= 0x11A7 or 0xD7B0 <= cp <= 0xD7C6:
        return "V"
    if 0x11A8 <= cp <= 0x11FF or 0xD7CB <= cp <= 0xD7FB:
        return "T"
    if 0xAC00 <= cp <= 0xD7A3:
        return "LVT" if (cp - 0xAC00) % 28 else "LV"
    return ""


def _old_hangul_mask(text: str) -> list[bool]:
    mask = [False] * len(text)
    if not _OLD_JAMO.search(text):
        return mask
    types = [_hangul_type(c) for c in text]
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and (types[j - 1], types[j]) in _HANGUL_JOINS:
            j += 1
        block = types[i:j]
        syllable = block[0] in ("L", "LV", "LVT") and any(t in ("V", "LV", "LVT") for t in block)
        if syllable and _OLD_JAMO.search(text, i, j):
            mask[i:j] = [True] * (j - i)
        i = j
    return mask


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _cannot_join_back(ch: str) -> bool:
    """앞 조각과 NFKC 에서 합쳐질 수 없는 글자인가. ASCII 와 완성형 한글 음절이 그렇다."""
    return ch < "\x80" or 0xAC00 <= ord(ch) <= 0xD7A3


def _nfkc_segments(text: str, base: int = 0) -> list[tuple[int, int]]:
    """NFKC 를 조각마다 따로 해도 전체 결과와 같아지도록 문자열을 나눈다.

    결합 문자(악센트 등)는 앞 글자와 같은 조각에 둔다. 이웃 조각끼리 NFKC 에서 합쳐지면
    한 조각으로 묶는다. ASCII 와 완성형 음절은 앞 조각과 합쳐질 일이 없어서 비교 없이
    바로 끊는다 (긴 한국어 문서도 빠르게 처리하기 위해). 돌려주는 위치에는 base 를 더한다.
    """
    starts = [i for i, ch in enumerate(text) if i == 0 or unicodedata.combining(ch) == 0]
    bounds = list(zip(starts, starts[1:] + [len(text)], strict=True))
    if not bounds:
        return []
    out: list[tuple[int, int]] = []
    cur_s, cur_e = bounds[0]
    for s, e in bounds[1:]:
        nxt = text[s:e]
        cur = text[cur_s:cur_e]
        if _cannot_join_back(nxt[0]) or _nfkc(cur + nxt) == _nfkc(cur) + _nfkc(nxt):
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = e
    out.append((cur_s, cur_e))
    if "".join(_nfkc(text[s:e]) for s, e in out) != _nfkc(text):
        out = [(0, len(text))]  # 드문 경우의 안전장치: 위치는 거칠어져도 결과는 정확하게
    return [(s + base, e + base) for s, e in out]


def _split_jamo_runs(text: str) -> list[tuple[int, int, bool]]:
    """문자열을 (시작, 끝, 자모 덩어리인가) 구간으로 나눈다."""
    out: list[tuple[int, int, bool]] = []
    i = 0
    while i < len(text):
        is_jamo = _is_jamo_letter(text[i])
        j = i + 1
        while j < len(text) and _is_jamo_letter(text[j]) == is_jamo:
            j += 1
        out.append((i, j, is_jamo))
        i = j
    return out


def apply_nfkc(ctx: Context) -> None:
    """NFKC 정규화. 전각 ｉｇｎ → ign, 수학 영숫자 → ign, ㈜ → (주), NFD 한글 → 완성형.

    호환 자모(ㄱ, ㅏ …)는 NFKC 에 넣지 않는다. NFKC 는 자음+모음만 합치고 받침은 못
    붙여서 ㄱㅗㅇ → 고ㅇ 처럼 반쪽만 풀고, 정상 채팅 ㅋㅠㅠ → 큐ㅠ 까지 바꿔 버린다.
    자모 조립은 다음 단계(assemble_jamo)가 확실할 때만 한다. 반각 자모(ﾡ)는 호환 자모로만
    바꿔 둔다.
    """
    text = ctx.tt.text
    if unicodedata.is_normalized("NFKC", text) and not _STRAY_JAMO.search(text):
        return

    edits = []
    for s, e, is_jamo in _split_jamo_runs(text):
        if is_jamo:
            for i in range(s, e):
                if text[i] in _HALFWIDTH_TO_COMPAT:
                    edits.append(Edit(i, i + 1, _HALFWIDTH_TO_COMPAT[text[i]]))
            continue
        segments = _nfkc_segments(text[s:e], base=s)
        outs = []
        pending = []
        for ss, se in segments:
            seg = text[ss:se]
            new = _nfkc(seg)
            if _CONJOINING.search(seg):
                pending.append(len(outs))
            elif _CONJOINING.search(new):
                # ㈀ → (ᄀ) 처럼 조합형 자모가 새로 생기면 호환 자모로 되돌린다 → (ㄱ)
                new = "".join(_TO_COMPAT_JAMO.get(c, c) for c in new)
            outs.append(new)
        if pending:
            old = _old_hangul_mask("".join(outs))
            offsets = list(accumulate(map(len, outs), initial=0))
            for idx in pending:
                # NFKC 뒤에도 남은 현대 조합형 자모는 (옛한글 음절에 든 것을 빼면) 음절을 못 이룬
                # '홀로 남은' 자모다. 호환 자모로 바꿔야 ᄆㅜᄉㅣ·ㅁᅮㅅᅵ 처럼 섞어 쓴 우회를
                # assemble_jamo 가 조립한다
                outs[idx] = "".join(
                    c if old[offsets[idx] + k] else _STRAY_TO_COMPAT.get(c, c)
                    for k, c in enumerate(outs[idx])
                )
        for (ss, se), new in zip(segments, outs, strict=True):
            seg = text[ss:se]
            if new != seg:
                # 여러 글자 묶음(결합 문자 등)은 길이가 같아도 글자끼리 짝이 맞지 않을 수 있다.
                # (Ⅻ̈́ → XIḮ, 결합 문자 순서 정렬) 이때는 묶음 전체를 가리키게 한다
                edits.append(Edit(ss, se, new, block=len(seg) > 1))
    ctx.record("nfkc_changed", ctx.tt.apply(edits))


# ── 5. 자모 조립 ──

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ("", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
_VOWEL_PAIRS = {
    ("ㅗ", "ㅏ"): "ㅘ",
    ("ㅗ", "ㅐ"): "ㅙ",
    ("ㅗ", "ㅣ"): "ㅚ",
    ("ㅜ", "ㅓ"): "ㅝ",
    ("ㅜ", "ㅔ"): "ㅞ",
    ("ㅜ", "ㅣ"): "ㅟ",
    ("ㅡ", "ㅣ"): "ㅢ",
}
_FINAL_PAIRS = {
    ("ㄱ", "ㅅ"): "ㄳ",
    ("ㄴ", "ㅈ"): "ㄵ",
    ("ㄴ", "ㅎ"): "ㄶ",
    ("ㄹ", "ㄱ"): "ㄺ",
    ("ㄹ", "ㅁ"): "ㄻ",
    ("ㄹ", "ㅂ"): "ㄼ",
    ("ㄹ", "ㅅ"): "ㄽ",
    ("ㄹ", "ㅌ"): "ㄾ",
    ("ㄹ", "ㅍ"): "ㄿ",
    ("ㄹ", "ㅎ"): "ㅀ",
    ("ㅂ", "ㅅ"): "ㅄ",
}
_HANGUL_WORD = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]+")
_JAMO_RUN = re.compile(r"[\u3131-\u318e]+")


def _assemble(run: str) -> list[tuple[int, int, str]]:
    """자모 덩어리를 한글 자판(두벌식) 규칙으로 조립한다.

    (덩어리 안 시작, 끝, 결과 글자) 목록을 돌려준다. 음절이 못 된 자모는 그대로 1글자.
    받침 후보 자음 뒤에 모음이 오면 받침이 아니라 다음 음절의 초성으로 넘긴다 (자판과 같음).
    """
    out: list[tuple[int, int, str]] = []
    n = len(run)

    def vowel_at(k: int) -> bool:
        return k < n and run[k] in _JUNG

    i = 0
    while i < n:
        start = i
        if run[i] in _CHO and vowel_at(i + 1):
            cho, jung = run[i], run[i + 1]
            i += 2
            if i < n and (jung, run[i]) in _VOWEL_PAIRS:
                jung = _VOWEL_PAIRS[(jung, run[i])]
                i += 1
            jong = ""
            if i < n and run[i] in _JONG and not vowel_at(i + 1):
                jong = run[i]
                i += 1
                if i < n and (jong, run[i]) in _FINAL_PAIRS and not vowel_at(i + 1):
                    jong = _FINAL_PAIRS[(jong, run[i])]
                    i += 1
            code = (_CHO.index(cho) * 21 + _JUNG.index(jung)) * 28 + _JONG.index(jong)
            out.append((start, i, chr(0xAC00 + code)))
        else:
            out.append((i, i + 1, run[i]))
            i += 1
    return out


def assemble_jamo(ctx: Context) -> None:
    """쪼개 쓴 자모를 음절로 조립한다. ㄱㅗㅇㄱㅕㄱ → 공격, 무ㅅㅣ해 → 무시해.

    정상 채팅(ㅋㅋㅋㅠㅠ, 좋아ㅇㅋ)을 바꾸지 않도록 확실할 때만 조립한다.
    - 원문에서 자모로 쓰인 덩어리 안에서만 조립한다. 완성형 음절(좋아)에는 받침을
      붙이지 않는다. 그래서 '좋아ㅇㅋ' 의 ㅇ 이 '앙' 이 되지 않는다.
    - 한글 단어(공백 등으로 나뉜 덩어리) 단위로, 조립한 결과에 자모가 하나도 남지 않고
      음절이 2개 이상일 때만 채택한다. 아니면 그 단어는 원문 그대로 둔다.
    알려진 한계: 한 음절 단어(ㄷㅏㄹㄱ), 띄어 쓴 자모(ㅁ ㅜ ㅅ ㅣ)는 풀지 않는다.
    """
    text = ctx.tt.text
    edits: list[Edit] = []
    for word in _HANGUL_WORD.finditer(text):
        runs = list(_JAMO_RUN.finditer(word.group()))
        if not runs:
            continue
        word_edits: list[Edit] = []
        leftover = 0
        new_syllables = 0
        for run in runs:
            base = word.start() + run.start()
            for s, e, out in _assemble(run.group()):
                if e - s == 1:
                    leftover += 1
                else:
                    new_syllables += 1
                    word_edits.append(Edit(base + s, base + e, out))
        old_syllables = sum(1 for c in word.group() if 0xAC00 <= ord(c) <= 0xD7A3)
        if leftover == 0 and old_syllables + new_syllables >= 2:
            edits.extend(word_edits)
    ctx.record("jamo_assembled", ctx.tt.apply(edits))


# ── 5. 공백 정리 ──


def tidy_whitespace(ctx: Context) -> None:
    """줄바꿈을 \\n 으로 통일하고, 연속 공백과 3줄 이상 빈 줄을 줄인다.

    소문자 변환은 하지 않는다 (모델이 대소문자를 구분해서 읽는다).
    """
    tt = ctx.tt
    tt.sub(r"\r\n?|[\u2028\u2029\u0085]", "\n")
    tt.sub(r"[^\S\n]+", " ")
    tt.sub(r"\n(?: ?\n){2,}", "\n\n")
