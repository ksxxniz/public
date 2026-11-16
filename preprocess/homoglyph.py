import re
import unicodedata

from confusable_homoglyphs import categories, confusables

from preprocess.context import Context
from preprocess.encoding import _COMMON
from preprocess.tracked import Edit

_OVERRIDES = {"\u0399": "I", "\u0406": "I", "\u04c0": "I", "\u04cf": "l"}

SUPPLEMENT: dict[str, str] = {
    "\u1d00": "a",
    "\u0299": "b",
    "\u1d04": "c",
    "\u1d05": "d",
    "\u1d07": "e",
    "\ua730": "f",
    "\u0262": "g",
    "\u029c": "h",
    "\u026a": "i",
    "\u1d0a": "j",
    "\u1d0b": "k",
    "\u029f": "l",
    "\u1d0d": "m",
    "\u0274": "n",
    "\u1d0f": "o",
    "\u1d18": "p",
    "\ua7af": "q",
    "\u0280": "r",
    "\ua731": "s",
    "\u1d1b": "t",
    "\u1d1c": "u",
    "\u1d20": "v",
    "\u1d21": "w",
    "\u028f": "y",
    "\u1d22": "z",
    "\u0251": "a",
    "\u0261": "g",
    "\u0269": "i",
    "\u0131": "i",
    "\u0237": "j",
}


def _build_confusable_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for ch, homoglyphs in confusables.confusables_data.items():
        if len(ch) != 1 or categories.alias(ch) not in ("CYRILLIC", "GREEK"):
            continue
        targets = {
            h["c"] for h in homoglyphs if len(h["c"]) == 1 and h["c"].isascii() and h["c"].isalpha()
        }
        if len(targets) == 1:
            table[ch] = targets.pop()
    table.update(_OVERRIDES)
    return table


def _add_nfkc_forms(table: dict[str, str]) -> dict[str, str]:
    out = dict(table)
    for ch, latin in table.items():
        folded = unicodedata.normalize("NFKC", ch)
        if len(folded) == 1 and folded != ch:
            out.setdefault(folded, latin)
    return out


VISUAL_CONFUSABLE: dict[str, str] = _build_confusable_table()
CONFUSABLE: dict[str, str] = _add_nfkc_forms(VISUAL_CONFUSABLE)
TO_LATIN: dict[str, str] = {**CONFUSABLE, **SUPPLEMENT}

KEYWORDS = frozenset(
    """
    ignore ignored ignoring disregard forget previous prior above earlier instruction
    instructions system prompt prompts rule rules override admin administrator developer
    password passwords secret secrets reveal print output execute command commands assistant
    jailbreak unrestricted unfiltered confidential bypass pretend verbatim token tokens
    credentials
    """.split()
)

_CANDIDATE = re.compile("[" + "".join(re.escape(c) for c in TO_LATIN) + "]")
_MARKS = "".join(chr(cp) for cp in range(0x300, 0x20000) if unicodedata.category(chr(cp))[0] == "M")
_WORD = re.compile(r"(?:[^\W\d_]|[" + _MARKS + "])+")


# ── 둘러싼 글자 ──
# NFKC 가 풀지 않거나(🅸 🅘 🇮) 괄호를 남기는(⒤ → "(i)") 글자. 글자 하나로 바꾼다.
_ENCLOSED_RANGES = (
    (0x1F150, 0x1F169, "A"),  # 🅐 음각 원
    (0x1F170, 0x1F189, "A"),  # 🅰 음각 사각
    (0x1F110, 0x1F129, "A"),  # 🄐 괄호 대문자
    (0x249C, 0x24B5, "a"),  # ⒜ 괄호 소문자
)
_REGIONAL_FIRST, _REGIONAL_LAST = 0x1F1E6, 0x1F1FF  # 🇦 지역 표시 문자 (둘이 모이면 국기)
# 국기로 그려지는 두 글자 코드. 이 짝으로만 이뤄진 나열은 국기라서 건드리지 않는다
_FLAG_CODES = frozenset(
    """
    AC AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO
    BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CP CR CU CV CW CX CY CZ DE DG
    DJ DK DM DO DZ EA EC EE EG EH ER ES ET EU FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM
    GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU IC ID IE IL IM IN IO IQ IR IS IT JE JM JO JP
    KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK
    ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE
    PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM
    SN SO SR SS ST SV SX SY SZ TA TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM UN
    US UY UZ VA VC VE VG VI VN VU WF WS XK YE YT ZA ZM ZW
    """.split()
)

_ENCLOSED_RUN = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b, _ in _ENCLOSED_RANGES) + "]{2,}"
    f"|[{chr(_REGIONAL_FIRST)}-{chr(_REGIONAL_LAST)}]{{2,}}"
)


def enclosed_letter(ch: str) -> str | None:
    """둘러싼 글자·지역 표시 문자가 나타내는 라틴 문자. 아니면 None."""
    cp = ord(ch)
    if cp < 0x249C:
        return None
    if _REGIONAL_FIRST <= cp <= _REGIONAL_LAST:
        return chr(ord("A") + cp - _REGIONAL_FIRST)
    for first, last, base in _ENCLOSED_RANGES:
        if first <= cp <= last:
            return chr(ord(base) + cp - first)
    return None


def _only_flags(letters: str) -> bool:
    if len(letters) % 2:
        return False
    return all(letters[k : k + 2] in _FLAG_CODES for k in range(0, len(letters), 2))


def fold_enclosed_letters(ctx: Context) -> None:
    """둘러싼 글자로 쓴 낱말을 보통 글자로 바꾼다. 🅸🅶🅽🅾🆁🅴 → IGNORE, ⒤⒢⒩ → ign.

    정상 글을 바꾸지 않도록 두 글자 이상 이어질 때만 바꾼다 (🅿 주차, ⒜ 항목 번호는 그대로).
    지역 표시 문자는 국기 짝으로만 이뤄져 있으면 (🇰🇷🇺🇸) 그대로 둔다.
    """
    text = ctx.tt.text
    edits: list[Edit] = []
    for m in _ENCLOSED_RUN.finditer(text):
        letters = "".join(enclosed_letter(c) or "" for c in m.group())
        is_regional = _REGIONAL_FIRST <= ord(m.group()[0]) <= _REGIONAL_LAST
        if is_regional and _only_flags(letters):
            continue
        edits.extend(Edit(m.start() + k, m.start() + k + 1, c) for k, c in enumerate(letters))
    if edits:
        ctx.record("homoglyph_replaced", ctx.tt.apply(edits))


def _is_latin(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith("LATIN")


def _disguised_keyword(word: str) -> bool:
    if not all(c in TO_LATIN for c in word):
        return False
    return "".join(TO_LATIN[c] for c in word).lower() in KEYWORDS


def _math_variable(word: str) -> bool:
    """그리스 글자 하나가 낱말 맨 앞이나 맨 뒤에 붙은 수식 표기인가 (αx, νmax, ρgh, xσ).

    이런 낱말은 라틴 문자로 바꾸면 정상 이공계 문서가 망가진다 (νx → vx). 바꾼 결과가
    흔한 영어 낱말이나 공격 키워드일 때만 위장으로 본다 (αll → all, ρassword → password).
    """
    greek = [k for k, c in enumerate(word) if unicodedata.name(c, "").startswith("GREEK")]
    if len(greek) != 1 or greek[0] not in (0, len(word) - 1):
        return False
    if not all(_is_latin(c) or k == greek[0] for k, c in enumerate(word)):
        return False
    latin = "".join(TO_LATIN.get(c, c) for c in word).lower()
    return latin not in KEYWORDS and latin not in _COMMON


def replace_homoglyphs(ctx: Context) -> None:
    text = ctx.tt.text
    if not _CANDIDATE.search(text):
        return
    edits: list[Edit] = []
    for m in _WORD.finditer(text):
        word = m.group()
        if not _CANDIDATE.search(word):
            continue
        mixed = any(_is_latin(c) for c in word) or _disguised_keyword(word)
        if mixed and _math_variable(word):
            mixed = False
        for k, ch in enumerate(word):
            if ch in SUPPLEMENT or (mixed and ch in CONFUSABLE):
                pos = m.start() + k
                end = pos + 1
                while end < len(text) and unicodedata.combining(text[end]):
                    end += 1
                new = unicodedata.normalize("NFKC", TO_LATIN[ch] + text[pos + 1 : end])
                if len(new) == end - pos and new[1:] == text[pos + 1 : end]:
                    edits.append(Edit(pos, pos + 1, TO_LATIN[ch]))
                else:
                    edits.append(Edit(pos, end, new, block=True))
    if edits:
        ctx.record("homoglyph_replaced", ctx.tt.apply(edits))
