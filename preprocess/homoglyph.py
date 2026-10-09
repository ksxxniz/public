import re
import unicodedata

from confusable_homoglyphs import categories, confusables

from preprocess.context import Context
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

_CANDIDATE = re.compile("[" + "".join(re.escape(c) for c in TO_LATIN) + "]")
_MARKS = "".join(chr(cp) for cp in range(0x300, 0x20000) if unicodedata.category(chr(cp))[0] == "M")
_WORD = re.compile(r"(?:[^\W\d_]|[" + _MARKS + "])+")


def _is_latin(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith("LATIN")


def replace_homoglyphs(ctx: Context) -> None:
    text = ctx.tt.text
    if not _CANDIDATE.search(text):
        return
    edits: list[Edit] = []
    for m in _WORD.finditer(text):
        word = m.group()
        if not _CANDIDATE.search(word):
            continue
        mixed = any(_is_latin(c) for c in word)
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
