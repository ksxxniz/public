import re
from bisect import bisect_left, bisect_right
from collections import defaultdict

from preprocess.context import Context
from preprocess.html_parse import ATTRIBUTE_REASONS, Piece, html5_parser_available, parse_html
from preprocess.tracked import Edit

_TAG_CHARS = re.compile("[\U000e0000-\U000e007f]+")
_SEPARATE = frozenset(
    {*ATTRIBUTE_REASONS.values(), "meta_content", "input_hidden", "comment", "cdata"}
)

_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")
_HTML_LIKE = re.compile(r"<[!/?A-Za-z]")
_NL = r"\n(?![ \t]*\n)"
_SEP = r"(?:[ \t]*\n[ \t]*|[ \t]+)"
_TITLE = (
    r"(?:\"([^\"\n]*(?:" + _NL + r"[^\"\n]*){0,4})\""
    r"|'([^'\n]*(?:" + _NL + r"[^'\n]*){0,4})'"
    r"|\(([^()\n]*(?:" + _NL + r"[^()\n]*){0,4})\))"
)
_DEST = r"(?:<[^<>\n]*>|(?:[^\s()]|\([^\s()]*\))*)"
_LABEL = r"\[([^\[\]\n]*)\]"
_MD_COMMENT = re.compile(
    r"^ {0,3}" + _LABEL + r":[ \t]*(?:#|<>)(?:" + _SEP + _TITLE + r")?[ \t]*$", re.M
)
_IMAGE = re.compile(r"!" + _LABEL + r"\([ \t]*" + _DEST + r"(?:" + _SEP + _TITLE + r")?[ \t]*\)")
_REF_IMAGE = re.compile(r"!" + _LABEL + r"\[[^\[\]\n]*\]")
_LINK = re.compile(r"(?<!!)" + _LABEL + r"\([ \t]*" + _DEST + _SEP + _TITLE + r"[ \t]*\)")
_REF_DEF = re.compile(
    r"^ {0,3}" + _LABEL + r":[ \t]*(?:<[^<>\n]*>|\S+)" + _SEP + _TITLE + r"[ \t]*$", re.M
)


def parse_markup(ctx: Context) -> None:
    if ctx.fmt == "html":
        _extract_html(ctx)
    elif ctx.fmt == "md":
        _mark_markdown(ctx)


def _extract_html(ctx: Context) -> None:
    tt = ctx.tt
    text = tt.text
    parsed = parse_html(text)
    kept = _non_overlapping(parsed.pieces + parsed.attributes)
    breaks = sorted(set(parsed.breaks))
    for s, e, reason in _hidden_regions(kept):
        ctx.record("hidden_text_found", [tt.raw_span(s, e)], note=reason)
    edits: list[Edit] = []
    prev: Piece | None = None
    for cur in [*kept, None]:
        a = prev.end if prev else 0
        b = cur.start if cur else len(text)
        if a < b:
            edits.extend(_gap_edits(text, a, b, _separator(prev, cur, breaks, a, b)))
        if cur is not None and text[cur.start : cur.end] != cur.text:
            edits.append(Edit(cur.start, cur.end, cur.text, block=True))
        prev = cur
    tt.apply(edits)


def _non_overlapping(pieces: list[Piece]) -> list[Piece]:
    out: list[Piece] = []
    pos = 0
    for p in sorted(pieces, key=lambda p: (p.start, p.end)):
        if p.start >= pos and p.end > p.start:
            out.append(p)
            pos = p.end
    return out


def _separator(prev: Piece | None, cur: Piece | None, breaks: list[int], a: int, b: int) -> str:
    if prev is None or cur is None:
        return ""
    k = bisect_left(breaks, a)
    if k < len(breaks) and breaks[k] < b:
        return "\n"
    if prev.hidden != cur.hidden or prev.hidden in _SEPARATE or cur.hidden in _SEPARATE:
        return " "
    return ""


def _gap_edits(text: str, a: int, b: int, sep: str) -> list[Edit]:
    edits: list[Edit] = []
    pos = a
    for m in _TAG_CHARS.finditer(text, a, b):
        if m.start() > pos:
            edits.append(Edit(pos, m.start(), sep))
            sep = ""
        pos = m.end()
    if pos < b:
        edits.append(Edit(pos, b, sep))
    return edits


def _hidden_regions(kept: list[Piece]) -> list[tuple[int, int, str]]:
    regions: list[list] = []
    current: list | None = None
    for p in kept:
        if not p.text.strip():
            continue
        if p.hidden is None:
            current = None
            continue
        if (
            current is not None
            and current[2] == p.hidden
            and (p.start == current[1] or p.hidden not in _SEPARATE)
        ):
            current[1] = p.end
        else:
            current = [p.start, p.end, p.hidden]
            regions.append(current)
    return [(s, e, reason) for s, e, reason in regions]


def _mark_markdown(ctx: Context) -> None:
    text = ctx.tt.text
    scan = _blank_code(text)
    found: list[tuple[int, int, str]] = []
    for m in _MD_COMMENT.finditer(scan):
        if any(m.group(g) is not None for g in (2, 3, 4)):
            found.append(_group_span(m, (2, 3, 4), "md_comment"))
        elif re.search(r"\w", m.group(1)):
            found.append(_group_span(m, (1,), "md_comment"))
    for m in _IMAGE.finditer(scan):
        found.append(_group_span(m, (1,), "attr_alt"))
        found.append(_group_span(m, (2, 3, 4), "attr_title"))
    for m in _REF_IMAGE.finditer(scan):
        found.append(_group_span(m, (1,), "attr_alt"))
    for rx in (_LINK, _REF_DEF):
        for m in rx.finditer(scan):
            found.append(_group_span(m, (2, 3, 4), "attr_title"))
    if _HTML_LIKE.search(scan) and html5_parser_available():
        parsed = parse_html(scan)
        for p in parsed.pieces + parsed.attributes:
            if p.hidden:
                found.append((p.start, p.end, p.hidden))
    for s, e, reason in _merge(found, text):
        ctx.record("hidden_text_found", [ctx.tt.raw_span(s, e)], note=reason)


def _group_span(m: re.Match[str], groups: tuple[int, ...], reason: str) -> tuple[int, int, str]:
    for g in groups:
        if m.group(g) is not None:
            return m.start(g), m.end(g), reason
    return 0, 0, reason


def _merge(found: list[tuple[int, int, str]], text: str) -> list[tuple[int, int, str]]:
    out: list[list] = []
    for s, e, reason in sorted(found, key=lambda f: (f[0], f[1])):
        if e <= s or not text[s:e].strip():
            continue
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, reason])
    return [(s, e, reason) for s, e, reason in out]


def _blank_code(text: str) -> str:
    chars = list(text)
    pos = 0
    fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        end = pos + len(line)
        m = _FENCE.match(line)
        if fence is None and m and not (m.group(1)[0] == "`" and "`" in line[m.end() :]):
            fence = (m.group(1)[0], len(m.group(1)))
            _blank(chars, pos, end)
        elif fence is not None:
            _blank(chars, pos, end)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]:
                if not line[m.end() :].strip():
                    fence = None
        else:
            for s, e in _code_spans(line):
                _blank(chars, pos + s, pos + e)
        pos = end
    return "".join(chars)


def _code_spans(line: str) -> list[tuple[int, int]]:
    runs = [m.span() for m in _BACKTICKS.finditer(line)]
    by_length: dict[int, list[int]] = defaultdict(list)
    for k, (s, e) in enumerate(runs):
        by_length[e - s].append(k)
    spans = []
    k = 0
    while k < len(runs):
        s, e = runs[k]
        if _escaped(line, s):
            k += 1
            continue
        same = by_length[e - s]
        i = bisect_right(same, k)
        if i < len(same):
            spans.append((s, runs[same[i]][1]))
            k = same[i] + 1
        else:
            k += 1
    return spans


def _escaped(line: str, pos: int) -> bool:
    k = pos
    while k > 0 and line[k - 1] == "\\":
        k -= 1
    return (pos - k) % 2 == 1


def _blank(chars: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = " "
