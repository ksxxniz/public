import colorsys
import html
import inspect
import math
import platform
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser

BLOCK_TAGS = frozenset(
    "address article aside blockquote br dd div dl dt figcaption figure footer form h1 h2 h3 h4 "
    "h5 h6 header hr li main nav ol p pre section table tbody td tfoot th thead tr ul".split()
)
VOID_TAGS = frozenset(
    "area base basefont bgsound br col embed frame hr image img input keygen link meta param "
    "source track wbr".split()
)
HIDDEN_TAGS = {
    tag: tag
    for tag in "template noscript script style iframe noembed noframes datalist video audio "
    "canvas".split()
}

_SCOPE = frozenset("applet caption html marquee object table td template th".split())
_BUTTON_SCOPE = _SCOPE | {"button"}
_LIST_SCOPE = _SCOPE | {"ol", "ul"}
_TABLE_SCOPE = frozenset({"html", "table", "template"})
_HEADINGS = frozenset("h1 h2 h3 h4 h5 h6".split())
_FORMATTING = frozenset("a b big code em font i nobr s small strike strong tt u".split())
_SPECIAL = frozenset(
    """address applet area article aside base basefont bgsound blockquote body br button caption
    center col colgroup dd details dialog dir div dl dt embed fieldset figcaption figure footer
    form frame frameset h1 h2 h3 h4 h5 h6 head header hgroup hr html iframe img input keygen li
    link listing main marquee menu meta nav noembed noframes noscript object ol p param plaintext
    pre script search section select source style summary table tbody td template textarea tfoot
    th thead title tr track ul wbr xmp""".split()
)
_CLOSES_P = frozenset(
    """address article aside blockquote center details dialog dir div dl fieldset figcaption
    figure footer form h1 h2 h3 h4 h5 h6 header hgroup hr listing main menu nav ol p plaintext pre
    search section summary table ul xmp""".split()
)
_TABLE_PARTS = frozenset("caption col colgroup tbody td tfoot th thead tr".split())
_TABLE_CONTEXT = frozenset("table tbody tfoot thead tr".split())
_TABLE_CONTENT = _TABLE_PARTS | frozenset("form input script style template".split())
_SVG_TEXT = frozenset("text tspan textpath".split())

SAME_COLOR_CONTRAST = 1.2
MIN_OPACITY = 0.05
TINY_FONT_PX = 1.0
TINY_BOX_PX = 1.0
OFFSCREEN_PX = -999.0
MAX_ACTIVE_FORMATTING = 64
_OFFSCREEN_PROPS = ("left", "top", "text-indent", "margin-left", "margin-top")

Color = tuple[float, float, float, float]
WHITE: Color = (255.0, 255.0, 255.0, 1.0)
BLACK: Color = (0.0, 0.0, 0.0, 1.0)

_NAMED_HEX = """
aliceblue f0f8ff antiquewhite faebd7 aqua 00ffff aquamarine 7fffd4 azure f0ffff beige f5f5dc
bisque ffe4c4 black 000000 blanchedalmond ffebcd blue 0000ff blueviolet 8a2be2 brown a52a2a
burlywood deb887 cadetblue 5f9ea0 chartreuse 7fff00 chocolate d2691e coral ff7f50
cornflowerblue 6495ed cornsilk fff8dc crimson dc143c cyan 00ffff darkblue 00008b darkcyan 008b8b
darkgoldenrod b8860b darkgray a9a9a9 darkgreen 006400 darkgrey a9a9a9 darkkhaki bdb76b
darkmagenta 8b008b darkolivegreen 556b2f darkorange ff8c00 darkorchid 9932cc darkred 8b0000
darksalmon e9967a darkseagreen 8fbc8f darkslateblue 483d8b darkslategray 2f4f4f
darkslategrey 2f4f4f darkturquoise 00ced1 darkviolet 9400d3 deeppink ff1493 deepskyblue 00bfff
dimgray 696969 dimgrey 696969 dodgerblue 1e90ff firebrick b22222 floralwhite fffaf0
forestgreen 228b22 fuchsia ff00ff gainsboro dcdcdc ghostwhite f8f8ff gold ffd700
goldenrod daa520 gray 808080 green 008000 greenyellow adff2f grey 808080 honeydew f0fff0
hotpink ff69b4 indianred cd5c5c indigo 4b0082 ivory fffff0 khaki f0e68c lavender e6e6fa
lavenderblush fff0f5 lawngreen 7cfc00 lemonchiffon fffacd lightblue add8e6 lightcoral f08080
lightcyan e0ffff lightgoldenrodyellow fafad2 lightgray d3d3d3 lightgreen 90ee90 lightgrey d3d3d3
lightpink ffb6c1 lightsalmon ffa07a lightseagreen 20b2aa lightskyblue 87cefa lightslategray 778899
lightslategrey 778899 lightsteelblue b0c4de lightyellow ffffe0 lime 00ff00 limegreen 32cd32
linen faf0e6 magenta ff00ff maroon 800000 mediumaquamarine 66cdaa mediumblue 0000cd
mediumorchid ba55d3 mediumpurple 9370db mediumseagreen 3cb371 mediumslateblue 7b68ee
mediumspringgreen 00fa9a mediumturquoise 48d1cc mediumvioletred c71585 midnightblue 191970
mintcream f5fffa mistyrose ffe4e1 moccasin ffe4b5 navajowhite ffdead navy 000080 oldlace fdf5e6
olive 808000 olivedrab 6b8e23 orange ffa500 orangered ff4500 orchid da70d6 palegoldenrod eee8aa
palegreen 98fb98 paleturquoise afeeee palevioletred db7093 papayawhip ffefd5 peachpuff ffdab9
peru cd853f pink ffc0cb plum dda0dd powderblue b0e0e6 purple 800080 rebeccapurple 663399
red ff0000 rosybrown bc8f8f royalblue 4169e1 saddlebrown 8b4513 salmon fa8072 sandybrown f4a460
seagreen 2e8b57 seashell fff5ee sienna a0522d silver c0c0c0 skyblue 87ceeb slateblue 6a5acd
slategray 708090 slategrey 708090 snow fffafa springgreen 00ff7f steelblue 4682b4 tan d2b48c
teal 008080 thistle d8bfd8 tomato ff6347 turquoise 40e0d0 violet ee82ee wheat f5deb3
white ffffff whitesmoke f5f5f5 yellow ffff00 yellowgreen 9acd32
""".split()
NAMED_COLORS: dict[str, Color] = {
    name: (float(int(h[0:2], 16)), float(int(h[2:4], 16)), float(int(h[4:6], 16)), 1.0)
    for name, h in zip(_NAMED_HEX[::2], _NAMED_HEX[1::2], strict=True)
}

_ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z][-.a-zA-Z0-9]*);?")
_NUMBER = re.compile(r"^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([a-z%]*)$")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_IMPORTANT = re.compile(r"!\s*important\s*$")
_URL = re.compile(r"url\([^)]*\)")
_COLOR_TOKEN = re.compile(r"(?:rgba?|hsla?)\([^)]*\)|#[0-9a-f]{3,8}\b|[a-z]+")
_SCALE = re.compile(r"\bscale(?:3d|x|y|z)?\(([^)]*)\)")
_ABSOLUTE_PX = {
    "px": 1.0,
    "pt": 4 / 3,
    "pc": 16.0,
    "in": 96.0,
    "cm": 96 / 2.54,
    "mm": 96 / 25.4,
    "q": 96 / 101.6,
}
_FONT_KEYWORDS = {
    "xx-small": 9.0,
    "x-small": 10.0,
    "small": 13.0,
    "medium": 16.0,
    "large": 18.0,
    "x-large": 24.0,
    "xx-large": 32.0,
    "xxx-large": 48.0,
}


@dataclass(frozen=True)
class Piece:
    start: int
    end: int
    text: str
    hidden: str | None = None


@dataclass
class ParsedHtml:
    raw: str
    pieces: list[Piece] = field(default_factory=list)
    breaks: list[int] = field(default_factory=list)

    @property
    def hidden_pieces(self) -> list[Piece]:
        return [p for p in self.pieces if p.hidden]

    def visible_text(self) -> str:
        return self._join([p for p in self.pieces if not p.hidden])

    def all_text(self) -> str:
        return self._join(self.pieces)

    def hidden_text(self) -> str:
        return self._join(self.hidden_pieces)

    def _join(self, pieces: list[Piece]) -> str:
        breaks = sorted(self.breaks)
        out: list[str] = []
        prev_end: int | None = None
        for p in sorted(pieces, key=lambda p: p.start):
            if prev_end is not None:
                k = bisect_left(breaks, prev_end)
                if k < len(breaks) and breaks[k] <= p.start:
                    out.append("\n")
            out.append(p.text)
            prev_end = p.end
        return "".join(out)


def parse_html(raw: str) -> ParsedHtml:
    parser = _Parser(raw)
    parser.feed(raw)
    parser.close()
    return parser.result


def html5_parser_available() -> bool:
    params = inspect.signature(HTMLParser.__init__).parameters
    return "scripting" in params and "iframe" in HTMLParser.CDATA_CONTENT_ELEMENTS


def _require_html5_parser() -> None:
    if not html5_parser_available():
        raise RuntimeError(
            f"HTML 파서에는 Python 3.11.15 이상이 필요합니다 (지금 {platform.python_version()}). "
            "그 전 버전의 html.parser 는 같은 문서를 다르게 읽어 결과가 달라집니다. "
            "uv python install 3.11.15 로 설치하고 .venv 폴더를 지운 뒤 uv sync 를 하세요."
        )


@dataclass(frozen=True)
class _Style:
    hidden: str | None = None
    invisible: bool = False
    opacity: float = 1.0
    font_px: float = 16.0
    color: Color = BLACK
    background: Color = WHITE
    svg: bool = False
    svg_text: bool = False


ROOT = _Style()


def _element_style(tag: str, attrs, parent: _Style) -> _Style:
    return _element_style_cached(tag, tuple(attrs), parent)


@lru_cache(maxsize=8192)
def _element_style_cached(tag: str, attrs: tuple, parent: _Style) -> _Style:
    attr = _first_attrs(attrs)
    style = _parse_style(attr.get("style") or "")
    font_px = _font_size(style.get("font-size"), parent.font_px)
    color = _color_value(style.get("color"), parent.color)
    svg = (parent.svg or tag == "svg") and tag != "foreignobject"
    return _Style(
        hidden=parent.hidden or _hides_subtree(tag, attr, style, font_px),
        invisible=_visibility(style.get("visibility"), parent.invisible),
        opacity=parent.opacity * _opacity(style.get("opacity")),
        font_px=font_px,
        color=color,
        background=_background(style, parent.background, color),
        svg=svg,
        svg_text=svg and (parent.svg_text or tag in _SVG_TEXT),
    )


@lru_cache(maxsize=8192)
def _text_reason(s: _Style) -> str | None:
    if s.hidden:
        return s.hidden
    if s.svg and not s.svg_text:
        return "svg_non_text"
    if s.invisible:
        return "visibility_hidden"
    if s.opacity <= MIN_OPACITY:
        return "opacity_zero"
    if s.font_px <= TINY_FONT_PX:
        return "font_size_zero"
    if s.color[3] == 0:
        return "transparent_color"
    if _contrast(s.color, s.background) < SAME_COLOR_CONTRAST:
        return "same_color"
    return None


@lru_cache(maxsize=4096)
def _hides_alone(tag: str, attrs: tuple) -> bool:
    return _text_reason(_element_style(tag, attrs, ROOT)) is not None


def _hides_subtree(tag: str, attr: dict[str, str], style: dict[str, str], font_px: float):
    if tag in HIDDEN_TAGS:
        return HIDDEN_TAGS[tag]
    if tag == "dialog" and "open" not in attr:
        return "dialog_closed"
    if "hidden" in attr:
        return "hidden_attr"
    if style.get("display") == "none":
        return "display_none"
    if style.get("content-visibility") == "hidden":
        return "content_visibility_hidden"
    if _scaled_to_zero(style.get("transform")):
        return "scale_zero"
    if any(_length(style.get(p), font_px) <= OFFSCREEN_PX for p in _OFFSCREEN_PROPS):
        return "offscreen"
    if _clipped(style, font_px):
        return "zero_size_box"
    return None


_MARKER_TAGS = frozenset("applet caption marquee object td template th".split())
_RECONSTRUCT_SPECIAL = frozenset(
    "applet area br button embed img input keygen marquee object select wbr xmp".split()
)
_RAW_TEXT = frozenset(
    "iframe noembed noframes noscript plaintext script style textarea title xmp".split()
)


@dataclass(eq=False)
class _Node:
    tag: str
    style: _Style
    attrs: tuple = ()
    pos: int = -1
    open: bool = True


class _Parser(HTMLParser):
    def __init__(self, raw: str):
        _require_html5_parser()
        super().__init__(convert_charrefs=False, scripting=True)
        self.raw = raw
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", raw)]
        self.result = ParsedHtml(raw)
        self._stack: list[_Node] = []
        self._formatting: list[_Node | None] = []
        self._at: dict[str, list[int]] = defaultdict(list)
        self._specials: list[int] = []
        self._listed: Counter[str] = Counter()

    def _pos(self) -> int:
        line, col = self.getpos()
        return self._line_starts[line - 1] + col

    def _top(self) -> str | None:
        return self._stack[-1].tag if self._stack else None

    def _push(self, node: _Node) -> None:
        node.pos = len(self._stack)
        self._stack.append(node)
        self._at[node.tag].append(node.pos)
        if node.tag in _SPECIAL:
            self._specials.append(node.pos)

    def _pop_to(self, k: int) -> None:
        for node in reversed(self._stack[k:]):
            node.open = False
            self._at[node.tag].pop()
            if node.tag in _SPECIAL:
                self._specials.pop()
            if node.tag in _MARKER_TAGS:
                self._clear_to_marker()
        del self._stack[k:]

    def _nearest(self, names) -> int:
        best = -1
        for name in names:
            at = self._at.get(name)
            if at and at[-1] > best:
                best = at[-1]
        return best

    def _find(self, names, boundary) -> int | None:
        k = self._nearest(names)
        if k < 0 or self._nearest(b for b in boundary if b not in names) > k:
            return None
        return k

    def _close(self, names, boundary) -> None:
        k = self._find(names, boundary)
        if k is not None:
            self._pop_to(k)

    def _index(self, node: _Node) -> int | None:
        return node.pos if node.open else None

    def _clear_to_marker(self) -> None:
        while self._formatting:
            entry = self._formatting.pop()
            if entry is None:
                return
            self._listed[entry.tag] -= 1

    def _unlist(self, i: int) -> _Node:
        entry = self._formatting.pop(i)
        self._listed[entry.tag] -= 1
        return entry

    def _remember(self, node: _Node) -> None:
        f = self._formatting
        first = len(f)
        while first > 0 and f[first - 1] is not None:
            first -= 1
        same = [i for i in range(first, len(f)) if (f[i].tag, f[i].attrs) == (node.tag, node.attrs)]
        if len(same) >= 3:
            self._unlist(same[0])
        elif len(f) - first >= MAX_ACTIVE_FORMATTING:
            plain = [i for i in range(first, len(f)) if not _hides_alone(f[i].tag, f[i].attrs)]
            self._unlist(plain[0] if plain else first)
        f.append(node)
        self._listed[node.tag] += 1

    def _active(self, tag: str) -> int | None:
        if not self._listed[tag]:
            return None
        for i in range(len(self._formatting) - 1, -1, -1):
            entry = self._formatting[i]
            if entry is None:
                return None
            if entry.tag == tag:
                return i
        return None

    def _reconstruct(self) -> None:
        f = self._formatting
        if not f or f[-1] is None or f[-1].open:
            return
        i = len(f) - 1
        while i > 0 and f[i - 1] is not None and not f[i - 1].open:
            i -= 1
        for j in range(i, len(f)):
            old = f[j]
            parent = self._current_style(foster=True)
            node = _Node(old.tag, _element_style(old.tag, old.attrs, parent), old.attrs)
            self._push(node)
            f[j] = node

    def _current_style(self, foster: bool) -> _Style:
        if foster and self._top() in _TABLE_CONTEXT and self._at["table"]:
            k = self._at["table"][-1]
            return self._stack[k - 1].style if k > 0 else ROOT
        return self._stack[-1].style if self._stack else ROOT

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs, self_closing=False)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, self_closing=True)

    def _start(self, tag, attrs, self_closing: bool) -> None:
        if tag in _TABLE_PARTS and self._find({"table"}, _TABLE_SCOPE) is None:
            return
        self._close_implied(tag)
        if tag in _FORMATTING or tag in _RECONSTRUCT_SPECIAL or tag not in _SPECIAL:
            self._reconstruct()
        if tag in BLOCK_TAGS:
            self.result.breaks.append(self._pos())
        if tag in VOID_TAGS:
            return
        parent = self._current_style(foster=tag not in _TABLE_CONTENT)
        if self_closing and (parent.svg or tag == "svg"):
            return
        attrs = tuple(attrs)
        node = _Node(tag, _element_style(tag, attrs, parent), attrs)
        self._push(node)
        if tag in _FORMATTING:
            self._remember(node)
        elif tag in _MARKER_TAGS:
            self._formatting.append(None)

    def _close_implied(self, tag: str) -> None:
        if tag in _CLOSES_P:
            self._close({"p"}, _BUTTON_SCOPE)
        if tag in _HEADINGS and self._top() in _HEADINGS:
            self._pop_to(len(self._stack) - 1)
        if tag == "li":
            self._close_list_item({"li"})
        elif tag in ("dd", "dt"):
            self._close_list_item({"dd", "dt"})
        elif tag in ("option", "optgroup") and self._top() == "option":
            self._pop_to(len(self._stack) - 1)
        elif tag == "a":
            i = self._active("a")
            if i is not None:
                k = self._index(self._unlist(i))
                if k is not None:
                    self._pop_to(k)
        elif tag in ("td", "th"):
            self._close({"td", "th"}, _TABLE_SCOPE)
        elif tag == "tr":
            self._close({"td", "th"}, _TABLE_SCOPE)
            self._close({"tr"}, _TABLE_SCOPE)
        elif tag in ("caption", "colgroup", "tbody", "tfoot", "thead"):
            k = self._find({"table"}, _TABLE_SCOPE)
            if k is not None:
                self._pop_to(k + 1)
        elif tag == "table" and self._top() in _TABLE_CONTEXT:
            self._close({"table"}, _TABLE_SCOPE)

    def _close_list_item(self, names: set[str]) -> None:
        k = self._nearest(names)
        blocked = False
        for s in reversed(self._specials):
            if s <= k:
                break
            if self._stack[s].tag not in ("address", "div", "p"):
                blocked = True
                break
        if k >= 0 and not blocked:
            self._pop_to(k)
        self._close({"p"}, _BUTTON_SCOPE)

    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS:
            self.result.breaks.append(self._pos())
        if tag in ("html", "body", "br"):
            return
        if tag == "p":
            self._close({"p"}, _BUTTON_SCOPE)
        elif tag == "li":
            self._close({"li"}, _LIST_SCOPE)
        elif tag in _HEADINGS:
            self._close(_HEADINGS, _SCOPE)
        elif tag in _TABLE_PARTS or tag == "table":
            self._close({tag}, _TABLE_SCOPE)
        elif tag == "template":
            self._close({"template"}, ())
        elif tag in _FORMATTING:
            self._end_formatting(tag)
        elif tag in _SPECIAL:
            self._close({tag}, _SCOPE)
        else:
            self._end_other(tag)

    def _end_formatting(self, tag: str) -> None:
        i = self._active(tag)
        if i is None:
            self._end_other(tag)
            return
        k = self._index(self._formatting[i])
        if k is None:
            self._unlist(i)
            return
        if self._nearest(_SCOPE) > k:
            return
        self._unlist(i)
        self._pop_to(k)

    def _end_other(self, tag: str) -> None:
        k = self._nearest((tag,))
        if k < 0 or (self._specials and self._specials[-1] > k):
            return
        self._pop_to(k)

    def _text_reason_here(self, text: str) -> str | None:
        in_table = self._top() in _TABLE_CONTEXT
        if self._top() not in _RAW_TEXT and (text.strip() or not in_table):
            self._reconstruct()
        return _text_reason(self._current_style(foster=bool(text.strip())))

    def handle_data(self, data):
        start = self._pos()
        reason = self._text_reason_here(data)
        self.result.pieces.append(Piece(start, start + len(data), data, reason))

    def handle_entityref(self, name):
        self._entity()

    def handle_charref(self, name):
        self._entity()

    def _entity(self):
        start = self._pos()
        m = _ENTITY.match(self.raw, start)
        end = m.end() if m else start + 1
        text = html.unescape(self.raw[start:end])
        self.result.pieces.append(Piece(start, end, text, self._text_reason_here(text)))

    def handle_comment(self, data):
        tag_start = self._pos()
        start = self.raw.find(data, tag_start) if data else tag_start
        self.result.pieces.append(Piece(start, start + len(data), data, "comment"))


def _first_attrs(attrs) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in attrs:
        out.setdefault(name, value if value is not None else "")
    return out


def _parse_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    important: set[str] = set()
    for decl in _CSS_COMMENT.sub("", style).split(";"):
        name, sep, value = decl.partition(":")
        if not sep:
            continue
        name, value = name.strip().lower(), value.strip().lower()
        is_important = bool(_IMPORTANT.search(value))
        if name in important and not is_important:
            continue
        out[name] = _IMPORTANT.sub("", value).strip()
        if is_important:
            important.add(name)
    return out


def _number(value: str) -> tuple[float, str] | None:
    m = _NUMBER.match(value.strip())
    if not m:
        return None
    n = float(m.group(1))
    return (n, m.group(2)) if math.isfinite(n) else None


def _length(value: str | None, font_px: float) -> float:
    if not value:
        return math.inf
    parsed = _number(value)
    if parsed is None:
        return math.inf
    n, unit = parsed
    if n == 0:
        return 0.0
    if unit in _ABSOLUTE_PX:
        return n * _ABSOLUTE_PX[unit]
    if unit == "em":
        return n * font_px
    if unit == "rem":
        return n * 16.0
    if unit in ("ex", "ch"):
        return n * font_px / 2
    return math.inf


def _font_size(value: str | None, parent_px: float) -> float:
    if not value:
        return parent_px
    v = value.strip()
    if v in _FONT_KEYWORDS:
        return _FONT_KEYWORDS[v]
    if v == "smaller":
        return parent_px / 1.2
    if v == "larger":
        return parent_px * 1.2
    if v in ("initial", "revert"):
        return _FONT_KEYWORDS["medium"]
    parsed = _number(v)
    if parsed is None or parsed[0] < 0:
        return parent_px
    n, unit = parsed
    if unit == "%":
        return parent_px * n / 100
    length = _length(v, parent_px)
    return length if math.isfinite(length) else parent_px


def _opacity(value: str | None) -> float:
    alpha = _alpha(value) if value else None
    return 1.0 if alpha is None else alpha


def _visibility(value: str | None, parent: bool) -> bool:
    if value in ("hidden", "collapse"):
        return True
    if value in ("visible", "initial"):
        return False
    return parent


def _scaled_to_zero(value: str | None) -> bool:
    for m in _SCALE.finditer(value or ""):
        for arg in re.split(r"[\s,]+", m.group(1).strip()):
            parsed = _number(arg) if arg else None
            if parsed is not None and parsed[0] == 0:
                return True
    return False


def _clipped(style: dict[str, str], font_px: float) -> bool:
    overflow = " ".join(style.get(p, "") for p in ("overflow", "overflow-x", "overflow-y"))
    clips = bool({"hidden", "clip"} & set(overflow.split())) or "clip" in style
    sizes = ("width", "height", "max-width", "max-height")
    if clips and any(_length(style.get(p), font_px) <= TINY_BOX_PX for p in sizes):
        return True
    m = re.fullmatch(r"rect\(([^)]*)\)", style.get("clip", ""))
    if m:
        edges = [_length(p, font_px) for p in re.split(r"[\s,]+", m.group(1).strip()) if p]
        if len(edges) == 4 and all(math.isfinite(e) for e in edges):
            top, right, bottom, left = edges
            if right - left <= TINY_BOX_PX or bottom - top <= TINY_BOX_PX:
                return True
    m = re.match(r"inset\(\s*([\d.]+)%", style.get("clip-path", ""))
    return bool(m) and float(m.group(1)) >= 50


def _color_value(value: str | None, parent: Color) -> Color:
    if not value or value in ("currentcolor", "inherit", "unset", "revert"):
        return parent
    if value == "initial":
        return BLACK
    color = _parse_color(value)
    return parent if color is None else color


def _background(style: dict[str, str], parent: Color, color: Color) -> Color:
    for name in ("background-color", "background"):
        value = _URL.sub(" ", style.get(name, ""))
        for token in _COLOR_TOKEN.findall(value):
            found = color if token == "currentcolor" else _parse_color(token)
            if found is not None:
                return _over(found, parent)
    return parent


def _over(fg: Color, bg: Color) -> Color:
    a = fg[3]
    return (*(fg[k] * a + bg[k] * (1 - a) for k in range(3)), 1.0)


def _parse_color(value: str | None) -> Color | None:
    if not value:
        return None
    v = value.strip().lower()
    if v == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if v in NAMED_COLORS:
        return NAMED_COLORS[v]
    if re.fullmatch(r"#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})", v):
        digits = v[1:]
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        r, g, b = (float(int(digits[k : k + 2], 16)) for k in (0, 2, 4))
        a = int(digits[6:8], 16) / 255 if len(digits) == 8 else 1.0
        return (r, g, b, a)
    m = re.fullmatch(r"(rgba?|hsla?)\(([^)]*)\)", v)
    if not m:
        return None
    parts = [p for p in re.split(r"[\s,/]+", m.group(2).strip()) if p]
    if len(parts) not in (3, 4):
        return None
    alpha = _alpha(parts[3]) if len(parts) == 4 else 1.0
    if m.group(1).startswith("rgb"):
        r, g, b = (_rgb_channel(p) for p in parts[:3])
        if None in (r, g, b, alpha):
            return None
        return (r, g, b, alpha)
    hue, sat, light = _hue(parts[0]), _percent(parts[1]), _percent(parts[2])
    if None in (hue, sat, light, alpha):
        return None
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return (r * 255, g * 255, b * 255, alpha)


def _rgb_channel(text: str) -> float | None:
    parsed = _number(text)
    if parsed is None or parsed[1] not in ("%", ""):
        return None
    n = parsed[0] / 100 * 255 if parsed[1] == "%" else parsed[0]
    return min(max(n, 0.0), 255.0)


def _alpha(text: str) -> float | None:
    parsed = _number(text)
    if parsed is None or parsed[1] not in ("%", ""):
        return None
    n = parsed[0] / 100 if parsed[1] == "%" else parsed[0]
    return min(max(n, 0.0), 1.0)


def _percent(text: str) -> float | None:
    parsed = _number(text)
    if parsed is None or parsed[1] not in ("%", ""):
        return None
    return min(max(parsed[0] / 100, 0.0), 1.0)


def _hue(text: str) -> float | None:
    parsed = _number(text)
    if parsed is None:
        return None
    n, unit = parsed
    turns = {"": 1 / 360, "deg": 1 / 360, "turn": 1.0, "rad": 1 / (2 * math.pi), "grad": 1 / 400}
    if unit not in turns:
        return None
    return (n * turns[unit]) % 1.0


def _luminance(c: Color) -> float:
    def channel(x: float) -> float:
        x /= 255
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(c[0]) + 0.7152 * channel(c[1]) + 0.0722 * channel(c[2])


def _contrast(fg: Color, bg: Color) -> float:
    mixed = _over(fg, bg)
    lighter, darker = sorted((_luminance(mixed), _luminance(bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
