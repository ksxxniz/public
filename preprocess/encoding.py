import base64
import binascii
import codecs
import html
import re
import unicodedata
from collections.abc import Callable
from urllib.parse import unquote

from preprocess.context import Context, DocumentTooLongError

MAX_DEPTH = 3
PRINTABLE_RATIO = 0.9
MIN_ROT13_WORDS = 3
MAX_DECODED_SEGMENTS = 10_000
MAX_DECODED_CHARS = 1_000_000
REF_GAP = 64
REF_CONTEXT = 200

_B64 = r"A-Za-z0-9+/_\-"
_B64_TOKEN = re.compile(rf"(?<![{_B64}])[{_B64}]{{16,}}={{0,2}}(?![{_B64}])")
_B64_WRAPPED = re.compile(
    rf"(?<![{_B64}])[{_B64}]{{20,}}(?:[ \t]*\n[ \t]*[{_B64}]{{4,}})+={{0,2}}(?![{_B64}])"
)
_B64_FULL = re.compile(rf"[{_B64}]{{16,}}={{0,2}}")
_PATH_PART = re.compile(r"[A-Za-z0-9+_\-]{16,}={0,2}")
_HEX_FULL = re.compile(r"(?:[0-9A-Fa-f]{2}){8,}")
_HEX_ESCAPED = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){8,}")
_HEX_PREFIXED = re.compile(
    r"(?<![0-9A-Za-z])0[xX](?:[0-9A-Fa-f]{2}){8,}(?![0-9A-Za-z])"
    r"|(?<![0-9A-Za-z])0[xX][0-9A-Fa-f]{2}"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)0[xX][0-9A-Fa-f]{2}){7,}(?![0-9A-Za-z])"
)
_URL_RUN = re.compile(r"(?:%[0-9A-Fa-f]{2}){3,}")
_URL_CHAR = re.compile(r"[A-Za-z0-9%._~:/?#@!$&*+,;=\-]")
_WORDS_ONLY = re.compile(r"[A-Za-z'\"! ]+")
_WORD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]+(?:'[A-Za-z]+)?(?![A-Za-z0-9])")
_ROT13_GAP = re.compile(r"[\W\d_]{0,40}")
_ROT13_MENTION = re.compile(r"rot[-_ ]?13", re.I)
_REF = re.compile(
    r"&#[xX](?P<hex>[0-9A-Fa-f]{1,8});?|&#(?P<dec>[0-9]{1,8});?|&[A-Za-z][A-Za-z0-9]{1,31};"
    r"|\\u\{(?P<brace>[0-9A-Fa-f]{1,8})\}|\\u(?P<u4>[0-9A-Fa-f]{4})|\\U(?P<u8>[0-9A-Fa-f]{8})"
)
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cn", "Mn", "Me"})

_WordHit = tuple[re.Match[str], bool, bool]

_COMMON = frozenset(
    """
    a about above across act add admin administrator after again against agent ago ai all allow
    already also always am an and another answer any anyone anything are area around as ask
    assistant at away back bad be because become been before behind being below best better
    between big body book both bot box break bring build but buy by call can cannot car care
    case cause change chat check child city class clear click close code come command company
    complete confidential content continue control conversation copy could country course
    data day delete deny describe detail developer did different do document does doing done
    door down draft each early easy else email end enough enter even ever every everything
    example exactly explain eye face fact false family far feel few file final find first
    follow following food for forget form found free friend from full game gave get give go
    going good got great group grow had hand happen hard has have he head hear help her here
    hidden high him his history hold home hour house how however human i idea if ignore
    important in information input instead instruction instructions into is issue it its job
    just keep key kind know language large last later lead learn leave left less let life like
    line link list little live long look lose lot love low made make man many may me mean
    message might mind mode model money more most move much must my name need never new next
    night no none not note nothing now number of off often old on once one only open or order
    other our out output over own page part password people person place plan play please
    point policy power previous print prior private problem product program prompt public put
    question quick read real really reason recommend remove repeat reply report request respond
    response restriction restrictions result return reveal right rule rules run safe said same
    say school second secret see seem send set setting settings she should show side simply
    since small so some someone something sometimes soon start state still stop story student
    such summarize sure system take talk task teach tell text than that the their them then
    there these they thing things think this those though thought three through time to today
    together told too tool top translate true try turn two under understand unlimited
    unrestricted until up update upon url us use user users value very verbatim visit want was
    water way we website week well were what when where which while who whole why will with
    within without word words work world would write wrong year yes yet you your
    """.split()
)


def restore_encodings(ctx: Context) -> None:
    text = ctx.tt.text
    if not text:
        return
    taken = bytearray(len(text))
    found: list[tuple[int, int, str, str, int]] = []

    def free(s: int, e: int) -> bool:
        return not any(taken[s:e])

    def accept(s: int, e: int, method: str, decoded: str | None) -> bool:
        if decoded is None or not _acceptable(decoded):
            return False
        final, depth = _peel(decoded, 1)
        found.append((s, e, method, final, depth))
        taken[s:e] = b"\x01" * (e - s)
        return True

    def token(s: int, e: int) -> bool:
        t = text[s:e]
        if _HEX_FULL.fullmatch(t) and accept(s, e, "hex", _hex_text(t)):
            return True
        return accept(s, e, "base64", _from_base64(t))

    for m in _B64_WRAPPED.finditer(text):
        if free(*m.span()):
            accept(*m.span(), "base64", _from_base64(m.group()))
    for rx in (_HEX_ESCAPED, _HEX_PREFIXED):
        for m in rx.finditer(text):
            if free(*m.span()):
                accept(*m.span(), "hex", _from_hex(m.group()))
    done = 0
    for m in _URL_RUN.finditer(text):
        if m.start() < done:
            continue
        s, e = _url_token(text, m.start(), m.end())
        done = e
        if free(s, e):
            accept(s, e, "url", _from_url(text[s:e]))
    for m in _B64_TOKEN.finditer(text):
        s, e = m.span()
        if not free(s, e) or token(s, e):
            continue
        for part in _PATH_PART.finditer(text, s, e):
            if part.span() != (s, e) and free(*part.span()):
                token(*part.span())
    for s, e in _rot13_spans(text):
        if free(s, e):
            accept(s, e, "rot13", codecs.encode(text[s:e], "rot13"))
    found.extend(_reference_regions(text))

    total = len(ctx.decoded_segments) + len(found)
    if total > MAX_DECODED_SEGMENTS:
        raise DocumentTooLongError(
            f"복원 결과가 {total}개로 상한 {MAX_DECODED_SEGMENTS}개를 넘습니다. "
            "문서를 나눠서 넣어 주세요"
        )
    size = sum(len(seg.decoded) for seg in ctx.decoded_segments)
    size += sum(len(decoded) for _, _, _, decoded, _ in found)
    _check_decoded_size(size)

    renormalize = ctx.renormalize or (lambda t: t)
    size = sum(len(seg.decoded) for seg in ctx.decoded_segments)
    for s, e, method, decoded, depth in sorted(found):
        decoded = renormalize(decoded)
        size += len(decoded)
        _check_decoded_size(size)
        ctx.add_decoded(method, ctx.tt.raw_span(s, e), decoded, depth)


def _check_decoded_size(size: int) -> None:
    if size > MAX_DECODED_CHARS:
        raise DocumentTooLongError(
            f"복원 결과가 {size}자로 상한 {MAX_DECODED_CHARS}자를 넘습니다. "
            "문서를 나눠서 넣어 주세요"
        )


def _reference_regions(text: str) -> list[tuple[int, int, str, str, int]]:
    clusters: list[list[re.Match[str]]] = []
    for m in _REF.finditer(text):
        if clusters and m.start() - clusters[-1][-1].end() <= REF_GAP:
            clusters[-1].append(m)
        else:
            clusters.append([m])
    regions: list[tuple[int, int, str]] = []
    for cluster in clusters:
        method = _cluster_method(cluster)
        if method is None:
            continue
        s, e = _line_around(text, cluster[0].start(), cluster[-1].end())
        if regions and s <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(e, regions[-1][1]), regions[-1][2])
        else:
            regions.append((s, e, method))
    out: list[tuple[int, int, str, str, int]] = []
    for s, e, method in regions:
        decoded = _unescape_refs(text[s:e])
        if any(c.isalpha() for c in decoded):
            out.append((s, e, method, decoded, 1))
    return out


def _cluster_method(cluster: list[re.Match[str]]) -> str | None:
    i = 0
    while i < len(cluster):
        m = cluster[i]
        value = _ref_value(m)
        if value is not None and 0xD800 <= value <= 0xDBFF and i + 1 < len(cluster):
            low = _ref_value(cluster[i + 1])
            if cluster[i + 1].start() == m.end() and low is not None and 0xDC00 <= low <= 0xDFFF:
                value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00)
                i += 1
        if value is not None and _suspicious_char(chr(value), _named(m)):
            return "html_entity" if m.group().startswith("&") else "unicode_escape"
        i += 1
    return None


def _named(m: re.Match[str]) -> bool:
    return m.group().startswith("&") and m.group("hex") is None and m.group("dec") is None


def _ref_value(m: re.Match[str]) -> int | None:
    if _named(m):
        decoded = html.unescape(m.group())
        return ord(decoded) if len(decoded) == 1 and decoded != m.group() else None
    digits = m.group("hex") or m.group("brace") or m.group("u4") or m.group("u8")
    value = int(digits, 16) if digits else int(m.group("dec"))
    return value if value <= 0x10FFFF else None


def _suspicious_char(c: str, named: bool) -> bool:
    if c in "\t\n\r" or 0xD800 <= ord(c) <= 0xDFFF:
        return False
    if unicodedata.category(c) in _HIDDEN_CATEGORIES:
        return True
    return c.isalnum() and not named


def _line_around(text: str, s: int, e: int) -> tuple[int, int]:
    lo = max(0, s - REF_CONTEXT)
    start = text.rfind("\n", lo, s) + 1 or lo
    if start == lo and lo > 0:
        space = text.find(" ", lo, s)
        start = space + 1 if space != -1 else lo
    hi = min(len(text), e + REF_CONTEXT)
    end = text.find("\n", e, hi)
    if end == -1:
        end = hi
        if hi < len(text):
            space = text.rfind(" ", e, hi)
            end = space if space != -1 else hi
    return start, end


def _unescape_refs(text: str) -> str:
    def one(m: re.Match[str]) -> str:
        if m.group().startswith("&"):
            return html.unescape(m.group())
        digits = m.group("brace") or m.group("u4") or m.group("u8")
        value = int(digits, 16)
        return chr(value) if value <= 0x10FFFF else m.group()

    decoded = _REF.sub(one, text)
    return decoded.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _acceptable(decoded: str) -> bool:
    if not any(c.isalpha() for c in decoded):
        return False
    allowed = max(0.0, (1 - PRINTABLE_RATIO) * (len(decoded) - 10))
    bad = 0
    for c in decoded:
        if _unprintable(c):
            bad += 1
            if bad > allowed:
                return False
    return True


def _unprintable(c: str) -> bool:
    if c in "\n\r\t":
        return False
    return c == "\ufffd" or unicodedata.category(c) in ("Cc", "Cs", "Co", "Cn")


def _peel(decoded: str, depth: int) -> tuple[str, int]:
    while depth < MAX_DEPTH:
        inner = _inner(decoded.strip())
        if inner is None:
            break
        decoded, depth = inner, depth + 1
    return decoded, depth


def _inner(t: str) -> str | None:
    decoders: list[Callable[[str], str | None]] = []
    if _HEX_FULL.fullmatch(t):
        decoders.append(_hex_text)
    if _B64_FULL.fullmatch(t):
        decoders.append(_from_base64)
    if _URL_RUN.search(t) and not any(c.isspace() for c in t):
        decoders.append(_from_url)
    decoders.append(_unrot13)
    for decode in decoders:
        inner = decode(t)
        if inner is not None and _acceptable(inner):
            return inner
    return None


def _bytes_text(data: bytes) -> str:
    if len(data) >= 8 and data.count(0) * 3 >= len(data):
        for codec, high in (("utf-16-le", data[1::2]), ("utf-16-be", data[::2])):
            if high.count(0) * 10 >= len(high) * 9:
                return data.decode(codec, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if len(data) >= 8 and len(data) % 2 == 0:
        for codec in ("utf-16-le", "utf-16-be"):
            text = data.decode(codec, errors="replace").lstrip("\ufeff")
            if _mostly_hangul(text):
                return text
    return data.decode("utf-8", errors="replace")


def _mostly_hangul(text: str) -> bool:
    hangul = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
    plain = sum(1 for c in text if " " <= c <= "~")
    return hangul >= 2 and (hangul + plain) * 10 >= len(text) * 9


def _from_base64(token: str) -> str | None:
    t = re.sub(r"\s+", "", token).rstrip("=")
    t = t.replace("-", "+").replace("_", "/")
    if len(t) % 4 == 1:
        return None
    try:
        data = base64.b64decode(t + "=" * (-len(t) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return _bytes_text(data)


def _from_hex(token: str) -> str | None:
    t = re.sub(r"\\x|0[xX]|[\s,]", "", token)
    if len(t) % 2 or len(t) < 16:
        return None
    try:
        return _bytes_text(bytes.fromhex(t))
    except ValueError:
        return None


def _hex_text(token: str) -> str | None:
    decoded = _from_hex(token)
    if decoded is not None and token.isdigit() and not _plain_words(decoded):
        return None
    return decoded


def _plain_words(decoded: str) -> bool:
    return (
        _WORDS_ONLY.fullmatch(decoded) is not None
        and " " in decoded.strip()
        and sum(c.isalpha() for c in decoded) >= 4
    )


def _from_url(token: str) -> str | None:
    decoded = unquote(token, encoding="utf-8", errors="replace")
    return decoded if decoded != token else None


def _url_token(text: str, s: int, e: int) -> tuple[int, int]:
    while s > 0 and _URL_CHAR.match(text[s - 1]):
        s -= 1
    while e < len(text) and _URL_CHAR.match(text[e]):
        e += 1
    return s, e


def _unrot13(text: str) -> str | None:
    spans = _rot13_spans(text)
    if not spans:
        return None
    out: list[str] = []
    pos = 0
    for s, e in spans:
        out.append(text[pos:s])
        out.append(codecs.encode(text[s:e], "rot13"))
        pos = e
    out.append(text[pos:])
    return "".join(out)


def _rot13_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    run: list[re.Match[str]] = []
    prev_end = -1
    for m in _WORD.finditer(text):
        if run and _ROT13_GAP.fullmatch(text, prev_end, m.start()) is None:
            spans.extend(_rot13_in_run(text, run))
            run = []
        run.append(m)
        prev_end = m.end()
    spans.extend(_rot13_in_run(text, run))
    return spans


def _rot13_in_run(text: str, run: list[re.Match[str]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seq: list[_WordHit] = []
    for m in [*run, None]:
        if m is not None:
            word = m.group().lower()
            before = word in _COMMON
            after = codecs.encode(word, "rot13") in _COMMON
            if not before or after:
                seq.append((m, before, after))
                continue
        for lead, core, tail in _dense_parts(seq):
            for part in (core, lead + core + tail):
                if len(part) >= MIN_ROT13_WORDS and _looks_rot13(text, part):
                    out.append((part[0][0].start(), part[-1][0].end()))
                    break
        seq = []
    return out


def _dense_parts(
    seq: list[_WordHit],
) -> list[tuple[list[_WordHit], list[_WordHit], list[_WordHit]]]:
    parts: list[tuple[list[_WordHit], list[_WordHit], list[_WordHit]]] = []
    lead: list[_WordHit] = []
    current: list[_WordHit] = []
    pending: list[_WordHit] = []
    misses = 0
    for item in seq:
        if item[2]:
            if current:
                current.extend(pending)
            else:
                lead = pending
            current.append(item)
            pending, misses = [], 0
            continue
        pending.append(item)
        misses += len(item[0].group()) >= 2
        if misses >= MIN_ROT13_WORDS:
            if current:
                parts.append((lead, current, pending[:-1]))
            lead, current, pending, misses = [], [], [], 0
    if current:
        parts.append((lead, current, pending))
    return parts


def _looks_rot13(text: str, seq: list[_WordHit]) -> bool:
    words = [(m, b, a) for m, b, a in seq if len(m.group()) >= 2]
    if len(words) < MIN_ROT13_WORDS:
        return False
    before = sum(b for _, b, _ in words) / len(words)
    hits = sum(a for _, _, a in words)
    after = hits / len(words)
    if hits >= 3 and after >= 0.5 and before <= 0.2:
        return True
    s, e = seq[0][0].start(), seq[-1][0].end()
    nearby = text[max(0, s - 120) : e + 120]
    return bool(_ROT13_MENTION.search(nearby)) and hits >= 2 and after >= before + 0.3
