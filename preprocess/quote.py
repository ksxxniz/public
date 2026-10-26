from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

from common.schema import Chunk, Span
from preprocess.normalize import normalize_text

MIN_SCORE = 0.8
MIN_FUZZY_CHARS = 8

Method = Literal["exact", "loose", "fuzzy"]

_QUOTE_MARKS = "\"'`“”‘’「」『』"


@dataclass(frozen=True)
class QuoteMatch:
    span: Span
    method: Method
    score: float
    segment: int | None = None


def find_quote(chunk: Chunk, quote: str, min_score: float = MIN_SCORE) -> QuoteMatch | None:
    variants = _variants(quote)
    if not variants:
        return None
    targets: list[tuple[str, int | None]] = [(chunk.text, None)]
    targets += [(seg.decoded, k) for k, seg in enumerate(chunk.decoded_segments)]
    for method, search in (("exact", _exact), ("loose", _loose)):
        for text, k in targets:
            for q in variants:
                hit = search(text, q)
                if hit is not None:
                    return _match(chunk, k, hit, method, 1.0)
    best: tuple[tuple[int, int], float, int | None] | None = None
    for text, k in targets:
        for q in variants:
            found = _fuzzy(text, q, min_score)
            if found is not None and (best is None or found[1] > best[1]):
                best = (found[0], found[1], k)
    if best is None:
        return None
    return _match(chunk, best[2], best[0], "fuzzy", best[1])


def _variants(quote: str) -> list[str]:
    q = quote.strip().strip(_QUOTE_MARKS).strip()
    if not q:
        return []
    out = [q]
    normalized = normalize_text(q, "txt").strip()
    if normalized and normalized != q:
        out.append(normalized)
    return out


def _match(
    chunk: Chunk, k: int | None, hit: tuple[int, int], method: Method, score: float
) -> QuoteMatch:
    if k is None:
        span = chunk.raw_span(*hit)
    else:
        seg = chunk.decoded_segments[k].span
        span = Span(start=seg.start, end=seg.end)
    return QuoteMatch(span=span, method=method, score=score, segment=k)


def _exact(text: str, q: str) -> tuple[int, int] | None:
    i = text.find(q)
    return None if i < 0 else (i, i + len(q))


def _compress(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index: list[int] = []
    for i, c in enumerate(text):
        if c.isspace():
            continue
        for f in c.casefold():
            chars.append(f)
            index.append(i)
    return "".join(chars), index


def _loose(text: str, q: str) -> tuple[int, int] | None:
    cq, _ = _compress(q)
    if not cq:
        return None
    ct, index = _compress(text)
    j = ct.find(cq)
    if j < 0:
        return None
    return index[j], index[j + len(cq) - 1] + 1


def _fuzzy(text: str, q: str, min_score: float) -> tuple[tuple[int, int], float] | None:
    cq, _ = _compress(q)
    ct, index = _compress(text)
    m, n = len(cq), len(ct)
    if m < MIN_FUZZY_CHARS or n == 0:
        return None
    if m >= n:
        score = fuzz.ratio(cq, ct) / 100
        return ((index[0], index[-1] + 1), score) if score >= min_score else None
    step = max(1, m // 16)
    lengths = sorted({m, *range(max(1, m * 3 // 4), min(n, m * 4 // 3) + 1, step)})
    cands = [
        (s, length)
        for length in lengths
        for s in sorted({*range(0, n - length + 1, step), n - length})
    ]
    s, length, score = _best(cq, ct, cands)
    radius = step
    while radius > 0:
        stride = max(1, radius // 4)
        cands = [
            (s2, l2)
            for l2 in range(max(1, length - radius), min(n, length + radius) + 1, stride)
            for s2 in range(max(0, s - radius), min(n - l2, s + radius) + 1, stride)
        ]
        cands.append((s, length))
        s, length, score = _best(cq, ct, cands)
        radius = 0 if stride == 1 else stride
    if score < min_score:
        return None
    return (index[s], index[s + length - 1] + 1), score


def _best(cq: str, ct: str, cands: list[tuple[int, int]]) -> tuple[int, int, float]:
    windows = [ct[s : s + length] for s, length in cands]
    _, score, k = process.extractOne(cq, windows, scorer=fuzz.ratio)
    s, length = cands[k]
    return s, length, score / 100
