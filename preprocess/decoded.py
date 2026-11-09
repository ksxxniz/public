from dataclasses import dataclass

from common.schema import Chunk, Span
from preprocess.chunk import split_document
from preprocess.tokens import Tokenizer, token_windows


@dataclass(frozen=True)
class DecodedInput:
    text: str
    span: Span
    method: str
    depth: int
    segment: int
    window: int


def decoded_inputs(chunk: Chunk, tokenizer: Tokenizer | None = None) -> list[DecodedInput]:
    out: list[DecodedInput] = []
    for k, seg in enumerate(chunk.decoded_segments):
        for w, (s, e) in enumerate(token_windows(seg.decoded, tokenizer)):
            out.append(
                DecodedInput(
                    text=seg.decoded[s:e],
                    span=seg.span,
                    method=seg.method,
                    depth=seg.depth,
                    segment=k,
                    window=w,
                )
            )
    return out


def decoded_texts(
    raw: str, fmt: str | None = None, tokenizer: Tokenizer | None = None
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for chunk in split_document("train", raw, fmt, tokenizer):
        for item in decoded_inputs(chunk, tokenizer):
            if item.text not in seen:
                seen.add(item.text)
                out.append(item.text)
    return out
