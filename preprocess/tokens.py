import os
from functools import lru_cache
from typing import Protocol

from common import config
from preprocess.normalize import normalize_text

ENV_VAR = "PREPROCESS_TOKENIZER"
FAKE_NAME = "char"

_MAX_TOKENS_PER_CHAR = 4
_MAX_SPECIAL_TOKENS = 4


class TokenizerLoadError(RuntimeError):
    pass


class Tokenizer(Protocol):
    name: str
    num_special_tokens: int

    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class CharTokenizer:
    name = FAKE_NAME
    num_special_tokens = 2

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i, ch in enumerate(text) if not ch.isspace()]


class HFTokenizer:
    def __init__(self, tokenizer, name: str):
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tok = tokenizer
        self.name = name
        self.num_special_tokens = tokenizer.num_special_tokens_to_add(False)

    @classmethod
    def load(cls, name: str) -> "HFTokenizer":
        from tokenizers import Tokenizer as RawTokenizer

        try:
            if name.endswith(".json"):
                return cls(RawTokenizer.from_file(name), name)
            return cls(RawTokenizer.from_pretrained(name), name)
        except Exception as e:
            raise TokenizerLoadError(
                f"토크나이저 '{name}' 을(를) 불러오지 못했습니다 ({type(e).__name__}). "
                "인터넷 연결과 config.TOKENIZER_NAME(또는 PREPROCESS_TOKENIZER)을 확인하세요."
            ) from e

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [tuple(o) for o in self._tok.encode(text, add_special_tokens=False).offsets]


@lru_cache(maxsize=8)
def load_tokenizer(name: str) -> Tokenizer:
    if name == FAKE_NAME:
        return CharTokenizer()
    return HFTokenizer.load(name)


def get_tokenizer() -> Tokenizer:
    return load_tokenizer(os.environ.get(ENV_VAR) or config.TOKENIZER_NAME)


def content_size(tokenizer: Tokenizer) -> int:
    return config.CHUNK_SIZE_TOKENS - tokenizer.num_special_tokens


def count_tokens(raw: str, fmt: str | None = "txt", tokenizer: Tokenizer | None = None) -> int:
    tok = tokenizer or get_tokenizer()
    return len(tok.offsets(normalize_text(raw, fmt))) + tok.num_special_tokens


def fits_in_chunk(raw: str, fmt: str | None = "txt", tokenizer: Tokenizer | None = None) -> bool:
    return count_tokens(raw, fmt, tokenizer) <= config.CHUNK_SIZE_TOKENS


def token_windows(
    text: str,
    tokenizer: Tokenizer | None = None,
    size: int | None = None,
    overlap: int | None = None,
) -> list[tuple[int, int]]:
    if not text:
        return []
    if size is not None:
        bound = size
    elif tokenizer is not None:
        bound = content_size(tokenizer)
    else:
        bound = config.CHUNK_SIZE_TOKENS - _MAX_SPECIAL_TOKENS
    if _MAX_TOKENS_PER_CHAR * len(text) + 1 <= bound:
        return [(0, len(text))]

    tok = tokenizer or get_tokenizer()
    size = content_size(tok) if size is None else size
    overlap = config.CHUNK_OVERLAP_TOKENS if overlap is None else overlap
    if not 0 <= overlap < size:
        raise ValueError(f"겹침은 0 이상, 크기({size}) 미만이어야 합니다: {overlap}")

    offs = tok.offsets(text)
    n = len(offs)
    if n <= size:
        return [(0, len(text))]

    windows: list[tuple[int, int]] = []
    i = 0
    while True:
        j = min(i + size, n)
        start = 0 if i == 0 else offs[i][0]
        end = _window_end(text, offs, start, j)
        while j - i > 1 and len(tok.offsets(text[start:end])) > size:
            j -= 1
            end = _window_end(text, offs, start, j)
        windows.append((start, end))
        if j == n:
            return windows
        i = max(j - overlap, i + 1)


def _window_end(text: str, offs: list[tuple[int, int]], start: int, j: int) -> int:
    if j == len(offs):
        return len(text)
    end = offs[j][0]
    if end <= start:
        end = max(offs[j - 1][1], start + 1)
    return end
