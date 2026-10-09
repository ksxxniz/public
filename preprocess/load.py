"""문서 파일을 읽어 문자열로 돌려준다 (① 입력 디코딩).

원문은 읽는 순간부터 바뀌면 안 된다. 그래서
- 바이트로 읽고 직접 디코딩한다. read_text() 는 \\r\\n 을 \\n 으로 몰래 바꾼다.
- 줄바꿈은 여기서 건드리지 않는다. 통일은 정규화(③ 공백 정리)에서 위치를 추적하며 한다.
- 맨 앞 BOM 은 지운다. 남겨두면 ③ 에서 제로폭 문자로 세어 정상 파일이 의심 신호를 받는다.
- UTF-8 이 아니면 charset-normalizer 로 판별한다 (CP949/EUC-KR 등).
- 지원 형식(decisions.md ②)이 아니면 거부한다.

원문 좌표의 기준은 여기서 돌려준 문자열이다 (바이트가 아님, decisions.md ⑥).
"""

import codecs
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

from common.config import SUPPORTED_EXTENSIONS
from preprocess.normalize import resolve_format

# 맨 앞 바이트로 확실히 알 수 있는 인코딩. 윈도우 메모장의 "유니코드" 저장이 UTF-16 이다
_BOMS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF32_LE, "utf-32-le"),  # UTF-16 LE BOM(FF FE)으로 시작하므로 먼저 본다
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


class UnsupportedDocumentError(ValueError):
    """지원하지 않는 형식이거나, 글자로 읽을 수 없는 파일."""


@dataclass(frozen=True)
class LoadedDocument:
    raw: str  # 원문. 이후 모든 위치의 기준 (정리본 Chunk.text 와 헷갈리지 않게 raw)
    fmt: str  # "txt" / "md" / "html"
    encoding: str  # 실제로 쓴 인코딩 (예: "utf-8", "cp949")
    had_bom: bool


def read_document(path: str | Path) -> LoadedDocument:
    """파일을 읽어 원문과 형식·인코딩 정보를 함께 돌려준다."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"지원하지 않는 형식입니다: {path.name} (지원: {', '.join(SUPPORTED_EXTENSIONS)})"
        )
    raw, encoding, had_bom = decode_bytes(path.read_bytes())
    return LoadedDocument(raw=raw, fmt=resolve_format(raw, ext), encoding=encoding, had_bom=had_bom)


def load_document(path: str | Path) -> str:
    """원문 문자열만 돌려준다. 형식도 필요하면 read_document() 를 쓴다."""
    return read_document(path).raw


def decode_bytes(data: bytes) -> tuple[str, str, bool]:
    """바이트를 (원문, 인코딩 이름, BOM 이 있었는지) 로 바꾼다."""
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            body = data[len(bom) :]
            try:
                return body.decode(encoding), encoding, True
            except UnicodeDecodeError as e:
                raise UnsupportedDocumentError(f"{encoding} BOM 이 있지만 읽을 수 없습니다") from e

    try:
        return data.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        pass

    if b"\x00" in data:
        # BOM 없는 UTF-16 은 드물고, 대부분 이미지·실행 파일 같은 바이너리다
        raise UnsupportedDocumentError("글자 파일이 아닌 것 같습니다 (0x00 바이트 포함)")

    # 짧은 한국어 문서는 charset-normalizer 가 big5·utf_16 으로 오판한다 ("안녕" → "寰喟").
    # 한국어 문서가 대부분이므로 CP949(EUC-KR 포함)를 먼저 시도하고, 한글 비율로 확인한다
    try:
        text = data.decode("cp949")
        if _looks_korean(text):
            japanese = _as_japanese(data, text)
            if japanese is not None:
                return japanese, "cp932", False
            return text, "cp949", False
    except UnicodeDecodeError:
        pass

    best = from_bytes(data).best()
    if best is None or best.encoding.startswith(("utf_16", "utf_32")):
        # BOM 도 0x00 도 없는 UTF-16/32 판별은 오판이다
        raise UnsupportedDocumentError("문자 인코딩을 알아낼 수 없습니다")
    return str(best), best.encoding, False


def _looks_korean(text: str) -> bool:
    """ASCII 가 아닌 글자 중 한글(음절·호환 자모)이 60% 이상인가."""
    non_ascii = [c for c in text if ord(c) >= 0x80]
    if not non_ascii:
        return True
    hangul = sum(1 for c in non_ascii if "\uac00" <= c <= "\ud7a3" or "\u3131" <= c <= "\u318e")
    return hangul / len(non_ascii) >= 0.6


_COMMON_HANGUL = frozenset(
    bytes((lead, trail)).decode("cp949")
    for lead in range(0xB0, 0xC9)
    for trail in range(0xA1, 0xFF)
)


def _as_japanese(data: bytes, korean: str) -> str | None:
    syllables = [c for c in korean if "\uac00" <= c <= "\ud7a3"]
    if 2 * sum(1 for c in syllables if c in _COMMON_HANGUL) >= len(syllables):
        return None
    try:
        text = data.decode("cp932")
    except UnicodeDecodeError:
        return None
    non_ascii = [c for c in text if ord(c) >= 0x80]
    hiragana = sum(1 for c in non_ascii if "\u3041" <= c <= "\u3096")
    return text if non_ascii and hiragana / len(non_ascii) >= 0.2 else None
