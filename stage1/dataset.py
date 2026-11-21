"""3단계 학습 데이터 읽기. JSONL 한 줄 = 샘플 하나.

형식은 datagen/evasion.py 가 읽고 쓰는 형식을 그대로 따른다.
  text          원문 (정규화 전). 필수
  label         "benign" / "injection". 필수
  fmt           "txt" / "md" / "html". 없으면 txt
  source_group  같은 원본에서 나온 변형끼리 같은 값. train/val 분할 단위
  나머지 칸(id, lang, attack_type, techniques, concealment, answer, variant_of,
  hard_negative, quoted_from …)은 학습에 쓰지 않고 Example.extra 에 보관한다.
  유형별 성능(예: Base64 공격, 하드 네거티브)을 볼 때 쓴다.

규칙
- 학습 입력은 preprocess.normalize.normalize_text() 를 거친 정리본이다.
  실전에서 1차 모델이 보는 글과 같아야 한다. (복사하지 말고 import 해서 쓴다)
- train/val 은 source_group 단위로 나눈다. 같은 원본의 변형(번역·Base64·제로폭 등)이
  train 과 val 에 흩어지면 점수가 부풀어 오른다 (데이터 누수).
- test 는 데이터 담당이 따로 보관한다. 이 파일은 test 를 나누거나 읽지 않는다.

복원 결과 (decisions.md ⑥, 2026-10-26 확정)
- 서빙에서는 Base64 등을 푼 복원 결과(decoded_segments)를 조각마다 따로 1차 모델에 넣는다.
  학습도 같게 하려고 add_decoded() 가 샘플마다 preprocess.decoded.decoded_texts() 로
  같은 조각을 만들어 학습 샘플(kind="decoded")로 더한다.
- 라벨: 조각에 그 샘플의 공격 문장(answer)이 들어 있으면 injection, 아니면 benign.
  공격 샘플의 복원 결과라도 개발 문서 예시 Base64 처럼 공격과 상관없는 글일 수 있다.
  answer 가 없는 공격 샘플은 조각의 라벨을 정할 수 없어 조각을 만들지 않는다.
- 조각은 원래 샘플과 같은 group 에 넣는다 (train/val 에 갈라지지 않게).
- 조각은 정리본이 아니라 복원 결과 그대로다. 서빙에서도 그대로 넣는다.
"""

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from common.config import SEED
from common.schema import Label
from preprocess.context import DocumentTooLongError
from preprocess.decoded import decoded_texts
from preprocess.html_parse import HtmlParserUnavailableError
from preprocess.normalize import normalize_text

# 칸 이름. 데이터 형식이 바뀌면 여기만 고친다
TEXT_KEY = "text"
LABEL_KEY = "label"
FMT_KEY = "fmt"
ANSWER_KEY = "answer"  # 공격 문장. 복원 결과 조각의 라벨을 정할 때 쓴다
GROUP_KEYS = ("source_group", "variant_of", "id")  # 앞에서부터 찾고, 모두 없으면 원문

LABEL2ID: dict[str, int] = {"benign": 0, "injection": 1}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class DatasetFormatError(ValueError):
    """JSONL 형식 오류. 몇 번째 줄인지 메시지에 담는다."""


@dataclass
class Example:
    text: str  # 정리본 (normalize_text 결과)
    label: Label
    raw: str  # 원문
    fmt: str
    group: str  # train/val 분할 단위
    extra: dict = field(default_factory=dict)  # 학습에 안 쓰는 나머지 칸
    doc: str = ""  # 원래 샘플 (파일:줄). 문서 단위 지표를 낼 때 조각을 묶는다
    kind: str = "body"  # "body" = 본문 정리본, "decoded" = 복원 결과 조각

    @property
    def label_id(self) -> int:
        return LABEL2ID[self.label]


def load_jsonl(path: str | Path) -> list[Example]:
    """JSONL 을 읽어 정리본으로 바꾼다. BOM 이 있어도 읽는다 (evasion.py 와 같음)."""
    path = Path(path)
    examples = []
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                examples.append(_parse(line, f"{path.name}:{n}"))
    return examples


def _parse(line: str, where: str) -> Example:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"{where}: JSON 이 아닙니다 ({e.msg})") from e
    if not isinstance(rec, dict):
        raise DatasetFormatError(f"{where}: 한 줄은 JSON 객체여야 합니다")

    raw = rec.pop(TEXT_KEY, None)
    label = rec.pop(LABEL_KEY, None)
    if not isinstance(raw, str):
        raise DatasetFormatError(f"{where}: {TEXT_KEY} 칸(문자열)이 없습니다")
    if label not in LABEL2ID:
        raise DatasetFormatError(
            f"{where}: {LABEL_KEY} 는 {'/'.join(LABEL2ID)} 중 하나여야 합니다: {label!r}"
        )
    fmt = rec.pop(FMT_KEY, None) or "txt"
    try:
        text = normalize_text(raw, fmt)
    except ValueError as e:  # 지원하지 않는 fmt
        raise DatasetFormatError(f"{where}: {e}") from e

    group = next((rec[k] for k in GROUP_KEYS if rec.get(k)), raw)
    return Example(text=text, label=label, raw=raw, fmt=fmt, group=str(group), extra=rec, doc=where)


def add_decoded(examples: Iterable[Example], tokenizer=None) -> tuple[list[Example], dict]:
    """본문 샘플 뒤에 복원 결과 조각 샘플을 붙여 돌려준다. 개수 통계도 함께.

    tokenizer 를 생략하면 전처리와 같은 토크나이저(config.TOKENIZER_NAME)로 조각을 자른다.
    """
    out: list[Example] = []
    stats = {"decoded_benign": 0, "decoded_injection": 0, "skipped_no_answer": 0}
    for ex in examples:
        out.append(ex)
        try:
            pieces = decoded_texts(ex.raw, ex.fmt, tokenizer)
        except (DocumentTooLongError, HtmlParserUnavailableError) as e:
            raise DatasetFormatError(f"{ex.doc}: 복원 결과를 만들 수 없습니다 ({e})") from e
        if not pieces:
            continue
        answer = ex.extra.get(ANSWER_KEY)
        if ex.label == "injection" and not answer:
            stats["skipped_no_answer"] += 1
            continue
        for piece in pieces:
            hit = ex.label == "injection" and _contains(piece, answer)
            label: Label = "injection" if hit else "benign"
            stats[f"decoded_{label}"] += 1
            out.append(
                Example(
                    text=piece,
                    label=label,
                    raw=ex.raw,
                    fmt=ex.fmt,
                    group=ex.group,
                    extra=ex.extra,
                    doc=ex.doc,
                    kind="decoded",
                )
            )
    return out, stats


def _contains(piece: str, answer: str) -> bool:
    """띄어쓰기·대소문자·전각 차이는 무시하고 answer 가 조각 안에 있나"""
    return _compact(answer) in _compact(piece)


def _compact(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).casefold().split())


def split_examples(
    examples: Iterable[Example], val_ratio: float = 0.15, seed: int = SEED
) -> tuple[list[Example], list[Example]]:
    """(train, val) 로 나눈다. group 단위로 해시를 써서 나누므로, 같은 데이터·같은 seed 면
    언제 돌려도 같은 결과가 나오고 같은 group 은 항상 같은 쪽에 들어간다."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio 는 0 과 1 사이여야 합니다: {val_ratio}")
    train, val = [], []
    for ex in examples:
        (val if _in_val(ex.group, val_ratio, seed) else train).append(ex)
    return train, val


def _in_val(group: str, ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < ratio


def label_counts(examples: Iterable[Example]) -> dict[str, int]:
    counts = dict.fromkeys(LABEL2ID, 0)
    for ex in examples:
        counts[ex.label] += 1
    return counts
