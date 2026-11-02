import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from common.config import SUPPORTED_EXTENSIONS
from common.schema import Chunk
from preprocess.chunk import split_document
from preprocess.context import DocumentTooLongError
from preprocess.html_parse import HtmlParserUnavailableError
from preprocess.load import UnsupportedDocumentError, read_document
from preprocess.normalize import VERSION
from preprocess.tokens import TokenizerLoadError

DELIBERATE = frozenset(
    """display_none visibility_hidden opacity_zero font_size_zero transparent_color same_color
    offscreen zero_size_box scale_zero content_visibility_hidden hidden_attr dialog_closed
    svg_non_text""".split()
)
COMMENTS = frozenset({"comment", "cdata", "md_comment"})

SIGNALS: tuple[tuple[str, str], ...] = (
    ("tag_chars_decoded", "강함"),
    ("zero_width_removed", "강함"),
    ("hangul_filler_removed", "강함"),
    ("homoglyph_replaced", "강함"),
    ("variation_selector_removed", "강함"),
    ("jamo_assembled", "강함"),
    ("hidden_deliberate", "강함"),
    ("exfil_image", "강함"),
    ("hidden_comment", "약함"),
    ("decoded_count", "약함"),
    ("bidi_removed", "약함"),
    ("nfkc_changed", "약함"),
    ("emoji_zwj", "정상"),
)
STRONG = frozenset(name for name, strength in SIGNALS if strength == "강함")


@dataclass
class Sample:
    name: str
    raw: str
    fmt: str | None


@dataclass
class ChunkSignals:
    doc: str
    chunk_id: str
    length: int
    counts: dict[str, int]
    preview: str

    @property
    def strong(self) -> list[str]:
        return [name for name in sorted(STRONG) if self.counts.get(name)]


@dataclass
class Baseline:
    docs: int = 0
    chunks: list[ChunkSignals] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def chunk_signals(doc: str, chunk: Chunk) -> ChunkSignals:
    counts = {k: v for k, v in chunk.transform_log.model_dump().items() if v}
    for ts in chunk.transform_spans:
        if ts.kind != "hidden_text_found":
            continue
        if ts.note in DELIBERATE:
            counts["hidden_deliberate"] = counts.get("hidden_deliberate", 0) + 1
        elif ts.note in COMMENTS:
            counts["hidden_comment"] = counts.get("hidden_comment", 0) + 1
        elif ts.note == "exfil_image":
            counts["exfil_image"] = counts.get("exfil_image", 0) + 1
    preview = " ".join(chunk.raw_text.split())[:120]
    return ChunkSignals(doc, chunk.chunk_id, len(chunk.raw_text), counts, preview)


def measure(samples: Iterable[Sample]) -> Baseline:
    out = Baseline()
    for sample in samples:
        out.docs += 1
        try:
            chunks = split_document(Path(sample.name).stem or "doc", sample.raw, sample.fmt)
        except (DocumentTooLongError, HtmlParserUnavailableError, TokenizerLoadError) as e:
            out.errors.append((sample.name, str(e)))
            continue
        out.chunks.extend(chunk_signals(sample.name, c) for c in chunks)
    return out


def read_samples(paths: Iterable[Path], errors: list[tuple[str, str]]) -> Iterator[Sample]:
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for f in files:
            if f.suffix.lower() == ".jsonl":
                yield from _jsonl_samples(f, errors)
            elif f.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    doc = read_document(f)
                except (UnsupportedDocumentError, DocumentTooLongError, OSError) as e:
                    errors.append((str(f), str(e)))
                    continue
                yield Sample(str(f), doc.raw, doc.fmt)


def _jsonl_samples(path: Path, errors: list[tuple[str, str]]) -> Iterator[Sample]:
    with path.open(encoding="utf-8") as fh:
        for k, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((f"{path}:{k}", f"JSON 을 읽을 수 없습니다 ({e.msg})"))
                continue
            if record.get("label") == "injection" or not isinstance(record.get("text"), str):
                continue
            name = f"{path}:{record.get('id', k)}"
            yield Sample(name, record["text"], record.get("fmt"))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def render(result: Baseline, top: int = 20) -> str:
    chunks = result.chunks
    n = len(chunks)
    flagged = [c for c in chunks if c.strong]
    rate = len(flagged) / n * 100 if n else 0.0
    lines = [
        f"# 정상 문서 기준선 (정규화 {VERSION})",
        "",
        f"문서 {result.docs}개, 청크 {n}개, 읽지 못한 문서 {len(result.errors)}개",
        "",
        f"강한 신호가 하나라도 켜진 청크: {len(flagged)}개 ({rate:.1f}%)",
        "",
        "| 신호 | 세기 | 켜진 청크 | 비율 | 1000자당 p50 | p95 | p99 | 최댓값 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, strength in SIGNALS:
        per_k = [c.counts.get(name, 0) / max(c.length, 1) * 1000 for c in chunks] or [0.0]
        on = sum(1 for c in chunks if c.counts.get(name))
        share = on / n * 100 if n else 0.0
        lines.append(
            f"| {name} | {strength} | {on} | {share:.1f}% | {_percentile(per_k, 0.5):.2f} "
            f"| {_percentile(per_k, 0.95):.2f} | {_percentile(per_k, 0.99):.2f} "
            f"| {max(per_k):.2f} |"
        )
    if flagged:
        lines += [
            "",
            f"## 강한 신호가 켜진 청크 (상위 {min(top, len(flagged))}개)",
            "",
            "| 문서 | 청크 | 신호 | 원문 앞부분 |",
            "|---|---|---|---|",
        ]
        ranked = sorted(flagged, key=lambda c: (-len(c.strong), -sum(c.counts.values())))
        for c in ranked[:top]:
            signals = ", ".join(f"{k} {c.counts[k]}" for k in c.strong)
            preview = c.preview.replace("|", "\\|")
            lines.append(f"| {c.doc} | {c.chunk_id} | {signals} | {preview} |")
    if result.errors:
        lines += ["", "## 읽지 못한 문서", ""]
        lines += [f"- {name}: {why}" for name, why in result.errors]
    return "\n".join(lines) + "\n"


def write_flagged(result: Baseline, path: Path) -> int:
    flagged = [c for c in result.chunks if c.strong]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for c in flagged:
            record = {
                "doc": c.doc,
                "chunk_id": c.chunk_id,
                "strong": c.strong,
                "counts": c.counts,
                "preview": c.preview,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(flagged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="정상 문서의 전처리 신호 분포와 오탐률")
    parser.add_argument("paths", nargs="+", type=Path, help="정상 문서 폴더·파일 또는 JSONL")
    parser.add_argument("--out", type=Path, help="표를 저장할 Markdown 파일")
    parser.add_argument("--flagged", type=Path, help="강한 신호가 켜진 청크를 저장할 JSONL")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    errors: list[tuple[str, str]] = []
    result = measure(read_samples(args.paths, errors))
    result.errors = errors + result.errors
    if not result.docs:
        print("오류: 읽을 수 있는 정상 문서가 없습니다", file=sys.stderr)
        return 2
    report = render(result, args.top)
    if args.out:
        args.out.write_text(report, encoding="utf-8", newline="\n")
    if args.flagged:
        write_flagged(result, args.flagged)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
