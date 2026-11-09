import argparse
import base64
import platform
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

from preprocess.chunk import split_document
from preprocess.normalize import VERSION, normalize_document
from preprocess.tokens import TokenizerLoadError, get_tokenizer

SIZES = (1_000, 10_000, 100_000, 1_000_000)
FORMATS = ("txt", "md", "html")

_KO = (
    "연차는 1년에 15일이며 신청은 인사팀 포털에서 합니다.",
    "출장비 정산은 귀국 후 7일 안에 영수증과 함께 제출해 주세요.",
    "보안 교육은 매 분기 첫째 주에 온라인으로 진행됩니다.",
    "회의실 예약은 그룹웨어 일정 메뉴에서 할 수 있습니다.",
    "신규 입사자는 첫 주에 장비 수령과 계정 발급을 마칩니다.",
)
_EN = (
    "Please submit the annual leave request to your manager.",
    "The quarterly report is due at the end of the month.",
    "See https://example.com/docs?page=2&lang=ko for details.",
    "Version 2.3.1 fixes the login timeout reported last week.",
    "The API token looks like " + base64.b64encode(b"example token for docs").decode() + ".",
)


def _para(rng: random.Random) -> str:
    return " ".join(rng.choice(_KO + _EN) for _ in range(rng.randint(2, 5)))


def _txt(size: int, rng: random.Random) -> str:
    out: list[str] = []
    total = 0
    while total < size:
        out.append(_para(rng))
        total += len(out[-1].encode("utf-8")) + 2
    return "\n\n".join(out)


def _md(size: int, rng: random.Random) -> str:
    out = ["# 사내 안내\n"]
    total = 0
    while total < size:
        k = rng.random()
        if k < 0.6:
            out.append(_para(rng) + "\n")
        elif k < 0.75:
            out.append(f"- {rng.choice(_KO)}\n- [{rng.choice(_EN)}](https://example.com)\n")
        elif k < 0.85:
            out.append(f"![{rng.choice(_KO)}](img/a.png)\n")
        elif k < 0.95:
            out.append(f"## {rng.choice(_KO)}\n")
        else:
            out.append("```python\nprint('hello')\n```\n")
        total += len(out[-1].encode("utf-8")) + 1
    return "\n".join(out)


def _html(size: int, rng: random.Random) -> str:
    out = [
        "<!DOCTYPE html><html><head><title>사내 안내</title><style>p{margin:0}</style></head><body>"
    ]
    total = len(out[0])
    while total < size:
        k = rng.random()
        if k < 0.6:
            out.append(f"<p>{_para(rng)}</p>")
        elif k < 0.75:
            out.append(f'<div class="card"><a href="/x" title="{rng.choice(_KO)}">링크</a></div>')
        elif k < 0.9:
            out.append(f"<ul><li>{rng.choice(_KO)}</li><li>{rng.choice(_EN)}</li></ul>")
        else:
            out.append(f'<img src="a.png" alt="{rng.choice(_KO)}"><br>')
        total += len(out[-1].encode("utf-8"))
    out.append("</body></html>")
    return "".join(out)


_MAKERS: dict[str, Callable[[int, random.Random], str]] = {"txt": _txt, "md": _md, "html": _html}


def sample_document(fmt: str, size: int, seed: int = 7) -> str:
    return _MAKERS[fmt](size, random.Random(f"{seed}|{fmt}|{size}"))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _repeats(size: int) -> int:
    return 50 if size <= 1_000 else 30 if size <= 10_000 else 10 if size <= 100_000 else 5


def _label(size: int) -> str:
    if size >= 1_000_000 and size % 1_000_000 == 0:
        return f"{size // 1_000_000}MB"
    if size >= 1_000 and size % 1_000 == 0:
        return f"{size // 1_000}KB"
    return f"{size}B"


def measure(fmt: str, size: int, repeats: int | None = None) -> dict[str, float | int]:
    raw = sample_document(fmt, size)
    norm: list[float] = []
    total: list[float] = []
    chunks = 0
    for _ in range(repeats or _repeats(size)):
        started = time.perf_counter()
        normalize_document(raw, fmt)
        norm.append(time.perf_counter() - started)
        started = time.perf_counter()
        chunks = len(split_document("perf", raw, fmt, max_chunks=10**9))
        total.append(time.perf_counter() - started)
    return {
        "chars": len(raw),
        "chunks": chunks,
        "norm_p50": _percentile(norm, 0.5) * 1000,
        "norm_p95": _percentile(norm, 0.95) * 1000,
        "total_p50": _percentile(total, 0.5) * 1000,
        "total_p95": _percentile(total, 0.95) * 1000,
    }


def render(rows: list[tuple[str, int, dict[str, float | int]]], tokenizer: str) -> str:
    lines = [
        f"정규화 {VERSION}, Python {platform.python_version()}, 토크나이저 {tokenizer}, "
        f"{platform.machine() or '알 수 없는 CPU'}",
        "",
        "| 형식 | 크기 | 글자 수 | 청크 | 정규화 p50 | 정규화 p95 | 전체 p50 | 전체 p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fmt, size, r in rows:
        lines.append(
            f"| {fmt} | {_label(size)} | {r['chars']} | {r['chunks']} "
            f"| {r['norm_p50']:.0f}ms | {r['norm_p95']:.0f}ms "
            f"| {r['total_p50']:.0f}ms | {r['total_p95']:.0f}ms |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="전처리 처리 시간 측정 (p50/p95)")
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)), help="바이트, 쉼표로 구분")
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--repeats", type=int, help="크기마다 반복 횟수 (기본: 크기별로 다름)")
    parser.add_argument("--out", type=Path, help="표를 저장할 Markdown 파일")
    args = parser.parse_args(argv)
    try:
        tokenizer = get_tokenizer().name
    except TokenizerLoadError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    rows = []
    for fmt in args.formats.split(","):
        for size in map(int, args.sizes.split(",")):
            rows.append((fmt, size, measure(fmt, size, args.repeats)))
            print(f"{fmt} {_label(size)} 측정 완료", file=sys.stderr)
    table = render(rows, tokenizer)
    if args.out:
        args.out.write_text(table, encoding="utf-8", newline="\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
