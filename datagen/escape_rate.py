import argparse
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from common import config
from datagen.evasion import CONCEALMENT, DEMO_SAMPLES, TECHNIQUES, _read_jsonl, generate
from preprocess.normalize import normalize_document, normalize_text

SIGNALS = (
    "zero_width_removed",
    "bidi_removed",
    "variation_selector_removed",
    "homoglyph_replaced",
    "nfkc_changed",
    "decoded_count",
)


def revealed(record: dict) -> bool:
    doc = normalize_document(record["text"], record.get("fmt", "txt"))
    target = normalize_text(record["answer"])
    if not target:
        return False
    if record["concealment"] == "visual":
        pos = doc.text.find(target)
        if pos < 0:
            return False
        r0, r1 = doc.starts[pos], doc.ends[pos + len(target) - 1]
        return any(
            ev.kind == "hidden_text_found" and ev.start < r1 and ev.end > r0 for ev in doc.events
        )
    if target in doc.text:
        return True
    return any(target in normalize_text(seg.decoded) for seg in doc.decoded_segments)


def signals(record: dict) -> Counter:
    doc = normalize_document(record["text"], record.get("fmt", "txt"))
    counts = Counter(ev.kind for ev in doc.events)
    counts["decoded_count"] = len(doc.decoded_segments)
    return counts


@dataclass
class Row:
    key: str
    concealment: str
    stage: str
    ready: bool
    total: int = 0
    shown: int = 0

    @property
    def escape_rate(self) -> float:
        return 1 - self.shown / self.total if self.total else 0.0


def escape_rows(variants: Iterable[dict]) -> list[Row]:
    rows: dict[str, Row] = {}
    for v in variants:
        if v.get("hard_negative"):
            continue
        key = "+".join(v["techniques"])
        if key not in rows:
            techs = [TECHNIQUES[n] for n in v["techniques"]]
            rows[key] = Row(
                key=key,
                concealment=v["concealment"],
                stage="".join(dict.fromkeys(t.stage for t in techs)),
                ready=all(t.ready for t in techs),
            )
        row = rows[key]
        row.total += 1
        row.shown += revealed(v)
    return sorted(rows.values(), key=lambda r: (not r.ready, "+" in r.key, r.stage, r.key))


def benign_rows(variants: Iterable[dict]) -> dict[str, dict[str, float]]:
    seen: dict[str, list[Counter]] = {}
    for v in variants:
        if v.get("hard_negative"):
            seen.setdefault(v["techniques"][0], []).append(signals(v))
    return {
        name: {s: sum(1 for c in cs if c[s] > 0) / len(cs) for s in SIGNALS}
        for name, cs in seen.items()
    }


def render(rows: list[Row], benign: dict[str, dict[str, float]]) -> str:
    lines = [
        "## 기법별 탈출률",
        "",
        "| 기법 | 축 2 | 단계 | 상태 | 샘플 | 드러남 | 탈출률 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        state = "구현" if r.ready else "3주차 예정"
        lines.append(
            f"| {r.key} | {CONCEALMENT[r.concealment]} | {r.stage} | {state} "
            f"| {r.total} | {r.shown} | {r.escape_rate:.0%} |"
        )
    ready = [r for r in rows if r.ready]
    total = sum(r.total for r in ready)
    leaked = sum(r.total - r.shown for r in ready)
    lines += [
        "",
        f"구현된 기법 {len(ready)}개 (조합 포함): 샘플 {total}개 중 {leaked}개 탈출 "
        f"(탈출률 {leaked / total if total else 0:.0%}, 목표 0%)",
    ]
    if benign:
        lines += [
            "",
            "## 정상 변형에서 켜진 신호 (샘플 비율)",
            "",
            "| 정상 변형 | " + " | ".join(SIGNALS) + " |",
            "|---|" + "---:|" * len(SIGNALS),
        ]
        for name, rates in benign.items():
            lines.append(f"| {name} | " + " | ".join(f"{rates[s]:.0%}" for s in SIGNALS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="기법별 탈출률 표")
    parser.add_argument("--input", type=Path, help="원본 JSONL (없으면 내장 예시)")
    parser.add_argument("--out", type=Path, help="표를 저장할 Markdown 파일")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--fail-on-escape", action="store_true", help="구현된 기법이 하나라도 새면 종료 코드 1"
    )
    args = parser.parse_args(argv)
    config.set_seed(args.seed)

    samples = _read_jsonl(args.input) if args.input else DEMO_SAMPLES
    variants = generate(samples, args.seed)
    rows = escape_rows(variants)
    report = render(rows, benign_rows(variants))
    if args.out:
        args.out.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    if args.fail_on_escape and any(r.ready and r.shown < r.total for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
