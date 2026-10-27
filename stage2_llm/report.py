"""results/ 의 실험 결과를 모아 모델 비교표(마크다운)를 만든다.

실행: uv run python -m stage2_llm.report [--prompt v0.3] [--out 파일.md]
run.py 로 모델마다 한 번씩 돌린 뒤 실행한다. 같은 프롬프트 버전끼리만 비교한다.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from stage2_llm.config import PROMPT_VERSION

RESULTS_DIR = Path(__file__).parent / "results"

LICENSES = {
    "exaone": "EXAONE AI Model License (비상업)",
    "kanana": "Apache 2.0",
    "llama": "Llama 3.1 Community",
    "qwen": "Apache 2.0",
    "safeguard": "Apache 2.0",
}


def pct(num: int, den: int) -> str:
    return "-" if den == 0 else f"{num / den:.0%} ({num}/{den})"


def p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=20, method="inclusive")[-1]


def load(prompt_version: str) -> dict[str, tuple[list[dict], dict]]:
    runs = {}
    for path in sorted(RESULTS_DIR.glob(f"*_{prompt_version}.jsonl")):
        key = path.name[: -len(f"_{prompt_version}.jsonl")]
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
        meta_path = path.with_name(f"{key}_{prompt_version}.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        runs[key] = (docs, meta)
    return runs


def summarize(docs: list[dict]) -> dict:
    inj = [d for d in docs if d["gold_label"] == "injection"]
    ben = [d for d in docs if d["gold_label"] == "benign"]
    hit = [d for d in inj if d["pred_label"] == "injection"]
    fp = [d for d in ben if d["pred_label"] == "injection"]

    def recall(lang: str) -> str:
        xs = [d for d in inj if d.get("lang") == lang]
        return pct(sum(d["pred_label"] == "injection" for d in xs), len(xs))

    hn = [d for d in docs if d.get("type") == "hard_negative"]
    chunks = [c for d in docs for c in d["chunks"]]
    parsed = [c for c in chunks if c["info"]["parse_ok"]]
    first_try = [c for c in parsed if c["info"]["attempts"] == 1]
    lat = [c["result"]["latency_ms"] for c in chunks]

    # 근거: 공격을 잡은 문서에서, 인용문이 있는 청크만 본다
    quoted = [
        c
        for d in hit
        for c in d["chunks"]
        if c["result"]["label"] == "injection" and c["info"]["quote_found"] is not None
    ]
    methods = Counter(c["info"]["quote_method"] or "not_found" for c in quoted)

    return {
        "recall": pct(len(hit), len(inj)),
        "recall_en": recall("en"),
        "recall_ko": recall("ko"),
        "fpr": pct(len(fp), len(ben)),
        "hard_negative": pct(sum(d["pred_label"] == "benign" for d in hn), len(hn)),
        "json_ok": pct(len(parsed), len(chunks)),
        "json_first_try": pct(len(first_try), len(chunks)),
        "quote_exact": pct(methods["exact"], len(quoted)),
        "quote_methods": dict(methods),
        "lat_mean": statistics.mean(lat) if lat else 0.0,
        "lat_p95": p95(lat),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
    }


def by_type(runs: dict) -> list[str]:
    types = sorted({d.get("type") or "-" for docs, _ in runs.values() for d in docs})
    keys = list(runs)
    lines = ["| 유형 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for t in types:
        cells = []
        for k in keys:
            xs = [d for d in runs[k][0] if (d.get("type") or "-") == t]
            cells.append(pct(sum(d["pred_label"] == d["gold_label"] for d in xs), len(xs)))
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


def mistakes(runs: dict) -> list[str]:
    lines = []
    for k, (docs, _) in runs.items():
        wrong = [d for d in docs if d["pred_label"] != d["gold_label"]]
        if not wrong:
            continue
        lines.append(f"\n**{k}** ({len(wrong)}건)\n")
        for d in wrong:
            reasons = [
                (c["info"].get("llm") or {}).get("reason") or "(답변 못 읽음)" for c in d["chunks"]
            ]
            lines.append(
                f"- `{d['id']}` ({d.get('type')}) 정답={d['gold_label']} "
                f"예측={d['pred_label']} — {reasons[-1]}"
            )
    return lines


def build(prompt_version: str) -> str:
    runs = load(prompt_version)
    if not runs:
        return f"results/ 에 *_{prompt_version}.jsonl 결과가 없습니다. run.py 를 먼저 돌리세요."

    head = (
        "| 모델 | 탐지율 | 탐지율(영어) | 탐지율(한국어) | 오탐률 | 하드 네거티브 정답률 "
        "| JSON 성공률 | 첫 시도 성공률 | 근거 exact | 평균 지연 | p95 지연 | VRAM | 라이선스 |"
    )
    lines = [
        f"# 2차 LLM 모델 비교 (프롬프트 {prompt_version})",
        "",
        head,
        "|---|" + "---|" * 12,
    ]
    for key, (docs, meta) in runs.items():
        s = summarize(docs)
        vram = f"{meta['vram_gb']}GB" if meta.get("vram_gb") else "-"
        lines.append(
            f"| {key} | {s['recall']} | {s['recall_en']} | {s['recall_ko']} | {s['fpr']} "
            f"| {s['hard_negative']} | {s['json_ok']} | {s['json_first_try']} "
            f"| {s['quote_exact']} | {s['lat_mean']:.0f}ms | {s['lat_p95']:.0f}ms | {vram} "
            f"| {LICENSES.get(key, '-')} |"
        )

    lines += ["", "## 유형별 정답률", "", *by_type(runs)]
    lines += ["", "## 근거 인용 방식 (공격을 잡은 청크)", ""]
    for key, (docs, _) in runs.items():
        lines.append(f"- {key}: {summarize(docs)['quote_methods']}")
    lines += ["", "## 틀린 문서", *mistakes(runs)]
    lines += [
        "",
        "## 읽는 법",
        "",
        "- 탐지율: 공격 문서 중 잡은 비율 (높을수록 좋음)",
        "- 오탐률: 정상 문서 중 잘못 막은 비율 (낮을수록 좋음)",
        "- 하드 네거티브 정답률: 공격 문장을 인용·설명하는 정상 문서를 정상으로 본 비율",
        "- JSON 성공률: 재시도 포함 양식을 지킨 청크 비율 (목표 99% 이상)",
        "- 근거 exact: 공격을 잡은 청크 중 근거 문장을 글자 그대로 인용한 비율",
        "- 지연: 청크 하나 판정 시간 (모델 로딩 제외)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="2차 LLM 모델 비교표")
    parser.add_argument("--prompt", default=PROMPT_VERSION, help="비교할 프롬프트 버전")
    parser.add_argument(
        "--out", default=None, help="저장할 .md 경로 (기본: results/report_버전.md)"
    )
    args = parser.parse_args()

    text = build(args.prompt)
    print(text)
    out = Path(args.out) if args.out else RESULTS_DIR / f"report_{args.prompt}.md"
    if RESULTS_DIR.exists():
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
