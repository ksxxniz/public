"""jsonl 데이터셋을 문서 단위로 판정하고 결과를 저장한다.

실행: uv run python -m stage2_llm.run --model exaone [--data 경로.jsonl] [--limit 5]
데이터 한 줄: {"id": "...", "text": "원문", "label": "benign"|"injection",
              "lang": "en"|"ko", "type": "...", "fmt": "txt"|"md"|"html"}
lang·type·fmt 는 없어도 된다. 문서 판정 = 청크 중 하나라도 injection 이면 injection.

결과: stage2_llm/results/{모델}_{프롬프트버전}.jsonl  (문서 한 줄씩)
      stage2_llm/results/{모델}_{프롬프트버전}.meta.json  (VRAM 등)
표 만들기: uv run python -m stage2_llm.report
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import ollama

from common.config import set_seed
from stage2_llm.config import MODELS, PROMPT_VERSION
from stage2_llm.judge import judge_document

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
DEFAULT_DATA = HERE / "evalsets" / "eval_v0.1.jsonl"
LABEL_ALIASES = {"malicious": "injection"}  # 예전 샘플 호환


def vram_gb(model_name: str) -> float | None:
    """지금 GPU 에 올라가 있는 이 모델의 VRAM(GB). 못 읽으면 None"""
    try:
        for m in ollama.ps().models:
            if m.model == model_name or m.name == model_name:
                return round(m.size_vram / 1024**3, 2)
    except Exception:
        return None
    return None


def main() -> None:
    set_seed()
    parser = argparse.ArgumentParser(description="2차 LLM 판정 실험")
    parser.add_argument("--model", default="exaone", choices=sorted(MODELS))
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="jsonl 데이터 경로")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 몇 개만")
    args = parser.parse_args()

    profile = MODELS[args.model]
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[: args.limit] if args.limit else rows

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{args.model}_{PROMPT_VERSION}"
    out_path = RESULTS_DIR / f"{stem}.jsonl"

    judge_document("warmup", "hello", profile)  # 모델을 GPU 에 미리 올림 (저장 안 함)
    vram = vram_gb(profile["name"])

    correct = 0
    started = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            gold = LABEL_ALIASES.get(row["label"], row["label"])
            judged = judge_document(row["id"], row["text"], profile, fmt=row.get("fmt", "txt"))
            pred = "injection" if any(r.label == "injection" for _, r, _ in judged) else "benign"
            correct += pred == gold
            record = {
                "id": row["id"],
                "lang": row.get("lang"),
                "type": row.get("type"),
                "gold_label": gold,
                "gold_attack_type": row.get("attack_type"),
                "pred_label": pred,
                "model": profile["name"],
                "prompt_version": PROMPT_VERSION,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "result": r.model_dump(),
                        "info": asdict(i),
                        "highlight": (
                            c.raw_text[r.evidence_span.start : r.evidence_span.end]
                            if r.evidence_span
                            else None
                        ),
                    }
                    for c, r, i in judged
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            latency = sum(r.latency_ms for _, r, _ in judged)
            mark = "✅" if pred == gold else "❌"
            print(f"{mark} {row['id']}  예측={pred}  정답={gold}  {latency:.0f}ms")

    meta = {
        "model_key": args.model,
        "model": profile["name"],
        "think": profile.get("think"),
        "prompt_version": PROMPT_VERSION,
        "data": str(args.data),
        "n_docs": len(rows),
        "vram_gb": vram,
        "elapsed_s": round(time.time() - started, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_DIR / f"{stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    vram_text = f"{vram}GB" if vram is not None else "확인 못 함"
    print(f"정답 {correct}/{len(rows)} · VRAM {vram_text} · 저장: {out_path}")


if __name__ == "__main__":
    main()
