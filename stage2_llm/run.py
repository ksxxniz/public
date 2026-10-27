"""jsonl 데이터셋을 문서 단위로 판정하고 결과를 저장한다.

실행: uv run python -m stage2_llm.run --model exaone --data 경로.jsonl [--limit 5]
데이터 한 줄: {"id": "kr_0001", "text": "...", "label": "benign" | "injection"}
문서 판정 = 청크 중 하나라도 injection 이면 injection.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from common.config import set_seed
from stage2_llm.config import MODELS, PROMPT_VERSION
from stage2_llm.judge import judge_document

RESULTS_DIR = Path(__file__).parent / "results"
LABEL_ALIASES = {"malicious": "injection"}  # 예전 샘플 호환


def main() -> None:
    set_seed()
    parser = argparse.ArgumentParser(description="2차 LLM 판정 실험")
    parser.add_argument("--model", default="exaone", choices=sorted(MODELS))
    parser.add_argument("--data", required=True, help="jsonl 데이터 경로")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 몇 개만")
    args = parser.parse_args()

    profile = MODELS[args.model]
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[: args.limit] if args.limit else rows

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.model}_{PROMPT_VERSION}.jsonl"

    judge_document("warmup", "hello", profile)  # 모델을 GPU 에 미리 올림 (저장 안 함)

    correct = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            gold = LABEL_ALIASES.get(row["label"], row["label"])
            judged = judge_document(row["id"], row["text"], profile)
            pred = "injection" if any(r.label == "injection" for _, r, _ in judged) else "benign"
            correct += pred == gold
            record = {
                "id": row["id"],
                "gold_label": gold,
                "pred_label": pred,
                "model": profile["name"],
                "prompt_version": PROMPT_VERSION,
                "chunks": [
                    {"chunk_id": c.chunk_id, "result": r.model_dump(), "info": asdict(i)}
                    for c, r, i in judged
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            latency = sum(r.latency_ms for _, r, _ in judged)
            mark = "✅" if pred == gold else "❌"
            print(f"{mark} {row['id']}  예측={pred}  정답={gold}  {latency:.0f}ms")

    print(f"정답 {correct}/{len(rows)} · 저장: {out_path}")


if __name__ == "__main__":
    main()
