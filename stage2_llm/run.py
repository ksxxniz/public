import json, sys
from pathlib import Path
from config import MODELS, PROMPT_VERSION
from judge import judge

BASE = Path(__file__).parent

model_key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
profile = MODELS[model_key]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

data_path = BASE / "data" / "sample.jsonl"
rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]
rows = rows[:limit] if limit else rows

results_dir = BASE / "results"
results_dir.mkdir(exist_ok=True)
out_path = results_dir / f"{model_key}_{PROMPT_VERSION}.jsonl"

judge("warmup", "hello", profile)   # 모델을 GPU에 미리 올리기 (결과 저장 안 함)

with open(out_path, "w", encoding="utf-8") as out:
    for row in rows:
        rec = judge(row["id"], row["text"], profile)
        rec["gold_label"] = row["label"]
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "✅" if rec["final_label"] == row["label"] else "❌"
        print(f'{mark} {row["id"]}  예측={rec["final_label"]}  정답={row["label"]}  {rec["latency_ms"]}ms')

print("저장:", out_path)