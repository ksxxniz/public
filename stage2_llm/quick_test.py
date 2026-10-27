import json, sys
from config import MODELS
from judge import judge

model_key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
profile = MODELS[model_key]
END = "END"   # 이 단어만 있는 줄이 나오면 입력 끝

print(f"[{model_key}] 문서를 붙여넣고, 마지막 줄에 {END} 입력 후 Enter. 종료: Ctrl+C\n")

while True:
    lines = []
    while True:
        line = input()
        if line.strip() == END:
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        print("내용이 비어 있어요. 다시 붙여넣으세요.\n")
        continue

    print("판정 중...")
    rec = judge("quick", text, profile)
    print("\n" + "=" * 50)
    print(json.dumps(rec.get("judge") or rec.get("raw"), ensure_ascii=False, indent=2))
    print(f'최종: {rec["final_label"]} | {rec["latency_ms"]}ms | {rec["prompt_tokens"]}토큰')
    print("=" * 50 + f"\n다음 문서를 붙여넣고 {END}로 끝내세요. 종료: Ctrl+C\n")

    # python quick_test.py 모델명