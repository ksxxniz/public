"""문서를 붙여넣고 바로 판정해 보는 도구.

실행: uv run python -m stage2_llm.quick_test exaone
문서를 붙여넣고, 마지막 줄에 END 만 입력한 뒤 Enter. 종료: Ctrl+C
"""

import json
import sys

from stage2_llm.config import MODELS
from stage2_llm.judge import highlight, judge_document

END = "END"


def read_document() -> str:
    lines = []
    while True:
        line = input()
        if line.strip() == END:
            return "\n".join(lines).strip()
        lines.append(line)


def main() -> None:
    model_key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    profile = MODELS[model_key]
    print(f"[{model_key}] 문서를 붙여넣고, 마지막 줄에 {END} 입력 후 Enter. 종료: Ctrl+C\n")

    while True:
        text = read_document()
        if not text:
            print("내용이 비어 있어요. 다시 붙여넣으세요.\n")
            continue
        print("판정 중...")
        for chunk, result, info in judge_document("quick", text, profile):
            print("\n" + "=" * 50)
            print(f"[{chunk.chunk_id}]")
            print(json.dumps(info.llm or info.raw, ensure_ascii=False, indent=2))
            print(
                f"판정: {result.label} | 위험도 {result.risk_score:.2f} | {result.latency_ms:.0f}ms"
            )
            if info.quote_found is False:
                print("근거: 원문에서 못 찾음 (LLM 이 문장을 지어냈을 수 있음)")
            elif result.evidence_span is not None:
                s = result.evidence_span
                method = f"{info.quote_method}, {info.quote_score:.2f}"
                print(f"근거 위치: [{s.start}, {s.end}) ({method}) → {highlight(chunk, result)!r}")
        print("=" * 50 + f"\n다음 문서를 붙여넣고 {END}로 끝내세요.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료")
