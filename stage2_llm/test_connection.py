"""Ollama 연결 확인. 실행: uv run python -m stage2_llm.test_connection exaone"""

import sys

import ollama

from stage2_llm.config import MODELS


def main() -> None:
    for m in ollama.list().models:
        print("설치됨:", m.model)
    model_key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    resp = ollama.chat(
        model=MODELS[model_key]["name"],
        messages=[{"role": "user", "content": "안녕? 한 문장으로 자기소개 해줘."}],
        options={"temperature": 0},
    )
    print(resp["message"]["content"])
    print("읽은 토큰 수:", resp["prompt_eval_count"])


if __name__ == "__main__":
    main()
