"""2차 LLM 판정 설정.

모델 이름은 `ollama list` 에 나온 이름 그대로 쓴다.
프롬프트를 고치면 prompts/system_vX.Y.txt 를 새로 만들고 PROMPT_VERSION 을 올린다.
"""

from common.config import SEED

# think: None = 생각 기능 없는 모델, False = 끄기, "low"/"medium"/"high" = 추론 수준
# use_format: Ollama 의 JSON 양식 강제 사용 여부
MODELS = {
    "exaone": {"name": "exaone3.5:7.8b-instruct-q8_0", "think": None, "use_format": True},
    "llama": {"name": "llama3.1:8b-instruct-q8_0", "think": None, "use_format": True},
    "qwen": {"name": "qwen3.5:9b-q8_0", "think": False, "use_format": True},
    "safeguard": {"name": "gpt-oss-safeguard:20b", "think": "low", "use_format": True},
    "kanana": {
        "name": "cookieshake/kanana-1.5-8b-instruct-2505:Q8_0",
        "think": None,
        "use_format": True,
    },
}

PROMPT_VERSION = "v0.3"
OPTIONS = {"temperature": 0, "num_ctx": 8192, "seed": SEED}

# JSON 이 깨졌을 때 다시 묻는 횟수. 그래도 못 읽으면 fail-closed(차단)
MAX_RETRIES = 1

# LLM 이 고른 확신도 단어 → 숫자. 임시 방식이며 logprob 기반 점수로 바꿀 예정
CONFIDENCE_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}
