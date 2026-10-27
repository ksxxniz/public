MODELS = {
    "exaone": {"name": "exaone3.5:7.8b-instruct-q8_0", "think": None,   "use_format": True},
    "llama":  {"name": "llama3.1:8b-instruct-q8_0",    "think": None,   "use_format": True},
    "qwen":   {"name": "qwen3.5:9b-q8_0",                "think": False,  "use_format": True},
    "safeguard": {"name": "gpt-oss-safeguard:20b",     "think": "low",  "use_format": True},
    "kanana": {"name": "cookieshake/kanana-1.5-8b-instruct-2505:Q8_0", "think": None, "use_format": True},
}
PROMPT_VERSION = "v0.2"
OPTIONS = {"temperature": 0, "num_ctx": 8192}