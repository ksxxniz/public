import os

import pytest

os.environ.setdefault("PREPROCESS_TOKENIZER", "char")

_HOW = "실제 토크나이저를 내려받는 테스트: uv run pytest -m real_tokenizer"
_HOW_MODEL = (
    "모델 파일·서버가 필요한 진짜 부품 테스트: GPU 환경에서 uv run pytest -m needs_model "
    "(5단계 ⑧ — CI 는 스텁으로만 돈다)"
)


def pytest_configure(config):
    config.addinivalue_line("markers", f"real_tokenizer: {_HOW}")
    config.addinivalue_line("markers", f"needs_model: {_HOW_MODEL}")


def pytest_collection_modifyitems(config, items):
    expr = config.option.markexpr or ""
    for name, how in (("real_tokenizer", _HOW), ("needs_model", _HOW_MODEL)):
        if name in expr:
            continue
        skip = pytest.mark.skip(reason=how)
        for item in items:
            if name in item.keywords:
                item.add_marker(skip)
