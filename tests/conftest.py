import os

import pytest

os.environ.setdefault("PREPROCESS_TOKENIZER", "char")

_HOW = "실제 토크나이저를 내려받는 테스트: uv run pytest -m real_tokenizer"


def pytest_configure(config):
    config.addinivalue_line("markers", f"real_tokenizer: {_HOW}")


def pytest_collection_modifyitems(config, items):
    if "real_tokenizer" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason=_HOW)
    for item in items:
        if "real_tokenizer" in item.keywords:
            item.add_marker(skip)
