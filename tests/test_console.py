"""윈도 cp949 출력에서 명령이 죽지 않는지 (common/console.py)."""

import io
import sys

import pytest

from common.console import safe_stdio


def test_cp949_output_survives_unsupported_characters(monkeypatch):
    out_buf, err_buf = io.BytesIO(), io.BytesIO()
    # newline="\n": 윈도에서는 기본값이 \n 을 \r\n 으로 바꿔 써서 비교가 어긋난다
    out = io.TextIOWrapper(out_buf, encoding="cp949", newline="\n")
    err = io.TextIOWrapper(err_buf, encoding="cp949", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    with pytest.raises(UnicodeEncodeError):  # 보호 전: 윈도에서 파일로 돌리면 이렇게 멈춤
        print("차단 — ✓", flush=True)
    out_buf.seek(0)
    out_buf.truncate()

    safe_stdio()
    print("차단 — ✓ █ 끝", flush=True)
    assert out_buf.getvalue().decode("cp949") == "차단 ? ? ? 끝\n"


def test_streams_that_cannot_be_reconfigured_are_left_alone(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)  # pythonw 처럼 출력이 없는 경우
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    safe_stdio()
