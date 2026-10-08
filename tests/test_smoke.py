"""저장소가 일단 돌아가는지 확인하는 최소 테스트.

각 담당자가 맡은 영역에 테스트를 추가하면 이 파일은 그대로 둬도 되고,
더 적절한 파일로 옮겨도 된다.
"""

import sys

import pytest

from common import examples
from common.logdb import connect, escalation_rate, log_verdict
from pipeline.run import main as run_main
from preprocess.chunk import split_document


def test_examples_build_without_error():
    assert examples.CHUNK.chunk_id == "example-c0"
    assert examples.VERDICT.final_label == "injection"


def test_logdb_roundtrip():
    conn = connect(":memory:")
    n = log_verdict(conn, examples.VERDICT, request_id="r1", doc_id="example", policy="default")
    assert n == len(examples.VERDICT.stage_results) + 1  # preprocess 줄 포함
    assert escalation_rate(conn) == 1.0


def test_split_document_short_text_is_one_chunk():
    chunks = split_document("doc1", "hello world")
    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"


# ── pipeline.run ──


def _run(monkeypatch, path):
    monkeypatch.setattr(sys, "argv", ["pipeline.run", "--file", str(path)])
    run_main()


def test_run_prints_chunks(monkeypatch, capsys, tmp_path):
    path = tmp_path / "rule.txt"
    path.write_text("연차는 1년에 15일입니다.", encoding="utf-8")
    _run(monkeypatch, path)
    assert "rule (txt, utf-8): 1개 청크" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("없는파일.txt", "파일이 없습니다"),
        ("폴더.txt", "파일을 읽을 수 없습니다"),
        ("a.pdf", "지원하지 않는"),
    ],
)
def test_run_reports_errors_without_traceback(monkeypatch, capsys, tmp_path, name, message):
    if name == "폴더.txt":
        (tmp_path / name).mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path / name)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_run_reports_tokenizer_failure_not_missing_file(monkeypatch, capsys, tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("가나다라마바사 " * 30, encoding="utf-8")
    monkeypatch.setenv("PREPROCESS_TOKENIZER", str(tmp_path / "없는토크나이저.json"))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, path)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "토크나이저" in err and "파일이 없습니다" not in err
