from pathlib import Path

import pytest
from tokenizers import Tokenizer as RawTokenizer
from tokenizers import models, pre_tokenizers, processors

from common import config
from preprocess.chunk import split_document
from preprocess.normalize import normalize_text
from preprocess.tokenizer_report import inspect_tokenizer, render
from preprocess.tokens import (
    CharTokenizer,
    HFTokenizer,
    TokenizerLoadError,
    content_size,
    count_tokens,
    fits_in_chunk,
    get_tokenizer,
    load_tokenizer,
    token_windows,
)

FAKE = CharTokenizer()


def test_tests_use_fake_tokenizer():
    assert get_tokenizer().name == "char"


def test_content_size_is_384_minus_special_tokens():
    assert content_size(FAKE) == config.CHUNK_SIZE_TOKENS - 2 == 382


def test_short_text_does_not_call_tokenizer():
    class Boom:
        name = "boom"
        num_special_tokens = 2

        def offsets(self, text):
            raise AssertionError("짧은 글은 토크나이저를 부르지 않는다")

    assert token_windows("짧은 문서", Boom()) == [(0, 5)]
    assert token_windows("") == []


def test_windows_respect_size_and_overlap():
    text = " ".join(f"w{k:03d}" for k in range(300))
    windows = token_windows(text, FAKE, size=100, overlap=20)
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    for s, e in windows:
        assert len(FAKE.offsets(text[s:e])) <= 100
    for (_, e1), (s2, _) in zip(windows, windows[1:], strict=False):
        assert s2 < e1
        assert len(FAKE.offsets(text[s2:e1])) == 20


def test_windows_cover_whole_text():
    text = "가나다 라마바\n사아자 " * 40
    windows = token_windows(text, FAKE, size=30, overlap=5)
    covered = [False] * len(text)
    for s, e in windows:
        for k in range(s, e):
            covered[k] = True
    assert all(covered)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        token_windows("가" * 500, FAKE, size=10, overlap=10)


def test_window_shrinks_when_retokenized_text_is_longer():
    class EdgeSensitive(CharTokenizer):
        def offsets(self, text):
            offs = super().offsets(text)
            return offs[:1] + offs if offs and not text.startswith("가") else offs

    text = "가" + "나" * 40
    for s, e in token_windows(text, EdgeSensitive(), size=10, overlap=2):
        assert len(EdgeSensitive().offsets(text[s:e])) <= 10


def test_split_document_cuts_by_tokens():
    raw = "가" * 1000
    chunks = split_document("d", raw, tokenizer=FAKE)
    assert [c.chunk_id for c in chunks] == ["d-c0", "d-c1", "d-c2"]
    assert all(len(FAKE.offsets(c.text)) <= 382 for c in chunks)
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    assert split_document("d", raw) == chunks


def test_count_tokens_is_after_normalization_with_special_tokens():
    assert count_tokens("가나", tokenizer=FAKE) == 2 + 2
    assert count_tokens("가\u200b나", tokenizer=FAKE) == 2 + 2
    assert count_tokens("㈜", tokenizer=FAKE) == 3 + 2


def test_fits_in_chunk_uses_384_with_special_tokens():
    assert fits_in_chunk("가" * 382, tokenizer=FAKE)
    assert not fits_in_chunk("가" * 383, tokenizer=FAKE)


def _tiny_raw() -> RawTokenizer:
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "안녕": 3, "세계": 4}
    tok = RawTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tok.enable_truncation(4)
    return tok


def test_hf_tokenizer_offsets_are_characters():
    hf = HFTokenizer(_tiny_raw(), "tiny")
    text = "😀😀 안녕 세계"
    assert hf.num_special_tokens == 2
    assert [text[s:e] for s, e in hf.offsets(text)] == ["😀😀", "안녕", "세계"]


def test_hf_tokenizer_does_not_truncate():
    assert len(HFTokenizer(_tiny_raw(), "tiny").offsets("안녕 " * 50)) == 50


def test_load_tokenizer_from_json_file(tmp_path: Path):
    path = tmp_path / "tiny.json"
    _tiny_raw().save(str(path))
    tok = load_tokenizer(str(path))
    assert tok.num_special_tokens == 2 and len(tok.offsets("안녕 세계")) == 2


class _StubFastTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offs = CharTokenizer().offsets(text)
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    class backend_tokenizer:
        @staticmethod
        def save(path):
            Path(path).write_text("{}", encoding="utf-8")


def test_tokenizer_report_with_stub(tmp_path: Path):
    def no_tokenizer_json(name):
        raise OSError("tokenizer.json 없음")

    r = inspect_tokenizer("stub/model", _StubFastTokenizer(), tmp_path, no_tokenizer_json)
    assert r.is_fast and r.offsets_ok and r.special_tokens == 2 and r.content_tokens == 382
    assert r.chars_per_token_ko > 1
    assert (tmp_path / "stub__model.json").exists()
    assert "| stub/model | O | O | 2 | 382 |" in render([r])
    render([r]).encode("cp949")


@pytest.mark.real_tokenizer
def test_real_tokenizer_windows_fit():
    tok = load_tokenizer(config.TOKENIZER_NAME)
    assert tok.num_special_tokens == 2
    text = normalize_text("연차는 1년에 15일이며, 미사용 연차는 이월되지 않습니다. " * 120)
    offs = tok.offsets(text)
    assert offs and all(0 <= s <= e <= len(text) for s, e in offs)
    assert all(b[0] >= a[0] for a, b in zip(offs, offs[1:], strict=False))
    chunks = split_document("d", text, tokenizer=tok)
    assert len(chunks) > 1
    assert all(len(tok.offsets(c.text)) <= content_size(tok) for c in chunks)


def test_tokenizer_load_failure_is_not_a_missing_file_error(tmp_path: Path):
    with pytest.raises(TokenizerLoadError) as exc:
        load_tokenizer(str(tmp_path / "없는토크나이저.json"))
    assert not isinstance(exc.value, OSError)
