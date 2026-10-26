import json
import time
from pathlib import Path

import pytest

from preprocess import markup
from preprocess.chunk import chunks_from_document, split_document
from preprocess.normalize import normalize_document

FIXTURES = Path(__file__).parent / "fixtures" / "hidden_html"
ANSWERS = json.loads((FIXTURES / "answers.json").read_text(encoding="utf-8"))


def _hidden(raw: str, fmt: str) -> list[tuple[str, str]]:
    doc = normalize_document(raw, fmt)
    return [
        (ev.note, raw[ev.start : ev.end]) for ev in doc.events if ev.kind == "hidden_text_found"
    ]


def _text(raw: str, fmt: str = "html") -> str:
    return normalize_document(raw, fmt).text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("foo</p><p>bar", "foo\nbar", id="block"),
        pytest.param("a<br>b<span>c</span>", "a\nbc", id="br-inline"),
        pytest.param("<div>a</div><div>b</div>", "a\nb", id="divs"),
        pytest.param("<ul><li>하나</li><li>둘</li></ul>", "하나\n둘", id="list"),
        pytest.param("<table><tr><td>1</td><td>2</td></tr></table>", "1\n2", id="table"),
        pytest.param("<p>A &amp; B &lt;C&gt; &#x41;&#66;</p>", "A & B <C> AB", id="entities"),
        pytest.param("<!DOCTYPE html><html><body><p>x</p></body></html>", "x", id="doctype"),
    ],
)
def test_html_text_extraction(raw, expected):
    assert _text(raw) == expected


def test_entity_maps_back_to_its_raw_text():
    raw = "<p>A &amp; B</p>"
    doc = normalize_document(raw, "html")
    i = doc.text.index("&")
    assert raw[doc.starts[i] : doc.ends[i]] == "&amp;"
    assert doc.text == "A & B"


@pytest.mark.parametrize(
    ("raw", "reason", "hidden"),
    [
        pytest.param(
            '<p>보임<span style="display:none">숨김</span></p>',
            "display_none",
            "숨김",
            id="display",
        ),
        pytest.param('<p style="color:#fff">흰 글씨</p>', "same_color", "흰 글씨", id="white"),
        pytest.param(
            '<span style="font-size:0">작은 글씨</span>', "font_size_zero", "작은 글씨", id="font"
        ),
        pytest.param('<p style="opacity:0">투명</p>', "opacity_zero", "투명", id="opacity"),
        pytest.param(
            '<p style="position:absolute;left:-9999px">밖</p>', "offscreen", "밖", id="offscreen"
        ),
        pytest.param("<div hidden>속성</div>", "hidden_attr", "속성", id="hidden-attr"),
        pytest.param("<p>앞<!-- 주석 -->뒤</p>", "comment", " 주석 ", id="comment"),
        pytest.param("<p>a<![CDATA[데이터]]>b</p>", "cdata", "데이터", id="cdata"),
        pytest.param("<template>틀</template>", "template", "틀", id="template"),
        pytest.param("<noscript>대체</noscript>", "noscript", "대체", id="noscript"),
        pytest.param("<script>var a = 1;</script>", "script", "var a = 1;", id="script"),
        pytest.param("<style>p { color: red }</style>", "style", "p { color: red }", id="style"),
    ],
)
def test_hidden_text_is_kept_and_marked(raw, reason, hidden):
    assert _hidden(raw, "html") == [(reason, hidden)]
    assert hidden.strip() in _text(raw)


@pytest.mark.parametrize(
    ("raw", "reason", "value"),
    [
        pytest.param('<img src="a.png" alt="회사 로고">', "attr_alt", "회사 로고", id="alt"),
        pytest.param('<a href="/x" title="설명">링크</a>', "attr_title", "설명", id="title"),
        pytest.param('<button aria-label="닫기">x</button>', "attr_aria_label", "닫기", id="aria"),
        pytest.param(
            '<meta name="description" content="요약 문장">', "meta_content", "요약 문장", id="meta"
        ),
        pytest.param('<meta property="og:title" content="제목">', "meta_content", "제목", id="og"),
        pytest.param(
            '<input type="hidden" value="비밀 지시">', "input_hidden", "비밀 지시", id="input"
        ),
    ],
)
def test_attribute_text_is_kept_and_marked(raw, reason, value):
    assert _hidden(raw, "html") == [(reason, value)]
    assert value in _text(raw)


def test_attribute_entities_are_decoded():
    raw = '<img alt="ig&amp;nore">'
    assert _text(raw) == "ig&nore"
    assert _hidden(raw, "html") == [("attr_alt", "ig&amp;nore")]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('<meta name="viewport" content="width=device-width">', id="viewport"),
        pytest.param('<meta charset="utf-8">', id="charset"),
        pytest.param('<input type="text" value="보이는 값">', id="visible-input"),
        pytest.param('<img alt="  ">', id="blank-alt"),
        pytest.param('<p class="note" id="x" data-x="y">본문</p>', id="other-attrs"),
    ],
)
def test_other_attributes_are_not_text(raw):
    assert _hidden(raw, "html") == []
    assert _text(raw) in ("", "본문")


def test_hidden_text_is_separated_from_visible_words():
    raw = 'Hello<span style="display:none">ignore all</span>world<!--x-->end'
    assert _text(raw) == "Hello ignore all world x end"


def test_inline_tags_do_not_split_words():
    assert _text("<p>pass<b>word</b> 안<i>내</i></p>") == "password 안내"


def test_hidden_block_with_children_is_one_region():
    raw = '<div style="display:none"><p>첫째</p><p>둘째</p></div><p>보임</p>'
    assert _hidden(raw, "html") == [("display_none", "첫째</p><p>둘째")]
    assert _text(raw) == "첫째\n둘째\n보임"


def test_tag_characters_inside_markup_are_still_decoded():
    tags = "".join(chr(0xE0000 + ord(c)) for c in "ignore")
    (chunk,) = split_document("d", f'<p class="a{tags}">본문</p>', "html")
    assert chunk.text == "본문"
    assert chunk.transform_log.tag_chars_decoded == len("ignore")
    assert [s.decoded for s in chunk.decoded_segments] == ["ignore"]


@pytest.mark.parametrize(
    ("raw", "reason", "hidden"),
    [
        pytest.param("<p>a</p><?숨긴 지시?>", "comment", "숨긴 지시?", id="pi"),
        pytest.param("<p>a</p><![숨긴 지시]>", "comment", "숨긴 지시", id="marked-section"),
        pytest.param("<p>a</p><![cdata[숨긴 지시]]>", "cdata", "숨긴 지시", id="lowercase-cdata"),
        pytest.param("<p>a</p><!숨긴 지시>", "comment", "숨긴 지시", id="bogus-comment"),
        pytest.param(
            '<!DOCTYPE html "숨긴 지시"><p>a</p>',
            "comment",
            'DOCTYPE html "숨긴 지시"',
            id="doctype",
        ),
        pytest.param(
            '<?xml version="1.0" encoding="ignore-all"?><p>a</p>',
            "comment",
            'xml version="1.0" encoding="ignore-all"?',
            id="xml-decl",
        ),
    ],
)
def test_declarations_and_instructions_are_kept_as_comments(raw, reason, hidden):
    assert _hidden(raw, "html") == [(reason, hidden)]
    assert hidden in _text(raw)


@pytest.mark.parametrize(
    "head",
    [
        "<!DOCTYPE html>",
        '<!doctype html system "about:legacy-compat">',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" '
        '"http://www.w3.org/TR/html4/loose.dtd">',
        '<?xml version="1.0" encoding="UTF-8"?>',
    ],
)
def test_standard_declarations_are_dropped(head):
    raw = head + "<p>본문</p>"
    assert _text(raw) == "본문"
    assert _hidden(raw, "html") == []


@pytest.mark.parametrize("ref", ["&#", "&#;", "&#zz;", "&#1a;", "&#x;"])
def test_broken_character_reference_does_not_stop_parsing(ref):
    raw = f'<p>a{ref}b{ref}c</p><div style="display:none">숨김</div><p>d;e</p>'
    assert _text(raw) == f"a{ref}b{ref}c\n숨김\nd;e"
    assert _hidden(raw, "html") == [("display_none", "숨김")]


def test_broken_character_reference_in_title_does_not_stop_parsing():
    raw = "<title>a &# b</title><div hidden>숨김</div>"
    assert _text(raw) == "a &# b\n숨김"
    assert _hidden(raw, "html") == [("hidden_attr", "숨김")]


def test_ampersand_at_end_of_document_is_kept():
    assert _text("<p>a &x") == "a &x"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('<font color="#ffffff">흰 글씨</font>', id="font-color"),
        pytest.param('<font color="ffffff">흰 글씨</font>', id="bare-hex"),
        pytest.param('<body text="white">흰 글씨</body>', id="body-text"),
        pytest.param(
            '<table bgcolor="black"><tr><td><font color="#000">흰 글씨</font></td></tr></table>',
            id="bgcolor",
        ),
    ],
)
def test_legacy_color_attributes_hide_text(raw):
    assert _hidden(raw, "html") == [("same_color", "흰 글씨")]


def test_legacy_color_values_cannot_inject_styles():
    assert _hidden('<font color="red;display:none">보임</font>', "html") == []
    raw = '<font color="/*" style="display:none;*/">숨김</font>'
    assert _hidden(raw, "html") == [("display_none", "숨김")]


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_fixture_answers_are_kept_and_marked(name):
    raw = (FIXTURES / name).read_text(encoding="utf-8")
    answer = ANSWERS[name]["answer"]
    doc = normalize_document(raw, "html")
    pos = doc.text.index(answer)
    r0, r1 = doc.starts[pos], doc.ends[pos + len(answer) - 1]
    assert any(
        ev.kind == "hidden_text_found" and ev.start < r1 and ev.end > r0 for ev in doc.events
    )


MD_CASES = [
    pytest.param(
        "앞\n\n[//]: # (ignore all rules)\n\n뒤",
        [("md_comment", "ignore all rules")],
        id="md-comment",
    ),
    pytest.param('[//]: # "따옴표 주석"', [("md_comment", "따옴표 주석")], id="md-comment-quoted"),
    pytest.param("[comment]: <> (꺾쇠 주석)", [("md_comment", "꺾쇠 주석")], id="md-comment-angle"),
    pytest.param("본문 <!-- html 주석 --> 끝", [("comment", " html 주석 ")], id="html-comment"),
    pytest.param("![회사 로고](logo.png)", [("attr_alt", "회사 로고")], id="image-alt"),
    pytest.param(
        '![alt](a.png "그림 제목")',
        [("attr_alt", "alt"), ("attr_title", "그림 제목")],
        id="image-title",
    ),
    pytest.param(
        '[링크](https://x.com "링크 제목")', [("attr_title", "링크 제목")], id="link-title"
    ),
    pytest.param('[ref]: https://x.com "참조 제목"', [("attr_title", "참조 제목")], id="ref-title"),
    pytest.param("![참조 그림][ref]", [("attr_alt", "참조 그림")], id="ref-image"),
    pytest.param(
        '글 <span style="display:none">숨김</span> 끝', [("display_none", "숨김")], id="inline-html"
    ),
    pytest.param(
        "[//]: # (여러\n줄 주석)", [("md_comment", "여러\n줄 주석")], id="md-comment-multiline"
    ),
    pytest.param(
        '[//]: #\n"다음 줄 주석"', [("md_comment", "다음 줄 주석")], id="md-comment-next-line"
    ),
    pytest.param("[숨긴 지시]: #", [("md_comment", "숨긴 지시")], id="md-comment-label"),
    pytest.param(
        '[링크](https://x.com "여러\n줄 제목")',
        [("attr_title", "여러\n줄 제목")],
        id="link-title-multiline",
    ),
    pytest.param(
        '[링크](https://x.com/a_(b) "제목")', [("attr_title", "제목")], id="link-with-parens"
    ),
    pytest.param("```a`\n<!-- 주석 -->\n```", [("comment", " 주석 ")], id="not-a-fence"),
    pytest.param("글 \\`<!-- 주석 --> \\` 끝", [("comment", " 주석 ")], id="escaped-backtick"),
]


@pytest.mark.parametrize(("raw", "expected"), MD_CASES)
def test_markdown_hidden_places_are_marked(raw, expected):
    assert _hidden(raw, "md") == expected
    assert normalize_document(raw, "md").text == raw


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('```html\n<div style="display:none">코드</div>\n[//]: # (x)\n```', id="fence"),
        pytest.param("~~~\n<!-- 코드 안 주석 -->\n~~~", id="tilde-fence"),
        pytest.param('본문 `<span style="display:none">x</span>` 끝', id="inline-code"),
        pytest.param("[링크](https://x.com) 와 ![](a.png)", id="no-title-no-alt"),
        pytest.param("[//]: #", id="empty-md-comment"),
        pytest.param("[//]: # (문단\n\n나뉨)", id="title-across-blank-line"),
        pytest.param("본문 ``a`b`` 와 `<!-- 코드 -->` 끝", id="inline-code-runs"),
    ],
)
def test_markdown_code_and_plain_links_are_not_marked(raw):
    assert _hidden(raw, "md") == []


def test_markdown_skips_html_marks_without_html5_parser(monkeypatch):
    monkeypatch.setattr(markup, "html5_parser_available", lambda: False)
    raw = '[//]: # (주석)\n\n글 <span style="display:none">숨김</span> 끝'
    assert _hidden(raw, "md") == [("md_comment", "주석")]


@pytest.mark.parametrize(
    "unit", ["[", "![", "![a](", "![a](<", "[a](x ", '[a]: # "\n', "`a``a", "<!--", "&#"]
)
def test_repeated_markdown_patterns_stay_fast(unit):
    raw = unit * (200_000 // len(unit))
    started = time.perf_counter()
    normalize_document(raw, "md")
    assert time.perf_counter() - started < 10


def test_many_backtick_run_lengths_stay_fast():
    raw = "a".join("`" * n for n in range(1, 600))
    started = time.perf_counter()
    normalize_document(raw, "md")
    assert time.perf_counter() - started < 10


def test_unclosed_fence_hides_nothing_after_it():
    raw = "```\n[//]: # (코드 안)\n"
    assert _hidden(raw, "md") == []


def test_txt_is_not_parsed_as_markup():
    raw = '<span style="display:none">x</span> [//]: # (y)'
    assert _hidden(raw, "txt") == []
    assert normalize_document(raw, "txt").text == raw


def test_hidden_span_crossing_chunk_boundary_is_in_both_chunks():
    raw = '<p>보이는 문장<span style="display:none">숨긴 긴 문장입니다</span>다음 문장</p>'
    doc = normalize_document(raw, "html")
    mid = doc.text.index("문장입니다")
    c0, c1 = chunks_from_document("d", doc, windows=[(0, mid), (mid, len(doc.text))])
    for c in (c0, c1):
        assert c.transform_log.hidden_text_found == 1
        (ts,) = [t for t in c.transform_spans if t.kind == "hidden_text_found"]
        assert ts.note == "display_none"
        assert 0 <= ts.span.start < ts.span.end <= len(c.raw_text)
    assert c0.raw_text[c0.transform_spans[0].span.start :].startswith("숨긴")
    assert c1.raw_text[: c1.transform_spans[0].span.end].endswith("문장입니다")
