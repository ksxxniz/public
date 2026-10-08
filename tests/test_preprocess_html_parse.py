import json
import time
from pathlib import Path

import pytest

from datagen.evasion import html_samples
from preprocess import html_parse
from preprocess.html_parse import parse_html

FIXTURES = Path(__file__).parent / "fixtures" / "hidden_html"
ANSWERS = json.loads((FIXTURES / "answers.json").read_text(encoding="utf-8"))
REASON = {
    "white_text": "same_color",
    "font_size_zero": "font_size_zero",
    "display_none": "display_none",
    "html_comment": "comment",
}


def test_pieces_point_to_their_raw_text():
    raw = "<p>안녕\r\n하세요</p>\n<div>둘째 &amp; 셋째&#8203;</div>"
    result = parse_html(raw)
    for p in result.pieces:
        if p.text in ("&", "\u200b"):
            assert raw[p.start : p.end] in ("&amp;", "&#8203;")
        else:
            assert raw[p.start : p.end] == p.text


@pytest.mark.parametrize(
    ("html_text", "reason"),
    [
        ('<div style="display:none">x</div>', "display_none"),
        ('<span style="visibility:hidden">x</span>', "visibility_hidden"),
        ('<span style="opacity:0">x</span>', "opacity_zero"),
        ('<span style="font-size:0px">x</span>', "font_size_zero"),
        ('<span style="font-size:1px">x</span>', "font_size_zero"),
        ('<span style="color:transparent">x</span>', "transparent_color"),
        ('<span style="color:#fff">x</span>', "same_color"),
        ('<div style="background:#000"><span style="color:#111">x</span></div>', "same_color"),
        ('<p style="position:absolute;left:-9999px">x</p>', "offscreen"),
        ('<div style="width:0;height:0;overflow:hidden">x</div>', "zero_size_box"),
        ("<div hidden>x</div>", "hidden_attr"),
        ("<template>x</template>", "template"),
        ("<noscript>x</noscript>", "noscript"),
        ("<script>x</script>", "script"),
        ("<style>x</style>", "style"),
        ("<!--x-->", "comment"),
    ],
)
def test_hidden_reasons(html_text, reason):
    result = parse_html(html_text)
    assert [(p.text, p.hidden) for p in result.pieces if p.text.strip()] == [("x", reason)]


@pytest.mark.parametrize(
    "html_text",
    [
        '<span style="color:#fff;background:#000">x</span>',
        '<span style="font-size:12px">x</span>',
        '<span style="opacity:0.8">x</span>',
        '<p style="left:10px">x</p>',
    ],
)
def test_visible_styles(html_text):
    result = parse_html(html_text)
    assert [p.hidden for p in result.pieces if p.text.strip()] == [None]


def test_hidden_parent_hides_children():
    result = parse_html('<div style="display:none"><p>a<b>b</b></p></div><p>c</p>')
    assert result.hidden_text() == "ab"
    assert result.visible_text() == "c"


def test_block_tags_become_line_breaks():
    assert parse_html("foo</p><p>bar").visible_text() == "foo\nbar"
    assert parse_html("a<br>b<span>c</span>").visible_text() == "a\nbc"


def test_unclosed_tags_are_closed_by_parent():
    result = parse_html('<div style="display:none"><p>secret</div><p>shown')
    assert result.visible_text() == "shown"
    assert result.hidden_text() == "secret"


def test_all_text_keeps_hidden_text():
    result = parse_html('<p>보이는 글</p><div style="display:none">숨긴 글</div>')
    assert result.all_text() == "보이는 글\n숨긴 글"


def test_fixture_files_match_generator():
    expected = {name: page for name, page, _ in html_samples(20)}
    files = {p.name: p.read_text(encoding="utf-8") for p in FIXTURES.glob("*.html")}
    assert files == expected
    assert len(ANSWERS) == 20


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_parser_finds_hidden_answer(name):
    info = ANSWERS[name]
    result = parse_html((FIXTURES / name).read_text(encoding="utf-8"))
    hits = [p for p in result.hidden_pieces if info["answer"] in p.text]
    assert hits and hits[0].hidden == REASON[info["technique"]]
    assert info["answer"] not in result.visible_text()


def test_self_closing_div_still_opens():
    result = parse_html('<div style="display:none"/>secret</div><p>shown</p>')
    assert result.hidden_text() == "secret"
    assert result.visible_text() == "shown"
    assert parse_html("a<br/>b").visible_text() == "a\nb"


@pytest.mark.parametrize(
    "html_text",
    [
        '<div style="color:#fff;background:#000"><span style="background:#fff">x</span></div>',
        '<div style="color:#000"><p style="background-color:black">x</p></div>',
    ],
)
def test_inherited_color_on_same_background_is_hidden(html_text):
    assert [p.hidden for p in parse_html(html_text).pieces if p.text == "x"] == ["same_color"]


SR_ONLY = "position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)"


def _words(html_text: str) -> tuple[str, str]:
    pieces = parse_html(html_text).pieces
    hidden = " ".join(p.text for p in pieces if p.hidden)
    visible = " ".join(p.text for p in pieces if not p.hidden)
    return hidden, visible


@pytest.mark.parametrize(
    "html_text",
    [
        '<div style="display:none" style="color:red">SECRET</div>',
        '<div style="display:none"><table><tr><td></div>SECRET</td></tr></table></div>',
        '<span style="display:none"><div></span>SECRET</div>',
        '<div style="display:none"><textarea></div></textarea>SECRET</div>',
        '<div style="display:none"><iframe></div></iframe>SECRET</div>',
        '<div style="display:none"><noscript></div></noscript>SECRET</div>',
        '<body style="display:none">a</body>SECRET',
        '<p><b style="display:none">x</p>SECRET',
        '<span style="color:ghostwhite">SECRET</span>',
        '<span style="color:hsl(0, 0%, 100%)">SECRET</span>',
        '<span style="color:rgb(100%,100%,100%)">SECRET</span>',
        '<span style="color:#123456;background-color:currentcolor">SECRET</span>',
        '<div style="display:/**/none">SECRET</div>',
        '<div style="display:none !important; display:block">SECRET</div>',
        '<div style="opacity:1%">SECRET</div>',
        '<div style="opacity:0.2"><span style="opacity:0.2">SECRET</span></div>',
        '<div style="font-size:0.05em">SECRET</div>',
        '<div style="font-size:5%">SECRET</div>',
        f'<span style="{SR_ONLY}">SECRET</span>',
        '<div style="clip-path:inset(50%)">SECRET</div>',
        '<div style="transform:scale(0)">SECRET</div>',
        '<div style="content-visibility:hidden">SECRET</div>',
        "<datalist><option>SECRET</option></datalist>",
        "<video>SECRET</video>",
        "<dialog>SECRET</dialog>",
        "<svg><desc>SECRET</desc></svg>",
        "<svg>SECRET</svg>",
    ],
)
def test_browser_hides_it_so_do_we(html_text):
    hidden, visible = _words(html_text)
    assert "SECRET" in hidden and "SECRET" not in visible


@pytest.mark.parametrize(
    "html_text",
    [
        '<td style="display:none">SHOWN',
        '<p style="display:none">x<div>SHOWN</div>',
        '<p style="display:none">x<h1>SHOWN</h1>',
        '<ul><li style="display:none">x<li>SHOWN</ul>',
        '<table style="display:none">SHOWN<tr><td>x</td></tr></table>',
        '<div style="visibility:hidden">x<span style="visibility:visible">SHOWN</span></div>',
        '<div style="font-size:0"><span style="font-size:14px">SHOWN</span></div>',
        '<div style="color:#fff"><span style="color:#000">SHOWN</span></div>',
        "<dialog open>SHOWN</dialog>",
        "<svg><foreignObject><p>SHOWN</p></foreignObject></svg>",
        "<svg><text>SHOWN</text></svg>",
        '<div style="font-size:-5px">SHOWN</div>',
        '<p style="left:-9999">SHOWN</p>',
        '<div style="background:#000"><span style="color:white">SHOWN</span></div>',
        '<b style="display:none">x<div>y</b>SHOWN</div>',
        '<image style="display:none">SHOWN',
    ],
)
def test_browser_shows_it_so_do_we(html_text):
    _, visible = _words(html_text)
    assert "SHOWN" in visible


def test_old_html_parser_is_rejected(monkeypatch):
    monkeypatch.setattr(html_parse, "html5_parser_available", lambda: False)
    with pytest.raises(RuntimeError, match="3.11.15"):
        parse_html("<p>x</p>")


@pytest.mark.parametrize(
    "html_text",
    [
        pytest.param("<div>" * 20000 + "x", id="nested-div"),
        pytest.param("<p><b>" * 5000 + "x", id="reopened-formatting"),
        pytest.param("<span>" * 20000 + "</x>" * 20000, id="unknown-end-tags"),
        pytest.param("<b><table><td>" + "<span>" * 20000 + "</b>" * 20000, id="blocked-end-tags"),
    ],
)
def test_deep_nesting_stays_fast(html_text):
    started = time.perf_counter()
    parse_html(html_text)
    assert time.perf_counter() - started < 5
