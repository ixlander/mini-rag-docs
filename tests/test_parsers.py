from __future__ import annotations

import pytest

from ingest.parsers import (
    Section,
    _normalize_text,
    parse_markdown,
    parse_html,
)


class TestNormalizeText:
    def test_collapses_blank_lines(self):
        assert _normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_trailing_spaces(self):
        assert _normalize_text("hello   \nworld") == "hello\nworld"

    def test_converts_crlf(self):
        assert _normalize_text("a\r\nb") == "a\nb"

    def test_empty_string(self):
        assert _normalize_text("") == ""


class TestParseMarkdown:
    def test_single_heading(self):
        md = "# Title\n\nSome content here."
        title, sections = parse_markdown(md)
        assert title == "Title"
        assert len(sections) == 1
        assert sections[0].title == "Title"
        assert "Some content here." in sections[0].text

    def test_multiple_headings(self):
        md = "# Doc\n\nIntro\n\n## Section A\n\nText A\n\n## Section B\n\nText B"
        title, sections = parse_markdown(md)
        assert title == "Doc"
        assert len(sections) == 3
        assert sections[1].title == "Section A"
        assert sections[2].title == "Section B"

    def test_no_headings(self):
        md = "Just plain text\nwith no headings."
        title, sections = parse_markdown(md)
        assert title == "Untitled"
        assert len(sections) == 1
        assert "Just plain text" in sections[0].text

    def test_empty_markdown(self):
        title, sections = parse_markdown("")
        assert title == "Untitled"
        assert len(sections) == 1


class TestParseHtml:
    def test_basic_html(self):
        html = "<html><head><title>My Page</title></head><body><h1>Hello</h1><p>World</p></body></html>"
        title, sections = parse_html(html)
        assert title == "My Page"
        assert any("World" in s.text for s in sections)

    def test_strips_scripts(self):
        html = "<html><body><script>alert('x')</script><p>Safe text</p></body></html>"
        title, sections = parse_html(html)
        text = " ".join(s.text for s in sections)
        assert "alert" not in text
        assert "Safe text" in text

    def test_empty_html(self):
        title, sections = parse_html("<html><body></body></html>")
        assert len(sections) == 1
