from __future__ import annotations

import pytest

from ingest.chunking import (
    Chunk,
    estimate_tokens,
    split_into_paragraphs,
    chunk_paragraphs,
    chunk_document,
)
from ingest.parsers import Document, Section


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hello") >= 1

    def test_proportional(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world this is a longer sentence")
        assert long > short


class TestSplitIntoParagraphs:
    def test_single_paragraph(self):
        assert split_into_paragraphs("Hello world") == ["Hello world"]

    def test_two_paragraphs(self):
        result = split_into_paragraphs("Para one\n\nPara two")
        assert len(result) == 2
        assert result[0] == "Para one"
        assert result[1] == "Para two"

    def test_strips_whitespace(self):
        result = split_into_paragraphs("  Para one  \n\n  Para two  ")
        assert result[0] == "Para one"
        assert result[1] == "Para two"


class TestChunkParagraphs:
    def test_small_text_single_chunk(self):
        paragraphs = ["Short paragraph."]
        chunks = chunk_paragraphs(paragraphs, max_tokens=100, overlap_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == "Short paragraph."

    def test_splits_large_text(self):
        paragraphs = ["This is a sentence. " * 200]
        chunks = chunk_paragraphs(paragraphs, max_tokens=50, overlap_tokens=5)
        assert len(chunks) > 1

    def test_preserves_content(self):
        paragraphs = ["First paragraph.", "Second paragraph."]
        chunks = chunk_paragraphs(paragraphs, max_tokens=1000, overlap_tokens=10)
        combined = " ".join(chunks)
        assert "First paragraph." in combined
        assert "Second paragraph." in combined


class TestChunkDocument:
    def test_produces_chunks(self):
        doc = Document(
            doc_id="test.md",
            source_path="/fake/test.md",
            source_group="test",
            title="Test Doc",
            sections=[Section(title="Intro", text="Some content here about testing.")],
        )
        chunks = chunk_document(doc, max_tokens=500, overlap_tokens=50)
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "test.md"
        assert chunks[0].title == "Test Doc"
        assert "test.md::chunk" in chunks[0].chunk_id

    def test_empty_section_skipped(self):
        doc = Document(
            doc_id="test.md",
            source_path="/fake/test.md",
            source_group="test",
            title="Test Doc",
            sections=[Section(title="Empty", text="")],
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 0
