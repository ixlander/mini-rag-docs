from __future__ import annotations

import pytest

from app.prompts import build_context_block, build_user_prompt


class TestBuildContextBlock:
    def test_basic_output(self):
        chunks = [
            {"chunk_id": "doc.md::chunk0001", "title": "My Doc", "text": "Hello world"},
        ]
        result = build_context_block(chunks)
        assert "chunk_id=doc.md::chunk0001" in result
        assert "Hello world" in result

    def test_respects_max_chars(self):
        chunks = [
            {"chunk_id": f"doc.md::chunk{i:04d}", "title": "Doc", "text": "x" * 5000}
            for i in range(10)
        ]
        result = build_context_block(chunks, max_chars=6000)
        assert len(result) <= 6000 + 200

    def test_empty_chunks(self):
        result = build_context_block([])
        assert result == ""


class TestBuildUserPrompt:
    def test_contains_context_and_question(self):
        prompt = build_user_prompt(question="What is X?", context_block="Some context")
        assert "CONTEXT:" in prompt
        assert "Some context" in prompt
        assert "QUESTION:" in prompt
        assert "What is X?" in prompt
