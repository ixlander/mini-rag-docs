from __future__ import annotations

import pytest

from app.prompts import build_context_block, build_user_prompt, build_conversation_history


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

    def test_without_history(self):
        prompt = build_user_prompt(question="Q?", context_block="C")
        assert "CONVERSATION HISTORY:" not in prompt

    def test_with_history(self):
        prompt = build_user_prompt(
            question="Follow up?",
            context_block="Some context",
            conversation_history="USER: First question\nASSISTANT: First answer",
        )
        assert "CONVERSATION HISTORY:" in prompt
        assert "USER: First question" in prompt
        assert "CONTEXT:" in prompt
        assert "Follow up?" in prompt


class TestBuildConversationHistory:
    def test_empty_messages(self):
        assert build_conversation_history([]) == ""

    def test_formats_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = build_conversation_history(msgs)
        assert "USER: Hello" in result
        assert "ASSISTANT: Hi there" in result

    def test_respects_max_chars(self):
        msgs = [
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
            {"role": "user", "content": "z" * 500},
        ]
        result = build_conversation_history(msgs, max_chars=600)
        # Should drop oldest messages that don't fit
        assert len(result) <= 700  # some tolerance for role prefix

    def test_preserves_chronological_order(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = build_conversation_history(msgs)
        pos_first = result.find("first")
        pos_second = result.find("second")
        pos_third = result.find("third")
        assert pos_first < pos_second < pos_third
