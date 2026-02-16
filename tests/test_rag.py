from __future__ import annotations

import json
import pytest

from app.rag_workspace import WorkspaceRAG, WorkspaceRAGConfig


class TestSafeParseJson:
    def test_valid_json(self):
        result = WorkspaceRAG._safe_parse_json('{"answer": "hello", "citations": []}')
        assert result == {"answer": "hello", "citations": []}

    def test_json_with_surrounding_text(self):
        result = WorkspaceRAG._safe_parse_json('Some text {"answer": "hello"} more text')
        assert result == {"answer": "hello"}

    def test_empty_string(self):
        assert WorkspaceRAG._safe_parse_json("") is None

    def test_none_input(self):
        assert WorkspaceRAG._safe_parse_json(None) is None

    def test_invalid_json(self):
        assert WorkspaceRAG._safe_parse_json("not json at all") is None

    def test_array_returns_none(self):
        assert WorkspaceRAG._safe_parse_json("[1, 2, 3]") is None


class TestWorkspaceRAGConfig:
    def test_default_values(self):
        cfg = WorkspaceRAGConfig()
        assert cfg.top_k == 8
        assert cfg.top_final == 4
        assert cfg.temperature == 0.0

    def test_custom_values(self):
        cfg = WorkspaceRAGConfig(top_k=20, temperature=0.5)
        assert cfg.top_k == 20
        assert cfg.temperature == 0.5
