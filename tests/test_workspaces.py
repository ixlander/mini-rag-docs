from __future__ import annotations

import pytest

from app.workspaces import make_workspace_id, validate_workspace_id


class TestMakeWorkspaceId:
    def test_returns_string(self):
        wid = make_workspace_id()
        assert isinstance(wid, str)

    def test_length_in_range(self):
        wid = make_workspace_id()
        assert 8 <= len(wid) <= 32

    def test_unique(self):
        ids = {make_workspace_id() for _ in range(50)}
        assert len(ids) == 50


class TestValidateWorkspaceId:
    def test_valid_id(self):
        validate_workspace_id("abc12345")

    def test_valid_with_hyphens_underscores(self):
        validate_workspace_id("my-workspace_01")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            validate_workspace_id("../../etc/passwd")

    def test_rejects_too_short(self):
        with pytest.raises(ValueError):
            validate_workspace_id("abc")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            validate_workspace_id("a" * 33)

    def test_rejects_spaces(self):
        with pytest.raises(ValueError):
            validate_workspace_id("has space!")

    def test_rejects_slashes(self):
        with pytest.raises(ValueError):
            validate_workspace_id("path/inject")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_workspace_id("")
