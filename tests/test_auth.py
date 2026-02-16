"""Tests for app.auth — authentication dependency."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.database import Database


@pytest.fixture()
def db(tmp_path):
    db_path = str(tmp_path / "test_auth.db")
    _db = Database(db_path)
    with patch("app.auth.get_db", return_value=_db):
        yield _db


class TestRequireWorkspaceAccess:
    def test_owner_has_access(self, db):
        from app.auth import require_workspace_access

        uid, _ = db.create_user("owner")
        db.register_workspace("ws-auth1", uid)

        with patch("app.auth.get_db", return_value=db):
            # Should not raise
            require_workspace_access("ws-auth1", uid)

    def test_non_owner_rejected(self, db):
        from fastapi import HTTPException
        from app.auth import require_workspace_access

        uid, _ = db.create_user("owner")
        other, _ = db.create_user("other")
        db.register_workspace("ws-auth2", uid)

        with patch("app.auth.get_db", return_value=db):
            with pytest.raises(HTTPException) as exc_info:
                require_workspace_access("ws-auth2", other)
            assert exc_info.value.status_code == 403
