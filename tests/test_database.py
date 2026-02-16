"""Tests for app.database — SQLite persistent store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.database import Database


@pytest.fixture()
def db(tmp_path):
    """Return a fresh in-memory-like Database backed by a temp file."""
    db_path = str(tmp_path / "test.db")
    return Database(db_path)


# ── Users ───────────────────────────────────────────────────────────


class TestUsers:
    def test_create_user_returns_id_and_key(self, db):
        user_id, api_key = db.create_user("alice")
        assert isinstance(user_id, int)
        assert len(api_key) > 20

    def test_authenticate_valid_key(self, db):
        _, api_key = db.create_user("bob")
        user = db.authenticate(api_key)
        assert user is not None
        assert user["name"] == "bob"

    def test_authenticate_invalid_key(self, db):
        db.create_user("carol")
        assert db.authenticate("wrong-key") is None

    def test_unique_keys(self, db):
        _, k1 = db.create_user("a")
        _, k2 = db.create_user("b")
        assert k1 != k2


# ── Workspaces ──────────────────────────────────────────────────────


class TestWorkspaces:
    def test_register_and_get(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-0001", uid, "test workspace")
        ws = db.get_workspace("ws-0001")
        assert ws is not None
        assert ws["owner_id"] == uid
        assert ws["description"] == "test workspace"

    def test_user_owns_workspace(self, db):
        uid, _ = db.create_user("owner")
        other_uid, _ = db.create_user("other")
        db.register_workspace("ws-0002", uid)
        assert db.user_owns_workspace("ws-0002", uid)
        assert not db.user_owns_workspace("ws-0002", other_uid)

    def test_list_workspaces(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-a", uid)
        db.register_workspace("ws-b", uid)
        wss = db.list_workspaces(uid)
        assert len(wss) == 2

    def test_touch_workspace(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-touch", uid)
        ws_before = db.get_workspace("ws-touch")
        db.touch_workspace("ws-touch")
        ws_after = db.get_workspace("ws-touch")
        assert ws_after["updated_at"] >= ws_before["updated_at"]

    def test_get_nonexistent(self, db):
        assert db.get_workspace("nope") is None


# ── Documents ───────────────────────────────────────────────────────


class TestDocuments:
    def test_register_and_list(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-doc", uid)
        db.register_document("ws-doc", "report.pdf", 12345, ".pdf")
        docs = db.list_documents("ws-doc")
        assert len(docs) == 1
        assert docs[0]["filename"] == "report.pdf"
        assert docs[0]["file_size"] == 12345

    def test_upsert_on_conflict(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-dup", uid)
        db.register_document("ws-dup", "file.txt", 100, ".txt")
        db.register_document("ws-dup", "file.txt", 200, ".txt")
        docs = db.list_documents("ws-dup")
        assert len(docs) == 1
        assert docs[0]["file_size"] == 200

    def test_mark_indexed(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-idx", uid)
        db.register_document("ws-idx", "a.md", 10, ".md")
        db.mark_documents_indexed("ws-idx")
        docs = db.list_documents("ws-idx")
        assert docs[0]["is_indexed"] == 1

    def test_document_count(self, db):
        uid, _ = db.create_user("owner")
        db.register_workspace("ws-cnt", uid)
        assert db.get_document_count("ws-cnt") == 0
        db.register_document("ws-cnt", "a.md", 10, ".md")
        db.register_document("ws-cnt", "b.md", 10, ".md")
        assert db.get_document_count("ws-cnt") == 2


# ── Conversations & Messages ───────────────────────────────────────


class TestConversations:
    def test_create_and_list(self, db):
        uid, _ = db.create_user("chatter")
        db.register_workspace("ws-conv", uid)
        cid = db.create_conversation("ws-conv", uid, "first chat")
        assert isinstance(cid, int)
        convs = db.list_conversations("ws-conv", uid)
        assert len(convs) == 1
        assert convs[0]["title"] == "first chat"

    def test_conversation_belongs_to_user(self, db):
        uid, _ = db.create_user("owner")
        other, _ = db.create_user("other")
        db.register_workspace("ws-own", uid)
        cid = db.create_conversation("ws-own", uid)
        assert db.conversation_belongs_to_user(cid, uid)
        assert not db.conversation_belongs_to_user(cid, other)

    def test_messages_round_trip(self, db):
        uid, _ = db.create_user("user")
        db.register_workspace("ws-msg", uid)
        cid = db.create_conversation("ws-msg", uid)
        db.add_message(cid, "user", "Hello?")
        db.add_message(cid, "assistant", "Hi there!", '["chunk1"]')
        msgs = db.get_messages(cid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["citations"] == '["chunk1"]'

    def test_get_recent_messages(self, db):
        uid, _ = db.create_user("user")
        db.register_workspace("ws-recent", uid)
        cid = db.create_conversation("ws-recent", uid)
        for i in range(20):
            db.add_message(cid, "user", f"msg {i}")
        recent = db.get_recent_messages(cid, limit=5)
        assert len(recent) == 5
        # Should be oldest-first within the window
        assert recent[0]["content"] == "msg 15"
        assert recent[-1]["content"] == "msg 19"

    def test_get_nonexistent_conversation(self, db):
        assert db.get_conversation(99999) is None
