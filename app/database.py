"""SQLite-backed persistent metadata store.

Tables
------
- users          – API-key authentication
- workspaces     – workspace ownership, description, timestamps
- documents      – per-file tracking (name, size, upload time, indexed flag)
- conversations  – conversation sessions per workspace+user
- messages       – individual messages within a conversation
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "minirag.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_hash    TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id    TEXT    PRIMARY KEY,
    owner_id        INTEGER NOT NULL REFERENCES users(id),
    description     TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id    TEXT    NOT NULL REFERENCES workspaces(workspace_id),
    filename        TEXT    NOT NULL,
    file_size       INTEGER NOT NULL DEFAULT 0,
    file_type       TEXT    NOT NULL DEFAULT '',
    uploaded_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    is_indexed      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(workspace_id, filename)
);

CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id    TEXT    NOT NULL REFERENCES workspaces(workspace_id),
    user_id         INTEGER NOT NULL REFERENCES users(id),
    title           TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role            TEXT    NOT NULL CHECK(role IN ('user','assistant')),
    content         TEXT    NOT NULL,
    citations       TEXT    NOT NULL DEFAULT '[]',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_workspaces_owner  ON workspaces(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_ws      ON documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_conversations_ws  ON conversations(workspace_id, user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conv     ON messages(conversation_id);
"""


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class Database:
    """Thread-safe SQLite wrapper. One instance per process."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        logger.info("Database initialised at %s", self._db_path)

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── users / auth ────────────────────────────────────────────────

    def create_user(self, name: str = "") -> Tuple[int, str]:
        """Create a user and return (user_id, raw_api_key)."""
        raw_key = secrets.token_urlsafe(32)
        key_hash = _hash_key(raw_key)
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO users (api_key_hash, name) VALUES (?, ?)",
                (key_hash, name),
            )
            user_id = cur.lastrowid
        logger.info("Created user id=%d name=%r", user_id, name)
        return user_id, raw_key

    def authenticate(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Return user row if key is valid & active, else None."""
        key_hash = _hash_key(api_key)
        row = self._get_conn().execute(
            "SELECT id, name, created_at FROM users WHERE api_key_hash = ? AND is_active = 1",
            (key_hash,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "name": row["name"], "created_at": row["created_at"]}

    # ── workspaces ──────────────────────────────────────────────────

    def register_workspace(self, workspace_id: str, owner_id: int, description: str = "") -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO workspaces (workspace_id, owner_id, description) VALUES (?, ?, ?)",
                (workspace_id, owner_id, description),
            )

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        row = self._get_conn().execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        return dict(row) if row else None

    def user_owns_workspace(self, workspace_id: str, user_id: int) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM workspaces WHERE workspace_id = ? AND owner_id = ?",
            (workspace_id, user_id),
        ).fetchone()
        return row is not None

    def list_workspaces(self, owner_id: int) -> List[Dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT * FROM workspaces WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def touch_workspace(self, workspace_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE workspaces SET updated_at = datetime('now') WHERE workspace_id = ?",
                (workspace_id,),
            )

    # ── documents ───────────────────────────────────────────────────

    def register_document(self, workspace_id: str, filename: str, file_size: int, file_type: str) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO documents (workspace_id, filename, file_size, file_type)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(workspace_id, filename) DO UPDATE SET
                       file_size  = excluded.file_size,
                       file_type  = excluded.file_type,
                       uploaded_at = datetime('now'),
                       is_indexed = 0""",
                (workspace_id, filename, file_size, file_type),
            )
            return cur.lastrowid

    def mark_documents_indexed(self, workspace_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE documents SET is_indexed = 1 WHERE workspace_id = ?",
                (workspace_id,),
            )

    def list_documents(self, workspace_id: str) -> List[Dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT * FROM documents WHERE workspace_id = ? ORDER BY uploaded_at DESC",
            (workspace_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_document_count(self, workspace_id: str) -> int:
        row = self._get_conn().execute(
            "SELECT COUNT(*) AS cnt FROM documents WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        return row["cnt"]

    # ── conversations ───────────────────────────────────────────────

    def create_conversation(self, workspace_id: str, user_id: int, title: str = "") -> int:
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (workspace_id, user_id, title) VALUES (?, ?, ?)",
                (workspace_id, user_id, title),
            )
            return cur.lastrowid

    def list_conversations(self, workspace_id: str, user_id: int) -> List[Dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT * FROM conversations WHERE workspace_id = ? AND user_id = ? ORDER BY updated_at DESC",
            (workspace_id, user_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        row = self._get_conn().execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return dict(row) if row else None

    def conversation_belongs_to_user(self, conversation_id: int, user_id: int) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        return row is not None

    # ── messages ────────────────────────────────────────────────────

    def add_message(self, conversation_id: int, role: str, content: str, citations: str = "[]") -> int:
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, citations) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, citations),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )
            return cur.lastrowid

    def get_messages(self, conversation_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_messages(self, conversation_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the last N messages, oldest-first."""
        rows = self._get_conn().execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ── singleton ────────────────────────────────────────────────────────
_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
