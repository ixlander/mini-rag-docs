from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,32}$")


@dataclass(frozen=True)
class WorkspacePaths:
    workspace_id: str
    raw_dir: Path
    artifacts_dir: Path


def make_workspace_id() -> str:
    return secrets.token_urlsafe(12)


def validate_workspace_id(workspace_id: str) -> None:
    if not _WORKSPACE_ID_RE.match(workspace_id):
        raise ValueError(
            "Invalid workspace_id: must be 8-32 alphanumeric characters, hyphens, or underscores."
        )


def get_paths(workspace_id: str) -> WorkspacePaths:
    validate_workspace_id(workspace_id)

    raw_dir = BASE_DIR / "data" / "workspaces" / workspace_id / "raw"
    artifacts_dir = BASE_DIR / "artifacts" / "workspaces" / workspace_id

    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return WorkspacePaths(workspace_id=workspace_id, raw_dir=raw_dir, artifacts_dir=artifacts_dir)
