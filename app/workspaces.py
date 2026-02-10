from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkspacePaths:
    workspace_id: str
    raw_dir: Path
    artifacts_dir: Path


def make_workspace_id() -> str:
    return secrets.token_urlsafe(12)


def get_paths(workspace_id: str) -> WorkspacePaths:
    raw_dir = BASE_DIR / "data" / "workspaces" / workspace_id / "raw"
    artifacts_dir = BASE_DIR / "artifacts" / "workspaces" / workspace_id

    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return WorkspacePaths(workspace_id=workspace_id, raw_dir=raw_dir, artifacts_dir=artifacts_dir)
