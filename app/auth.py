"""FastAPI dependency for Bearer-token authentication."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.database import get_db

_scheme = HTTPBearer(auto_error=False)


async def require_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_scheme),
) -> Dict[str, Any]:
    """Validate the Bearer token and return the user dict.

    Raises 401 if missing/invalid, so every route that depends on this
    is automatically protected.
    """
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    user = get_db().authenticate(creds.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    return user


def require_workspace_access(workspace_id: str, user_id: int) -> None:
    """Raise 403 if the user does not own the workspace."""
    if not get_db().user_owns_workspace(workspace_id, user_id):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this workspace",
        )
