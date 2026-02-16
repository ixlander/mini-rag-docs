from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import require_user, require_workspace_access
from app.database import get_db
from app.prompts import build_conversation_history
from app.rag_workspace import WorkspaceRAG, WorkspaceRAGConfig
from app.workspaces import get_paths, make_workspace_id
from ingest.build_index_lib import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(title="Mini-RAG (Workspaces)")
rag = WorkspaceRAG(WorkspaceRAGConfig())


# ── Pydantic models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str = ""

class RegisterResponse(BaseModel):
    user_id: int
    api_key: str
    message: str = "Save this API key — it cannot be retrieved later."

class CreateWorkspaceRequest(BaseModel):
    description: str = ""

class CreateWorkspaceResponse(BaseModel):
    workspace_id: str

class BuildIndexResponse(BaseModel):
    workspace_id: str
    num_docs: int
    num_chunks: int

class UploadDirRequest(BaseModel):
    directory: str = Field(..., min_length=1)

class QueryRequest(BaseModel):
    workspace_id: str
    question: str = Field(..., min_length=1)
    conversation_id: Optional[int] = None
    debug: bool = False

class CreateConversationRequest(BaseModel):
    workspace_id: str
    title: str = ""


# ── Health (no auth) ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── User registration (no auth) ────────────────────────────────────

@app.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    """Create a new user and return their API key."""
    user_id, api_key = get_db().create_user(name=req.name)
    return RegisterResponse(user_id=user_id, api_key=api_key)


# ── Workspaces (auth required) ─────────────────────────────────────

@app.post("/workspaces", response_model=CreateWorkspaceResponse)
def create_workspace(
    req: CreateWorkspaceRequest = CreateWorkspaceRequest(),
    user: Dict[str, Any] = Depends(require_user),
):
    wid = make_workspace_id()
    get_paths(wid)
    get_db().register_workspace(wid, owner_id=user["id"], description=req.description)
    return {"workspace_id": wid}


@app.get("/workspaces")
def list_workspaces(user: Dict[str, Any] = Depends(require_user)):
    """List all workspaces owned by the authenticated user."""
    return get_db().list_workspaces(user["id"])


@app.get("/status/{workspace_id}")
def status(workspace_id: str, user: Dict[str, Any] = Depends(require_user)):
    require_workspace_access(workspace_id, user["id"])

    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db = get_db()
    docs = db.list_documents(workspace_id)
    ws = db.get_workspace(workspace_id)

    has_index = (paths.artifacts_dir / "faiss.index").exists() and (paths.artifacts_dir / "chunks.parquet").exists()

    return {
        "workspace_id": workspace_id,
        "description": ws["description"] if ws else "",
        "created_at": ws["created_at"] if ws else None,
        "documents": docs,
        "document_count": len(docs),
        "has_index": has_index,
    }


ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}


@app.post("/upload/{workspace_id}")
async def upload_files(
    workspace_id: str,
    files: List[UploadFile] = File(...),
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(workspace_id, user["id"])

    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db = get_db()
    saved = []
    saved_paths = []

    for f in files:
        name = Path(f.filename or "file").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        content = await f.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty file uploaded: {name}")

        dst = paths.raw_dir / name
        dst.write_bytes(content)
        saved.append(name)
        saved_paths.append(str(dst.resolve()))

        db.register_document(workspace_id, name, len(content), ext)

    db.touch_workspace(workspace_id)

    return {
        "workspace_id": workspace_id,
        "raw_dir": str(paths.raw_dir.resolve()),
        "saved": saved,
        "saved_paths": saved_paths,
        "count": len(saved),
    }


@app.post("/upload_dir/{workspace_id}")
def upload_directory(
    workspace_id: str,
    req: UploadDirRequest,
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(workspace_id, user["id"])

    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    src = Path(req.directory)
    if not src.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.directory}")
    if not src.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {req.directory}")

    db = get_db()
    saved = []
    saved_paths = []
    skipped = []

    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(f.name)
            continue

        content = f.read_bytes()
        if not content:
            skipped.append(f.name)
            continue

        dst = paths.raw_dir / f.name
        dst.write_bytes(content)
        saved.append(f.name)
        saved_paths.append(str(dst.resolve()))

        db.register_document(workspace_id, f.name, len(content), ext)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files found. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    db.touch_workspace(workspace_id)

    return {
        "workspace_id": workspace_id,
        "raw_dir": str(paths.raw_dir.resolve()),
        "saved": saved,
        "saved_paths": saved_paths,
        "count": len(saved),
        "skipped": skipped,
    }


@app.post("/build_index/{workspace_id}", response_model=BuildIndexResponse)
def build_index_for_workspace(
    workspace_id: str,
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(workspace_id, user["id"])

    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw_files = [p for p in paths.raw_dir.iterdir() if p.is_file()] if paths.raw_dir.exists() else []
    if not raw_files:
        raise HTTPException(status_code=400, detail="No uploaded files found. Call /upload/{workspace_id} first.")

    try:
        stats = build_index(
            raw_dir=str(paths.raw_dir),
            artifacts_dir=str(paths.artifacts_dir),
            embed_model="intfloat/multilingual-e5-small",
            use_e5_prefix=True,
            batch_size=64,
            device=None,
            max_tokens=550,
            overlap_tokens=80,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rag.invalidate_cache(str(paths.artifacts_dir))
    get_db().mark_documents_indexed(workspace_id)
    get_db().touch_workspace(workspace_id)

    return {"workspace_id": workspace_id, "num_docs": stats["num_docs"], "num_chunks": stats["num_chunks"]}


# ── Conversations (auth required) ──────────────────────────────────

@app.post("/conversations")
def create_conversation(
    req: CreateConversationRequest,
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(req.workspace_id, user["id"])
    conv_id = get_db().create_conversation(req.workspace_id, user["id"], req.title)
    return {"conversation_id": conv_id}


@app.get("/conversations/{workspace_id}")
def list_conversations(
    workspace_id: str,
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(workspace_id, user["id"])
    return get_db().list_conversations(workspace_id, user["id"])


@app.get("/conversations/{workspace_id}/{conversation_id}/messages")
def get_conversation_messages(
    workspace_id: str,
    conversation_id: int,
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(workspace_id, user["id"])
    db = get_db()
    if not db.conversation_belongs_to_user(conversation_id, user["id"]):
        raise HTTPException(status_code=403, detail="Conversation not accessible")
    return db.get_messages(conversation_id)


# ── Query (auth required, with optional conversation memory) ───────

@app.post("/query")
def query(
    req: QueryRequest,
    stream: bool = False,
    user: Dict[str, Any] = Depends(require_user),
):
    require_workspace_access(req.workspace_id, user["id"])

    try:
        paths = get_paths(req.workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not (paths.artifacts_dir / "faiss.index").exists():
        raise HTTPException(status_code=400, detail="Index not built for this workspace. Call /build_index/{workspace_id} first.")

    db = get_db()

    # ── Build conversation history if a conversation_id is provided ────
    history_text = ""
    if req.conversation_id is not None:
        if not db.conversation_belongs_to_user(req.conversation_id, user["id"]):
            raise HTTPException(status_code=403, detail="Conversation not accessible")
        recent = db.get_recent_messages(req.conversation_id, limit=10)
        history_text = build_conversation_history(recent)

        # Persist the user message
        db.add_message(req.conversation_id, "user", req.question)

    if stream:
        return StreamingResponse(
            rag.answer_stream(
                artifacts_dir=str(paths.artifacts_dir),
                question=req.question,
                conversation_history=history_text,
            ),
            media_type="text/event-stream",
        )

    result = rag.answer(
        artifacts_dir=str(paths.artifacts_dir),
        question=req.question,
        debug=req.debug,
        conversation_history=history_text,
    )

    # Persist the assistant reply
    if req.conversation_id is not None:
        answer_text = result.get("answer", "") if isinstance(result, dict) else str(result)
        citations_json = json.dumps(result.get("citations", []) if isinstance(result, dict) else [])
        db.add_message(req.conversation_id, "assistant", answer_text, citations_json)

    return result
