from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from app.workspaces import make_workspace_id, get_paths
from ingest.build_index_lib import build_index
from app.rag_workspace import WorkspaceRAG, WorkspaceRAGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(title="Mini-RAG (Workspaces)")
rag = WorkspaceRAG(WorkspaceRAGConfig())


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
    debug: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/workspaces", response_model=CreateWorkspaceResponse)
def create_workspace():
    wid = make_workspace_id()
    get_paths(wid)
    return {"workspace_id": wid}


@app.get("/status/{workspace_id}")
def status(workspace_id: str):
    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw_files = []
    if paths.raw_dir.exists():
        raw_files = [p.name for p in paths.raw_dir.iterdir() if p.is_file()]

    has_index = (paths.artifacts_dir / "faiss.index").exists() and (paths.artifacts_dir / "chunks.parquet").exists()

    return {
        "workspace_id": workspace_id,
        "raw_files_count": len(raw_files),
        "raw_files": raw_files[:200],
        "has_index": has_index,
        "raw_dir": str(paths.raw_dir.resolve()),
        "artifacts_dir": str(paths.artifacts_dir.resolve()),
    }

ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}


@app.post("/upload/{workspace_id}")
async def upload_files(workspace_id: str, files: List[UploadFile] = File(...)):
    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

    return {
        "workspace_id": workspace_id,
        "raw_dir": str(paths.raw_dir.resolve()),
        "saved": saved,
        "saved_paths": saved_paths,
        "count": len(saved),
    }


@app.post("/upload_dir/{workspace_id}")
def upload_directory(workspace_id: str, req: UploadDirRequest):
    try:
        paths = get_paths(workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    src = Path(req.directory)
    if not src.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.directory}")
    if not src.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {req.directory}")

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

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files found. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    return {
        "workspace_id": workspace_id,
        "raw_dir": str(paths.raw_dir.resolve()),
        "saved": saved,
        "saved_paths": saved_paths,
        "count": len(saved),
        "skipped": skipped,
    }


@app.post("/build_index/{workspace_id}", response_model=BuildIndexResponse)
def build_index_for_workspace(workspace_id: str):
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

    return {"workspace_id": workspace_id, "num_docs": stats["num_docs"], "num_chunks": stats["num_chunks"]}


@app.post("/query")
def query(req: QueryRequest):
    try:
        paths = get_paths(req.workspace_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not (paths.artifacts_dir / "faiss.index").exists():
        raise HTTPException(status_code=400, detail="Index not built for this workspace. Call /build_index/{workspace_id} first.")

    return rag.answer(artifacts_dir=str(paths.artifacts_dir), question=req.question, debug=req.debug)
