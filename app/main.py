from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from app.workspaces import make_workspace_id, get_paths
from ingest.build_index_lib import build_index
from app.rag_workspace import WorkspaceRAG, WorkspaceRAGConfig


app = FastAPI(title="Mini-RAG (Workspaces)")

rag = WorkspaceRAG(WorkspaceRAGConfig())


class CreateWorkspaceResponse(BaseModel):
    workspace_id: str


class BuildIndexResponse(BaseModel):
    workspace_id: str
    num_docs: int
    num_chunks: int


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


@app.post("/upload/{workspace_id}")
async def upload_files(workspace_id: str, files: List[UploadFile] = File(...)):
    paths = get_paths(workspace_id)

    allowed = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}
    saved = []

    for f in files:
        name = Path(f.filename or "file").name
        ext = Path(name).suffix.lower()
        if ext not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        dst = paths.raw_dir / name
        content = await f.read()
        dst.write_bytes(content)
        saved.append(name)

    return {"workspace_id": workspace_id, "saved": saved}


@app.post("/build_index/{workspace_id}", response_model=BuildIndexResponse)
def build_index_for_workspace(workspace_id: str):
    paths = get_paths(workspace_id)
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
    return {"workspace_id": workspace_id, "num_docs": stats["num_docs"], "num_chunks": stats["num_chunks"]}


@app.post("/query")
def query(req: QueryRequest):
    paths = get_paths(req.workspace_id)
    if not (paths.artifacts_dir / "faiss.index").exists():
        raise HTTPException(status_code=400, detail="Index not built for this workspace. Call /build_index/{workspace_id} first.")
    return rag.answer(artifacts_dir=str(paths.artifacts_dir), question=req.question, debug=req.debug)
