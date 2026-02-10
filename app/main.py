from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import LocalRAG, RAGConfig

app = FastAPI(title="Mini-RAG Docs (Local LLM)")

rag = LocalRAG(RAGConfig())


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    debug: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    return rag.answer(req.question, debug=req.debug)
