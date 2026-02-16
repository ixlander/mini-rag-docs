from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt, build_conversation_history

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceRAGConfig:
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"))
    use_e5_prefix: bool = True
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    top_k: int = 8
    top_final: int = 4
    context_k: int = 3
    min_retrieval_score: float = 0.25
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"))
    ollama_timeout_s: int = 180
    num_predict: int = 180
    temperature: float = 0.0


class WorkspaceRAG:
    def __init__(self, cfg: WorkspaceRAGConfig) -> None:
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        logger.info("Loading embedding model: %s", cfg.embed_model)
        self.embedder = SentenceTransformer(cfg.embed_model, device=device)
        logger.info("Loading reranker model: %s", cfg.rerank_model)
        self.reranker = CrossEncoder(cfg.rerank_model, device=device)
        self._artifact_cache: Dict[str, Dict[str, Any]] = {}
        self._warmup()

    def _warmup(self) -> None:
        logger.info("Warming up models...")
        t0 = time.perf_counter()
        self.embedder.encode(["warmup"], convert_to_numpy=True)
        self.reranker.predict([("warmup", "warmup")])
        logger.info("Warmup done in %.2fs", time.perf_counter() - t0)

    def _load_artifacts(self, artifacts_dir: str) -> Tuple[faiss.Index, pd.DataFrame, Dict[str, str], Dict[str, Dict[str, Any]]]:
        art = Path(artifacts_dir)
        index_path = art / "faiss.index"
        mtime = index_path.stat().st_mtime

        cached = self._artifact_cache.get(artifacts_dir)
        if cached and cached["mtime"] == mtime:
            logger.debug("Using cached artifacts for %s", artifacts_dir)
            return cached["index"], cached["df"], cached["id_map"], cached["chunk_by_id"]

        logger.info("Loading artifacts from disk for %s", artifacts_dir)
        index = faiss.read_index(str(index_path))
        df = pd.read_parquet(art / "chunks.parquet")
        id_map = json.loads((art / "id_map.json").read_text(encoding="utf-8"))
        chunk_by_id: Dict[str, Dict[str, Any]] = {
            str(r["chunk_id"]): {str(k): v for k, v in r.items()}
            for r in df.to_dict(orient="records")
        }
        self._artifact_cache[artifacts_dir] = {
            "mtime": mtime, "index": index, "df": df,
            "id_map": id_map, "chunk_by_id": chunk_by_id,
        }
        return index, df, id_map, chunk_by_id

    def invalidate_cache(self, artifacts_dir: str) -> None:
        self._artifact_cache.pop(artifacts_dir, None)

    def _embed_query(self, query: str) -> np.ndarray:
        q = query.strip()
        if self.cfg.use_e5_prefix:
            q = f"query: {q}"
        emb = self.embedder.encode([q], convert_to_numpy=True).astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores.tolist(), candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: self.cfg.top_final]

    def _ollama_generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "format": "json",
            "stream": stream,
            "options": {
                "temperature": float(self.cfg.temperature),
                "num_predict": int(self.cfg.num_predict),
            },
        }
        logger.debug("Sending request to Ollama at %s", self.cfg.ollama_url)
        if not stream:
            r = requests.post(self.cfg.ollama_url, json=payload, timeout=self.cfg.ollama_timeout_s)
            r.raise_for_status()
            return r.json().get("response", "")
        return self._stream_ollama(payload)

    def _stream_ollama(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        with requests.post(
            self.cfg.ollama_url, json=payload,
            timeout=self.cfg.ollama_timeout_s, stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token

    @staticmethod
    def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(s[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    def answer(self, artifacts_dir: str, question: str, debug: bool = False, conversation_history: str = "") -> Dict[str, Any]:
        logger.info("Answering question for workspace at %s", artifacts_dir)
        index, _, id_map, chunk_by_id = self._load_artifacts(artifacts_dir)

        q_emb = self._embed_query(question)
        D, I = index.search(q_emb, self.cfg.top_k)
        scores, vids = D[0], I[0]

        candidates: List[Dict[str, Any]] = []
        best_score = float(scores[0]) if len(scores) else -1.0

        for score, vid in zip(scores, vids):
            if int(vid) < 0:
                continue
            chunk_id = id_map.get(str(int(vid)))
            if not chunk_id:
                continue
            row = chunk_by_id.get(chunk_id)
            if not row:
                continue
            c = dict(row)
            c["retrieval_score"] = float(score)
            candidates.append(c)

        if not candidates or best_score < self.cfg.min_retrieval_score:
            out: Dict[str, Any] = {"answer": "I couldn't find this in the documentation.", "citations": [], "confidence": "low"}
            if debug:
                out["debug"] = {"best_retrieval_score": best_score}
            return out

        reranked = self._rerank(question, candidates)
        top_chunks = [c for _, c in reranked[: self.cfg.context_k]]
        context_block = build_context_block(top_chunks)
        prompt = build_user_prompt(question=question, context_block=context_block, conversation_history=conversation_history)

        raw = self._ollama_generate(prompt)
        parsed = self._safe_parse_json(raw) or {"answer": raw.strip(), "citations": [c["chunk_id"] for c in top_chunks], "confidence": "medium"}

        allowed = {c["chunk_id"] for c in top_chunks}
        cits = parsed.get("citations")
        if not isinstance(cits, list):
            cits = []
        cits = [c for c in cits if isinstance(c, str) and c in allowed]
        parsed["citations"] = cits

        if debug:
            parsed["debug"] = {
                "best_retrieval_score": best_score,
                "context_preview": [
                    {"chunk_id": c["chunk_id"], "title": c.get("title"), "section": c.get("section"), "text_preview": (c.get("text") or "")[:220].replace("\n", " ")}
                    for c in top_chunks
                ],
                "raw_model_output_preview": (raw or "")[:500],
            }

        return parsed
    def answer_stream(self, artifacts_dir: str, question: str, conversation_history: str = "") -> Generator[str, None, None]:
        index, _, id_map, chunk_by_id = self._load_artifacts(artifacts_dir)
        q_emb = self._embed_query(question)
        D, I = index.search(q_emb, self.cfg.top_k)
        scores, vids = D[0], I[0]

        candidates: List[Dict[str, Any]] = []
        best_score = float(scores[0]) if len(scores) else -1.0

        for score, vid in zip(scores, vids):
            if int(vid) < 0:
                continue
            chunk_id = id_map.get(str(int(vid)))
            if not chunk_id:
                continue
            row = chunk_by_id.get(chunk_id)
            if not row:
                continue
            c = dict(row)
            c["retrieval_score"] = float(score)
            candidates.append(c)

        if not candidates or best_score < self.cfg.min_retrieval_score:
            yield json.dumps({"answer": "I couldn't find this in the documentation.", "citations": [], "confidence": "low"})
            return

        reranked = self._rerank(question, candidates)
        top_chunks = [c for _, c in reranked[: self.cfg.context_k]]
        context_block = build_context_block(top_chunks)
        prompt = build_user_prompt(question=question, context_block=context_block, conversation_history=conversation_history)

        for token in self._ollama_generate(prompt, stream=True):
            yield token