from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt


@dataclass(frozen=True)
class RAGConfig:
    artifacts_dir: str = "artifacts"
    embed_model: str = "intfloat/multilingual-e5-small"
    use_e5_prefix: bool = True
    
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    top_k: int = 20
    top_final: int = 6
    context_k: int = 5  

    min_retrieval_score: float = 0.25

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_timeout_s: int = 180


class LocalRAG:
    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        art = Path(cfg.artifacts_dir)

        self.index = faiss.read_index(str(art / "faiss.index"))
        self.df = pd.read_parquet(art / "chunks.parquet")
        self.id_map = json.loads((art / "id_map.json").read_text(encoding="utf-8"))

        self.embedder = SentenceTransformer(cfg.embed_model)
        self.reranker = CrossEncoder(cfg.rerank_model)

        self._chunk_by_id: Dict[str, Dict[str, Any]] = {}
        for row in self.df.to_dict(orient="records"):
            self._chunk_by_id[row["chunk_id"]] = row

    def _embed_query(self, query: str) -> np.ndarray:
        q = query.strip()
        if self.cfg.use_e5_prefix:
            q = f"query: {q}"
        emb = self.embedder.encode([q], convert_to_numpy=True).astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb

    def _retrieve(self, query_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search(query_emb, self.cfg.top_k)
        return D[0], I[0]

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)  # higher is better
        scored = list(zip(scores.tolist(), candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: self.cfg.top_final]

    def _ollama_generate(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
        }
        r = requests.post(self.cfg.ollama_url, json=payload, timeout=self.cfg.ollama_timeout_s)
        r.raise_for_status()
        return r.json().get("response", "")

    def answer(self, question: str, debug: bool = False) -> Dict[str, Any]:
        t0 = time.time()

        q_emb = self._embed_query(question)
        t_embed = time.time()

        scores, vids = self._retrieve(q_emb)
        t_retr = time.time()

        candidates: List[Dict[str, Any]] = []
        best_score = float(scores[0]) if len(scores) else -1.0
        for score, vid in zip(scores, vids):
            if vid < 0:
                continue
            chunk_id = self.id_map.get(str(int(vid)))
            if not chunk_id:
                continue
            row = self._chunk_by_id.get(chunk_id)
            if not row:
                continue
            c = dict(row)
            c["retrieval_score"] = float(score)
            candidates.append(c)

        if not candidates or best_score < self.cfg.min_retrieval_score:
            result = {
                "answer": "I couldn't find this in the documentation.",
                "citations": [],
                "confidence": "low",
            }
            if debug:
                result["debug"] = {
                    "best_retrieval_score": best_score,
                    "timing_ms": {
                        "embed": int((t_embed - t0) * 1000),
                        "retrieve": int((t_retr - t_embed) * 1000),
                    },
                }
            return result

        reranked = self._rerank(question, candidates)
        t_rerank = time.time()

        top_chunks = [c for _, c in reranked[: self.cfg.context_k]]
        context_block = build_context_block(top_chunks)

        user_prompt = build_user_prompt(question=question, context_block=context_block)

        raw = self._ollama_generate(user_prompt)
        t_gen = time.time()

        parsed = self._safe_parse_json(raw)
        if parsed is None:
            parsed = {
                "answer": raw.strip() if raw.strip() else "I couldn't find this in the documentation.",
                "citations": [c["chunk_id"] for c in top_chunks],
                "confidence": "medium",
            }

        if not isinstance(parsed.get("citations"), list):
            parsed["citations"] = []
        parsed["citations"] = [c for c in parsed["citations"] if isinstance(c, str)]

        if debug:
            parsed["debug"] = {
                "best_retrieval_score": best_score,
                "top_retrieval": [
                    {
                        "chunk_id": c["chunk_id"],
                        "retrieval_score": c.get("retrieval_score"),
                        "title": c.get("title"),
                        "section": c.get("section"),
                    }
                    for c in candidates[: min(5, len(candidates))]
                ],
                "reranked": [
                    {
                        "chunk_id": c["chunk_id"],
                        "rerank_score": float(s),
                        "title": c.get("title"),
                        "section": c.get("section"),
                    }
                    for s, c in reranked
                ],
                "timing_ms": {
                    "embed": int((t_embed - t0) * 1000),
                    "retrieve": int((t_retr - t_embed) * 1000),
                    "rerank": int((t_rerank - t_retr) * 1000),
                    "generate": int((t_gen - t_rerank) * 1000),
                    "total": int((t_gen - t0) * 1000),
                },
            }

        return parsed

    @staticmethod
    def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return None

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None