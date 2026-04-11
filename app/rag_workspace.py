from __future__ import annotations

import json
import logging
import os
import hashlib
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

from app.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
from app.text_utils import redact_pii, tokenize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceRAGConfig:
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"))
    use_e5_prefix: bool = True
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    top_k: int = 8
    max_top_k: int = 20
    top_final: int = 4
    context_k: int = 3
    min_retrieval_score: float = 0.25
    min_rerank_score: float = -2.0
    hybrid_dense_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"))
    ollama_model_complex: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL_COMPLEX", os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")))
    ollama_timeout_s: int = 180
    num_predict: int = 180
    temperature: float = 0.0
    enable_answer_cache: bool = True
    answer_cache_ttl_s: int = 120


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
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._answer_cache: Dict[str, Dict[str, Any]] = {}
        self._warmup()

    def _warmup(self) -> None:
        logger.info("Warming up models...")
        t0 = time.perf_counter()
        self.embedder.encode(["warmup"], convert_to_numpy=True)
        self.reranker.predict([("warmup", "warmup")])
        logger.info("Warmup done in %.2fs", time.perf_counter() - t0)

    def _load_artifacts(self, artifacts_dir: str) -> Tuple[faiss.Index, pd.DataFrame, Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        art = Path(artifacts_dir)
        index_path = art / "faiss.index"
        mtime = index_path.stat().st_mtime

        cached = self._artifact_cache.get(artifacts_dir)
        if cached and cached["mtime"] == mtime:
            logger.debug("Using cached artifacts for %s", artifacts_dir)
            return cached["index"], cached["df"], cached["id_map"], cached["chunk_by_id"], cached["bm25"]

        logger.info("Loading artifacts from disk for %s", artifacts_dir)
        index = faiss.read_index(str(index_path))
        df = pd.read_parquet(art / "chunks.parquet")
        id_map = json.loads((art / "id_map.json").read_text(encoding="utf-8"))
        bm25_path = art / "bm25.json"
        bm25 = json.loads(bm25_path.read_text(encoding="utf-8")) if bm25_path.exists() else {}
        chunk_by_id: Dict[str, Dict[str, Any]] = {
            str(r["chunk_id"]): {str(k): v for k, v in r.items()}
            for r in df.to_dict(orient="records")
        }
        self._artifact_cache[artifacts_dir] = {
            "mtime": mtime, "index": index, "df": df,
            "id_map": id_map, "chunk_by_id": chunk_by_id,
            "bm25": bm25,
        }
        return index, df, id_map, chunk_by_id, bm25

    def invalidate_cache(self, artifacts_dir: str) -> None:
        self._artifact_cache.pop(artifacts_dir, None)

    def _embed_query(self, query: str) -> np.ndarray:
        q = query.strip()
        if self.cfg.use_e5_prefix:
            q = f"query: {q}"
        cached = self._embedding_cache.get(q)
        if cached is not None:
            return cached
        emb = self.embedder.encode([q], convert_to_numpy=True).astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        self._embedding_cache[q] = emb
        return emb

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return tokenize_text(text)

    @staticmethod
    def _redact_pii(text: str) -> str:
        return redact_pii(text)

    def _analyze_query(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        ql = q.lower()
        is_cyr = bool(any("а" <= ch <= "я" or ch == "ё" for ch in ql))
        language = "ru" if is_cyr else "en"
        intent = "factoid"
        if any(k in ql for k in ("list", "перечис", "steps", "шаг")):
            intent = "list"
        elif any(k in ql for k in ("compare", "difference", "сравн")):
            intent = "comparison"
        elif any(k in ql for k in ("what is", "что такое", "define", "опред")):
            intent = "definition"
        tokens = self._tokenize(ql)
        ambiguous = len(tokens) < 4 or any(k in ql for k in ("it", "that", "this", "это", "то"))
        return {"language": language, "intent": intent, "ambiguous": ambiguous, "token_count": len(tokens)}

    def _rewrite_query(self, question: str, analysis: Dict[str, Any], conversation_history: str = "") -> str:
        q = (question or "").strip()
        if not analysis.get("ambiguous"):
            return q
        if conversation_history:
            history = conversation_history.splitlines()[-2:]
            if history:
                return f"{q}\n\nContext hint:\n" + "\n".join(history)
        return q

    def _dynamic_top_k(self, analysis: Dict[str, Any]) -> int:
        base = self.cfg.top_k
        tokens = int(analysis.get("token_count", 0))
        if tokens > 15:
            base += 4
        if analysis.get("intent") == "comparison":
            base += 2
        if analysis.get("ambiguous"):
            base += 2
        return max(4, min(self.cfg.max_top_k, base))

    def _bm25_rank(self, question: str, bm25: Dict[str, Any], limit: int) -> Dict[str, float]:
        docs = bm25.get("docs") or []
        if not docs:
            return {}
        q_tokens = self._tokenize(question)
        if not q_tokens:
            return {}
        df = bm25.get("df") or {}
        n_docs = max(1, int(bm25.get("N", len(docs))))
        avgdl = max(1e-9, float(bm25.get("avgdl", 1.0)))
        k1 = float(bm25.get("k1", 1.5))
        b = float(bm25.get("b", 0.75))
        scores: Dict[str, float] = {}
        for doc in docs:
            tf: Dict[str, int] = doc.get("tf", {})
            dl = max(1, int(doc.get("len", 0)))
            s = 0.0
            for tok in q_tokens:
                f = float(tf.get(tok, 0))
                if f <= 0.0:
                    continue
                dfi = int(df.get(tok, 0))
                idf = np.log(1.0 + ((n_docs - dfi + 0.5) / (dfi + 0.5)))
                den = f + k1 * (1.0 - b + b * (dl / avgdl))
                s += float(idf * ((f * (k1 + 1.0)) / max(1e-9, den)))
            if s > 0:
                scores[str(doc.get("chunk_id"))] = s
        if not scores:
            return {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return dict(ranked)

    @staticmethod
    def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def _retrieve_candidates(
        self,
        index: faiss.Index,
        id_map: Dict[str, str],
        chunk_by_id: Dict[str, Dict[str, Any]],
        bm25: Dict[str, Any],
        question: str,
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
        q_emb = self._embed_query(question)
        D, I = index.search(q_emb, top_k)
        dense_scores_raw: Dict[str, float] = {}
        dense_top = float(D[0][0]) if len(D[0]) else -1.0
        for score, vid in zip(D[0], I[0]):
            if int(vid) < 0:
                continue
            chunk_id = id_map.get(str(int(vid)))
            if chunk_id:
                dense_scores_raw[chunk_id] = float(score)
        bm25_scores_raw = self._bm25_rank(question, bm25, limit=top_k)
        dense_scores = self._normalize_scores(dense_scores_raw)
        bm25_scores = self._normalize_scores(bm25_scores_raw)
        all_chunk_ids = set(dense_scores) | set(bm25_scores)
        combined: List[Tuple[float, Dict[str, Any]]] = []
        for cid in all_chunk_ids:
            row = chunk_by_id.get(cid)
            if not row:
                continue
            ds = dense_scores.get(cid, 0.0)
            bs = bm25_scores.get(cid, 0.0)
            score = self.cfg.hybrid_dense_weight * ds + self.cfg.hybrid_bm25_weight * bs
            c = dict(row)
            c["retrieval_score"] = score
            c["dense_score"] = dense_scores_raw.get(cid, 0.0)
            c["bm25_score"] = bm25_scores_raw.get(cid, 0.0)
            combined.append((score, c))
        combined.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in combined[:top_k]]
        diagnostics = {
            "dense_top_score": dense_top,
            "dense_hits": len(dense_scores_raw),
            "bm25_hits": len(bm25_scores_raw),
            "candidate_count": len(candidates),
        }
        return candidates, dense_top, diagnostics

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores.tolist(), candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: self.cfg.top_final]

    def _select_model(self, analysis: Dict[str, Any]) -> str:
        if analysis.get("intent") in {"comparison", "list"} or analysis.get("token_count", 0) > 18:
            return self.cfg.ollama_model_complex
        return self.cfg.ollama_model

    def _ollama_generate(self, prompt: str, model: str, stream: bool = False) -> str | Generator[str, None, None]:
        payload = {
            "model": model,
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
        started = time.perf_counter()
        index, _, id_map, chunk_by_id, bm25 = self._load_artifacts(artifacts_dir)
        analysis = self._analyze_query(question)
        rewritten_question = self._rewrite_query(question, analysis, conversation_history)
        top_k = self._dynamic_top_k(analysis)

        cache_key = f"{artifacts_dir}|{rewritten_question}|{conversation_history}"
        cached = self._answer_cache.get(cache_key)
        if cached and (time.monotonic() - cached.get("ts", 0.0)) <= self.cfg.answer_cache_ttl_s:
            return cached["value"]

        candidates, best_score, retrieval_diag = self._retrieve_candidates(
            index=index,
            id_map=id_map,
            chunk_by_id=chunk_by_id,
            bm25=bm25,
            question=rewritten_question,
            top_k=top_k,
        )

        if not candidates or best_score < self.cfg.min_retrieval_score:
            out: Dict[str, Any] = {"answer": "I couldn't find this in the documentation.", "citations": [], "confidence": "low"}
            if debug:
                out["debug"] = {"best_retrieval_score": best_score, "query_analysis": analysis, "retrieval": retrieval_diag}
            return out

        reranked = self._rerank(rewritten_question, candidates)
        best_rerank = float(reranked[0][0]) if reranked else -999.0
        if best_rerank < self.cfg.min_rerank_score:
            out = {"answer": "I couldn't find this in the documentation.", "citations": [], "confidence": "low"}
            if debug:
                out["debug"] = {
                    "best_retrieval_score": best_score,
                    "best_rerank_score": best_rerank,
                    "query_analysis": analysis,
                    "retrieval": retrieval_diag,
                }
            return out
        top_chunks = [c for _, c in reranked[: self.cfg.context_k]]
        context_block = build_context_block(top_chunks)
        prompt = build_user_prompt(question=rewritten_question, context_block=context_block, conversation_history=conversation_history)
        model = self._select_model(analysis)

        raw = self._ollama_generate(prompt, model=model)
        parsed = self._safe_parse_json(raw) or {"answer": raw.strip(), "citations": [c["chunk_id"] for c in top_chunks], "confidence": "medium"}

        allowed = {c["chunk_id"] for c in top_chunks}
        cits = parsed.get("citations")
        if not isinstance(cits, list):
            cits = []
        cits = [c for c in cits if isinstance(c, str) and c in allowed]
        parsed["citations"] = cits
        parsed["answer"] = self._redact_pii(str(parsed.get("answer", "")))

        if not cits and parsed.get("confidence") != "low":
            parsed["confidence"] = "low"
            parsed["answer"] = "I couldn't find this in the documentation."

        parsed["provenance"] = {
            c["chunk_id"]: self._provenance_hash(c)
            for c in top_chunks
            if c["chunk_id"] in cits
        }

        if debug:
            parsed["debug"] = {
                "best_retrieval_score": best_score,
                "best_rerank_score": best_rerank,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "query_analysis": analysis,
                "rewritten_question": rewritten_question,
                "model": model,
                "retrieval": retrieval_diag,
                "prompt_version": "2026.1",
                "context_preview": [
                    {"chunk_id": c["chunk_id"], "title": c.get("title"), "section": c.get("section"), "text_preview": (c.get("text") or "")[:220].replace("\n", " ")}
                    for c in top_chunks
                ],
                "raw_model_output_preview": (raw or "")[:500],
            }

        if self.cfg.enable_answer_cache:
            self._answer_cache[cache_key] = {"ts": time.monotonic(), "value": parsed}
        return parsed
    def answer_stream(self, artifacts_dir: str, question: str, conversation_history: str = "") -> Generator[str, None, None]:
        index, _, id_map, chunk_by_id, bm25 = self._load_artifacts(artifacts_dir)
        analysis = self._analyze_query(question)
        rewritten_question = self._rewrite_query(question, analysis, conversation_history)
        top_k = self._dynamic_top_k(analysis)
        candidates, best_score, _ = self._retrieve_candidates(
            index=index,
            id_map=id_map,
            chunk_by_id=chunk_by_id,
            bm25=bm25,
            question=rewritten_question,
            top_k=top_k,
        )

        if not candidates or best_score < self.cfg.min_retrieval_score:
            yield json.dumps({"answer": "I couldn't find this in the documentation.", "citations": [], "confidence": "low"})
            return

        reranked = self._rerank(rewritten_question, candidates)
        top_chunks = [c for _, c in reranked[: self.cfg.context_k]]
        context_block = build_context_block(top_chunks)
        prompt = build_user_prompt(question=rewritten_question, context_block=context_block, conversation_history=conversation_history)
        model = self._select_model(analysis)

        for token in self._ollama_generate(prompt, model=model, stream=True):
            yield token
    @staticmethod
    def _provenance_hash(chunk: Dict[str, Any]) -> str:
        key = "|".join(
            [
                str(chunk.get("chunk_id", "")),
                str(chunk.get("doc_id", "")),
                str(chunk.get("title", "")),
                str((chunk.get("text") or "")[:200]),
            ]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
