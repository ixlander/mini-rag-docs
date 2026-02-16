from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import logging

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from ingest.parsers import iter_docs
from ingest.chunking import chunk_corpus

logger = logging.getLogger(__name__)


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _format_for_e5_passage(text: str) -> str:
    return f"passage: {text}"


def build_index(
    raw_dir: str,
    artifacts_dir: str,
    embed_model: str = "intfloat/multilingual-e5-small",
    use_e5_prefix: bool = True,
    batch_size: int = 64,
    device: str | None = None,
    max_tokens: int = 550,
    overlap_tokens: int = 80,
) -> Dict[str, int]:
    art = Path(artifacts_dir)
    _ensure_dir(art)

    docs = iter_docs(raw_dir)
    chunks = chunk_corpus(docs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        raise RuntimeError("No chunks produced from uploaded documents.")

    device = _resolve_device(device)
    logger.info("Using device: %s", device)
    model = SentenceTransformer(embed_model, device=device)

    texts: List[str] = []
    for c in chunks:
        t = c.text.strip()
        if use_e5_prefix:
            t = _format_for_e5_passage(t)
        texts.append(t)

    emb_list: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        emb_list.append(embs)

    E = np.vstack(emb_list).astype(np.float32)
    E = _normalize_l2(E)

    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)

    faiss.write_index(index, str(art / "faiss.index"))

    df = pd.DataFrame(
        [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_group": c.source_group,
                "title": c.title,
                "section": c.section,
                "text": c.text,
                "url": c.url,
            }
            for c in chunks
        ]
    )
    df.to_parquet(art / "chunks.parquet", index=False)

    id_map = {i: chunks[i].chunk_id for i in range(len(chunks))}
    (art / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "embedding_dim": int(E.shape[1]),
    }
    (art / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"num_docs": len(docs), "num_chunks": len(chunks)}
