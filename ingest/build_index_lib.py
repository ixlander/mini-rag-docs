from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from collections import Counter

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


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 1]


def _build_bm25_meta(rows: List[Dict[str, object]]) -> Dict[str, object]:
    doc_freq: Counter[str] = Counter()
    docs_meta: List[Dict[str, object]] = []
    total_len = 0
    for row in rows:
        tokens = _tokenize(str(row.get("text", "")))
        tf = Counter(tokens)
        total_len += len(tokens)
        for tok in tf.keys():
            doc_freq[tok] += 1
        docs_meta.append({
            "chunk_id": row["chunk_id"],
            "len": len(tokens),
            "tf": dict(tf),
        })
    n_docs = max(1, len(docs_meta))
    avgdl = total_len / n_docs
    return {
        "version": 1,
        "k1": 1.5,
        "b": 0.75,
        "N": len(docs_meta),
        "avgdl": avgdl,
        "df": dict(doc_freq),
        "docs": docs_meta,
    }


def _doc_signature(source_path: str) -> str:
    st = Path(source_path).stat()
    return f"{int(st.st_mtime)}:{st.st_size}"


def build_index(
    raw_dir: str,
    artifacts_dir: str,
    embed_model: str = "intfloat/multilingual-e5-small",
    use_e5_prefix: bool = True,
    batch_size: int = 64,
    device: str | None = None,
    max_tokens: int = 550,
    overlap_tokens: int = 80,
    incremental: bool = True,
) -> Dict[str, int]:
    art = Path(artifacts_dir)
    _ensure_dir(art)

    docs = iter_docs(raw_dir)
    if not docs:
        raise RuntimeError("No supported documents found for indexing.")

    doc_sigs = {d.doc_id: _doc_signature(d.source_path) for d in docs}

    rows: List[Dict[str, object]] = []
    E_parts: List[np.ndarray] = []
    reused_chunks = 0
    changed_docs = docs

    prev_df_path = art / "chunks.parquet"
    prev_emb_path = art / "embeddings.npy"
    prev_sig_path = art / "doc_signatures.json"

    if (
        incremental
        and prev_df_path.exists()
        and prev_emb_path.exists()
        and prev_sig_path.exists()
    ):
        try:
            prev_df = pd.read_parquet(prev_df_path)
            prev_E = np.load(prev_emb_path)
            prev_sigs = json.loads(prev_sig_path.read_text(encoding="utf-8"))
            if len(prev_df) == len(prev_E):
                unchanged_doc_ids = {
                    doc_id
                    for doc_id, sig in doc_sigs.items()
                    if prev_sigs.get(doc_id) == sig
                }
                if unchanged_doc_ids:
                    mask = prev_df["doc_id"].isin(unchanged_doc_ids).to_numpy()
                    if mask.any():
                        unchanged_df = prev_df.loc[mask]
                        rows.extend(unchanged_df.to_dict(orient="records"))
                        E_parts.append(prev_E[mask].astype(np.float32))
                        reused_chunks = int(mask.sum())
                changed_docs = [d for d in docs if d.doc_id not in unchanged_doc_ids]
        except Exception:
            logger.exception("Incremental index load failed; falling back to full rebuild.")
            rows = []
            E_parts = []
            reused_chunks = 0
            changed_docs = docs

    new_chunks = chunk_corpus(changed_docs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if not rows and not new_chunks:
        raise RuntimeError("No chunks produced from uploaded documents.")

    if new_chunks:
        device = _resolve_device(device)
        logger.info("Using device: %s", device)
        model = SentenceTransformer(embed_model, device=device)
        texts: List[str] = []
        for c in new_chunks:
            t = c.text.strip()
            if use_e5_prefix:
                t = _format_for_e5_passage(t)
            texts.append(t)
        emb_list: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
            emb_list.append(embs)
        new_E = np.vstack(emb_list).astype(np.float32)
        new_E = _normalize_l2(new_E)
        E_parts.append(new_E)
        rows.extend(
            [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source_group": c.source_group,
                    "title": c.title,
                    "section": c.section,
                    "section_path": c.section_path,
                    "text": c.text,
                    "url": c.url,
                    "doc_type": c.doc_type,
                    "source_quality": float(c.source_quality),
                    "ingested_at": c.ingested_at,
                }
                for c in new_chunks
            ]
        )

    E = np.vstack(E_parts).astype(np.float32)
    E = _normalize_l2(E)

    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)

    faiss.write_index(index, str(art / "faiss.index"))

    df = pd.DataFrame(rows)
    if "section_path" not in df.columns:
        df["section_path"] = df.get("section", "Main")
    if "doc_type" not in df.columns:
        df["doc_type"] = "unknown"
    if "source_quality" not in df.columns:
        df["source_quality"] = 1.0
    if "ingested_at" not in df.columns:
        df["ingested_at"] = ""
    df.to_parquet(art / "chunks.parquet", index=False)
    np.save(art / "embeddings.npy", E)

    id_map = {i: str(df.iloc[i]["chunk_id"]) for i in range(len(df))}
    (art / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (art / "doc_signatures.json").write_text(json.dumps(doc_sigs, ensure_ascii=False, indent=2), encoding="utf-8")

    bm25_meta = _build_bm25_meta(rows)
    (art / "bm25.json").write_text(json.dumps(bm25_meta, ensure_ascii=False), encoding="utf-8")

    meta = {
        "num_docs": len(docs),
        "num_chunks": len(df),
        "embedding_dim": int(E.shape[1]),
        "reused_chunks": reused_chunks,
        "new_chunks": max(0, len(df) - reused_chunks),
        "incremental": bool(incremental),
    }
    (art / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"num_docs": len(docs), "num_chunks": len(df)}
