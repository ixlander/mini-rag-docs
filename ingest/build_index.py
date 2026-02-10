from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.parsers import iter_docs
from ingest.chunking import chunk_corpus, Chunk


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    # x: (n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _format_for_e5(text: str) -> str:
    return f"passage: {text}"


def _build_embeddings(
    model_name: str,
    chunks: List[Chunk],
    batch_size: int,
    device: str | None,
    use_e5_prefix: bool,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)

    texts = []
    for c in chunks:
        t = c.text.strip()
        if use_e5_prefix:
            t = _format_for_e5(t)
        texts.append(t)

    emb_list: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False, 
        )
        emb_list.append(embs.astype(np.float32))

    E = np.vstack(emb_list)
    E = _normalize_l2(E)
    return E


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Cosine similarity = inner product for L2-normalized vectors.
    """
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    assert index.ntotal == n
    return index


def _chunks_to_dataframe(chunks: List[Chunk]) -> pd.DataFrame:
    rows = []
    for c in chunks:
        rows.append(
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_group": c.source_group,
                "title": c.title,
                "section": c.section,
                "text": c.text,
                "url": c.url,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Input docs root")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Output artifacts dir")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-small", help="Embedding model")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--device", type=str, default=None, help='e.g. "cpu" or "cuda". Default: auto')
    parser.add_argument("--max_tokens", type=int, default=550, help="Chunk size (approx tokens)")
    parser.add_argument("--overlap_tokens", type=int, default=80, help="Chunk overlap (approx tokens)")
    parser.add_argument(
        "--use_e5_prefix",
        action="store_true",
        help="Add 'passage: ' prefix for E5 models (recommended for intfloat/*e5*).",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    artifacts_dir = Path(args.artifacts_dir)
    _ensure_dir(artifacts_dir)

    use_e5_prefix = args.use_e5_prefix or ("e5" in args.model.lower())

    print(f"[1/5] Parsing docs from: {raw_dir}")
    docs = iter_docs(raw_dir)
    print(f"  docs: {len(docs)}")

    print(f"[2/5] Chunking (max_tokens={args.max_tokens}, overlap={args.overlap_tokens})")
    chunks = chunk_corpus(docs, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    print(f"  chunks: {len(chunks)}")
    if len(chunks) == 0:
        raise RuntimeError("No chunks produced. Add docs to data/raw/** and rerun.")

    print(f"[3/5] Building embeddings: model={args.model}, device={args.device or 'auto'}, e5_prefix={use_e5_prefix}")
    E = _build_embeddings(
        model_name=args.model,
        chunks=chunks,
        batch_size=args.batch_size,
        device=args.device,
        use_e5_prefix=use_e5_prefix,
    )
    print(f"  embeddings shape: {E.shape} dtype={E.dtype}")

    print("[4/5] Building FAISS index (IndexFlatIP)")
    index = _build_faiss_index(E)

    print("[5/5] Saving artifacts")
    index_path = artifacts_dir / "faiss.index"
    faiss.write_index(index, str(index_path))

    df = _chunks_to_dataframe(chunks)
    chunks_path = artifacts_dir / "chunks.parquet"
    df.to_parquet(chunks_path, index=False)

    # vector_id -> chunk_id
    id_map: Dict[int, str] = {i: chunks[i].chunk_id for i in range(len(chunks))}
    id_map_path = artifacts_dir / "id_map.json"
    id_map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "raw_dir": raw_dir,
        "embedding_model": args.model,
        "device": args.device,
        "use_e5_prefix": use_e5_prefix,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "embedding_dim": int(E.shape[1]),
    }
    (artifacts_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"  - {index_path}")
    print(f"  - {chunks_path}")
    print(f"  - {id_map_path}")
    print(f"  - {artifacts_dir / 'build_meta.json'}")


if __name__ == "__main__":
    main()
    
    