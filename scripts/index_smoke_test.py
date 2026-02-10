import json
from pathlib import Path

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

ART = Path("artifacts")
index = faiss.read_index(str(ART / "faiss.index"))
df = pd.read_parquet(ART / "chunks.parquet")
id_map = json.loads((ART / "id_map.json").read_text(encoding="utf-8"))

model = SentenceTransformer("intfloat/multilingual-e5-small")
q = "query: how to reset password?"
q_emb = model.encode([q], convert_to_numpy=True).astype(np.float32)
q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

D, I = index.search(q_emb, 5)
for rank, (score, vid) in enumerate(zip(D[0], I[0]), start=1):
    if vid == -1:
        continue
    chunk_id = id_map[str(vid)]
    row = df[df["chunk_id"] == chunk_id].iloc[0]
    print("\n---")
    print("rank:", rank, "score:", float(score))
    print("chunk_id:", chunk_id)
    print("title:", row["title"])
    print("section:", row["section"])
    print("text:", row["text"][:220].replace("\n", " "))
