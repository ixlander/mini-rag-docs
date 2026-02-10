import pandas as pd

df = pd.read_parquet("artifacts/chunks.parquet")

q = "парол"
hits = df[df["text"].str.contains(q, case=False, na=False)]

print("hits:", len(hits))
if len(hits):
    for _, r in hits.head(10).iterrows():
        print("\n---")
        print("chunk_id:", r["chunk_id"])
        print("title:", r["title"])
        print("section:", r["section"])
        print(r["text"][:400].replace("\n", " "))
