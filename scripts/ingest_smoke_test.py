import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.parsers import iter_docs
from ingest.chunking import chunk_corpus

docs = iter_docs("data/raw")
print("docs:", len(docs))

chunks = chunk_corpus(docs, max_tokens=550, overlap_tokens=80)
print("chunks:", len(chunks))

for c in chunks[:2]:
    print("\n---")
    print("chunk_id:", c.chunk_id)
    print("title:", c.title)
    print("section:", c.section)
    print("text_preview:", c.text[:200].replace("\n", " "))