# Mini-RAG Docs (Workspaces)

A FastAPI service for creating per-workspace document indexes and answering questions using a local LLM (Ollama) with embeddings.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
│  Upload API │ -> │   Parsers    │ -> │   Chunking    │ -> │  FAISS   │
│  (FastAPI)  │    │ MD/HTML/PDF  │    │ Token +Overlap│    │  Index   │
│             │    │  DOCX/TXT    │    │               │    │          │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────┘
                                                                      │
                                                                      ▼
                                                           ┌─────────────┐
                                                           │  Query API  │
                                                           │  (FastAPI)  │
                                                           └─────┬───────┘
                                                                 │
                                                                 ▼
                       ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────┐
                       │  Embed  │ -> │ Retrieve │ -> │   Rerank    │ -> │   LLM   │
                       │  (E5)   │    │  (FAISS) │    │ Cross-Enc.  │    │ (Ollama)│
                       └─────────┘    └──────────┘    └─────────────┘    └─────────┘
                                                                 │
                                                                 ▼
                                                      ┌────────────────────┐
                                                      │ Answer + Citations │
                                                      └────────────────────┘

```

## Key Features

- **Multi-workspace isolation** — each workspace has its own document store and index
- **5 file formats** — Markdown, HTML, TXT, PDF, DOCX
- **Two-stage retrieval** — FAISS vector search → cross-encoder reranking for precision
- **Multilingual** — E5-small embeddings with automatic language detection in prompts
- **Structured output** — JSON responses with answer, citations, and confidence level
- **Evaluation metrics** — built-in support for measuring retrieval and answer quality
- **Configurable via environment** — all key settings (models, URLs) via env vars
- **Docker-ready** — single command deployment with docker-compose

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (`intfloat/multilingual-e5-small`), CUDA auto-detected |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| LLM | Ollama (default: `qwen2.5:3b-instruct`) |
| Document Parsing | BeautifulSoup4, pypdf, python-docx |
| Evaluation | Precision, Recall, MRR, NDCG, Faithfulness, Answer Relevance, LLM-as-Judge |
| Testing | pytest (79 unit tests) |

Prerequisites

Python 3.10+

Ollama installed and running locally. Start Ollama:
```bash
ollama serve
```

Pull a compatible model:
```bash
ollama pull qwen2.5:3b-instruct
```

## Setup

### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### GPU Acceleration (Optional)

The project auto-detects CUDA at runtime. If a GPU is available, embedding and reranking models run on it automatically, significantly speeding up index building and queries.

By default, `requirements.txt` installs CPU-only PyTorch. To enable GPU acceleration, reinstall PyTorch with CUDA support after the base install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version if different (e.g. `cu124`, `cu128`). Check your CUDA driver version with `nvidia-smi`.

## Docker Setup

### Prerequisites for Docker:
- Docker and Docker Compose installed
- Ollama running on host machine

Build and run with Docker Compose:
```bash
docker-compose up -d
```

Or build and run manually:
```bash
# Build image
docker build -t mini-rag-docs .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data/workspaces:/app/data/workspaces \
  -v $(pwd)/artifacts/workspaces:/app/artifacts/workspaces \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  --add-host host.docker.internal:host-gateway \
  mini-rag-docs
```

### Windows (PowerShell):
```powershell
docker run -d `
  -p 8000:8000 `
  -v ${PWD}/data/workspaces:/app/data/workspaces `
  -v ${PWD}/artifacts/workspaces:/app/artifacts/workspaces `
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate `
  --add-host host.docker.internal:host-gateway `
  mini-rag-docs
```

The API will be available at `http://localhost:8000`

Stop container:
```bash
docker-compose down
```

## Testing

Run the test suite:
```bash
pip install pytest
python -m pytest tests/ -v
```

## API Endpoints

- `POST /workspaces` - create a new workspace
- `GET /status/{workspace_id}` - check workspace status and index availability
- `POST /upload/{workspace_id}` - upload files (md, txt, html, pdf, docx)
- `POST /upload_dir/{workspace_id}` - upload all supported files from a local directory
- `POST /build_index/{workspace_id}` - build the FAISS index for the workspace
- `POST /query` - ask a question about uploaded documents

## Complete Workflow Example

1. Create workspace
```bash
curl.exe -X POST http://127.0.0.1:8000/workspaces
```
Response: `{"workspace_id":"WRPgUdZzpPAoPyYv"}`

2. Upload files
```bash
curl.exe -X POST http://127.0.0.1:8000/upload/WRPgUdZzpPAoPyYv -F "files=@document.pdf"
```

Or upload an entire directory:
```bash
curl.exe -X POST http://127.0.0.1:8000/upload_dir/WRPgUdZzpPAoPyYv -H "Content-Type: application/json" -d "{\"directory\": \"C:\\Users\\Admin\\Desktop\\company_policies\"}"
```

3. Check status
```bash
curl.exe http://127.0.0.1:8000/status/WRPgUdZzpPAoPyYv
```

4. Build index
```bash
curl.exe -X POST http://127.0.0.1:8000/build_index/WRPgUdZzpPAoPyYv
```

5. Query (Linux/macOS)
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"WRPgUdZzpPAoPyYv","question":"What is this document about?","debug":false}'
```

6. Query (Windows PowerShell)
```powershell
# Create query.json
@'
{"workspace_id":"WRPgUdZzpPAoPyYv","question":"What is this document about?","debug":false}
'@ | Set-Content query.json

# Send request
curl.exe -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" --data-binary "@query.json"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | LLM model name |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |

Additional settings in `app/rag_workspace.py`:
- `temperature`: default `0.0`
- `num_predict`: default `180`

## Storage

- Index artifacts: `artifacts/workspaces/{workspace_id}/`
- Uploaded files: `data/workspaces/{workspace_id}/raw/`

## Project Structure

```
mini-rag-docs/
├── app/
│   ├── main.py              # FastAPI routes and endpoints
│   ├── rag_workspace.py     # RAG engine (retrieve → rerank → generate)
│   ├── workspaces.py        # Workspace ID generation and path management
│   ├── prompts.py           # System prompts and prompt builders
│   └── evaluation.py        # Evaluation metrics and utilities
├── ingest/
│   ├── build_index_lib.py   # FAISS index building pipeline
│   ├── parsers.py           # Multi-format document parsers
│   └── chunking.py          # Token-based chunking with overlap
├── tests/                   # Unit tests (pytest)
├── examples/
│   ├── run_evaluation.py    # Example evaluation script
│   ├── evaluation_dataset_example.json  # Sample evaluation dataset
│   └── evaluation_dataset_bcc.json      # Real evaluation dataset (25 questions, BCC docs)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Evaluation

The system includes built-in evaluation metrics to measure RAG performance:

### Retrieval Metrics
- **Precision@K** — fraction of retrieved chunks that are relevant
- **Recall@K** — fraction of relevant chunks that were retrieved
- **MRR (Mean Reciprocal Rank)** — measures how high the first relevant result appears
- **NDCG@K** — considers both relevance and ranking position

### Answer Quality Metrics (Embedding-Based)
- **Faithfulness** — cosine similarity between the answer and retrieved context embeddings; measures if the answer is grounded in what was retrieved
- **Answer Relevance** — weighted cosine similarity between the answer, the question, and the ground truth answer (40% question similarity + 60% ground truth similarity)

### LLM-as-Judge Metrics (optional)
When `--judge` is passed, each answer is scored by a second LLM call on a 1-5 integer scale:
- **Faithfulness** — is the answer supported by the retrieved context?
- **Relevance** — does the answer address the question?
- **Completeness** — does the answer cover the key points of the ground-truth?

The judge prompt enforces structured JSON output and uses temperature 0 for reproducibility. You can specify a different (larger) model for judging with `--judge-model`.

### Running Evaluation

1. Prepare an evaluation dataset (JSON format):
```json
[
  {
    "question": "What is RAG?",
    "ground_truth_answer": "RAG is Retrieval-Augmented Generation...",
    "relevant_chunk_ids": ["doc1::chunk0001", "doc2::chunk0005"],
    "workspace_id": "your_workspace_id",
    "metadata": {"category": "definition"}
  }
]
```

2. Run the evaluation script:
```bash
python examples/run_evaluation.py \
  --dataset examples/evaluation_dataset_example.json \
  --output results.json \
  --k 5 \
  --verbose
```

With LLM-as-judge scoring:
```bash
python examples/run_evaluation.py \
  --dataset examples/evaluation_dataset_bcc.json \
  --output results.json \
  --k 5 \
  --judge \
  --judge-model qwen2.5:3b-instruct \
  --verbose
```

3. View results:
```bash
cat results.json
```

The results include both aggregated metrics and detailed per-question results.

### Using Evaluation in Code

```python
from app.evaluation import (
    load_evaluation_dataset,
    evaluate_rag_system,
    save_evaluation_results
)

# Load dataset
items = load_evaluation_dataset("eval_data.json")

# Create RAG function wrapper
def rag_fn(workspace_id, question):
    # Your RAG logic here
    return {
        'answer': '...',
        'citations': [...],
        'retrieved_chunks': [...]
    }

# Run evaluation
results = evaluate_rag_system(items, rag_fn, k=5)

# Save results
save_evaluation_results(results, "results.json")
```

### Experiment: Bank CenterCredit Public Policies

We tested this RAG system on a dataset of 25 questions based on publicly available Bank CenterCredit (BCC) policy documents, including sustainability reports, climate strategy, information security policy, E&S risk management policy, and consolidated financial statements.

**Source documents:** [Google Drive](https://drive.google.com/drive/folders/1QWn1dX7XO0H_Co7U0GItJQC73-GlGzwd?usp=drive_link)

**Evaluation dataset:** `examples/evaluation_dataset_bcc.json` (24 questions across 7 categories: financial, climate strategy, ESG policy, information security, governance, social, and reporting)

**Results (K=5, 24 samples, embedding-based):**

| Metric | Score |
|--------|-------|
| Faithfulness (embedding) | 0.834 |
| Answer Relevance (embedding) | 0.900 |

> Retrieval metrics (Precision, Recall, MRR, NDCG) are implemented but omitted here — they require human-annotated `relevant_chunk_ids` to be meaningful. The codebase includes an optional **LLM-as-judge** mode (`--judge`) that scores each answer on faithfulness, relevance, and completeness (1-5 scale) for a more credible evaluation.

**Setup:** `qwen2.5:3b-instruct` via Ollama, `intfloat/multilingual-e5-small` embeddings, `ms-marco-MiniLM-L-6-v2` reranker, NVIDIA RTX 4050 GPU.

## Troubleshooting

**Ollama not responding:** Ensure Ollama is running with `ollama serve`

**Empty citations:** Normal behavior; citations are filtered to only include referenced chunks

**Windows JSON body errors:** Use a JSON file with `--data-binary "@file.json"` instead of inline JSON

**Slow index building:** Install PyTorch with CUDA support (see GPU Acceleration section above). Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
