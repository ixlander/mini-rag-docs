# Mini-RAG Docs

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-67%20passed-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

A local-first RAG system that indexes documents per workspace and answers questions with citations — evaluated across 2 domains with 54 ground-truth questions.

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
| Evaluation | Faithfulness, Answer Relevance (embedding), LLM-as-Judge |
| Testing | pytest (67 unit tests) |

## Setup

### Prerequisites

- Python 3.10+
- Ollama installed and running (`ollama serve`)
- A compatible model pulled: `ollama pull qwen2.5:3b-instruct`

### Install & Run

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Quick Start

```bash
# 1. Create workspace
curl -X POST http://127.0.0.1:8000/workspaces
# → {"workspace_id":"WRPgUdZzpPAoPyYv"}

# 2. Upload files
curl -X POST http://127.0.0.1:8000/upload/WRPgUdZzpPAoPyYv -F "files=@document.pdf"

# 3. Build index
curl -X POST http://127.0.0.1:8000/build_index/WRPgUdZzpPAoPyYv

# 4. Query
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"WRPgUdZzpPAoPyYv","question":"What is this document about?"}'
```

<details>
<summary>Windows PowerShell syntax</summary>

```powershell
# Upload
curl.exe -X POST http://127.0.0.1:8000/upload/WRPgUdZzpPAoPyYv -F "files=@document.pdf"

# Query (write JSON to file to avoid escaping issues)
@'{"workspace_id":"WRPgUdZzpPAoPyYv","question":"What is this document about?"}'
'@ | Set-Content query.json
curl.exe -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" --data-binary "@query.json"
```
</details>

### GPU Acceleration (Optional)

The project auto-detects CUDA at runtime. If a GPU is available, embedding and reranking models run on it automatically, significantly speeding up index building and queries.

By default, `requirements.txt` installs CPU-only PyTorch. To enable GPU acceleration, reinstall PyTorch with CUDA support after the base install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version if different (e.g. `cu124`, `cu128`). Check your CUDA driver version with `nvidia-smi`.

## Frontend (Streamlit)

A browser-based chat UI is included. Start the API server first, then run:

```bash
streamlit run frontend.py
```

The UI opens at `http://localhost:8501` and connects to the API at `http://127.0.0.1:8000`.
Override the API URL with the `API_URL` environment variable if needed.

Features:
- Create / select workspaces
- Upload documents (drag & drop)
- Build FAISS index
- Chat with citations

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
├── frontend.py              # Streamlit chat UI
├── ingest/
│   ├── build_index_lib.py   # FAISS index building pipeline
│   ├── parsers.py           # Multi-format document parsers
│   └── chunking.py          # Token-based chunking with overlap
├── tests/                   # Unit tests (pytest)
├── examples/
│   ├── run_evaluation.py    # Example evaluation script
│   ├── evaluation_dataset_example.json  # Sample evaluation dataset
│   ├── evaluation_dataset_bcc.json      # BCC evaluation dataset (24 questions)
│   └── evaluation_dataset_privacy.json  # Privacy policy evaluation dataset (30 questions)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Evaluation

### Experiment 1: Bank CenterCredit Public Policies

We tested this RAG system on a dataset of 24 questions based on publicly available Bank CenterCredit (BCC) policy documents, including sustainability reports, climate strategy, information security policy, E&S risk management policy, and consolidated financial statements.

**Source documents:** [Google Drive](https://drive.google.com/drive/folders/1QWn1dX7XO0H_Co7U0GItJQC73-GlGzwd?usp=drive_link)

**Evaluation dataset:** `examples/evaluation_dataset_bcc.json` (24 questions across 7 categories: financial, climate strategy, ESG policy, information security, governance, social, and reporting)

**Results (24 samples):**

| Metric | Score |
|--------|-------|
| Faithfulness (embedding) | 0.837 |
| Answer Relevance (embedding) | 0.901 |
| Faithfulness (LLM judge, 1-5) | 2.08 |
| Relevance (LLM judge, 1-5) | 2.92 |
| Completeness (LLM judge, 1-5) | 1.62 |

**Setup:** `qwen2.5:3b-instruct` via Ollama, `intfloat/multilingual-e5-small` embeddings, `ms-marco-MiniLM-L-6-v2` reranker, NVIDIA RTX 4050 GPU.

### Experiment 2: Tech Company Privacy Policies

To validate cross-domain generalization, we evaluated the same pipeline on a completely different corpus — 97 annotated privacy policy PDFs from major tech companies and online platforms.

**Source:** [Annotated Privacy Policies of 100 Online Platforms](https://data.mendeley.com/datasets/pcgvm6zh43/1) (Mendeley Data, CC BY 4.0, DOI: `10.17632/pcgvm6zh43.1`)

**Corpus:** 97 PDFs from 28+ companies across 13 sectors (social media, streaming, e-commerce, fintech, gaming, etc.) → 2,429 chunks after indexing.

**Evaluation dataset:** `examples/evaluation_dataset_privacy.json` (30 questions across 7 categories: data collection, data sharing, data processing, data retention, user rights, cookies & tracking, children's privacy)

**Results (30 samples):**

| Metric | Score |
|--------|-------|
| Faithfulness (embedding) | 0.873 |
| Answer Relevance (embedding) | 0.922 |
| Faithfulness (LLM judge, 1-5) | 2.87 |
| Relevance (LLM judge, 1-5) | 3.40 |
| Completeness (LLM judge, 1-5) | 1.83 |

### Cross-Domain Comparison

| Metric | BCC (24 samples) | Privacy (30 samples) |
|--------|:-:|:-:|
| Faithfulness (embedding) | 0.837 | 0.873 |
| Answer Relevance (embedding) | 0.901 | 0.922 |
| Faithfulness (judge) | 2.08 | 2.87 |
| Relevance (judge) | 2.92 | 3.40 |
| Completeness (judge) | 1.62 | 1.83 |

The privacy policy corpus scored higher across all metrics, likely due to the structured and repetitive nature of privacy policy documents (standard GDPR/CCPA sections) compared to the more varied BCC financial/ESG reports.

### Metrics

**Embedding-based (automatic):**
- **Faithfulness** — cosine similarity between the answer and retrieved context; measures grounding
- **Answer Relevance** — weighted similarity between the answer, question, and ground truth (40/60 split)

**LLM-as-judge (optional, `--judge` flag):**
- **Faithfulness / Relevance / Completeness** — each scored 1–5 by a second LLM call with structured JSON output and temperature 0

<details>
<summary>Running your own evaluation</summary>

```bash
python examples/run_evaluation.py \
  --dataset examples/evaluation_dataset_bcc.json \
  --output results.json \
  --k 5 \
  --judge \
  --judge-model qwen2.5:3b-instruct \
  --verbose
```

Dataset format:
```json
[
  {
    "question": "What is RAG?",
    "ground_truth_answer": "RAG is Retrieval-Augmented Generation...",
    "workspace_id": "your_workspace_id",
    "metadata": {"category": "definition"}
  }
]
```
</details>

## Limitations

- **Small LLM** — the 3B-parameter model limits completeness scores; a larger model (7B+) would likely improve judge metrics
- **No authentication** — workspaces are not access-controlled; intended for local/internal use
- **No persistent database** — FAISS indexes live on disk; no metadata DB for search filtering
- **Single-query context** — no conversation memory across queries

## Troubleshooting

**Ollama not responding:** Ensure Ollama is running with `ollama serve`

**Empty citations:** Normal behavior; citations are filtered to only include referenced chunks

**Windows JSON body errors:** Use a JSON file with `--data-binary "@file.json"` instead of inline JSON

**Slow index building:** Install PyTorch with CUDA support (see GPU Acceleration section above). Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## License

MIT
