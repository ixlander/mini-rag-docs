Mini-RAG Docs (Workspaces)

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
- **Configurable via environment** — all key settings (models, URLs) via env vars
- **Docker-ready** — single command deployment with docker-compose

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (`intfloat/multilingual-e5-small`) |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| LLM | Ollama (default: `qwen2.5:3b-instruct`) |
| Document Parsing | BeautifulSoup4, pypdf, python-docx |
| Testing | pytest (45 unit tests) |

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

Setup

Windows
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

Docker Setup

Prerequisites for Docker:
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

Windows (PowerShell):
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

API Endpoints

- `POST /workspaces` - create a new workspace
- `GET /status/{workspace_id}` - check workspace status and index availability
- `POST /upload/{workspace_id}` - upload files (md, txt, html, pdf, docx)
- `POST /build_index/{workspace_id}` - build the FAISS index for the workspace
- `POST /query` - ask a question about uploaded documents

Complete Workflow Example

1. Create workspace
```bash
curl.exe -X POST http://127.0.0.1:8000/workspaces
```
Response: `{"workspace_id":"WRPgUdZzpPAoPyYv"}`

2. Upload files
```bash
curl.exe -X POST http://127.0.0.1:8000/upload/WRPgUdZzpPAoPyYv -F "files=@document.pdf"
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

Configuration

Settings can be configured via environment variables or in code:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | LLM model name |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |

Additional settings in `app/rag_workspace.py`:
- `temperature`: default `0.0`
- `num_predict`: default `180`

Storage

- Index artifacts: `artifacts/workspaces/{workspace_id}/`
- Uploaded files: `data/workspaces/{workspace_id}/raw/`

## Project Structure

```
mini-rag-docs/
├── app/
│   ├── main.py              # FastAPI routes and endpoints
│   ├── rag_workspace.py     # RAG engine (retrieve → rerank → generate)
│   ├── workspaces.py        # Workspace ID generation and path management
│   └── prompts.py           # System prompts and prompt builders
├── ingest/
│   ├── build_index_lib.py   # FAISS index building pipeline
│   ├── parsers.py           # Multi-format document parsers
│   └── chunking.py          # Token-based chunking with overlap
├── tests/                   # Unit tests (pytest)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Troubleshooting

Ollama not responding: Ensure Ollama is running with `ollama serve`

Empty citations: Normal behavior; citations are filtered to only include referenced chunks

Windows JSON body errors: Use a JSON file with `--data-binary "@file.json"` instead of inline JSON
