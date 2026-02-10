Mini-RAG Docs (Workspaces)

A FastAPI service for creating per-workspace document indexes and answering questions using a local LLM (Ollama) with embeddings.

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

Ollama settings are in `app/rag_workspace.py`:
- `ollama_url`: default `http://localhost:11434/api/generate`
- `ollama_model`: default `qwen2.5:3b-instruct`
- `temperature`: default `0.0`
- `num_predict`: default `180`

Embedding model: `intfloat/multilingual-e5-small`

Storage

- Index artifacts: `artifacts/workspaces/{workspace_id}/`
- Uploaded files: `data/workspaces/{workspace_id}/raw/`

Troubleshooting

Ollama not responding: Ensure Ollama is running with `ollama serve`

Empty citations: Normal behavior; citations are filtered to only include referenced chunks

Windows JSON body errors: Use a JSON file with `--data-binary "@file.json"` instead of inline JSON
