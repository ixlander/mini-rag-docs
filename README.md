Mini-RAG Docs (Workspaces)

This project provides a simple FastAPI service to create per-workspace document indexes and answer questions using a local LLM (Ollama) and embeddings.

Requirements
- Python 3.10+
- Ollama running locally

Setup (Windows)
1. Create and activate a virtual environment
	- `python -m venv .venv`
	- `.\.venv\Scripts\activate`
2. Install dependencies
	- `pip install -r requirements.txt`
3. Start the API
	- `uvicorn app.main:app --reload`

Setup (Linux/macOS)
1. Create and activate a virtual environment
	- `python3 -m venv .venv`
	- `source .venv/bin/activate`
2. Install dependencies
	- `pip install -r requirements.txt`
3. Start the API
	- `uvicorn app.main:app --reload`

API overview
- `POST /workspaces` -> create a workspace
- `POST /upload/{workspace_id}` -> upload files (md, txt, html, pdf, docx)
- `POST /build_index/{workspace_id}` -> build the index for the workspace
- `POST /query` -> ask a question
- `GET /status/{workspace_id}` -> check workspace files and index status

Example workflow (curl)
1. Create a workspace
	- `curl -X POST http://127.0.0.1:8000/workspaces`
2. Upload files
	- `curl -X POST http://127.0.0.1:8000/upload/{workspace_id} -F "files=@/path/to/file.md"`
3. Build index
	- `curl -X POST http://127.0.0.1:8000/build_index/{workspace_id}`
4. Query
	- `curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"workspace_id":"...","question":"...","debug":false}'`

Notes
- The service expects Ollama at `http://localhost:11434`.
- Index artifacts are stored under `artifacts/workspaces/{workspace_id}`.
- Raw uploaded files are stored under `data/workspaces/{workspace_id}/raw`.
