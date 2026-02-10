import requests
import streamlit as st

st.set_page_config(page_title="Mini-RAG Workspaces", layout="wide")

API_BASE_DEFAULT = "http://127.0.0.1:8000"


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_files(url: str, files, timeout: int = 600) -> dict:
    r = requests.post(url, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()


def create_workspace(api_base: str) -> str:
    data = post_json(f"{api_base}/workspaces", {})
    return data["workspace_id"]


def upload_to_workspace(api_base: str, workspace_id: str, uploaded_files) -> dict:
    files = []
    for f in uploaded_files:
        files.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
    return post_files(f"{api_base}/upload/{workspace_id}", files=files)


def build_index(api_base: str, workspace_id: str) -> dict:
    return post_json(f"{api_base}/build_index/{workspace_id}", {})


def query(api_base: str, workspace_id: str, question: str, debug: bool) -> dict:
    return post_json(
        f"{api_base}/query",
        {"workspace_id": workspace_id, "question": question, "debug": debug},
        timeout=600,
    )


def render_pills(conf: str | None, citations: list[str] | None):
    cols = st.columns([1, 4])
    with cols[0]:
        if conf:
            st.markdown(f"**confidence:** {conf}")
    with cols[1]:
        if citations:
            st.markdown("**Citations:**")
            st.code("\n".join(citations), language="text")


def render_debug(dbg: dict):
    with st.expander("Debug"):
        timing = dbg.get("timing_ms")
        if timing:
            st.markdown("**Timing (ms)**")
            st.json(timing)

        top_ret = dbg.get("top_retrieval")
        if top_ret:
            st.markdown("**Top retrieval**")
            st.json(top_ret)

        reranked = dbg.get("reranked")
        if reranked:
            st.markdown("**Reranked**")
            st.json(reranked)

        ctx_prev = dbg.get("context_preview")
        if ctx_prev:
            st.markdown("**Context preview (sent to LLM)**")
            for c in ctx_prev:
                st.markdown(f"**{c.get('chunk_id','')}** — {c.get('title','')} / {c.get('section','')}")
                st.write(c.get("text_preview", ""))

        raw_prev = dbg.get("raw_model_output_preview")
        if raw_prev:
            st.markdown("**Raw model output preview**")
            st.code(raw_prev, language="text")


st.title("Mini-RAG Docs (Workspaces)")

with st.sidebar:
    st.subheader("Connection")
    api_base = st.text_input("FastAPI base URL", value=API_BASE_DEFAULT)
    debug_mode = st.toggle("Debug", value=True)

    st.divider()
    st.subheader("Workspace")

    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = ""

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Create new"):
            try:
                wid = create_workspace(api_base)
                st.session_state.workspace_id = wid
                st.success(f"workspace_id: {wid}")
            except Exception as e:
                st.error(str(e))

    with col2:
        if st.button("Clear chat"):
            st.session_state.messages = []

    st.session_state.workspace_id = st.text_input(
        "workspace_id",
        value=st.session_state.workspace_id,
        placeholder="Create a workspace or paste an id",
    )

    st.divider()
    st.subheader("Upload & Index")
    uploaded = st.file_uploader(
        "Upload files",
        type=["md", "markdown", "txt", "html", "htm", "pdf", "docx"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Upload files", disabled=not (uploaded and st.session_state.workspace_id)):
            try:
                res = upload_to_workspace(api_base, st.session_state.workspace_id, uploaded)
                st.success(f"Uploaded: {len(res.get('saved', []))}")
                st.json(res)
            except Exception as e:
                st.error(str(e))

    with c2:
        if st.button("Build index", disabled=not st.session_state.workspace_id):
            try:
                res = build_index(api_base, st.session_state.workspace_id)
                st.success("Index built")
                st.json(res)
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption("Backend")
    st.code("uvicorn app.main:app --reload", language="bash")


if "messages" not in st.session_state:
    st.session_state.messages = []


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        payload = m.get("payload")
        if payload:
            render_pills(payload.get("confidence"), payload.get("citations"))
            dbg = payload.get("debug")
            if isinstance(dbg, dict):
                render_debug(dbg)


question = st.chat_input("Ask a question about YOUR uploaded docs…")

if question:
    if not st.session_state.workspace_id:
        st.error("Create/paste a workspace_id first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    payload = query(api_base, st.session_state.workspace_id, question, debug_mode)
                except requests.RequestException as e:
                    st.error(f"API error: {e}")
                    payload = {"answer": "API error", "citations": [], "confidence": "low"}

            answer = (payload.get("answer") or "").strip() or "No answer."
            st.markdown(answer)
            render_pills(payload.get("confidence"), payload.get("citations"))
            dbg = payload.get("debug")
            if isinstance(dbg, dict):
                render_debug(dbg)

        st.session_state.messages.append({"role": "assistant", "content": answer, "payload": payload})
