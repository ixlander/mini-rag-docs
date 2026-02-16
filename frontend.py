from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Docs", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{API_URL}{path}"
    return requests.request(method, url, timeout=300, **kwargs)


with st.sidebar:
    st.title("📄 RAG Docs")
    st.caption("Upload documents, build an index, ask questions.")

    st.divider()

    # workspace selector
    st.subheader("Workspace")

    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = ""

    wid = st.text_input("Workspace ID", value=st.session_state.workspace_id, placeholder="paste or create new")
    st.session_state.workspace_id = wid

    if st.button("➕ Create new workspace"):
        try:
            r = api("POST", "/workspaces")
            r.raise_for_status()
            new_id = r.json()["workspace_id"]
            st.session_state.workspace_id = new_id
            st.success(f"Created: {new_id}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    # workspace status
    if wid:
        try:
            r = api("GET", f"/status/{wid}")
            r.raise_for_status()
            info = r.json()
            col1, col2 = st.columns(2)
            col1.metric("Files", info["raw_files_count"])
            col2.metric("Index", "✅" if info["has_index"] else "❌")
        except Exception:
            st.warning("Could not fetch workspace status.")

    st.divider()

    # file upload
    st.subheader("Upload files")
    uploaded = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=["pdf", "docx", "md", "markdown", "txt", "html", "htm"],
    )

    if uploaded and wid:
        if st.button("📤 Upload"):
            files = [("files", (f.name, f.getvalue())) for f in uploaded]
            try:
                r = api("POST", f"/upload/{wid}", files=files)
                r.raise_for_status()
                data = r.json()
                st.success(f"Uploaded {data['count']} file(s)")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()

    # build index
    st.subheader("Build index")
    if wid and st.button("🔨 Build index"):
        with st.spinner("Building FAISS index..."):
            try:
                r = api("POST", f"/build_index/{wid}")
                r.raise_for_status()
                data = r.json()
                st.success(f"Indexed {data['num_docs']} docs → {data['num_chunks']} chunks")
                st.rerun()
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(detail or str(e))
            except Exception as e:
                st.error(str(e))


# ── chat area ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    if not wid:
        st.error("Select or create a workspace first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = api("POST", "/query", json={
                        "workspace_id": wid,
                        "question": prompt,
                        "debug": False,
                    })
                    r.raise_for_status()
                    data = r.json()

                    answer = data.get("answer", "No answer returned.")
                    citations = data.get("citations", [])

                    st.markdown(answer)

                    if citations:
                        with st.expander(f"📎 Citations ({len(citations)})"):
                            for c in citations:
                                title = c.get("title", "")
                                section = c.get("section", "")
                                label = title or section or c.get("chunk_id", "source")
                                st.markdown(f"**{label}**")
                                if "text_preview" in c:
                                    st.caption(c["text_preview"][:300])
                                st.divider()

                    full_reply = answer
                    if citations:
                        sources = ", ".join(
                            c.get("title", c.get("chunk_id", ""))
                            for c in citations
                        )
                        full_reply += f"\n\n*Sources: {sources}*"

                    st.session_state.messages.append({"role": "assistant", "content": full_reply})

                except requests.exceptions.HTTPError as e:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:
                        pass
                    err = detail or str(e)
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
                except Exception as e:
                    st.error(str(e))
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
