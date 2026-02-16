from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Docs", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{API_URL}{path}"
    headers = kwargs.pop("headers", {})
    if "api_key" in st.session_state and st.session_state.api_key:
        headers["Authorization"] = f"Bearer {st.session_state.api_key}"
    return requests.request(method, url, timeout=300, headers=headers, **kwargs)


# ── Initialise session state ─────────────────────────────────────────
for key, default in [
    ("api_key", ""),
    ("workspace_id", ""),
    ("conversation_id", None),
    ("messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.title("📄 RAG Docs")
    st.caption("Upload documents, build an index, ask questions.")

    st.divider()

    st.subheader("🔑 Authentication")
    api_key_input = st.text_input(
        "API Key", value=st.session_state.api_key, type="password", placeholder="paste your API key",
    )
    st.session_state.api_key = api_key_input

    col_reg, col_check = st.columns(2)
    with col_reg:
        reg_name = st.text_input("Name (optional)", key="reg_name", placeholder="your name")
        if st.button("📝 Register"):
            try:
                r = requests.post(f"{API_URL}/register", json={"name": reg_name}, timeout=30)
                r.raise_for_status()
                data = r.json()
                st.session_state.api_key = data["api_key"]
                st.success(f"Registered! Key: `{data['api_key']}`")
                st.rerun()
            except Exception as e:
                st.error(f"Registration failed: {e}")

    if not st.session_state.api_key:
        st.warning("Enter an API key or register to continue.")
        st.stop()

    st.divider()

    st.subheader("Workspace")

    wid = st.text_input("Workspace ID", value=st.session_state.workspace_id, placeholder="paste or create new")
    st.session_state.workspace_id = wid

    if st.button("➕ Create new workspace"):
        try:
            r = api("POST", "/workspaces", json={"description": ""})
            r.raise_for_status()
            new_id = r.json()["workspace_id"]
            st.session_state.workspace_id = new_id
            st.session_state.conversation_id = None
            st.session_state.messages = []
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
            col1.metric("Docs", info.get("document_count", 0))
            col2.metric("Index", "✅" if info["has_index"] else "❌")
        except Exception:
            st.warning("Could not fetch workspace status.")

    st.divider()

    # ── File upload ─────────────────────────────────────────────────
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

    # ── Build index ─────────────────────────────────────────────────
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

    st.divider()

    st.subheader("💬 Conversations")

    if wid and st.button("➕ New conversation"):
        try:
            r = api("POST", "/conversations", json={"workspace_id": wid, "title": ""})
            r.raise_for_status()
            cid = r.json()["conversation_id"]
            st.session_state.conversation_id = cid
            st.session_state.messages = []
            st.success(f"Conversation #{cid}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if wid:
        try:
            r = api("GET", f"/conversations/{wid}")
            r.raise_for_status()
            convs = r.json()
            if convs:
                labels = {c["id"]: f"#{c['id']}  {c.get('title','')[:30]}  ({c['updated_at'][:16]})" for c in convs}
                current = st.session_state.conversation_id
                options = list(labels.keys())
                index = options.index(current) if current in options else 0
                selected = st.selectbox("Select conversation", options, index=index, format_func=lambda x: labels[x])
                if selected != st.session_state.conversation_id:
                    st.session_state.conversation_id = selected
                    # Load existing messages
                    try:
                        r2 = api("GET", f"/conversations/{wid}/{selected}/messages")
                        r2.raise_for_status()
                        msgs = r2.json()
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]} for m in msgs
                        ]
                    except Exception:
                        st.session_state.messages = []
                    st.rerun()
        except Exception:
            pass

    if st.session_state.conversation_id:
        st.caption(f"Active conversation: #{st.session_state.conversation_id}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    if not wid:
        st.error("Select or create a workspace first.")
    else:
        # Auto-create conversation if none active
        if st.session_state.conversation_id is None:
            try:
                r = api("POST", "/conversations", json={"workspace_id": wid, "title": prompt[:50]})
                r.raise_for_status()
                st.session_state.conversation_id = r.json()["conversation_id"]
            except Exception as e:
                st.error(f"Could not create conversation: {e}")
                st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = api("POST", "/query", json={
                        "workspace_id": wid,
                        "question": prompt,
                        "conversation_id": st.session_state.conversation_id,
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
