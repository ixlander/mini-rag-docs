import json
import requests
import streamlit as st

st.set_page_config(page_title="Mini-RAG Docs (Local)", layout="wide")

API_URL_DEFAULT = "http://127.0.0.1:8000/query"


def call_api(api_url: str, question: str, debug: bool) -> dict:
    r = requests.post(
        api_url,
        json={"question": question, "debug": debug},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def pill(text: str) -> str:
    return f"<span style='display:inline-block;padding:4px 8px;border-radius:999px;border:1px solid #ddd;margin:2px 6px 2px 0;font-size:12px;'>{text}</span>"


st.title("Mini-RAG Docs (Local LLM)")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("FastAPI /query URL", value=API_URL_DEFAULT)
    debug = st.toggle("Debug (timings, rerank, context preview)", value=True)
    st.divider()
    st.caption("Run backend:")
    st.code("uvicorn app.main:app --reload", language="bash")

if "messages" not in st.session_state:
    st.session_state.messages = []  

question = st.chat_input("Ask a question about the docs… (RU/EN)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("payload"):
            payload = m["payload"]
            citations = payload.get("citations", [])
            conf = payload.get("confidence", "")
            if conf:
                st.markdown(pill(f"confidence: {conf}"), unsafe_allow_html=True)
            if citations:
                st.markdown("**Citations:**")
                st.code("\n".join(citations), language="text")

            dbg = payload.get("debug")
            if dbg:
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
                            st.markdown(
                                pill(c.get("chunk_id", ""))
                                + pill(c.get("title", ""))
                                + pill(c.get("section", "")),
                                unsafe_allow_html=True
                            )
                            st.write(c.get("text_preview", ""))

                    raw_prev = dbg.get("raw_model_output_preview")
                    if raw_prev:
                        st.markdown("**Raw model output preview**")
                        st.code(raw_prev, language="text")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                payload = call_api(api_url, question, debug)
            except requests.RequestException as e:
                st.error(f"API error: {e}")
                st.stop()

        answer = payload.get("answer", "").strip() or "No answer."
        st.markdown(answer)

        citations = payload.get("citations", [])
        conf = payload.get("confidence", "")
        if conf:
            st.markdown(pill(f"confidence: {conf}"), unsafe_allow_html=True)

        if citations:
            st.markdown("**Citations:**")
            st.code("\n".join(citations), language="text")

        if payload.get("debug"):
            with st.expander("Debug"):
                st.json(payload["debug"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "payload": payload})

st.caption("Tip: if you keep getting “I couldn't find this…”, check your docs actually contain the info, then rebuild index: `python ingest/build_index.py`.")