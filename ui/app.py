"""Small Streamlit interface for the CouncilAI phase-1 vertical slice."""
import json
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
st.set_page_config(page_title="CouncilAI Starter", page_icon="🧠", layout="wide")
st.title("🧠 CouncilAI — Deliberation Lab")
st.caption("Phase-1 implementation: Go concurrent council + Python document retrieval. Mock mode generates illustrative, not factual, answers.")
if "doc_id" not in st.session_state:
    st.session_state.doc_id = ""

with st.sidebar:
    st.header("Document")
    upload = st.file_uploader("Upload a PDF, TXT or Markdown file", type=["pdf", "txt", "md"])
    if st.button("Ingest", disabled=upload is None):
        try:
            r = requests.post(f"{API}/api/v1/ingest", files={"file": (upload.name, upload.getvalue())}, timeout=120)
            r.raise_for_status()
            info = r.json()
            st.session_state.doc_id = info["doc_id"]
            st.success(f"Indexed {info['chunks']} chunks on {info['pages']} pages")
        except requests.RequestException as exc:
            st.error(f"Ingestion failed: {exc}")
    if st.session_state.doc_id:
        st.code(st.session_state.doc_id, language=None)
        if st.button("Detach"):
            st.session_state.doc_id = ""
    st.divider()
    strategy = st.selectbox("Execution strategy", ["auto", "direct", "council", "council_deep"], index=2)
    streaming = st.checkbox("Show streaming stage events", value=True)
    st.caption("Model keys and providers are configured using .env, not this UI.")

question = st.text_area("Ask a question", "How do the alternatives compare, and what are the tradeoffs?", height=100)
if st.button("Ask council", type="primary", disabled=not question.strip()):
    body = {"question": question, "strategy": strategy, "doc_id": st.session_state.doc_id, "top_k": 4}
    try:
        if not streaming:
            with st.spinner("Deliberating…"):
                r = requests.post(f"{API}/api/v1/query", json=body, timeout=150)
                r.raise_for_status()
                result = r.json()
        else:
            result = None
            display = st.empty()
            events = []
            with requests.post(f"{API}/api/v1/query", json=body, headers={"Accept": "text/event-stream"}, stream=True, timeout=150) as r:
                r.raise_for_status()
                event_type = ""
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: "):
                        payload = json.loads(line[6:])
                        events.append(event_type)
                        if event_type == "final":
                            result = payload
                        elif event_type == "error":
                            raise ValueError(payload.get("error", "Unknown error"))
                        display.info(f"Stages: {' → '.join(events)}")
        if result:
            st.subheader("Final answer")
            st.write(result.get("answer", ""))
            st.caption(f"Strategy: {result.get('strategy')} | Server latency: {result.get('duration_ms')} ms | Cache hit: {result.get('cache_hit')}")
            st.caption(f"Answer source: {result.get('source', '')} | Peer reviewed: {result.get('peer_reviewed', False)} | Top peer-ranked: {result.get('selected') or '—'}")
            st.caption(result.get("confidence_note", ""))
            with st.expander("Candidate responses"):
                for c in result.get("candidates", []):
                    label = f"Response {c['label']} — " if c.get("label") else ""
                    score = f" (peer score {c['score']})" if c.get("score") else ""
                    st.markdown(f"**{label}{c['model']}**{score}")
                    st.write(c.get("answer", ""))
            with st.expander("Peer review and evidence"):
                reviews = result.get("reviews", [])
                if reviews:
                    st.table([
                        {
                            "reviewer": r.get("reviewer", ""),
                            "ranking": ", ".join(r.get("ranking") or []) or "—",
                            "reasoning": r.get("reasoning", ""),
                            "error": r.get("error", ""),
                            "latency_ms": r.get("latency_ms", ""),
                        }
                        for r in reviews
                    ])
                st.json({"reviews": reviews, "evidence": result.get("sources", []), "reflection": result.get("reflection", "")})
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Request failed: {exc}")

with st.expander("Backend metrics"):
    try:
        metrics = requests.get(f"{API}/metrics", timeout=3).json()
        st.json(metrics)
    except requests.RequestException:
        st.caption("Backend unavailable")
