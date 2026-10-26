"""Streamlit interface for CouncilAI: login, conversations, documents and
the full deliberation trace. All API calls carry the user's Bearer token."""
import json
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
st.set_page_config(page_title="CouncilAI Starter", page_icon="🧠", layout="wide")
st.title("🧠 CouncilAI — Deliberation Lab")
st.caption("Phase-1 implementation: Go concurrent council + Python document retrieval. Mock mode generates illustrative, not factual, answers.")
for key, default in {"doc_id": "", "token": "", "username": "", "session_id": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def auth_headers(extra=None):
    """Bearer token for every protected call; the token lives only in this
    browser session's server-side state and is never displayed or logged."""
    h = {"Authorization": f"Bearer {st.session_state.token}"}
    h.update(extra or {})
    return h


def explain(r):
    """Turn auth/limit statuses into user-facing messages; logs out on 401."""
    if r.status_code == 401:
        st.session_state.token = ""
        return "Your session expired or is invalid. Please log in again."
    if r.status_code == 429:
        return f"Rate limit reached. Try again in {r.headers.get('Retry-After', 'a few')} second(s)."
    try:
        return r.json().get("error", r.text)
    except ValueError:
        return r.text


def new_session_id():
    return "chat-" + os.urandom(6).hex()


# ── Login / registration ─────────────────────────────────────────────
if not st.session_state.token:
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    for tab, path, label in ((login_tab, "login", "Log in"), (register_tab, "register", "Create account")):
        with tab, st.form(f"{path}_form"):
            username = st.text_input("Username", key=f"{path}_user",
                                     help="3–32 characters: letters, digits, '.', '_' or '-'" if path == "register" else None)
            password = st.text_input("Password", type="password", key=f"{path}_pw",
                                     help="8–72 characters" if path == "register" else None)
            if st.form_submit_button(label, type="primary"):
                try:
                    r = requests.post(f"{API}/api/v1/{path}", json={"username": username, "password": password}, timeout=15)
                    if r.ok:
                        data = r.json()
                        st.session_state.token = data["token"]
                        st.session_state.username = data["username"]
                        st.session_state.session_id = new_session_id()
                        st.session_state.doc_id = ""
                        st.rerun()
                    else:
                        st.error(explain(r))
                except requests.RequestException as exc:
                    st.error(f"Backend unavailable: {exc}")
    st.stop()

with st.sidebar:
    st.markdown(f"Signed in as **{st.session_state.username}**")
    if st.button("Log out"):
        for key in ("token", "username", "session_id", "doc_id"):
            st.session_state[key] = ""
        st.rerun()
    st.divider()
    st.header("Conversation")
    remember = st.checkbox("Remember this conversation", value=True,
                           help="Earlier questions and answers in this session are sent to the models as context "
                                "(kept 24 h in Redis). Follow-up answers are not cached.")
    st.session_state.session_id = st.text_input("Session", st.session_state.session_id or new_session_id(),
                                                max_chars=64, help="Letters, digits, '-' and '_'. Reuse an ID to continue a conversation.")
    col_new, col_clear = st.columns(2)
    if col_new.button("New session"):
        st.session_state.session_id = new_session_id()
        st.rerun()
    if col_clear.button("Clear session"):
        try:
            r = requests.post(f"{API}/api/v1/conversation/clear", json={"session_id": st.session_state.session_id},
                              headers=auth_headers(), timeout=10)
            st.success("Session cleared.") if r.ok else st.error(explain(r))
        except requests.RequestException as exc:
            st.error(f"Clear failed: {exc}")
    st.divider()
    st.header("Document")
    upload = st.file_uploader("Upload a PDF, image (OCR), TXT or Markdown file", type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "txt", "md"])
    if st.button("Ingest", disabled=upload is None):
        try:
            r = requests.post(f"{API}/api/v1/ingest", files={"file": (upload.name, upload.getvalue())},
                              headers=auth_headers(), timeout=120)
            if not r.ok:
                raise requests.RequestException(explain(r))
            info = r.json()
            st.session_state.doc_id = info["doc_id"]
            meta = info.get("metadata") or {}
            st.success(f"Indexed {info.get('chunk_count')} chunks from {meta.get('page_count', '?')} page(s)"
                       + (" (replaced the earlier copy of this file)" if info.get("replaced") else ""))
            st.caption(f"Extraction: {info.get('ocr_method')} · type {meta.get('file_type')} · text layer {meta.get('has_text_layer')} · "
                       f"scanned {meta.get('is_scanned')} · tables {meta.get('has_tables')}")
            if info.get("chunk_types"):
                st.caption("Chunk types: " + ", ".join(f"{k} × {v}" for k, v in sorted(info["chunk_types"].items())))
            if info.get("embedding_model"):
                st.caption(f"Embedding model: {info['embedding_model']}")
            status = info.get("summary_status", "unknown")
            if info.get("summary"):
                st.caption(f"Document summary ({info.get('summary_model', 'ingestion model')}):")
                st.info(info["summary"])
            else:
                st.warning(f"No document summary ({status}). Queries will still use this document. {info.get('summary_error', '')}")
        except requests.RequestException as exc:
            st.error(f"Ingestion failed: {exc}")
    if st.session_state.doc_id:
        st.code(st.session_state.doc_id, language=None)
        if st.button("Detach"):
            st.session_state.doc_id = ""
    st.divider()
    strategy = st.selectbox(
        "Execution strategy",
        ["auto", "direct", "council", "council_deep"],
        index=0,
        help="auto: the LLM router chooses the strategy and whether to use the document. Other values are diagnostic overrides that skip the router.",
    )
    streaming = st.checkbox("Show streaming stage events", value=True)
    st.caption("Model keys and providers are configured using .env, not this UI.")

if remember and st.session_state.session_id:
    try:
        r = requests.post(f"{API}/api/v1/conversation/history", json={"session_id": st.session_state.session_id},
                          headers=auth_headers(), timeout=10)
        turns = r.json().get("turns") or [] if r.ok else []
        if not r.ok:
            st.caption(f"Conversation history unavailable: {explain(r)}")
        if turns:
            with st.expander(f"Conversation so far ({len(turns) // 2} exchange(s))", expanded=False):
                for t in turns:
                    with st.chat_message("user" if t.get("role") == "user" else "assistant"):
                        st.write(t.get("content", ""))
    except (requests.RequestException, ValueError) as exc:
        st.caption(f"Conversation history unavailable: {exc}")

question = st.text_area("Ask a question", "How do the alternatives compare, and what are the tradeoffs?", height=100)
if st.button("Ask council", type="primary", disabled=not question.strip()):
    body = {"question": question, "strategy": strategy, "doc_id": st.session_state.doc_id, "top_k": 4}
    if remember and st.session_state.session_id:
        body["session_id"] = st.session_state.session_id
    try:
        if not streaming:
            with st.spinner("Deliberating…"):
                r = requests.post(f"{API}/api/v1/query", json=body, headers=auth_headers(), timeout=150)
                if not r.ok:
                    raise ValueError(explain(r))
                result = r.json()
        else:
            result = None
            display = st.empty()
            events = []
            with requests.post(f"{API}/api/v1/query", json=body, headers=auth_headers({"Accept": "text/event-stream"}),
                               stream=True, timeout=150) as r:
                if not r.ok:
                    raise ValueError(explain(r))
                event_type = ""
                # Split on bytes (only \n / \r), then decode: str.splitlines()
                # would also split on U+0085 and similar characters, which can
                # appear raw inside JSON strings (e.g. web-search text).
                for raw_line in r.iter_lines():
                    line = raw_line.decode("utf-8")
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
            session = result.get("session")
            if session:
                line = f"Session {session.get('id')}: {session.get('history_turns_used', 0) // 2} earlier exchange(s) used as context"
                line += " · saved" if session.get("saved") else " · NOT saved"
                st.caption(line)
                if session.get("error"):
                    st.warning(f"Conversation memory: {session['error']}")
            cinfo = result.get("cache") or {}
            if cinfo:
                tier = {"l1_exact": "exact (L1)", "l2_semantic": "semantic (L2)"}.get(cinfo.get("tier"), "miss")
                line = f"Cache: {tier} · L1 {cinfo.get('l1')} · L2 {cinfo.get('l2')} · lookup {cinfo.get('lookup_ms')} ms"
                if cinfo.get("similarity") is not None:
                    line += f" · similarity {cinfo['similarity']:.3f} (threshold {cinfo.get('threshold')})"
                line += " · models ran for this request" if cinfo.get("inference_executed") else " · no council/direct model ran for this request"
                if cinfo.get("stored"):
                    line += " · stored in " + ", ".join(cinfo["stored"])
                st.caption(line)
                if cinfo.get("write_error"):
                    st.caption(f"Cache write skipped: {cinfo['write_error']}")
            routing = result.get("routing") or {}
            if routing:
                how = "LLM router" if routing.get("mode") == "auto" else "explicit override (diagnostic)"
                if routing.get("fallback"):
                    how += " — router failed, default plan used"
                st.caption(f"Routing: {routing.get('strategy')} via {how}" + (f" ({routing.get('router_model')})" if routing.get("router_model") else ""))
                if routing.get("reasoning"):
                    st.caption(f"Router reasoning: {routing['reasoning']}")
                if routing.get("document_attached"):
                    st.caption(f"Document attached: summary {routing.get('document_summary')}, needed: {routing.get('needs_doc')}, used: {routing.get('document_used')}")
            grounding = result.get("grounding", "none")
            labels = {"document": "attached document", "web": "web search results (unverified)", "none": "none — answered from model knowledge only"}
            st.caption(f"Grounding: {labels.get(grounding, grounding)}")
            if routing.get("grounding_error"):
                st.warning(f"Grounding unavailable: {routing['grounding_error']}")
            for src in routing.get("web_sources") or []:
                st.caption(f"[{src['rank']}] [{src['title'] or src['url']}]({src['url']}) — {src['content_source'].replace('_', ' ')}")
            usage = result.get("token_usage")
            st.caption(f"Token usage (provider-reported): {usage['total_tokens']} total" if usage else "Token usage: not reported by the providers")
            st.caption(result.get("confidence_note", ""))
            for stage in result.get("degraded") or []:
                st.warning(f"Degraded: {stage.replace('-', ' ')}")
            synthesis = result.get("synthesis")
            reflection = result.get("reflection")
            if synthesis or reflection:
                with st.expander("Chairman synthesis and self-review", expanded=True):
                    st.caption("Everything here is the models' own assessment, not an independent evaluation. "
                               "A self-reported confidence of 0.9 does not mean 90% of such answers are correct.")
                    if synthesis:
                        if synthesis.get("status") == "synthesized":
                            st.markdown(f"**Synthesized by** `{synthesis.get('model')}`")
                        else:
                            st.error(f"Chairman synthesis failed; showing the top peer-ranked candidate instead. {synthesis.get('error', '')}")
                    if result.get("reasoning"):
                        st.markdown(f"**Synthesis rationale:** {result['reasoning']}")
                    conf = result.get("confidence")
                    if conf is not None:
                        st.markdown(f"**Model self-reported confidence:** {conf:.2f} (from {result.get('confidence_source', 'model')})")
                    else:
                        st.caption("No confidence was reported by the model.")
                    if reflection:
                        if reflection.get("status") != "completed":
                            st.warning(f"Self-review unavailable: {reflection.get('error', '')}. This is not evidence the answer is faithful.")
                        else:
                            faithful = reflection.get("faithful")
                            st.markdown(f"**Self-review quality:** {reflection.get('quality')} · **faithful to sources (model's view):** {faithful}")
                            if reflection.get("issues"):
                                st.markdown(f"**Issues:** {reflection['issues']}")
                            if reflection.get("suggestion"):
                                st.markdown(f"**Suggestion:** {reflection['suggestion']}")
                        revision = reflection.get("revision")
                        if revision:
                            if revision.get("status") == "succeeded":
                                st.success("Answer revised once after self-review.")
                            else:
                                st.warning(f"Revision failed; the original synthesized answer is kept. {revision.get('error', '')}")
                            with st.expander("Answer before revision"):
                                st.caption(f"Source: {revision.get('previous_source')}")
                                st.write(revision.get("previous_answer", ""))
            retrieval = result.get("retrieval")
            if retrieval:
                with st.expander(f"Retrieved document chunks ({len(retrieval.get('chunks') or [])})"):
                    if retrieval.get("reranked"):
                        st.caption(f"Ordered by cross-encoder score (raw logits, higher = more relevant; not probabilities) from {retrieval.get('candidates')} vector candidates.")
                    else:
                        st.caption(f"Vector order ({retrieval.get('distance_metric')} distance, lower = closer)."
                                   + (f" Reranking unavailable: {retrieval['rerank_error']}" if retrieval.get("rerank_error") else ""))
                    for ch in retrieval.get("chunks") or []:
                        where = []
                        if ch.get("page_number"):
                            where.append(f"page {ch['page_number']}" + (f"–{ch['page_end']}" if ch.get("page_end") else ""))
                        where.append(ch.get("chunk_type", ""))
                        if ch.get("ocr_method"):
                            where.append(ch["ocr_method"])
                        scores = []
                        if ch.get("rerank_score") is not None:
                            scores.append(f"rerank {ch['rerank_score']:.2f}")
                        if ch.get("vector_distance") is not None:
                            scores.append(f"distance {ch['vector_distance']:.2f}")
                        if ch.get("ocr_confidence") is not None:
                            scores.append(f"OCR confidence {ch['ocr_confidence']:.2f}")
                        st.markdown(f"**Chunk {ch.get('chunk_index')}** · {' · '.join(where)} · {' · '.join(scores)}")
                        if ch.get("chunk_type") == "table":
                            st.markdown(ch.get("content", ""))
                        else:
                            st.text(ch.get("content", ""))
            with st.expander("Candidate responses"):
                for c in result.get("candidates", []):
                    label = f"Response {c['label']} — " if c.get("label") else ""
                    score = f" (peer score {c['score']})" if c.get("score") else ""
                    st.markdown(f"**{label}{c['model']}**{score}")
                    details = []
                    if c.get("served_model") and c["served_model"] != c["model"]:
                        details.append(f"served by {c['served_model']}")
                    if c.get("usage"):
                        details.append(f"{c['usage']['total_tokens']} tokens")
                    if c.get("error"):
                        details.append(f"error: {c['error']}")
                    if details:
                        st.caption(" | ".join(details))
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
                st.json({"reviews": reviews, "evidence": result.get("sources", []), "synthesis": result.get("synthesis"), "reflection": result.get("reflection")})
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Request failed: {exc}")

with st.expander("Backend health"):
    # Prometheus metrics are internal-only; see Grafana (observability profile).
    try:
        st.json(requests.get(f"{API}/health", timeout=3).json())
    except requests.RequestException:
        st.caption("Backend unavailable")
