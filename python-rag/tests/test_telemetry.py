"""W3C trace-context propagation into the RAG service (OpenTelemetry).

The global tracer provider can be set only once per process, so one
module-scoped provider with an in-memory exporter serves all tests here.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import telemetry
from app.models import Chunk
from app.retrieval.chroma_store import ChromaStore

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_SPAN = "00f067aa0ba902b7"


@pytest.fixture(scope="module")
def traced():
    exporter = InMemorySpanExporter()
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve():
        with telemetry.span("rag.vector_search", **{"rag.fetch_k": 6}):
            pass
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    provider = telemetry.setup_tracing(app, span_exporter=exporter)
    assert provider is not None
    return TestClient(app), exporter


def test_incoming_traceparent_becomes_parent(traced):
    client, exporter = traced
    exporter.clear()
    r = client.post("/retrieve", json={"question": "secret question text", "doc_id": "d"},
                    headers={"traceparent": f"00-{TRACE_ID}-{PARENT_SPAN}-01"})
    assert r.status_code == 200
    spans = exporter.get_finished_spans()
    server = [s for s in spans if s.kind.name == "SERVER"]
    assert len(server) == 1, [s.name for s in spans]
    srv = server[0]
    assert format(srv.context.trace_id, "032x") == TRACE_ID
    assert format(srv.parent.span_id, "016x") == PARENT_SPAN
    assert srv.parent.is_remote
    child = next(s for s in spans if s.name == "rag.vector_search")
    assert child.context.trace_id == srv.context.trace_id
    assert child.parent.span_id == srv.context.span_id
    # No request content in any attribute.
    for s in spans:
        for v in (s.attributes or {}).values():
            assert "secret question" not in str(v)


def test_without_traceparent_a_new_root_trace_starts(traced):
    client, exporter = traced
    exporter.clear()
    client.post("/retrieve", json={})
    srv = next(s for s in exporter.get_finished_spans() if s.kind.name == "SERVER")
    assert srv.parent is None
    assert format(srv.context.trace_id, "032x") != TRACE_ID


def test_unsampled_parent_is_respected(traced):
    client, exporter = traced
    exporter.clear()
    client.post("/retrieve", json={}, headers={"traceparent": f"00-{TRACE_ID}-{PARENT_SPAN}-00"})
    assert exporter.get_finished_spans() == ()


def test_health_probes_are_not_traced(traced):
    client, exporter = traced
    exporter.clear()
    client.get("/health", headers={"traceparent": f"00-{TRACE_ID}-{PARENT_SPAN}-01"})
    assert exporter.get_finished_spans() == ()


def test_store_retrieval_emits_child_spans(traced, tmp_path, fake_embedder):
    _, exporter = traced
    exporter.clear()
    store = ChromaStore(str(tmp_path / "chroma"), fake_embedder)
    chunks = [Chunk(content=t, chunk_index=i) for i, t in enumerate(["alpha beta", "gamma delta", "alpha omega"])]
    store.ingest(chunks, "doc-trace")
    exporter.clear()
    with telemetry.span("request"):
        store.retrieve("alpha", "doc-trace", top_k=2, rerank=False)
    names = [s.name for s in exporter.get_finished_spans()]
    assert "rag.embed_query" in names and "rag.vector_search" in names


def test_tracing_off_installs_nothing():
    assert telemetry.setup_tracing(FastAPI(), exporter="none") is None
