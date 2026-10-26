"""HTTP contract tests with an injected test encoder (no model downloads)."""
import pytest
from fastapi.testclient import TestClient

import fixtures as fx
from app import state
from app.config import Settings
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.reranker import Reranker
from app.retrieval.search_agent import WebSearchAgent
from app.sampling import bme_preview
from app.services import RagServices
from conftest import FakeCrossEncoder, FakeEmbedder


@pytest.fixture()
def client(tmp_path, monkeypatch):
    emb = FakeEmbedder()
    settings = Settings(chroma_db_path=str(tmp_path / "chroma"), embedding_model="test/fake-embedder")
    svc = RagServices(settings, embedder=emb, reranker=Reranker("fake", model=FakeCrossEncoder()),
                      store=ChromaStore(settings.chroma_db_path, emb), autoload=False)
    monkeypatch.setattr(state, "services", svc)
    hits = [{"title": "Go", "href": "", "body": "Go is a programming language."}]
    monkeypatch.setattr(state, "search_agent", WebSearchAgent(search_fn=lambda q, n: hits[:n], resolver=lambda h: ["93.184.216.34"]))
    from app.main import app
    with TestClient(app) as c:
        yield c


def upload(client, name, data, **params):
    return client.post("/ingest", files={"file": (name, data)}, params=params)


def test_ingest_returns_preview_and_counts(client):
    r = upload(client, "notes.md", b"Zogland's capital is Mirevale.")
    assert r.status_code == 200
    body = r.json()
    assert body["chunk_count"] == 1 and body["ocr_method"] == "plain_text"
    assert body["preview_text"] == "Zogland's capital is Mirevale."
    assert body["doc_id"].startswith("notes_") and body["metadata"]["page_count"] == 1


def test_ingest_pdf_contract(client):
    r = upload(client, "revenue.pdf", fx.table_pdf())
    body = r.json()
    assert r.status_code == 200 and body["ocr_method"] == "layout_aware" and body["metadata"]["has_tables"]
    assert body["chunk_types"] == {"paragraph": 2, "table": 1} and "layout_aware" in body["message"]
    all_chunks = client.post("/retrieve-all", json={"doc_id": body["doc_id"]}).json()
    assert all_chunks["total_chunks"] == 3
    assert body["preview_text"] == bme_preview([c["content"] for c in all_chunks["chunks"]])
    assert [c["chunk_index"] for c in all_chunks["chunks"]] == [0, 1, 2]
    assert all_chunks["chunks"][1]["chunk_type"] == "table" and "| North | 120 | 135 |" in all_chunks["chunks"][1]["content"]


def test_duplicate_upload_is_idempotent(client):
    first = upload(client, "a.md", b"Same bytes about Mirevale.").json()
    second = upload(client, "a.md", b"Same bytes about Mirevale.").json()
    other = upload(client, "a.md", b"Different bytes about Halvard.").json()
    assert first["doc_id"] == second["doc_id"] and second["replaced"] and not first["replaced"]
    assert other["doc_id"] != first["doc_id"]
    assert client.post("/retrieve-all", json={"doc_id": first["doc_id"]}).json()["total_chunks"] == 1


def test_retrieve_contract(client):
    paras = [b"Mirevale is the capital city of Zogland on the Lune river.",
             b"Copper is mined in Halvard by about four thousand workers.",
             b"Wool is exported from the southern farms to many nations."]
    doc = upload(client, "z.md", b"\n\n".join(paras)).json()["doc_id"]
    r = client.post("/retrieve", json={"question": "Where is copper mined?", "doc_id": doc, "top_k": 1})
    body = r.json()
    assert r.status_code == 200 and body["reranked"] and body["candidates"] == 3  # min(3 * top_k, 3 chunks)
    assert body["chunks"][0]["content"] == paras[1].decode() and "rerank_score" in body["chunks"][0]["metadata"]
    assert body["distance_metric"] == "l2"
    off = client.post("/retrieve", json={"question": "copper", "doc_id": doc, "top_k": 2, "rerank": False}).json()
    assert not off["reranked"] and "rerank_score" not in off["chunks"][0]["metadata"]


def test_retrieve_errors(client):
    assert client.post("/retrieve", json={"question": "q", "doc_id": "nope"}).status_code == 404
    assert client.post("/retrieve", json={"question": "q"}).status_code == 400
    assert client.post("/retrieve", json={"question": "", "doc_id": "x"}).status_code == 422
    assert client.post("/retrieve-all", json={"doc_id": "nope"}).status_code == 404


def test_embed(client):
    body = client.post("/embed", json={"text": "hello world"}).json()
    assert len(body["embedding"]) == 384 and body["dimension"] == 384 and body["model"] == "test/fake-embedder"
    assert client.post("/embed", json={"text": "  "}).status_code == 400


@pytest.mark.parametrize("name, data, status", [
    ("empty.txt", b"   ", 422),
    ("bad.pdf", fx.CORRUPT_PDF, 400),
    ("bad.docx", b"PK", 400),
    ("empty.pdf", b"", 400),
])
def test_ingest_errors(client, name, data, status):
    assert upload(client, name, data).status_code == status


def test_ingest_size_limit_and_doc_id_validation(client, monkeypatch):
    monkeypatch.setattr(state.services.settings, "max_upload_bytes", 10)
    assert upload(client, "a.md", b"x" * 11).status_code == 413
    monkeypatch.setattr(state.services.settings, "max_upload_bytes", 10_000)
    assert upload(client, "a.md", b"hello there", doc_id="../etc").status_code == 400
    assert upload(client, "a.md", b"hello there", doc_id="custom-1").json()["doc_id"] == "custom-1"


def test_health_and_readiness(client, monkeypatch):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["ready"] and h["models"]["embedding"]["status"] == "ready"
    assert client.get("/ready").status_code == 200
    not_ready = RagServices(Settings(), autoload=False)
    not_ready.errors["embedding"] = "OSError: model not found"
    monkeypatch.setattr(state, "services", not_ready)
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["models"]["embedding"]["status"] == "failed"
    assert client.get("/health").status_code == 200  # liveness is separate
    assert client.post("/embed", json={"text": "x"}).status_code == 503
    assert upload(client, "a.md", b"text").status_code == 503
    assert client.post("/retrieve", json={"question": "q", "doc_id": "d"}).status_code == 503


def test_search_endpoint_format(client):
    r = client.post("/search", json={"query": "what is go", "max_results": 3})
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "results": ["Source [1]: Go\nURL: \nContent: Go is a programming language.\n"],
        "sources": [{"rank": 1, "title": "Go", "url": "", "content_source": "search_snippet"}],
        "error": None,
    }


def test_search_validation(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "max_results": 50}).status_code == 422


def test_doc_id_uses_whole_file_not_first_4kb():
    from app.routers.ingest import make_doc_id

    head = b"%PDF-1.4 shared template header " * 200  # > 4096 bytes
    a, b = make_doc_id("report.pdf", head + b"first tail"), make_doc_id("report.pdf", head + b"second tail")
    assert a != b, "the reference (MD5 of the first 4 KB) gives these the same ID"
    assert a == make_doc_id("report.pdf", head + b"first tail")  # deterministic
    assert make_doc_id("../we!rd name.pdf", b"x").startswith("we_rd_name_")
    import re
    assert re.fullmatch(r"[A-Za-z0-9._-]{1,64}", make_doc_id("x" * 300 + ".pdf", b"y"))
