import importlib

import pytest
from fastapi.testclient import TestClient

from app.search_agent import WebSearchAgent


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "docs.sqlite3"))
    import app.main as main
    main = importlib.reload(main)
    hits = [{"title": "Go", "href": "", "body": "Go is a programming language."}]
    main.search_agent = WebSearchAgent(search_fn=lambda q, n: hits[:n], resolver=lambda h: ["93.184.216.34"])
    return TestClient(main.app)


def test_ingest_returns_preview_and_counts(client):
    r = client.post("/ingest", files={"file": ("notes.md", b"Zogland's capital is Mirevale.")})
    assert r.status_code == 200
    body = r.json()
    assert body["chunk_count"] == body["chunks"] == 1
    assert body["preview_text"] == "Zogland's capital is Mirevale."
    assert body["doc_id"] and body["pages"] == 1


def test_ingest_empty_document(client):
    r = client.post("/ingest", files={"file": ("empty.txt", b"   ")})
    assert r.status_code == 400


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
