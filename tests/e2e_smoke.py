"""HTTP smoke test for Go backend + Python RAG (both must already be running)."""
import os
import uuid

import requests

api = os.getenv("API_URL", "http://localhost:8080").rstrip("/")


def main():
    assert requests.get(api + "/health", timeout=5).status_code == 200
    text = ("Go runs concurrent council models. Python indexes document evidence. " * 10).encode()
    uploaded = requests.post(api + "/api/v1/ingest", files={"file": ("example.txt", text)}, timeout=25)
    uploaded.raise_for_status()
    doc_id = uploaded.json()["doc_id"]
    question = {"doc_id": doc_id, "question": "What does Go do and what does Python do?", "strategy": "council_deep"}
    result = requests.post(api + "/api/v1/query", json=question, timeout=130)
    result.raise_for_status()
    answer = result.json()
    assert answer["sources"], "expected retrieval evidence"
    assert answer["answer"], "expected nonempty answer"
    second = requests.post(api + "/api/v1/query", json=question, timeout=130)
    second.raise_for_status()
    assert second.json()["cache_hit"], "expected warm in-process cache hit"
    q = {"question": "compare alternatives " + uuid.uuid4().hex, "strategy": "council"}
    event_response = requests.post(api + "/api/v1/query", json=q, headers={"Accept": "text/event-stream"}, timeout=130)
    event_response.raise_for_status()
    assert "event: final" in event_response.text
    print("PASS: upload → retrieve → council → cache → SSE")
    print("Note: mock answers are intentionally not factual QA evaluations.")


if __name__ == "__main__":
    main()
