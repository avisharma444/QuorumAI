"""HTTP smoke test for Go backend + Python RAG (both must already be running)."""
import os
import time
import uuid

import requests

api = os.getenv("API_URL", "http://localhost:8080").rstrip("/")


def authed_session(base):
    """Register a throwaway user and return a requests.Session that sends its
    Bearer token and waits out 429 rate-limit responses (Retry-After)."""
    s = requests.Session()
    name = "bench" + uuid.uuid4().hex[:12]
    for _ in range(20):
        r = s.post(base + "/api/v1/register", json={"username": name, "password": uuid.uuid4().hex}, timeout=15)
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("Retry-After", "5")))
    r.raise_for_status()
    s.headers["Authorization"] = "Bearer " + r.json()["token"]
    plain_request = s.request

    def request(method, url, **kw):
        while True:
            resp = plain_request(method, url, **kw)
            if resp.status_code != 429:
                return resp
            time.sleep(int(resp.headers.get("Retry-After", "1")))

    s.request = request
    return s


def main():
    assert requests.get(api + "/health", timeout=5).status_code == 200
    assert requests.post(api + "/api/v1/query", json={"question": "x"}, timeout=5).status_code == 401, "query must require a token"
    http = authed_session(api)
    text = ("Go runs concurrent council models. Python indexes document evidence. " * 10).encode()
    uploaded = http.post(api + "/api/v1/ingest", files={"file": ("example.txt", text)}, timeout=25)
    uploaded.raise_for_status()
    doc_id = uploaded.json()["doc_id"]
    question = {"doc_id": doc_id, "question": "What does Go do and what does Python do?", "strategy": "council_deep"}
    result = http.post(api + "/api/v1/query", json=question, timeout=130)
    result.raise_for_status()
    answer = result.json()
    assert answer["sources"], "expected retrieval evidence"
    assert answer["answer"], "expected nonempty answer"
    second = http.post(api + "/api/v1/query", json=question, timeout=130)
    second.raise_for_status()
    assert second.json()["cache_hit"], "expected warm L1 (Redis) cache hit"
    q = {"question": "compare alternatives " + uuid.uuid4().hex, "strategy": "council"}
    event_response = http.post(api + "/api/v1/query", json=q, headers={"Accept": "text/event-stream"}, timeout=130)
    event_response.raise_for_status()
    assert "event: final" in event_response.text
    print("PASS: auth → upload → retrieve → council → cache → SSE")
    print("Note: mock answers are intentionally not factual QA evaluations.")


if __name__ == "__main__":
    main()
