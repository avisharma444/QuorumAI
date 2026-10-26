"""Measure cache cold/warm latency on an already running backend.

The default mock model makes results useful for *system-level* latency only;
never present them as real LLM inference or answer-quality benchmarks.
"""
import json
import os
import statistics
import time
import uuid

import requests

BASE = os.getenv("API_URL", "http://localhost:8080")
N = int(os.getenv("N", "20"))


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


def run():
    cold, warm = [], []
    http = authed_session(BASE)
    # Questions are grounded in one uploaded document so the result does not
    # depend on live web search (failed web grounding is never cached).
    text = ("Indexing builds vector and keyword structures ahead of time. Retrieval "
            "queries them at request time and reranks the candidates. " * 5).encode()
    up = http.post(BASE + "/api/v1/ingest", files={"file": (f"bench_{uuid.uuid4().hex[:8]}.txt", text)}, timeout=120)
    up.raise_for_status()
    doc_id = up.json()["doc_id"]
    for i in range(N):
        payload = {"question": f"Compare indexing and retrieval. id={uuid.uuid4().hex}", "strategy": "council", "doc_id": doc_id}
        a = time.perf_counter()
        one = http.post(BASE + "/api/v1/query", json=payload, timeout=90)
        one.raise_for_status()
        cold.append((time.perf_counter() - a) * 1000)
        b = time.perf_counter()
        two = http.post(BASE + "/api/v1/query", json=payload, timeout=90)
        two.raise_for_status()
        warm.append((time.perf_counter() - b) * 1000)
        assert not one.json()["cache_hit"] and two.json()["cache_hit"]
    result = {"samples": N, "mode": "mock-or-live-depending-on-backend-config", "cold_ms_p50": statistics.median(cold), "warm_ms_p50": statistics.median(warm), "cold_ms_p95": sorted(cold)[int(.95 * (N - 1))], "warm_ms_p95": sorted(warm)[int(.95 * (N - 1))]}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run()
