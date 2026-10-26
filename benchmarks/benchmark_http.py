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


def run():
    cold, warm = [], []
    for i in range(N):
        payload = {"question": f"Compare indexing and retrieval. id={uuid.uuid4().hex}", "strategy": "council"}
        a = time.perf_counter()
        one = requests.post(BASE + "/api/v1/query", json=payload, timeout=90)
        one.raise_for_status()
        cold.append((time.perf_counter() - a) * 1000)
        b = time.perf_counter()
        two = requests.post(BASE + "/api/v1/query", json=payload, timeout=90)
        two.raise_for_status()
        warm.append((time.perf_counter() - b) * 1000)
        assert not one.json()["cache_hit"] and two.json()["cache_hit"]
    result = {"samples": N, "mode": "mock-or-live-depending-on-backend-config", "cold_ms_p50": statistics.median(cold), "warm_ms_p50": statistics.median(warm), "cold_ms_p95": sorted(cold)[int(.95 * (N - 1))], "warm_ms_p95": sorted(warm)[int(.95 * (N - 1))]}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run()
