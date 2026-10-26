#!/usr/bin/env python3
"""Cache benchmark: real BGE embeddings, real Redis Stack, mock LLMs.

Layers measured separately:
  A. /embed latency (real BAAI/bge-small-en-v1.5 on CPU)
  B. semantic decision quality on a labelled set: 25 paraphrase pairs
     (should reuse) and 25 near-miss pairs (must not reuse), cosine computed
     from real BGE vectors with the same rule the service uses
     (similarity >= threshold AND same numbers). RediSearch's distance was
     checked against this cosine to within 1e-3 in the Go integration tests.
  C. end-to-end through the Go API (MOCK_LLM=true, so model latency is ~0):
     cold request, exact repeat (L1), partner question (L2 or miss). Each
     pair gets its own tiny document, so a hit can only come from its pair.

Redis-only GET/SET/KNN latency is measured by the Go harness
(internal/cache/bench_report_test.go).

Run on the compose network, e.g.:
  docker run --rm --network councilai-reproduced_default -v "$PWD/benchmarks:/b" \
    councilai-rag-test python /b/cache_benchmark.py --repeats 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import string
import time
import uuid

import requests

API = os.getenv("API_BASE", "http://go-backend:8080")
RAG = os.getenv("RAG_BASE", "http://python-rag:8000")

# (cached question, later question) — should reuse the cached answer.
PARAPHRASES = [
    ("Who sets household water prices?", "Which organisation decides what households pay for water?"),
    ("What is the refund policy?", "How do refunds work?"),
    ("How long does shipping take?", "What is the delivery time for orders?"),
    ("Who is the CEO of the company?", "Who leads the company as chief executive?"),
    ("What are the office opening hours?", "When is the office open?"),
    ("How do I reset my password?", "What are the steps to change a forgotten password?"),
    ("What does the warranty cover?", "Which defects are included in the warranty?"),
    ("Where is the headquarters located?", "In which city is the main office?"),
    ("What are the main risks identified in the report?", "Which risks does the report highlight?"),
    ("How is customer data stored?", "Where and how does the company keep customer data?"),
    ("What languages does the product support?", "Which languages is the product available in?"),
    ("Who can apply for the grant?", "What are the eligibility criteria for the grant?"),
    ("What is the late payment penalty?", "What fee is charged when a payment is late?"),
    ("How many employees does the firm have?", "What is the size of the firm's workforce?"),
    ("What causes the river to flood?", "Why does the river overflow its banks?"),
    ("What is the purpose of the study?", "What did the study aim to find out?"),
    ("Which method was used to collect the data?", "How was the data gathered?"),
    ("What is the minimum system requirement?", "What hardware is needed at minimum?"),
    ("How do I contact support?", "What is the way to reach customer support?"),
    ("What is the cancellation deadline?", "By when must I cancel?"),
    ("Summarise the main findings.", "What are the key findings?"),
    ("What are the side effects of the treatment?", "Which adverse effects can the treatment cause?"),
    ("Who wrote the report?", "Who is the author of the report?"),
    ("What is the capital of Zogland?", "Which city is Zogland's capital?"),
    ("How is the board elected?", "What is the procedure for electing the board?"),
]

# (cached question, later question) — reusing the cached answer would be WRONG.
NEAR_MISSES = [
    ("What was revenue in 2024?", "What was revenue in 2025?"),
    ("Which countries are supported?", "Which countries are unsupported?"),
    ("What is the refund policy?", "What is the cancellation policy?"),
    ("What was profit in Q1?", "What was profit in Q2?"),
    ("Who is the CEO?", "Who is the CFO?"),
    ("What is the minimum age to apply?", "What is the maximum age to apply?"),
    ("What are the benefits of the plan?", "What are the drawbacks of the plan?"),
    ("How do I enable two-factor authentication?", "How do I disable two-factor authentication?"),
    ("What is included in the basic tier?", "What is included in the premium tier?"),
    ("What was the population in 1990?", "What was the population in 2020?"),
    ("Which drugs interact with aspirin?", "Which drugs interact with ibuprofen?"),
    ("What happens if I pay early?", "What happens if I pay late?"),
    ("What is the import tax on cars?", "What is the export tax on cars?"),
    ("How many days of annual leave do employees get?", "How many days of sick leave do employees get?"),
    ("What is the boiling point of water?", "What is the freezing point of water?"),
    ("Who won the 2018 election?", "Who won the 2022 election?"),
    ("What are the requirements for a student visa?", "What are the requirements for a work visa?"),
    ("Is the product safe for children?", "Is the product safe for pets?"),
    ("What is the interest rate on savings accounts?", "What is the interest rate on loans?"),
    ("When does the northern office open?", "When does the southern office open?"),
    ("What is the speed limit in towns?", "What is the speed limit on motorways?"),
    ("Which features were added in version 3?", "Which features were removed in version 3?"),
    ("What is the price of the monthly plan?", "What is the price of the annual plan?"),
    ("How do I file a complaint?", "How do I withdraw a complaint?"),
    ("What did the first experiment show?", "What did the second experiment show?"),
]

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_match(a: str, b: str) -> bool:
    return sorted(NUM.findall(a)) == sorted(NUM.findall(b))


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def pct(xs):
    s = sorted(xs)
    at = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "mean_ms": round(statistics.mean(s), 2), "p50_ms": round(at(0.5), 2), "p95_ms": round(at(0.95), 2), "p99_ms": round(at(0.99), 2)}


def embed(text):
    r = requests.post(f"{RAG}/embed", json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))



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


HTTP = None


def ask(question, doc_id):
    t = time.perf_counter()
    r = HTTP.post(f"{API}/api/v1/query", json={"question": question, "doc_id": doc_id}, timeout=120)
    ms = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    return r.json(), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--embed-iterations", type=int, default=200)
    args = ap.parse_args()
    report = {"config": {"api": API, "rag": RAG, "pairs_paraphrase": len(PARAPHRASES), "pairs_near_miss": len(NEAR_MISSES), "repeats": args.repeats, "llm": "MOCK_LLM=true (model latency ~0; no real tokens)"}}

    # A. /embed latency
    texts = [q for pair in PARAPHRASES + NEAR_MISSES for q in pair]
    embed(texts[0])  # warm-up
    lat = []
    for i in range(args.embed_iterations):
        t = time.perf_counter()
        embed(texts[i % len(texts)])
        lat.append((time.perf_counter() - t) * 1000)
    report["embed_latency"] = pct(lat)

    # B. labelled decision quality
    sims = {"paraphrase": [], "near_miss": []}
    for label, pairs in (("paraphrase", PARAPHRASES), ("near_miss", NEAR_MISSES)):
        for a, b in pairs:
            sims[label].append({"a": a, "b": b, "similarity": round(cos(embed(a), embed(b)), 4), "numbers_match": numbers_match(a, b)})
    sweep = {}
    for thr in (0.80, 0.85, 0.90, 0.95):
        row = {}
        for guard in (False, True):
            ok = lambda s: s["similarity"] >= thr and (s["numbers_match"] or not guard)
            row["with_number_guard" if guard else "threshold_only"] = {
                "paraphrase_reuse_rate": round(sum(ok(s) for s in sims["paraphrase"]) / len(PARAPHRASES), 3),
                "near_miss_false_positive_rate": round(sum(ok(s) for s in sims["near_miss"]) / len(NEAR_MISSES), 3),
            }
        sweep[f"{thr:.2f}"] = row
    report["semantic_decisions"] = sweep
    report["similarities"] = sims

    # C. end-to-end through Go (one small document per pair). Needs the
    # backend started with SEMANTIC_CACHE_ENABLED=true (L2 reuse is off by default).
    global HTTP
    HTTP = authed_session(API)
    probe = HTTP.post(f"{API}/api/v1/query", json={"question": "semantic cache probe " + uuid.uuid4().hex}, timeout=120).json()
    if (probe.get("cache") or {}).get("l2") == "disabled_by_policy":
        raise SystemExit("L2 reuse is disabled on the backend; restart go-backend with SEMANTIC_CACHE_ENABLED=true for the end-to-end section")
    cold, l1, l2_hit, l2_miss = [], [], [], []
    e2e = {"paraphrase": {"hits": 0, "n": 0}, "near_miss": {"hits": 0, "n": 0}}
    calls_avoided = 0
    for rep in range(args.repeats):
        nonce = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
        for label, pairs in (("paraphrase", PARAPHRASES), ("near_miss", NEAR_MISSES)):
            for i, (a, b) in enumerate(pairs):
                body = f"Benchmark document {nonce} {label} {chr(97 + i % 26)}{i}.\n\nThis document discusses: {a} It also mentions: {b}"
                doc = HTTP.post(f"{API}/api/v1/ingest", files={"file": (f"bench_{nonce}_{label}_{i}.md", body.encode())}, timeout=120).json()["doc_id"]
                res, ms = ask(a, doc)
                assert not res["cache_hit"] and res["cache"]["inference_executed"], res["cache"]
                cold.append(ms)
                res, ms = ask(a, doc)
                assert res["cache"]["tier"] == "l1_exact", res["cache"]
                l1.append(ms)
                calls_avoided += 8  # router + 3 candidates + 3 reviews + chairman
                res, ms = ask(b, doc)
                e2e[label]["n"] += 1
                if res["cache"]["tier"] == "l2_semantic":
                    e2e[label]["hits"] += 1
                    l2_hit.append(ms)
                    calls_avoided += 7  # the router still runs before L2
                else:
                    l2_miss.append(ms)
    report["end_to_end"] = {
        "cold_request": pct(cold), "l1_hit_request": pct(l1),
        "l2_hit_request": pct(l2_hit) if l2_hit else None, "l2_miss_request": pct(l2_miss) if l2_miss else None,
        "paraphrase_l2_hit_rate": round(e2e["paraphrase"]["hits"] / e2e["paraphrase"]["n"], 3),
        "near_miss_l2_false_positive_rate": round(e2e["near_miss"]["hits"] / e2e["near_miss"]["n"], 3),
        "model_calls_avoided": calls_avoided,
        "token_savings": "not reported: mock models report no usage",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cache_benchmark.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "similarities"}, indent=2))
    worst = sorted(sims["near_miss"], key=lambda s: -s["similarity"])[:6]
    print("\nmost similar near-misses:")
    for s in worst:
        print(f"  {s['similarity']:.4f} numbers_match={s['numbers_match']}  {s['a']!r} vs {s['b']!r}")
    lowest = sorted(sims["paraphrase"], key=lambda s: s["similarity"])[:4]
    print("least similar paraphrases:")
    for s in lowest:
        print(f"  {s['similarity']:.4f}  {s['a']!r} vs {s['b']!r}")


if __name__ == "__main__":
    main()
