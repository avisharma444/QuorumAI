"""Neural retrieval evaluation with the real RAG components.

Every document goes through the application's own pipeline (inspection,
plain-text extraction, layout chunker, BAAI/bge-small-en-v1.5 embeddings,
ChromaDB). Each question is retrieved within its own document, as the API
does. For each question we fetch 3*k vector candidates once and compare:

  vector    the candidates in vector (L2 distance) order
  reranked  the same candidates reordered by cross-encoder/ms-marco-MiniLM-L-6-v2

so the two configurations see identical candidates (reranking can only
reorder what vector search returned).

Relevance comes from the dataset's evidence sentences, AFTER chunking: a
chunk is relevant if it contains a gold evidence sentence. Items without
evidence (DROP) have unknown relevance and are excluded from recall metrics.
Multi-hop items also report "all evidence in top k".

Run inside the test image (models in /models):
  python -m councileval.retrieval_eval --data data/councileval-v1 --split heldout --out results/retrieval
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from .metrics import all_recall_at_k, mrr_at_k, ndcg_at_k, normalize, recall_at_k
from .schema import load_dataset
from .stats import bootstrap_mean_ci, paired_bootstrap_diff, wilson_interval

KS = (1, 3, 5)


def relevant_chunks(chunks, evidence: list[str]) -> tuple[set[int], list[set[int]]]:
    """(union of relevant chunk indexes, per-evidence-sentence chunk sets)."""
    norm_chunks = [normalize(c.content) for c in chunks]
    per = []
    for ev in evidence:
        ne = normalize(ev)
        per.append({c.chunk_index for c, nc in zip(chunks, norm_chunks) if ne and ne in nc})
    union = set().union(*per) if per else set()
    return union, per


def all_evidence_at_k(ranked, per_evidence: list[set[int]], k: int):
    if not per_evidence or any(not s for s in per_evidence):
        return None
    top = set(ranked[:k])
    return 1.0 if all(s & top for s in per_evidence) else 0.0


def auroc(pos: list[float], neg: list[float]):
    """Probability a random answerable item scores above a random unanswerable one."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def run(data: Path, split: str, out: Path, k: int = 5, limit: int | None = None) -> dict:
    from app.chunking.layout_chunker import chunk_document
    from app.config import get_settings
    from app.embedding.transformer import TransformerEmbeddings
    from app.inspection.inspector import inspect_document
    from app.ocr.router import route_ocr
    from app.retrieval.chroma_store import ChromaStore
    from app.retrieval.reranker import Reranker

    items, docs = load_dataset(data)
    items = [it for it in items if it.split == split or split == "all"]
    if limit:
        items = items[:limit]
    needed = sorted({it.doc_id for it in items})

    t0 = time.perf_counter()
    settings = get_settings()  # the service's own model configuration
    embedder = TransformerEmbeddings(settings.embedding_model, settings.embedding_dimension, settings.embedding_batch_size)
    reranker = Reranker(settings.reranker_model)
    model_load_s = time.perf_counter() - t0
    tmp = tempfile.mkdtemp(prefix="councileval-chroma-")
    store = ChromaStore(tmp, embedder, embedding_model=settings.embedding_model)

    doc_chunks, index_ms = {}, {}
    for doc_id in needed:
        d = docs[doc_id]
        raw = d.text.encode()
        t = time.perf_counter()
        meta = inspect_document(raw, f"{doc_id}.md")
        chunks = chunk_document(route_ocr(raw, f"{doc_id}.md", meta, None))
        store.ingest(chunks, doc_id)
        index_ms[doc_id] = (time.perf_counter() - t) * 1000
        doc_chunks[doc_id] = chunks

    rows = []
    for it in items:
        chunks = doc_chunks[it.doc_id]
        rel, per = relevant_chunks(chunks, it.evidence)
        vec = store.retrieve(it.question, it.doc_id, top_k=3 * k, rerank=False)
        rr = store.retrieve(it.question, it.doc_id, top_k=k, rerank=True, reranker=reranker)
        vec_rank = [c.chunk_index for c in vec.chunks][:k]
        rr_rank = [c.chunk_index for c in rr.chunks]
        candidates = [c.chunk_index for c in vec.chunks]
        top_rerank = max((c.metadata.get("rerank_score", float("-inf")) for c in rr.chunks), default=None)
        row = {
            "id": it.id, "source": it.source, "category": it.category, "answerable": it.answerable,
            "doc_chunks": len(chunks), "relevant_chunks": len(rel), "candidates": len(candidates),
            "relevant_in_candidates": bool(rel & set(candidates)) if rel else None,
            "embed_ms": rr.timings_ms.get("embed_query"), "search_ms": rr.timings_ms.get("vector_search"),
            "rerank_ms": rr.timings_ms.get("rerank"), "reranked": rr.reranked, "rerank_error": rr.rerank_error,
            "top_rerank_score": top_rerank,
            "relevant_chunk_types": sorted({c.chunk_type.value for c in chunks if c.chunk_index in rel}),
        }
        for name, ranked in (("vector", vec_rank), ("reranked", rr_rank)):
            for kk in KS:
                row[f"{name}_recall@{kk}"] = recall_at_k(ranked, rel, kk)
                row[f"{name}_all@{kk}"] = all_evidence_at_k(ranked, per, kk) if len(per) > 1 else None
            row[f"{name}_mrr@{k}"] = mrr_at_k(ranked, rel, k)
            row[f"{name}_ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
        rows.append(row)

    summary = summarize(rows, k)
    summary["setup"] = {
        "split": split, "items": len(rows), "documents": len(needed),
        "chunks_total": sum(len(c) for c in doc_chunks.values()),
        "chunks_per_doc": {"min": min(len(c) for c in doc_chunks.values()), "median": statistics.median(len(c) for c in doc_chunks.values()),
                           "max": max(len(c) for c in doc_chunks.values())},
        "embedding_model": getattr(embedder, "model_name", ""), "reranker": getattr(reranker, "model_name", ""),
        "k": k, "candidates_per_query": 3 * k, "model_load_s": round(model_load_s, 2),
        "indexing_ms_per_doc": _dist(list(index_ms.values())),
        "indexing_ms_total": round(sum(index_ms.values()), 1),
        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    with open(out / "retrieval_items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary


def _dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "mean": round(statistics.mean(s), 2), "p50": round(s[len(s) // 2], 2),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2), "max": round(s[-1], 2)}


def summarize(rows: list[dict], k: int) -> dict:
    labelled = [r for r in rows if r["relevant_chunks"]]

    def block(sel: list[dict]) -> dict:
        out = {"n": len(sel)}
        for metric in [f"recall@{kk}" for kk in KS] + [f"mrr@{k}", f"ndcg@{k}"] + [f"all@{kk}" for kk in KS]:
            a = [r[f"vector_{metric}"] for r in sel]
            b = [r[f"reranked_{metric}"] for r in sel]
            if all(x is None for x in a):
                continue
            diff = paired_bootstrap_diff(a, b)
            out[metric] = {"vector": diff["mean_a"], "reranked": diff["mean_b"], "diff": diff["diff"],
                           "diff_ci95": diff["ci"], "n": diff["n"],
                           "vector_ci95": bootstrap_mean_ci(a), "reranked_ci95": bootstrap_mean_ci(b)}
        return out

    strata = defaultdict(list)
    for r in labelled:
        strata[f"category={r['category']}"].append(r)
        strata[f"source={r['source']}"].append(r)
        size = "small(<=4 chunks)" if r["doc_chunks"] <= 4 else ("medium(5-20)" if r["doc_chunks"] <= 20 else "large(>20)")
        strata[f"doc_size={size}"].append(r)
        strata["hops=" + ("multi" if r["category"].startswith("multi_hop") else "single")].append(r)
        for t in r["relevant_chunk_types"] or ["none"]:
            strata[f"relevant_chunk_type={t}"].append(r)
    in_cands = [r["relevant_in_candidates"] for r in labelled]
    ans = [r["top_rerank_score"] for r in rows if r["source"].startswith("squad") and r["answerable"] and r["top_rerank_score"] is not None]
    unans = [r["top_rerank_score"] for r in rows if r["source"].startswith("squad") and not r["answerable"] and r["top_rerank_score"] is not None]
    hits = sum(1 for x in in_cands if x)
    return {
        "overall": block(labelled),
        "strata": {k_: block(v) for k_, v in sorted(strata.items())},
        "relevant_in_candidate_pool": {"rate": hits / len(in_cands) if in_cands else None, "ci95": wilson_interval(hits, len(in_cands)),
                                       "note": "upper bound for any reordering of the 3k candidates"},
        "unlabelled_items": len(rows) - len(labelled),
        "unanswerable_signal": {
            "auroc_top_rerank_score": auroc(ans, unans), "answerable_n": len(ans), "unanswerable_n": len(unans),
            "answerable_median": statistics.median(ans) if ans else None,
            "unanswerable_median": statistics.median(unans) if unans else None,
            "note": "Can the top cross-encoder score separate answerable from unanswerable SQuAD questions? 0.5 = no signal.",
        },
        "latency_ms": {"embed_query": _dist([r["embed_ms"] for r in rows]), "vector_search": _dist([r["search_ms"] for r in rows]),
                       "rerank": _dist([r["rerank_ms"] for r in rows])},
        "rerank_errors": sum(1 for r in rows if r["rerank_error"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/councileval-v1"))
    ap.add_argument("--split", default="heldout", choices=["dev", "heldout", "all"])
    ap.add_argument("--out", type=Path, default=Path("results/retrieval"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    s = run(a.data, a.split, a.out, a.k, a.limit)
    print(json.dumps({"setup": s["setup"], "overall": s["overall"], "unanswerable_signal": s["unanswerable_signal"],
                      "latency_ms": s["latency_ms"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
