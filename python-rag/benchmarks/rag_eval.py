"""Small, reproducible retrieval evaluation.

Corpus: 40 short paragraphs about a FICTIONAL country (written for this
benchmark, so no model has seen the facts). Each question is a paraphrase of
one fact. The relevant chunk for a question is found AFTER chunking, as the
chunk containing that fact's answer phrase, so ground truth follows the real
pipeline.

Configurations (same chunks, same questions):
  hash       — starter's hashed bag-of-words vectors (benchmark baseline only)
  neural     — BAAI/bge-small-en-v1.5 mean-pooled vectors in Chroma (L2), no rerank
  neural+rr  — the same 3*k candidates reranked by cross-encoder/ms-marco-MiniLM-L-6-v2

Run inside the test image:  python -m benchmarks.rag_eval [--pdf]
This is a functionality-scale check (40 chunks, 30 queries), not a benchmark
of general retrieval quality.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path

FACTS = [
    ("Mirevale", "The capital of Velmora is Mirevale, a river city where the Ardent and Lune rivers meet."),
    ("1842", "Velmora declared independence in 1842 after a short and almost bloodless dispute with its neighbours."),
    ("National Water Board", "Household water prices in Velmora are set every quarter by the National Water Board."),
    ("Halvard", "Most of the country's copper comes from deep mines around the northern town of Halvard."),
    ("four thousand", "The Halvard mines employ about four thousand people, many of them on rotating night shifts."),
    ("merino", "Sheep farmers in the southern hills raise merino flocks whose wool is sold mainly abroad."),
    ("Lune Bridge", "The Lune Bridge, built of grey granite in 1871, is the oldest public structure still in use."),
    ("velmoran crown", "Prices are quoted in the velmoran crown, which has been pegged to gold reserves since 1903."),
    ("Saint Orla", "The national holiday honours Saint Orla, a healer who, according to legend, cured a plague with river water."),
    ("eleven provinces", "The country is divided into eleven provinces, each with an elected governor serving five-year terms."),
    ("compulsory until sixteen", "School attendance is compulsory until sixteen, after which most students choose technical colleges."),
    ("Kessel Observatory", "Astronomers work at the Kessel Observatory on Mount Dray, famous for its record of comet sightings."),
    ("tidal turbines", "Since 2015 a growing share of electricity has come from tidal turbines installed in the Brenn estuary."),
    ("Grand Assembly", "Laws are passed by the Grand Assembly, a single chamber of one hundred and twenty members."),
    ("amber beetle", "The amber beetle, found only in the eastern marsh forests, is the national insect and a protected species."),
    ("Velmora Rail", "Velmora Rail operates the passenger network and runs an overnight sleeper train between the two largest cities."),
    ("fermented plum", "A fermented plum drink called sorrel is served at weddings and is traditionally brewed by the bride's family."),
    ("Port Calder", "Port Calder handles almost all container shipping and is ice-free throughout the winter."),
    ("Dray mountains", "The highest peaks lie in the Dray mountains, where the summit of Mount Dray reaches 2,940 metres."),
    ("forty percent", "Forest covers roughly forty percent of the land, most of it birch and spruce managed by the state."),
    ("press freedom", "A 1992 charter guarantees press freedom and forbids the government from licensing newspapers."),
    ("blue and silver", "The national flag shows two horizontal bands of blue and silver with a white heron in the centre."),
    ("heron", "The grey heron is the national bird and appears on every coin minted since the currency reform."),
    ("Ostrava Cup", "Football fans follow the Ostrava Cup, a knockout competition contested by clubs from every province."),
    ("Old Velmoran", "Many villages in the far west still speak Old Velmoran, a language related to no other living tongue."),
    ("universal healthcare", "Universal healthcare is funded by a payroll tax and free at the point of use for all residents."),
    ("forty-two hours", "The legal working week is forty-two hours, with overtime paid at one and a half times the normal rate."),
    ("Brenn estuary", "Migrating geese gather each autumn in the Brenn estuary, drawing thousands of birdwatchers."),
    ("glass-blowing", "The town of Lisk is known for glass-blowing workshops that have operated since the fifteenth century."),
    ("salt flats", "Commercial salt is harvested from the coastal salt flats near Marrow by evaporating seawater in shallow pans."),
    ("rye bread", "Dense rye bread with caraway seeds is eaten at almost every meal and is baked in communal ovens in villages."),
    ("Archive of the Realm", "Historical records, including the 1842 declaration, are kept in the Archive of the Realm in Mirevale."),
    ("left side", "Vehicles drive on the left side of the road, a custom kept from the era of horse-drawn carriages."),
    ("voting age", "The voting age was lowered from eighteen to seventeen in a referendum held in 2009."),
    ("hydrogen buses", "Mirevale replaced its diesel fleet with hydrogen buses, refuelled at a depot beside the eastern ring road."),
    ("chess", "Chess is taught in primary schools, and the national championship is broadcast live every spring."),
    ("ferry", "A ferry links the mainland to the island of Ness twice a day, weather permitting."),
    ("lighthouse", "The lighthouse at Cape Morrow has been automated since 1987 and now houses a small maritime museum."),
    ("pension age", "The state pension age is sixty-six for everyone born after 1970."),
    ("earthquakes", "Small earthquakes are recorded several times a year near the Dray fault, though serious damage is rare."),
]

QUESTIONS = [
    ("Which city serves as the seat of government?", "Mirevale"),
    ("In what year did the nation become independent?", "1842"),
    ("Who decides what families pay for tap water?", "National Water Board"),
    ("Where does most copper ore get extracted?", "Halvard"),
    ("How many workers do the copper mines have?", "four thousand"),
    ("What kind of sheep are raised in the south?", "merino"),
    ("What is the oldest structure still used by the public?", "Lune Bridge"),
    ("What money is used in the country and what backs it?", "velmoran crown"),
    ("Which saint is celebrated on the national holiday?", "Saint Orla"),
    ("How many administrative regions are there?", "eleven provinces"),
    ("Until what age must children attend school?", "compulsory until sixteen"),
    ("Where do scientists watch comets?", "Kessel Observatory"),
    ("How is renewable power generated from the sea?", "tidal turbines"),
    ("What body makes the laws?", "Grand Assembly"),
    ("Which protected insect lives in the eastern marshes?", "amber beetle"),
    ("Is there a night train between the main cities?", "Velmora Rail"),
    ("What do guests drink at weddings?", "fermented plum"),
    ("Which harbour stays open in winter?", "Port Calder"),
    ("How high is the tallest mountain?", "Dray mountains"),
    ("How much of the territory is wooded?", "forty percent"),
    ("Can the state license newspapers?", "press freedom"),
    ("What colours are on the national banner?", "blue and silver"),
    ("Which bird is printed on coins?", "heron"),
    ("What soccer tournament do supporters follow?", "Ostrava Cup"),
    ("Is there an isolated language spoken in the west?", "Old Velmoran"),
    ("How are hospitals and doctors paid for?", "universal healthcare"),
    ("How long is the legal work week?", "forty-two hours"),
    ("Where can people watch migrating geese?", "Brenn estuary"),
    ("Which town is famous for making glass?", "glass-blowing"),
    ("At what age do people get a state pension?", "pension age"),
]


def build_chunks(pdf: bool):
    from app.chunking.layout_chunker import chunk_document
    from app.inspection.inspector import inspect_document
    from app.ocr.router import route_ocr

    if pdf:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
        import fixtures as fx
        pages = [[("p", text) for _, text in FACTS[i:i + 5]] for i in range(0, len(FACTS), 5)]
        data, name = fx.text_pdf(pages), "velmora.pdf"
    else:
        data, name = "\n\n".join(text for _, text in FACTS).encode(), "velmora.md"
    result = route_ocr(data, name, inspect_document(data, name))
    return chunk_document(result), result.ocr_method


def gold_indices(chunks, phrase):
    return {c.chunk_index for c in chunks if phrase.lower() in c.content.lower()}


def metrics(ranked_lists, golds, ks=(1, 3, 5)):
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(bool(set(r[:k]) & g) for r, g in zip(ranked_lists, golds)) / len(golds)
    rr = []
    for r, g in zip(ranked_lists, golds):
        rank = next((i + 1 for i, idx in enumerate(r[:5]) if idx in g), None)
        rr.append(1 / rank if rank else 0.0)
    out["mrr@5"] = sum(rr) / len(rr)
    return out


def lat(samples):
    s = sorted(samples)
    return {"mean_ms": round(statistics.mean(s), 2), "p50_ms": round(s[len(s) // 2], 2), "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="ingest the corpus as a generated PDF instead of Markdown")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    from app.config import get_settings
    from app.embedding.transformer import TransformerEmbeddings
    from app.retrieval.chroma_store import ChromaStore
    from app.retrieval.reranker import Reranker
    from app.baselines.hash_retrieval import Embedder as HashEmbedder

    s = get_settings()
    chunks, method = build_chunks(args.pdf)
    golds = [gold_indices(chunks, phrase) for _, phrase in QUESTIONS]
    assert all(golds), "every question must have a gold chunk"

    t = time.perf_counter()
    embedder = TransformerEmbeddings(s.embedding_model, s.embedding_dimension, s.embedding_batch_size)
    reranker = Reranker(s.reranker_model)
    load_s = time.perf_counter() - t

    results = {}
    # hash baseline: dot product of normalised hashed vectors over the same chunks
    hasher = HashEmbedder("hash")
    hv = hasher.embed([c.content for c in chunks])
    ranked, times = [], []
    for q, _ in QUESTIONS:
        t = time.perf_counter()
        qv = hasher.embed([q])[0]
        order = sorted(range(len(chunks)), key=lambda i: (-sum(a * b for a, b in zip(qv, hv[i])), i))
        times.append((time.perf_counter() - t) * 1000)
        ranked.append([chunks[i].chunk_index for i in order[:args.k]])
    results["hash (baseline)"] = {**metrics(ranked, golds), "latency": lat(times)}

    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaStore(os.path.join(tmp, "chroma"), embedder)
        t = time.perf_counter()
        store.ingest(chunks, "velmora")
        ingest_s = time.perf_counter() - t
        for name, rerank in (("neural (vector only)", False), ("neural + cross-encoder", True)):
            ranked, total, rr_times = [], [], []
            for q, _ in QUESTIONS:
                t = time.perf_counter()
                res = store.retrieve(q, "velmora", top_k=args.k, rerank=rerank, reranker=reranker)
                total.append((time.perf_counter() - t) * 1000)
                if "rerank" in res.timings_ms:
                    rr_times.append(res.timings_ms["rerank"])
                assert res.reranked == rerank, res.rerank_error
                ranked.append([c.chunk_index for c in res.chunks])
            entry = {**metrics(ranked, golds), "latency": lat(total)}
            if rr_times:
                entry["rerank_latency"] = lat(rr_times)
            results[name] = entry

    report = {
        "corpus": {"format": "pdf" if args.pdf else "markdown", "ocr_method": method, "chunks": len(chunks), "facts": len(FACTS), "queries": len(QUESTIONS), "k": args.k},
        "models": {"embedding": s.embedding_model, "reranker": s.reranker_model},
        "hardware": {"machine": platform.machine(), "cpus": os.cpu_count(), "python": platform.python_version()},
        "model_load_s": round(load_s, 2),
        "ingest_embed_store_s": round(ingest_s, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "results": results,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"rag_eval_{report['corpus']['format']}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(f"\n| config | R@1 | R@3 | R@5 | MRR@5 | mean ms | p95 ms |\n|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | {r['recall@5']:.2f} | {r['mrr@5']:.3f} | {r['latency']['mean_ms']} | {r['latency']['p95_ms']} |")
    if "rerank_latency" in results.get("neural + cross-encoder", {}):
        print(f"\nrerank step alone: {results['neural + cross-encoder']['rerank_latency']}")


if __name__ == "__main__":
    main()
