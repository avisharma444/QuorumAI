"""Build the answer-quality dataset (councileval-v1) from public benchmarks.

Inputs (downloaded once into evaluation/data/raw/, git-ignored; checksums
are verified so a changed upstream file is detected):

* SQuAD 2.0 dev            CC BY-SA 4.0   extractive + unanswerable
* HotpotQA distractor val  CC BY-SA 4.0   multi-hop (bridge, comparison)
* DROP dev                 CC BY-SA 4.0   numeric, date and span reasoning

Ground truth is copied from the source. Evidence sentences come from the
source annotations (SQuAD: the sentence containing the gold answer span;
HotpotQA: the gold supporting facts). DROP has no evidence annotation, so
its evidence list is empty and its retrieval relevance is "unknown".

Sampling is deterministic (seed). Each item is assigned to "dev" (debugging,
~20%) or "heldout" (reported results) by a hash of its id, so assignments
never change when the sample grows.

Run: python -m councileval.prepare --raw evaluation/data/raw --out evaluation/data/councileval-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

from . import DATASET_VERSION
from .schema import Document, EvalItem, validate_dataset

RAW_FILES = {
    "squad_v2_dev.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    "hotpot_distractor_validation.parquet": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    "drop_dataset.zip": "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6",
}
SOURCES = {
    "squad_v2_dev.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "hotpot_distractor_validation.parquet": "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet",
    "drop_dataset.zip": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
}
LICENSE = "CC BY-SA 4.0"
DEFAULT_COUNTS = {"squad_answerable": 150, "squad_unanswerable": 100, "hotpot_bridge": 75,
                  "hotpot_comparison": 75, "drop_number": 50, "drop_date": 25, "drop_span": 25}
DEV_FRACTION = 0.2

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_raw(raw: Path) -> dict:
    out = {}
    for name, want in RAW_FILES.items():
        p = raw / name
        if not p.exists():
            raise SystemExit(f"missing {p}; download it from {SOURCES[name]}")
        got = sha256(p)
        if got != want:
            raise SystemExit(f"checksum mismatch for {name}: {got} (expected {want})")
        out[name] = {"sha256": got, "source": SOURCES[name], "bytes": p.stat().st_size}
    return out


def split_for(item_id: str) -> str:
    h = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)
    return "dev" if (h % 1000) / 1000 < DEV_FRACTION else "heldout"


def sentence_at(context: str, offset: int, end: int | None = None) -> str:
    """The smallest run of consecutive sentences of context covering the
    character range [offset, end) (an answer can cross a splitter boundary,
    e.g. after an abbreviation)."""
    end = offset + 1 if end is None else max(end, offset + 1)
    bounds, start = [], 0
    for m in _SENT_END.finditer(context):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(context)))
    first = next(i for i, (s, e) in enumerate(bounds) if offset < e or i == len(bounds) - 1)
    last = next((i for i, (s, e) in enumerate(bounds) if end <= e), len(bounds) - 1)
    return context[bounds[first][0]:bounds[max(first, last)][1]].strip()


# ── SQuAD 2.0 ──────────────────────────────────────────────────────────


def squad(raw: Path, rng: random.Random, n_ans: int, n_unans: int):
    data = json.loads((raw / "squad_v2_dev.json").read_text())
    docs, answerable, unanswerable = {}, [], []
    for a_i, art in enumerate(data["data"]):
        title = art["title"].replace("_", " ")
        doc_id = f"squad-{a_i:02d}"
        paras = [p["context"] for p in art["paragraphs"]]
        docs[doc_id] = Document(doc_id, title, f"# {title}\n\n" + "\n\n".join(paras), "squad_v2/dev", LICENSE)
        for p_i, p in enumerate(art["paragraphs"]):
            for qa in p["qas"]:
                rec = (doc_id, p_i, p["context"], qa)
                (unanswerable if qa["is_impossible"] else answerable).append(rec)
    items = []
    for doc_id, p_i, ctx, qa in rng.sample(answerable, n_ans):
        golds = sorted({a["text"].strip() for a in qa["answers"] if a["text"].strip()})
        first = qa["answers"][0]
        items.append(EvalItem(
            id=f"squad:{qa['id']}", question=qa["question"].strip(), source="squad_v2/dev", source_id=qa["id"],
            category="extractive", answer_type="number" if all(_is_number(g) for g in golds) else "span",
            answers=golds, answerable=True, doc_id=doc_id, evidence=[sentence_at(ctx, first["answer_start"], first["answer_start"] + len(first["text"]))],
            split="", license=LICENSE, metadata={"paragraph_index": p_i}))
    for doc_id, p_i, ctx, qa in rng.sample(unanswerable, n_unans):
        items.append(EvalItem(
            id=f"squad:{qa['id']}", question=qa["question"].strip(), source="squad_v2/dev", source_id=qa["id"],
            category="unanswerable", answer_type="none", answers=[], answerable=False, doc_id=doc_id, evidence=[],
            split="", license=LICENSE,
            metadata={"paragraph_index": p_i, "plausible_answers": sorted({a["text"] for a in qa.get("plausible_answers", [])})}))
    return items, docs


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


# ── HotpotQA ───────────────────────────────────────────────────────────


def hotpot(raw: Path, rng: random.Random, n_bridge: int, n_comp: int):
    import pyarrow.parquet as pq

    rows = pq.read_table(raw / "hotpot_distractor_validation.parquet").to_pylist()
    by_type = {"bridge": [], "comparison": []}
    for r in rows:
        by_type[r["type"]].append(r)
    items, docs = [], {}
    for kind, n in (("bridge", n_bridge), ("comparison", n_comp)):
        for r in rng.sample(by_type[kind], n):
            titles, sents = r["context"]["title"], r["context"]["sentences"]
            paras = {t: s for t, s in zip(titles, sents)}
            doc_id = f"hotpot-{r['id']}"
            body = "\n\n".join(f"## {t}\n\n" + "".join(s).strip() for t, s in zip(titles, sents))
            docs[doc_id] = Document(doc_id, f"HotpotQA context {r['id']}", f"# Reference passages\n\n{body}",
                                    "hotpot_qa/distractor/validation", LICENSE)
            evidence = []
            for t, sid in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
                if t in paras and sid < len(paras[t]) and paras[t][sid].strip():
                    evidence.append(paras[t][sid].strip())
            ans = r["answer"].strip()
            yes_no = ans.lower() in ("yes", "no")
            items.append(EvalItem(
                id=f"hotpot:{r['id']}", question=r["question"].strip(), source="hotpot_qa/distractor/validation",
                source_id=r["id"], category=f"multi_hop_{kind}",
                answer_type="yes_no" if yes_no else ("number" if _is_number(ans) else "span"),
                answers=[ans], answerable=True, doc_id=doc_id, evidence=evidence, split="", license=LICENSE,
                metadata={"level": r["level"], "supporting_titles": sorted(set(r["supporting_facts"]["title"]))}))
    return items, docs


# ── DROP ──────────────────────────────────────────────────────────────

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _date_answers(d: dict) -> tuple[list[str], dict]:
    day, month, year = (d.get("day") or "").strip(), (d.get("month") or "").strip(), (d.get("year") or "").strip()
    parts = {"day": day, "month": month, "year": year}
    text = " ".join(x for x in (month, day, year) if x)
    variants = {text}
    if day and month and year:
        variants |= {f"{day} {month} {year}", f"{month} {day}, {year}"}
    return sorted(v for v in variants if v), parts


def drop(raw: Path, rng: random.Random, n_num: int, n_date: int, n_span: int):
    with zipfile.ZipFile(raw / "drop_dataset.zip") as z:
        data = json.loads(z.read("drop_dataset/drop_dataset_dev.json"))
    pools = {"number": [], "date": [], "span": []}
    for pid, p in data.items():
        for qa in p["qa_pairs"]:
            a = qa["answer"]
            if a.get("number"):
                pools["number"].append((pid, p, qa))
            elif any((a.get("date") or {}).get(k) for k in ("day", "month", "year")):
                pools["date"].append((pid, p, qa))
            elif a.get("spans") and len(a["spans"]) == 1:
                pools["span"].append((pid, p, qa))
    items, docs = [], {}
    for kind, n, cat in (("number", n_num, "numeric"), ("date", n_date, "temporal"), ("span", n_span, "span_reasoning")):
        for pid, p, qa in rng.sample(pools[kind], n):
            doc_id = f"drop-{pid}"
            docs.setdefault(doc_id, Document(doc_id, pid, f"# Passage {pid}\n\n{p['passage'].strip()}", "drop/dev", LICENSE))
            a = qa["answer"]
            meta = {"validated_answers": len(qa.get("validated_answers", []))}
            if kind == "number":
                answers, atype = [a["number"].strip()], "number"
            elif kind == "date":
                answers, parts = _date_answers(a["date"])
                atype, meta["date"] = "date", parts
            else:
                answers, atype = [a["spans"][0].strip()], "span"
            items.append(EvalItem(
                id=f"drop:{qa['query_id']}", question=qa["question"].strip(), source="drop/dev", source_id=qa["query_id"],
                category=cat, answer_type=atype, answers=answers, answerable=True, doc_id=doc_id, evidence=[],
                split="", license=LICENSE, metadata=meta))
    return items, docs


def build(raw: Path, out: Path, seed: int = 13, counts: dict | None = None) -> dict:
    counts = dict(DEFAULT_COUNTS, **(counts or {}))
    checksums = verify_raw(raw)
    items, docs = [], {}
    for fn, args in ((squad, (counts["squad_answerable"], counts["squad_unanswerable"])),
                     (hotpot, (counts["hotpot_bridge"], counts["hotpot_comparison"])),
                     (drop, (counts["drop_number"], counts["drop_date"], counts["drop_span"]))):
        rng = random.Random(f"{seed}:{fn.__name__}")  # independent streams per source
        it, dc = fn(raw, rng, *args)
        items += it
        docs.update(dc)
    for it in items:
        it.split = split_for(it.id)
    used = {it.doc_id for it in items}
    docs = {k: v for k, v in docs.items() if k in used}
    validate_dataset(items, docs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(it.to_json() + "\n" for it in sorted(items, key=lambda i: i.id)))
    (out / "documents.jsonl").write_text("".join(json.dumps(asdict(d), ensure_ascii=False, sort_keys=True) + "\n"
                                                 for d in sorted(docs.values(), key=lambda d: d.doc_id)))
    by = lambda key: {k: sum(1 for it in items if key(it) == k) for k in sorted({key(it) for it in items})}
    manifest = {"version": DATASET_VERSION, "seed": seed, "counts_requested": counts, "items": len(items),
                "documents": len(docs), "by_category": by(lambda i: i.category), "by_split": by(lambda i: i.split),
                "by_source": by(lambda i: i.source), "dev_fraction": DEV_FRACTION, "raw_files": checksums,
                "license": f"Derived from {LICENSE} datasets (SQuAD 2.0, HotpotQA, DROP); share-alike applies."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("evaluation/data/raw"))
    ap.add_argument("--out", type=Path, default=Path(f"evaluation/data/{DATASET_VERSION}"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    print(json.dumps(build(args.raw, args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
