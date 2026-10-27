"""Build the labelled question-pair set for the semantic-cache study.

Three sources, each with an explicit label basis:

1. authored     data/authored/cache_pairs_authored_v1.jsonl (in Git): 60
                meaning-preserving paraphrases and 80 near-misses in nine
                categories. Labels follow from how each pair was written;
                they were not produced by an independent annotator.
2. squad_rules  near-misses generated from SQuAD 2.0 dev questions by
                deterministic edits that change what is asked: a number or
                year, an inserted negation, an antonym swap, or a named
                entity swapped for another from the same article. Label =
                not equivalent, by construction. (Rule outputs can be
                ungrammatical; that does not change the label.)
3. paws         PAWS-Wiki labelled_final test: human-labelled sentence pairs
                (paraphrase vs. adversarial word-order/word-swap). These are
                statements, not questions: an external check of the
                embedding's behaviour, reported separately.

Each pair is assigned to "explore" (threshold/guard design, ~40%) or
"heldout" (reported) by a hash of its id.

Run: python -m councileval.cache_pairs --raw data/raw --out data/cache-pairs-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

EXPLORE_FRACTION = 0.4
ANTONYMS = [("first", "last"), ("most", "least"), ("largest", "smallest"), ("highest", "lowest"),
            ("earliest", "latest"), ("before", "after"), ("increase", "decrease"), ("increased", "decreased"),
            ("maximum", "minimum"), ("more", "less"), ("north", "south"), ("east", "west"), ("oldest", "youngest"),
            ("older", "younger"), ("longest", "shortest"), ("biggest", "smallest"), ("won", "lost"),
            ("start", "end"), ("began", "ended"), ("opened", "closed"), ("upper", "lower"), ("inside", "outside"),
            ("larger", "smaller"), ("higher", "lower"), ("major", "minor"), ("rise", "fall"), ("gain", "loss")]
_AUX = re.compile(r"\b(is|was|are|were|did|does|do|can|could|has|have|had|will|would|should)\b(?! not)", re.I)
_NUM = re.compile(r"\b\d{1,4}\b")
_ENTITY = re.compile(r"(?<!^)(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)")


def split_for(pair_id: str) -> str:
    h = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16)
    return "explore" if (h % 1000) / 1000 < EXPLORE_FRACTION else "heldout"


def change_number(q: str, rng: random.Random):
    m = list(_NUM.finditer(q))
    if not m:
        return None
    t = rng.choice(m)
    n = int(t.group(0))
    delta = rng.choice([1, 2, 3, 5, 10]) if n >= 1000 else rng.choice([1, 2, 3])
    new = n + delta if rng.random() < 0.5 or n - delta < 0 else n - delta
    if new == n:
        return None
    category = "year_change" if 1000 <= n <= 2100 else "number_change"
    return q[:t.start()] + str(new) + q[t.end():], category


def negate(q: str, _rng):
    m = _AUX.search(q)
    if not m:
        return None
    return q[:m.end()] + " not" + q[m.end():], "negation"


def antonym(q: str, rng: random.Random):
    options = []
    for a, b in ANTONYMS:
        for x, y in ((a, b), (b, a)):
            for m in re.finditer(rf"\b{x}\b", q, re.I):
                options.append((m, y))
    if not options:
        return None
    m, y = rng.choice(options)
    y = y.capitalize() if m.group(0)[0].isupper() else y
    return q[:m.start()] + y + q[m.end():], "antonym"


def swap_entity(q: str, rng: random.Random, pool: list[str]):
    ents = [m for m in _ENTITY.finditer(q) if m.start() > 0]
    if not ents:
        return None
    m = rng.choice(ents)
    candidates = [e for e in pool if e.lower() != m.group(0).lower() and e.lower() not in q.lower()]
    if not candidates:
        return None
    return q[:m.start()] + rng.choice(candidates) + q[m.end():], "entity_change"


def squad_rule_pairs(raw: Path, rng: random.Random, per_rule: int) -> list[dict]:
    data = json.loads((raw / "squad_v2_dev.json").read_text())
    questions, pools = [], {}
    for a_i, art in enumerate(data["data"]):
        ents = set()
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                questions.append((a_i, qa["id"], qa["question"].strip()))
                ents |= {m.group(0) for m in _ENTITY.finditer(qa["question"]) if m.start() > 0}
        pools[a_i] = sorted(ents)
    rng.shuffle(questions)
    rules = [("number", change_number), ("negation", negate), ("antonym", antonym), ("entity", None)]
    out, used = [], Counter()
    for a_i, qid, q in questions:
        for name, fn in rules:
            if used[name] >= per_rule:
                continue
            res = swap_entity(q, rng, pools[a_i]) if name == "entity" else fn(q, rng)
            if res and res[0] != q:
                used[name] += 1
                out.append({"id": f"squadrule-{name}-{qid}", "q1": q, "q2": res[0], "equivalent": False,
                            "category": res[1], "source": "squad_rules",
                            "label_basis": f"rule '{name}' changes what is asked"})
                break
        if all(used[n] >= per_rule for n, _ in rules):
            break
    return out


def paws_pairs(raw: Path, rng: random.Random, per_label: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(raw / "paws_wiki_labeled_final_test.parquet").to_pylist()
    out = []
    for label in (1, 0):
        pick = rng.sample([r for r in rows if r["label"] == label], per_label)
        for r in pick:
            out.append({"id": f"paws-{r['id']}", "q1": r["sentence1"], "q2": r["sentence2"], "equivalent": bool(label),
                        "category": "paws_paraphrase" if label else "paws_adversarial", "source": "paws",
                        "label_basis": "PAWS human label"})
    return out


def build(raw: Path, authored: Path, out: Path, seed: int = 29, per_rule: int = 100, paws_per_label: int = 150) -> dict:
    pairs = [json.loads(l) for l in authored.read_text().splitlines() if l.strip()]
    pairs += squad_rule_pairs(raw, random.Random(f"{seed}:squad"), per_rule)
    pairs += paws_pairs(raw, random.Random(f"{seed}:paws"), paws_per_label)
    ids = Counter(p["id"] for p in pairs)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate pair ids: {dup[:5]}")
    for p in pairs:
        if not p["q1"].strip() or not p["q2"].strip() or not isinstance(p["equivalent"], bool):
            raise SystemExit(f"malformed pair {p['id']}")
        p["split"] = split_for(p["id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in pairs))
    manifest = {"version": "cache-pairs-v1", "seed": seed, "pairs": len(pairs),
                "by_source": dict(Counter(p["source"] for p in pairs)),
                "by_category": dict(sorted(Counter(p["category"] for p in pairs).items())),
                "by_split": dict(Counter(p["split"] for p in pairs)),
                "equivalent": sum(p["equivalent"] for p in pairs), "not_equivalent": sum(not p["equivalent"] for p in pairs)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--authored", type=Path, default=Path("data/authored/cache_pairs_authored_v1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/cache-pairs-v1"))
    a = ap.parse_args()
    print(json.dumps(build(a.raw, a.authored, a.out), indent=2))


if __name__ == "__main__":
    main()
