"""Blind human-evaluation packet for long-form answers.

From a run's records it writes, for a seeded sample of items:

  packet.csv  item, question, reference answer(s), and the answers of every
              strategy under random labels (X1, X2, ...), shuffled per item,
              plus empty rubric columns for each label;
  key.csv     label -> strategy mapping (keep it away from the raters).

Rubric (1-5 each; see evaluation/README.md): factual correctness,
completeness, relevance, unsupported claims (5 = none), clarity.

Run: python -m councileval.blind results/runs/<id> --n 40 --seed 3
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .schema import load_dataset

RUBRIC = ["correctness", "completeness", "relevance", "unsupported_claims", "clarity"]


def build(run: Path, n: int, seed: int) -> tuple[Path, Path]:
    meta = json.loads((run / "metadata.json").read_text())
    items, _ = load_dataset(Path(meta["config"]["dataset"]))
    by_id = {i.id: i for i in items}
    answers = defaultdict(dict)
    for line in (run / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["status"] == 200 and r["response"] and r["repeat"] == 0:
            answers[r["item_id"]][r["strategy"]] = r["response"].get("answer", "")
    complete = sorted(i for i, a in answers.items() if len(a) == len(meta["config"]["strategies"]))
    rng = random.Random(seed)
    pick = rng.sample(complete, min(n, len(complete)))
    strategies = meta["config"]["strategies"]
    packet, key = run / "blind_packet.csv", run / "blind_key.csv"
    with open(packet, "w", newline="") as pf, open(key, "w", newline="") as kf:
        pw, kw = csv.writer(pf), csv.writer(kf)
        header = ["item", "question", "reference"]
        for j in range(len(strategies)):
            header += [f"X{j + 1}_answer"] + [f"X{j + 1}_{c}" for c in RUBRIC]
        pw.writerow(header)
        kw.writerow(["item", "label", "strategy"])
        for n_i, item_id in enumerate(pick, 1):
            order = list(strategies)
            rng.shuffle(order)
            it = by_id[item_id]
            row = [f"Q{n_i}", it.question, " | ".join(it.answers) if it.answerable else "(not answerable from the document)"]
            for j, st in enumerate(order):
                row += [answers[item_id][st]] + [""] * len(RUBRIC)
                kw.writerow([f"Q{n_i}", f"X{j + 1}", st])
            pw.writerow(row)
    return packet, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    p, k = build(a.run, a.n, a.seed)
    print(f"packet: {p}\nkey (keep from raters): {k}")


if __name__ == "__main__":
    main()
