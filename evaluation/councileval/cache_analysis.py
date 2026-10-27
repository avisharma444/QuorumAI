"""Analyse semantic-cache decisions measured by go-backend/cmd/cacheeval.

* Verifies the recorded production decisions: production_hit(t) must equal
  (similarity >= t and numbers_match), and the RediSearch similarity must
  equal the locally computed cosine. Any mismatch is reported.
* For every threshold: confusion matrix, TPR (equivalent pairs reused),
  FPR (different questions wrongly reused), precision, F1, with 95% Wilson
  intervals, per split, per source, and per near-miss category.
* Threshold selection uses the EXPLORE split only; the chosen threshold is
  then reported on the HELD-OUT split.
* Experimental guards (councileval.guards) are evaluated on the same pairs as
  extra vetoes on top of the production rule; paired bootstrap CIs give the
  change in TPR/FPR. Guards are not enabled in the application.

Run: python -m councileval.cache_analysis --pairs data/cache-pairs-v1/pairs.jsonl \
        --decisions results/cache/decisions.jsonl --out results/cache
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .guards import GUARDS, vetoed
from .metrics import confusion
from .stats import paired_bootstrap_diff, wilson_interval

FPR_TARGET = 0.02  # maximum false-positive rate allowed on the explore split when choosing a threshold
SIM_EPS = 1e-6     # float32 vs float64 tolerance at the threshold boundary


def load(pairs_path: Path, decisions_path: Path) -> list[dict]:
    pairs = {json.loads(l)["id"]: json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()}
    rows = []
    for l in decisions_path.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            p = pairs[d["id"]]
            if p["equivalent"] != d["equivalent"] or p["split"] != d["split"]:
                raise ValueError(f"{d['id']}: decision does not match its pair")
            d["q1"], d["q2"] = p["q1"], p["q2"]
            rows.append(d)
    return rows


def verify(rows: list[dict]) -> dict:
    """Check that recorded production decisions follow the threshold rule."""
    rule_mismatch, redis_mismatch, boundary = [], [], 0
    for r in rows:
        if r["redis_similarity"] is not None and abs(r["redis_similarity"] - r["similarity"]) > 1e-4:
            redis_mismatch.append(r["id"])
        for t, hit in r["production_hit"].items():
            t = float(t)
            if abs(r["similarity"] - t) <= SIM_EPS:
                boundary += 1
                continue
            if hit != (r["similarity"] >= t and r["numbers_match"]):
                rule_mismatch.append((r["id"], t))
    return {"rule_mismatches": len(rule_mismatch), "redis_similarity_mismatches": len(redis_mismatch),
            "boundary_cases_skipped": boundary, "examples": rule_mismatch[:5] + redis_mismatch[:5]}


def decide(r: dict, t: float, guards=()) -> bool:
    """Production rule at threshold t, optionally with extra guard vetoes."""
    key = f"{t:.3f}"
    hit = r["production_hit"][key] if key in r["production_hit"] else (r["similarity"] >= t and r["numbers_match"])
    return hit and not (guards and vetoed(r["q1"], r["q2"], guards))


def table(rows: list[dict], t: float, guards=()) -> dict:
    c = confusion([r["equivalent"] for r in rows], [decide(r, t, guards) for r in rows])
    pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
    c["tpr_ci95"] = wilson_interval(c["tp"], pos)
    c["fpr_ci95"] = wilson_interval(c["fp"], neg)
    c["precision_ci95"] = wilson_interval(c["tp"], c["tp"] + c["fp"])
    c["positives"], c["negatives"] = pos, neg
    return c


def choose_threshold(rows: list[dict], thresholds: list[float], guards=()) -> dict:
    """Lowest threshold (most reuse) whose explore FPR <= FPR_TARGET."""
    for t in sorted(thresholds):
        c = table(rows, t, guards)
        if c["fpr"] is not None and c["fpr"] <= FPR_TARGET:
            return {"threshold": t, "explore": c}
    return {"threshold": None, "explore": None}


def analyse(rows: list[dict]) -> dict:
    thresholds = sorted({float(t) for r in rows for t in r["production_hit"]})
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    explore, held = by_split["explore"], by_split["heldout"]

    out = {"verification": verify(rows), "counts": {s: {"pairs": len(v), "equivalent": sum(r["equivalent"] for r in v),
                                                       "not_equivalent": sum(not r["equivalent"] for r in v)} for s, v in by_split.items()},
           "fpr_target_for_selection": FPR_TARGET, "thresholds": thresholds}
    out["heldout_by_threshold"] = {f"{t:.3f}": table(held, t) for t in thresholds}
    out["heldout_by_source"] = {}
    for src in sorted({r["source"] for r in held}):
        sel = [r for r in held if r["source"] == src]
        out["heldout_by_source"][src] = {f"{t:.3f}": table(sel, t) for t in thresholds}
    cats = defaultdict(list)
    for r in held:
        if not r["equivalent"]:
            cats[r["category"]].append(r)
    out["heldout_near_miss_reuse_by_category"] = {
        c: {f"{t:.3f}": {"reused": sum(decide(r, t) for r in v), "n": len(v),
                         "rate": sum(decide(r, t) for r in v) / len(v), "ci95": wilson_interval(sum(decide(r, t) for r in v), len(v))}
            for t in (0.80, 0.85, 0.90, 0.95)} for c, v in sorted(cats.items())}
    # Similarity distributions (equivalent vs not), held-out.
    for name, sel in (("equivalent", [r for r in held if r["equivalent"]]), ("not_equivalent", [r for r in held if not r["equivalent"]])):
        s = sorted(r["similarity"] for r in sel)
        out.setdefault("heldout_similarity", {})[name] = {"n": len(s), "min": s[0], "p25": s[len(s) // 4], "median": s[len(s) // 2],
                                                           "p75": s[3 * len(s) // 4], "max": s[-1]} if s else None
    # Threshold selected on explore, evaluated on held-out; with and without guards.
    variants = {"production (cosine + numbers)": ()}
    variants.update({f"+ {g}": (g,) for g in GUARDS})
    variants["+ all guards"] = tuple(GUARDS)
    out["selected_threshold"] = {}
    for name, g in variants.items():
        ch = choose_threshold(explore, thresholds, g)
        t = ch["threshold"]
        out["selected_threshold"][name] = {"threshold": t, "explore": ch["explore"], "heldout": table(held, t, g) if t else None}
    # Guards at the reference threshold 0.85 on held-out: paired change vs production.
    base_neg = [float(decide(r, 0.85)) for r in held if not r["equivalent"]]
    base_pos = [float(decide(r, 0.85)) for r in held if r["equivalent"]]
    out["guards_at_0.85_heldout"] = {}
    for name, g in variants.items():
        if not g:
            continue
        neg = [float(decide(r, 0.85, g)) for r in held if not r["equivalent"]]
        pos = [float(decide(r, 0.85, g)) for r in held if r["equivalent"]]
        out["guards_at_0.85_heldout"][name] = {"table": table(held, 0.85, g), "fpr_change": paired_bootstrap_diff(base_neg, neg),
                                               "tpr_change": paired_bootstrap_diff(base_pos, pos)}
    # Remaining false positives at 0.85 with all guards (held-out), for inspection.
    out["heldout_false_positives_0.85_all_guards"] = [
        {"id": r["id"], "category": r["category"], "similarity": round(r["similarity"], 4), "q1": r["q1"], "q2": r["q2"]}
        for r in held if not r["equivalent"] and decide(r, 0.85, tuple(GUARDS))][:40]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/cache-pairs-v1/pairs.jsonl"))
    ap.add_argument("--decisions", type=Path, default=Path("results/cache/decisions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/cache"))
    a = ap.parse_args()
    res = analyse(load(a.pairs, a.decisions))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "cache_summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"verification": res["verification"], "counts": res["counts"]}, indent=2))


if __name__ == "__main__":
    main()
