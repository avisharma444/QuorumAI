"""Aggregate an experiment run into summary.json, scores.csv and report.md.

Offline (mock-model) runs are reported as WORKFLOW VALIDATION: the correctness
columns describe mock text and are not model accuracy.

Run: python -m councileval.report results/runs/<experiment_id>
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .cost import UsageLedger, load_pricing
from .metrics import brier_score, calibration_bins, expected_calibration_error
from .schema import load_dataset
from .scoring import score_record
from .stats import describe_difference, mcnemar_exact, paired_bootstrap_diff, wilson_interval

PAIRS = [("direct", "council"), ("council", "council_deep"), ("direct", "council_deep"), ("direct", "auto")]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def parse_prom(text: Optional[str]) -> dict[str, float]:
    out = {}
    for line in (text or "").splitlines():
        if line.startswith("councilai_"):
            k, v = line.rsplit(" ", 1)
            out[k] = float(v)
    return out


def prom_deltas(run: Path) -> Optional[dict]:
    b, a = run / "metrics_before.prom", run / "metrics_after.prom"
    if not (b.exists() and a.exists()):
        return None
    before, after = parse_prom(b.read_text()), parse_prom(a.read_text())
    d = {k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0) != 0}

    def grab(prefix: str, label: str) -> dict:
        res = defaultdict(lambda: {"count": 0.0, "sum_s": 0.0})
        for k, v in d.items():
            if k.startswith(prefix):
                m = re.search(label + r'="([^"]+)"', k)
                key = m.group(1) if m else "all"
                if k.startswith(prefix + "_count"):
                    res[key]["count"] += v
                elif k.startswith(prefix + "_sum"):
                    res[key]["sum_s"] += v
        return {k: {"count": int(v["count"]), "mean_ms": round(1000 * v["sum_s"] / v["count"], 2) if v["count"] else None}
                for k, v in res.items()}

    return {"council_stage": grab("councilai_council_stage_seconds", "stage"),
            "llm_request": grab("councilai_llm_request_seconds", "stage"),
            "rag_request": grab("councilai_rag_request_seconds", "endpoint"),
            "cache_lookup": grab("councilai_cache_lookup_seconds", "level"),
            "llm_failures": {k: v for k, v in d.items() if k.startswith("councilai_llm_failure_count_total")},
            "llm_tokens": {k: v for k, v in d.items() if k.startswith("councilai_llm_tokens_total")}}


def rate(rows, key) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    k = sum(1 for v in vals if v)
    lo, hi = wilson_interval(k, len(vals))
    return {"n": len(vals), "k": k, "rate": k / len(vals) if vals else None, "ci95": (lo, hi)}


def build(run: Path) -> dict:
    meta = json.loads((run / "metadata.json").read_text())
    cfg = meta["config"]
    data_dir = Path(cfg["dataset"])
    items, _ = load_dataset(data_dir if data_dir.is_absolute() else Path.cwd() / data_dir)
    by_id = {i.id: i for i in items}
    recs = [json.loads(l) for l in (run / "records.jsonl").read_text().splitlines() if l.strip()]
    scores = [score_record(by_id[r["item_id"]], r) for r in recs]
    mock = meta["backend_llm"].get("mode") == "mock"

    by_strategy = defaultdict(list)
    for s in scores:
        by_strategy[s["strategy"]].append(s)
    strategies = {}
    pricing = {}
    try:
        pricing = load_pricing(Path("config/pricing.yaml"))
    except Exception:  # pricing is optional for reporting
        pass
    for st, rows in sorted(by_strategy.items()):
        ok = [r for r in rows if r["ok"]]
        ledger = UsageLedger()
        for r in ok:
            for model, usage in zip(r["candidate_models"], r["candidate_usage"]):
                ledger.add_call(model, usage, pricing)
        tax = defaultdict(int)
        for r in ok:
            tax[r["taxonomy"]] += 1
        strategies[st] = {
            "requests": len(rows), "succeeded": len(ok), "failed": len(rows) - len(ok),
            "http_statuses": dict(sorted({str(k): sum(1 for r in rows if r["status"] == k) for k in {r["status"] for r in rows}}.items())),
            "automatic_correctness": rate(ok, "correct"),
            "answerable_correctness": rate([r for r in ok if r["answerable"]], "correct"),
            "unanswerable_abstention": rate([r for r in ok if not r["answerable"]], "correct"),
            "false_abstention_rate": rate([r for r in ok if r["answerable"]], "abstained"),
            "exact_match": rate(ok, "exact_match"), "short_answers_scored_by_em": sum(1 for r in ok if r["exact_match"] is not None),
            "token_f1_mean": statistics.mean([r["token_f1"] for r in ok if r["token_f1"] is not None]) if any(r["token_f1"] is not None for r in ok) else None,
            "taxonomy": dict(tax),
            "evidence_retrieved": rate(ok, "evidence_retrieved"),
            "cache_contaminated": sum(1 for r in ok if r["cache_contaminated"]),
            "latency_ms": {"p50": pct([r["latency_ms"] for r in ok], 0.5), "p95": pct([r["latency_ms"] for r in ok], 0.95),
                           "mean": statistics.mean([r["latency_ms"] for r in ok]) if ok else None},
            "model_calls_mean": statistics.mean([r["model_calls"] for r in ok]) if ok else None,
            "executed_strategies": dict(sorted({k: sum(1 for r in ok if r["executed_strategy"] == k) for k in {r["executed_strategy"] for r in ok}}.items())),
            "router_fallbacks": sum(1 for r in ok if r.get("router_fallback")),
            "reported_token_usage": {"prompt": sum((r["token_usage"] or {}).get("prompt_tokens", 0) for r in ok),
                                     "completion": sum((r["token_usage"] or {}).get("completion_tokens", 0) for r in ok),
                                     "requests_with_usage": sum(1 for r in ok if r["token_usage"])},
            "candidate_ledger": ledger.summary(mock),
            "confidence": {"reported": sum(1 for r in ok if r["confidence"] is not None),
                           "brier": brier_score([r["confidence"] for r in ok], [r["correct"] for r in ok]),
                           "ece_10": expected_calibration_error([r["confidence"] for r in ok], [r["correct"] for r in ok]),
                           "bins": calibration_bins([r["confidence"] for r in ok], [r["correct"] for r in ok])},
            "by_category": {c: rate([r for r in ok if r["category"] == c], "correct") for c in sorted({r["category"] for r in ok})},
        }

    # Paired comparisons on identical items (same item id and repeat).
    idx = {(s["item_id"], s["repeat"], s["strategy"]): s for s in scores if s["ok"]}
    comparisons = {}
    for a, b in PAIRS:
        keys = sorted({(i, rep) for (i, rep, st) in idx if st == a} & {(i, rep) for (i, rep, st) in idx if st == b})
        if not keys:
            continue
        xa = [idx[(i, rep, a)]["correct"] for i, rep in keys]
        xb = [idx[(i, rep, b)]["correct"] for i, rep in keys]
        diff = paired_bootstrap_diff([float(x) if x is not None else None for x in xa], [float(x) if x is not None else None for x in xb])
        lat = paired_bootstrap_diff([idx[(i, rep, a)]["latency_ms"] for i, rep in keys], [idx[(i, rep, b)]["latency_ms"] for i, rep in keys])
        verdict = "not interpreted (mock models)" if mock else describe_difference(diff)
        comparisons[f"{b} vs {a}"] = {"n_items": len(keys), "correctness": diff, "verdict": verdict,
                                      "mcnemar": mcnemar_exact(xa, xb), "latency_ms": lat}

    # Evidence identity: the same item must see the same retrieved chunks under every strategy.
    fps = defaultdict(set)
    for s in scores:
        if s["ok"] and s["grounding"] == "document":
            fps[(s["item_id"], s["repeat"])].add(s["evidence_fingerprint"])
    multi = [k for k, v in fps.items() if len(v) > 1]

    council_rows = [s for s in scores if s["ok"] and s["executed_strategy"] in ("council", "council_deep") and s["candidates_n"]]
    ablation = {
        "n": len(council_rows),
        "mean_single_candidate": statistics.mean([r["mean_candidate_correct"] for r in council_rows if r["mean_candidate_correct"] is not None]) if council_rows else None,
        "best_ranked_candidate": rate(council_rows, "best_ranked_correct"),
        "chairman_synthesis": rate(council_rows, "chairman_correct"),
        "final_answer": rate(council_rows, "correct"),
        "oracle_any_candidate": rate(council_rows, "any_candidate_correct"),
        "chairman_vs_best_ranked": paired_bootstrap_diff([float(r["best_ranked_correct"]) if r["best_ranked_correct"] is not None else None for r in council_rows],
                                                         [float(r["chairman_correct"]) if r["chairman_correct"] is not None else None for r in council_rows]),
    }
    deep = [s for s in scores if s["ok"] and s["executed_strategy"] == "council_deep" and s.get("reflection_quality")]
    reflection = {
        "n": len(deep),
        "revision_triggered": sum(1 for r in deep if r["revision_status"]),
        "flagged_when_chairman_wrong": sum(1 for r in deep if r["chairman_correct"] is False and r["reflection_quality"] == "needs_revision"),
        "chairman_wrong": sum(1 for r in deep if r["chairman_correct"] is False),
        "flagged_when_chairman_right": sum(1 for r in deep if r["chairman_correct"] and r["reflection_quality"] == "needs_revision"),
        "revision_fixed": sum(1 for r in deep if r["revision_status"] == "succeeded" and r["chairman_correct"] is False and r["correct"]),
        "revision_broke": sum(1 for r in deep if r["revision_status"] == "succeeded" and r["chairman_correct"] and r["correct"] is False),
    }
    run_state = json.loads((run / "run_state.json").read_text()) if (run / "run_state.json").exists() else None
    summary = {"experiment_id": meta["experiment_id"], "mode": cfg["mode"], "mock_models": mock,
               "interpretation": ("OFFLINE WORKFLOW VALIDATION: answers are mock text; correctness figures test the pipeline "
                                  "and scoring code and are NOT model accuracy.") if mock else
               "Real-model run: automatic correctness as defined in evaluation/README.md.",
               "metadata": meta, "run_state": run_state, "records": len(recs), "strategies": strategies,
               "paired_comparisons": comparisons,
               "evidence_identity": {"items": len(fps), "items_with_differing_evidence": len(multi), "examples": multi[:5]},
               "council_stage_ablation": ablation, "reflection": reflection, "prometheus_deltas": prom_deltas(run)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if scores:
        cols = [k for k in scores[0].keys() if k not in ("candidate_usage",)]
        with open(run / "scores.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for s in scores for k in s} - {"candidate_usage"}), extrasaction="ignore")
            w.writeheader()
            w.writerows(scores)
    (run / "report.md").write_text(markdown(summary))
    return summary


def _r(x, d=3):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def _rate(r: dict) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{r['rate']:.3f} ({r['k']}/{r['n']}; 95% CI {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f})"


def markdown(s: dict) -> str:
    m = s["metadata"]
    lines = [f"# Evaluation report: {s['experiment_id']}", "", f"> {s['interpretation']}", "",
             "## Setup", "",
             f"- mode: **{s['mode']}**, backend models: {m['backend_llm']}",
             f"- external model calls enabled: {m['external_model_calls_enabled']}",
             f"- dataset: {m['dataset_version']} split `{m['config']['split']}`, items {m['items']} (ids sha256 {m['item_ids_sha256'][:12]})",
             f"- strategies: {m['config']['strategies']}, repeats {m['config']['repeats']}, top_k {m['config']['top_k']}, seed {m['config']['seed']}",
             f"- git commit: {m['git_commit']}, started {m['started_utc']}, runtime {m['runtime']}",
             f"- run state: {s['run_state']}", "",
             "## Per strategy", "",
             "| strategy | ok/requests | automatic correctness | answerable | unanswerable abstention | false abstention | evidence retrieved | p50 ms | p95 ms | model calls | cache contaminated |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, v in s["strategies"].items():
        lines.append(f"| {st} | {v['succeeded']}/{v['requests']} | {_rate(v['automatic_correctness'])} | {_rate(v['answerable_correctness'])} | "
                     f"{_rate(v['unanswerable_abstention'])} | {_rate(v['false_abstention_rate'])} | {_rate(v['evidence_retrieved'])} | "
                     f"{_r(v['latency_ms']['p50'], 1)} | {_r(v['latency_ms']['p95'], 1)} | {_r(v['model_calls_mean'], 2)} | {v['cache_contaminated']} |")
    lines += ["", "## Paired comparisons (same items)", "", "| comparison | n | correctness A → B | verdict | McNemar p | latency diff ms |", "|---|---|---|---|---|---|"]
    for k, v in s["paired_comparisons"].items():
        c = v["correctness"]
        lines.append(f"| {k} | {v['n_items']} | {_r(c['mean_a'])} → {_r(c['mean_b'])} | {v['verdict']} | {_r(v['mcnemar']['p_value'])} | {_r(v['latency_ms']['diff'], 1)} |")
    ab = s["council_stage_ablation"]
    lines += ["", "## Council stages", "",
              f"- council responses analysed: {ab['n']}",
              f"- mean single-candidate correctness: {_r(ab['mean_single_candidate'])}",
              f"- best peer-ranked candidate: {_rate(ab['best_ranked_candidate'])}",
              f"- chairman synthesis: {_rate(ab['chairman_synthesis'])}",
              f"- final answer: {_rate(ab['final_answer'])}",
              f"- oracle (any candidate correct): {_rate(ab['oracle_any_candidate'])}",
              f"- chairman − best-ranked: {'not interpreted (mock models)' if s['mock_models'] else describe_difference(ab['chairman_vs_best_ranked'])}",
              f"- reflection: {s['reflection']}", "",
              "## Evidence identity", "", f"{s['evidence_identity']}", "",
              "## Prometheus deltas for this run", "", "```", json.dumps(s["prometheus_deltas"], indent=2), "```", "",
              "## Taxonomy, cost and calibration per strategy", ""]
    for st, v in s["strategies"].items():
        lines += [f"### {st}", "", f"- taxonomy: {v['taxonomy']}", f"- executed strategies: {v['executed_strategies']}, router fallbacks: {v['router_fallbacks']}",
                  f"- reported tokens: {v['reported_token_usage']}", f"- candidate-call ledger: {v['candidate_ledger']}",
                  f"- confidence: {v['confidence']}", f"- by category: " + ", ".join(f"{c}: {_rate(r)}" for c, r in v["by_category"].items()), ""]
    return "\n".join(lines) + "\n"


def main():
    run = Path(sys.argv[1])
    s = build(run)
    print(markdown(s))


if __name__ == "__main__":
    main()
