"""Experiment runner: sends dataset questions through the real CouncilAI API.

Modes
-----
offline  The backend must report llm.mode == "mock" in /health (checked
         BEFORE any query). Retrieval, routing, caching, council workflow
         and response schemas are real; answers are mock text. Results
         validate the workflow and metric code only and are labelled so.
real     Uses the configured provider models. Refuses to start unless ALL of:
           --authorize-paid-calls is given,
           /health reports llm.mode == "live",
           every configured model has a VERIFIED price in the pricing file,
           --max-requests and --max-spend-usd are set (> 0).
         A BudgetGuard stops the run before any request that could exceed
         either limit. Command-line flags are a technical safeguard; they do
         not replace the project owner's explicit approval to spend money.

Fairness controls
-----------------
* Every (item, strategy) pair is sent once per repeat by a FRESH user, so
  no answer can come from an earlier request's cache; responses with a
  cache hit are flagged anyway.
* The same document and top_k are used for every strategy, and the
  retrieved chunks are recorded so evidence identity can be verified.
* Strategy order is rotated per item (seeded) to avoid order effects.

Results go to results/runs/<experiment_id>/records.jsonl, one line per
request, appended immediately (a checkpoint): rerun with --resume <dir> to
continue after an interruption without repeating finished requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from . import DATASET_VERSION
from .client import APIError, Client, redact
from .cost import BudgetExceeded, BudgetGuard, load_pricing, require_verified, worst_case_request
from .schema import load_dataset

STRATEGIES = ("direct", "council", "council_deep", "auto")


class ModeError(RuntimeError):
    pass


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("name", "mode", "api_base", "dataset", "split", "strategies"):
        if key not in cfg:
            raise ValueError(f"config missing {key}")
    if cfg["mode"] not in ("offline", "real"):
        raise ValueError("mode must be offline or real")
    bad = [s for s in cfg["strategies"] if s not in STRATEGIES]
    if bad:
        raise ValueError(f"unknown strategies {bad}")
    cfg.setdefault("top_k", 4)
    cfg.setdefault("repeats", 1)
    cfg.setdefault("limit", None)
    cfg.setdefault("seed", 7)
    cfg.setdefault("retries_5xx", 1 if cfg["mode"] == "offline" else 0)
    cfg.setdefault("metrics_url", None)
    cfg.setdefault("max_output_tokens_estimate", 1024)
    return cfg


def check_mode(health: dict, mode: str, authorized: bool) -> dict:
    """Decide from /health alone (no model call) whether the run may start."""
    llm = health.get("llm") or {}
    backend = llm.get("mode")
    if mode == "offline" and backend != "mock":
        raise ModeError(f"offline mode needs a mock backend, but /health reports llm.mode={backend!r}; refusing to send queries")
    if mode == "real":
        if not authorized:
            raise ModeError("real-model mode requires --authorize-paid-calls (and the project owner's explicit approval)")
        if backend != "live":
            raise ModeError(f"real-model mode needs a live backend, but /health reports llm.mode={backend!r}")
    return llm


def configured_models(llm: dict) -> list[str]:
    return list(llm.get("members", [])) + [m for m in (llm.get("chairman"), llm.get("router")) if m]


def plan(items, strategies, repeats: int, seed: int) -> list[tuple]:
    """(item, strategy, repeat) jobs; strategy order rotated per item."""
    jobs = []
    rng = random.Random(seed)
    for rep in range(repeats):
        for it in items:
            order = list(strategies)
            rng.shuffle(order)
            jobs += [(it, s, rep) for s in order]
    return jobs


def job_key(item_id: str, strategy: str, rep: int) -> str:
    return f"{item_id}|{strategy}|{rep}"


def done_keys(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    out = set()
    for line in records_path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted write is redone
            out.add(job_key(r["item_id"], r["strategy"], r["repeat"]))
    return out


def experiment_id(cfg: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{cfg['name']}-{stamp}-{(os.environ.get('GIT_COMMIT') or 'nogit')[:8]}"


class Interrupted(Exception):
    pass


def run(cfg: dict, out_root: Path, resume: Optional[Path] = None, authorize_paid: bool = False,
        max_requests: Optional[int] = None, max_spend_usd: Optional[float] = None, pricing_path: Optional[Path] = None,
        client: Optional[Client] = None, stop_after: Optional[int] = None) -> Path:
    client = client or Client(cfg["api_base"])
    llm = check_mode(client.health(), cfg["mode"], authorize_paid)

    guard = None
    if cfg["mode"] == "real":
        if not max_requests or max_requests < 1 or max_spend_usd is None or max_spend_usd <= 0:
            raise ModeError("real-model mode needs --max-requests >= 1 and --max-spend-usd > 0")
        pricing = load_pricing(pricing_path or Path("config/pricing.yaml"))
        models = configured_models(llm)
        require_verified(pricing, models)
        guard = BudgetGuard(max_requests, max_spend_usd,
                            max(pricing[m].input_usd_per_mtok for m in models),
                            max(pricing[m].output_usd_per_mtok for m in models))

    items, docs = load_dataset(Path(cfg["dataset"]))
    items = sorted((it for it in items if cfg["split"] == "all" or it.split == cfg["split"]), key=lambda i: i.id)
    if cfg.get("categories"):
        items = [it for it in items if it.category in cfg["categories"]]
    if cfg.get("per_category"):
        # Seeded stratified sample: the same items every run.
        rng = random.Random(f"{cfg['seed']}:sample")
        picked = []
        for cat in sorted({it.category for it in items}):
            pool = [it for it in items if it.category == cat]
            picked += rng.sample(pool, min(int(cfg["per_category"]), len(pool)))
        items = sorted(picked, key=lambda i: i.id)
    if cfg["limit"]:
        items = items[: int(cfg["limit"])]

    run_dir = resume or out_root / experiment_id(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        meta = {
            "experiment_id": run_dir.name, "config": cfg, "dataset_version": DATASET_VERSION,
            "git_commit": os.environ.get("GIT_COMMIT", "unknown"), "started_utc": datetime.now(timezone.utc).isoformat(),
            "external_model_calls_enabled": cfg["mode"] == "real", "backend_llm": llm,
            "items": len(items), "item_ids_sha256": hashlib.sha256("\n".join(i.id for i in items).encode()).hexdigest(),
            "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                        "host": os.environ.get("HOST_DESCRIPTION", "unknown")},
            "budget": guard.state() if guard else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    if cfg.get("metrics_url"):
        txt = client.metrics_text(cfg["metrics_url"])
        if txt is not None and not (run_dir / "metrics_before.prom").exists():
            (run_dir / "metrics_before.prom").write_text(txt)

    finished = done_keys(records_path)
    jobs = [j for j in plan(items, cfg["strategies"], int(cfg["repeats"]), int(cfg["seed"])) if job_key(j[0].id, j[1], j[2]) not in finished]

    stop = {"flag": False}

    def on_signal(signum, frame):  # finish the current request, then stop cleanly
        stop["flag"] = True

    old = signal.signal(signal.SIGINT, on_signal)
    users: dict[int, tuple[Client, dict]] = {}
    sent = 0
    status = "completed"
    try:
        for it, strategy, rep in jobs:
            if stop["flag"] or (stop_after is not None and sent >= stop_after):
                status = "interrupted"
                break
            if rep not in users:
                c = client if not users else client.fresh()
                c.register()
                users[rep] = (c, {})
            c, docmap = users[rep]
            if it.doc_id not in docmap:
                info = c.ingest(f"{it.doc_id}.md", docs[it.doc_id].text.encode())
                docmap[it.doc_id] = info["doc_id"]
            body = {"question": it.question, "doc_id": docmap[it.doc_id], "top_k": cfg["top_k"]}
            if strategy != "auto":
                body["strategy"] = strategy
            est = worst_case_request(strategy, len(it.question), cfg["top_k"] * 1500, cfg["max_output_tokens_estimate"],
                                     routed=strategy == "auto")
            if guard:
                guard.check(est)  # raises BudgetExceeded before spending
            attempts, out = 0, None
            while True:
                out = c.query(body)
                if out.status < 500 or attempts >= cfg["retries_5xx"]:
                    break
                attempts += 1
            rec = {"key": job_key(it.id, strategy, rep), "item_id": it.id, "strategy": strategy, "repeat": rep,
                   "category": it.category, "status": out.status, "latency_ms": round(out.latency_ms, 2),
                   "trace_id": out.trace_id, "error": out.error, "retries_5xx": attempts, "waits_429": out.retries_429,
                   "response": strip_response(out.body), "utc": datetime.now(timezone.utc).isoformat()}
            if guard:
                guard.record(est, reported_upper_cost(out.body, guard))
                rec["budget"] = guard.state()
            with open(records_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sent += 1
    except BudgetExceeded as exc:
        status = f"stopped: {exc}"
    finally:
        signal.signal(signal.SIGINT, old)
        if cfg.get("metrics_url"):
            txt = client.metrics_text(cfg["metrics_url"])
            if txt is not None:
                (run_dir / "metrics_after.prom").write_text(txt)
        state = {"status": status, "sent_this_session": sent, "remaining": len(jobs) - sent,
                 "finished_utc": datetime.now(timezone.utc).isoformat(), "budget": guard.state() if guard else None}
        (run_dir / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")
    return run_dir


def reported_upper_cost(body: Optional[dict], guard: BudgetGuard) -> Optional[float]:
    """Upper bound of a finished request's cost from REPORTED usage (all
    tokens priced at the most expensive configured model). None, so the
    guard charges the worst-case estimate, when the request failed or any
    model call lacked reported usage (fail closed)."""
    if not body or not body.get("token_usage"):
        return None
    calls = [c for c in body.get("candidates") or []] + [r for r in body.get("reviews") or []]
    if any(not (c.get("usage") or {}).get("total_tokens") for c in calls if not c.get("error")):
        return None
    u = body["token_usage"]
    return (u.get("prompt_tokens", 0) / 1e6 * guard.price_in) + (u.get("completion_tokens", 0) / 1e6 * guard.price_out)


def strip_response(body: Optional[dict]) -> Optional[dict]:
    """Keep what the evaluation needs; drop session details."""
    if body is None:
        return None
    body = dict(body)
    body.pop("session", None)
    return body


def main():
    ap = argparse.ArgumentParser(description="Run an evaluation experiment against the CouncilAI API")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/runs"))
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--authorize-paid-calls", action="store_true",
                    help="required for mode: real. A technical guard only: real runs also need the owner's explicit approval.")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-spend-usd", type=float)
    ap.add_argument("--pricing", type=Path, default=Path("config/pricing.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        d = run(cfg, a.out, a.resume, a.authorize_paid_calls, a.max_requests, a.max_spend_usd, a.pricing)
    except (ModeError, APIError) as exc:
        raise SystemExit(f"refused: {redact(str(exc))}")
    print(json.dumps(json.loads((d / "run_state.json").read_text()), indent=2))
    print(f"results: {d}")


if __name__ == "__main__":
    main()
