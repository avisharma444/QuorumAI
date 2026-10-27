"""End-to-end QuorumAI requests with a RunPod-hosted vLLM council member.

Runs through the real Go backend (auth, retrieval, council, chairman), with
council members 1-2 on hosted APIs and member 3 = Phi-4-mini on RunPod
Serverless. Explicit `council` strategy (no router), unique questions (no L1
hits), L2 disabled. Phases:

  council   N document-grounded questions: end-to-end latency, per-member
            candidate and review latency, token usage, degraded stages
  fallback  run while the backend has a WRONG RunPod key: the council must
            still answer with the two hosted members and report the failure
  cancel    the client gives up mid-council; RunPod /health is sampled to see
            whether the remote job stops

QuorumAI's SSE candidate events are per-candidate, not per-token, so no TTFT
is derived here (see gpu_bench for token-level streaming).

GPU use is charged to the shared campaign ledger (gpu_budget.py) as session
e2e_<phase>: the warm-up job counts 1 request and every council counts 2
(Phi's candidate and its review; Phi is not asked to review when its
candidate failed, but the full 2 are counted). A council is refused if it
could still be running at the session's wall cap. The council and cancel
phases first wake the worker with an async /run job, because RunPod's
synchronous routes stop waiting after about 90 s.

  python -m councileval.e2e_vllm --config config/experiments/runpod_e2e.yaml --phase council \\
      --gpu-campaign config/experiments/runpod_campaign.yaml --authorize-paid-calls --max-requests 8 --max-spend-usd 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml

from .client import APIError, Client, redact
from .cost import BudgetExceeded, BudgetGuard, load_pricing, require_verified, worst_case_request
from .gpu_bench import (ENDPOINT_ENV, INFERENCE_KEY_ENV, READONLY_KEY_ENV, Abort, AllocationProbe, Endpoint, SpendMonitor,
                        load_config as load_bench_config, quiet_check, serverless_placement, workers_up)
from .gpu_budget import GpuBudgetExceeded, GpuSession, load_campaign, session_worst_case_usd
from .scoring import is_correct
from .cache_bench import select_questions

PHI = "local:microsoft/Phi-4-mini-instruct"
PHI_CALLS_PER_COUNCIL = 2  # candidate + review (strategy "council": no reflection stage)


def check_identity(llm: dict, cfg: dict) -> None:
    want = cfg["expected_models"]
    if llm.get("mode") != "live" or list(llm.get("members") or []) != want["members"] or llm.get("chairman") != want["chairman"]:
        raise SystemExit(f"backend models {llm} are not the approved {want}; refusing")


def summarise_response(body: Optional[dict]) -> dict:
    if not body:
        return {}
    cands = [{k: c.get(k) for k in ("model", "served_model", "latency_ms", "usage", "error")} for c in body.get("candidates") or []]
    failed = [{k: c.get(k) for k in ("model", "latency_ms", "error")} for c in body.get("failed_candidates") or []]
    revs = [{k: r.get(k) for k in ("reviewer", "served_model", "latency_ms", "usage", "error")} for r in body.get("reviews") or []]
    return {"candidates": cands, "failed_candidates": failed, "reviews": revs, "degraded": body.get("degraded") or [], "synthesis": body.get("synthesis"),
            "token_usage": body.get("token_usage"), "source": body.get("source"), "answer": body.get("answer"),
            "duration_ms": body.get("duration_ms"), "cache": body.get("cache")}


def run(cfg: dict, phase: str, out_root: Path, authorize_paid: bool, max_requests: int, max_spend_usd: float,
        pricing_path: Path = Path("config/pricing.yaml"), client: Optional[Client] = None, ep: Optional[Endpoint] = None,
        budget: Optional[GpuSession] = None, sleep=time.sleep, monitor: Optional[SpendMonitor] = None,
        placement=None, quiet=None) -> Path:
    if not authorize_paid:
        raise SystemExit("refusing: needs --authorize-paid-calls")
    if budget is None or budget.name != f"e2e_{phase}":
        raise SystemExit(f"refusing: needs the GPU campaign budget (session e2e_{phase})")
    if phase in ("council", "cancel") and ep is None:
        raise SystemExit(f"refusing: the {phase} phase needs {ENDPOINT_ENV} and {INFERENCE_KEY_ENV} for the warm-up job")
    client = client or Client(cfg["api_base"], timeout=cfg.get("client_timeout_s", 180))
    check_identity(client.health().get("llm") or {}, cfg)
    pricing = load_pricing(pricing_path)
    hosted = [m for m in cfg["expected_models"]["members"] if not m.startswith("local:")] + [cfg["expected_models"]["chairman"]]
    require_verified(pricing, hosted)
    # Hosted-API spend guard (GPU time is bounded separately by max workers 1 and the idle timeout).
    guard = BudgetGuard(max_requests, max_spend_usd, max(pricing[m].input_usd_per_mtok for m in hosted),
                        max(pricing[m].output_usd_per_mtok for m in hosted))
    order, _, docs = select_questions(cfg)
    cancel_item = None
    if phase == "council" and cfg.get("cancel_after_councils"):
        # The cancellation test reuses the warm worker in the same session:
        # one more (unique) question, given up after cancel_after_s.
        cancel_item = order[cfg["questions"]]
    order = order[: cfg["questions"]] if phase == "council" else order[:1]
    run_dir = out_root / f"{cfg['name']}-{phase}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"

    def write(rec):
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (run_dir / "metadata.json").write_text(json.dumps({"config": cfg, "phase": phase, "backend_llm": client.health().get("llm"),
                                                       "questions": [it.id for it in order], "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    client.register()
    status = "completed"
    per_council_s = cfg.get("client_timeout_s", 180)
    try:
        budget.open()
    except GpuBudgetExceeded as exc:
        raise SystemExit(f"refused by the GPU campaign budget: {exc}")
    try:
        if monitor is not None:
            monitor.out = monitor.out or (lambda rec: write({"phase": "spend_monitor", **rec}))
            monitor.start()
        if phase in ("council", "cancel"):
            # Wake the GPU worker first so the measured councils are not cold
            # starts (a cold Phi call would just hit the stage timeout). Async
            # job: the synchronous route stops waiting after ~90 s. Recorded.
            alloc_s, init_s = cfg["warmup_allocation_timeout_s"], cfg["warmup_init_timeout_s"]
            if placement is not None:  # fail closed: never submit a job RunPod cannot place
                pl = placement()
                write({"phase": "placement", **pl})
                if not pl.get("ok"):
                    raise GpuBudgetExceeded(f"no Serverless host matches the endpoint's GPU/CUDA filters: {pl.get('error') or pl.get('gpus')}")
            budget.take(1, "warm-up job", alloc_s + init_s)
            w = ep.job({"model": cfg["served_model"], "messages": [{"role": "user", "content": "Reply with OK."}],
                        "max_tokens": 8, "temperature": 0}, "warmup", 0, "warmup", timeout_s=init_s, poll_s=2, sleep=sleep,
                       on_poll=(monitor.check if monitor is not None else None),
                       alloc_timeout_s=alloc_s, allocated=AllocationProbe(ep))
            write({"phase": "warmup", "final_status": w.final_status, "client_total_s": w.client_total_s,
                   "allocated_s": w.allocated_s, "served_model": w.served_model, "completion_tokens": w.completion_tokens,
                   "worker_id": w.worker_id, "delay_ms": w.delay_ms,
                   "execution_ms": w.execution_ms, "error_class": w.error_class, "error": w.error, "status_trace": w.status_trace,
                   "utc": datetime.now(timezone.utc).isoformat()})
            if w.error_class:
                raise GpuBudgetExceeded(f"warm-up failed: {w.error_class}; not running councils")
        plan_items = [(it, phase) for it in order] + ([(cancel_item, "cancel")] if cancel_item is not None else [])
        for it, mode in plan_items:
            info = client.ingest(f"{it.doc_id}.md", docs[it.doc_id].text.encode())
            body = {"question": it.question, "doc_id": info["doc_id"], "top_k": cfg["top_k"], "strategy": "council"}
            est = worst_case_request("council", len(it.question), cfg["top_k"] * 1500, cfg["max_output_tokens_estimate"], routed=False)
            guard.check(est)
            if monitor is not None:
                monitor.check(force=True)  # provider-side spend and retries before every council
            budget.take(PHI_CALLS_PER_COUNCIL, f"council {it.id}", per_council_s)
            if mode == "cancel":
                c2 = Client(cfg["api_base"], timeout=cfg["cancel_after_s"])
                c2.token = client.token
                out = c2.query(body)
                guard.record(est, None)  # unknown how far it got: charge the worst case
                samples = []
                if ep is not None:
                    for _ in range(cfg.get("cancel_observe_samples", 10)):
                        try:
                            h = ep.health()
                            samples.append({"utc": datetime.now(timezone.utc).isoformat(), "jobs": h.get("jobs"), "workers": h.get("workers")})
                        except requests.RequestException as exc:
                            samples.append({"error": redact(str(exc))[:200]})
                        sleep(2)
                # Anything still queued or running for the abandoned council is purged, not left to bill.
                last = next((x for x in reversed(samples) if "jobs" in x), None)
                left = ((last or {}).get("jobs") or {})
                purge = ep.purge_queue() if (left.get("inQueue") or left.get("inProgress")) else None
                write({"phase": "cancel", "item_id": it.id, "status": out.status, "latency_ms": out.latency_ms, "error": out.error,
                       "runpod_health_after_cancel": samples, "purge_queue_status": purge,
                       "utc": datetime.now(timezone.utc).isoformat()})
                continue
            out = client.query(body)
            usage = (out.body or {}).get("token_usage") or {}
            guard.record(est, usage.get("cost_credits") if usage.get("cost_credits") else None)
            s = summarise_response(out.body)
            write({"phase": phase, "item_id": it.id, "category": it.category, "status": out.status, "latency_ms": out.latency_ms,
                   "trace_id": out.trace_id, "error": out.error, "correct": is_correct(it, s.get("answer")) if s else None,
                   **s, "budget": guard.state(), "utc": datetime.now(timezone.utc).isoformat()})
    except (BudgetExceeded, GpuBudgetExceeded, Abort) as exc:
        status = f"stopped: {exc}"
    except APIError as exc:
        status = f"stopped: {redact(str(exc))}"
    finally:
        zero, end_kind = None, None
        if ep is not None:
            # No running or billing worker may remain: wait (bounded) for
            # scale-down. A stopped, unbilled standby worker is allowed.
            quiet = quiet or (lambda: {"quiet": workers_up(ep.health()) == 0, "kind": "zero"})
            deadline = time.perf_counter() + cfg.get("scale_down_wait_s", 240)
            while time.perf_counter() < deadline:
                try:
                    q = quiet()
                    end_kind = q["kind"]
                    if q["quiet"]:
                        zero = True
                        break
                except requests.RequestException:
                    pass
                zero = False
                sleep(5)
        if monitor is not None and monitor.start_balance is not None:
            try:
                monitor.check(force=True)
            except Abort as exc:
                status = status if status != "completed" else f"completed; monitor: {exc}"
        gpu_state = budget.close(status)
        (run_dir / "run_state.json").write_text(json.dumps({"status": status, "budget": guard.state(), "gpu_session": gpu_state,
                                                            "scaled_to_zero": zero, "end_condition": end_kind,
                                                            "spend_monitor": monitor.last if monitor is not None else None,
                                                            "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    return run_dir


def analyse(run_dir: Path) -> dict:
    from .gpu_bench import dist
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r.get("phase") in ("council", "fallback", "cancel")]  # not warm-up, placement or monitor records
    ok = [r for r in recs if r.get("status") == 200]
    by_model: dict[str, dict[str, list]] = {}
    for r in ok:
        for c in r.get("candidates") or []:
            by_model.setdefault(c["model"], {"candidate_ms": [], "review_ms": [], "candidate_out_tokens": []})
            by_model[c["model"]]["candidate_ms"].append(c.get("latency_ms"))
            by_model[c["model"]]["candidate_out_tokens"].append((c.get("usage") or {}).get("completion_tokens"))
        for v in r.get("reviews") or []:
            by_model.setdefault(v["reviewer"], {"candidate_ms": [], "review_ms": [], "candidate_out_tokens": []})
            by_model[v["reviewer"]]["review_ms"].append(v.get("latency_ms"))
    slowest = [max(r["candidates"], key=lambda c: c.get("latency_ms") or 0)["model"] for r in ok if r.get("candidates")]
    phi_missing = [r["item_id"] for r in recs if r.get("status") == 200 and PHI not in [c["model"] for c in r.get("candidates") or []]]
    phi_review_errors = [v.get("error") for r in ok for v in r.get("reviews") or [] if v["reviewer"] == PHI and v.get("error")]
    phi_failures = [c.get("error") for r in ok for c in r.get("failed_candidates") or [] if c.get("model") == PHI]
    # A failed member must not be asked to review: Phi reviews only where its candidate succeeded.
    phi_reviewed_after_failure = [r["item_id"] for r in ok if PHI in [c.get("model") for c in r.get("failed_candidates") or []]
                                  and PHI in [v.get("reviewer") for v in r.get("reviews") or []]]
    out = {"requests": len(recs), "ok": len(ok), "end_to_end_ms": dist([r["latency_ms"] for r in ok]),
           "server_duration_ms": dist([r.get("duration_ms") for r in ok]),
           "per_model": {m: {"candidate_ms": dist(v["candidate_ms"]), "review_ms": dist(v["review_ms"]),
                             "candidate_output_tokens": dist(v["candidate_out_tokens"])} for m, v in by_model.items()},
           "slowest_candidate_counts": {m: slowest.count(m) for m in set(slowest)},
           "phi_candidate_missing": phi_missing, "phi_candidate_errors": phi_failures, "phi_review_errors": phi_review_errors,
           "phi_reviewed_after_failure": phi_reviewed_after_failure,
           "degraded": {r["item_id"]: r["degraded"] for r in ok if r.get("degraded")},
           "correct": sum(1 for r in ok if r.get("correct")), "hosted_cost_credits": sum(((r.get("token_usage") or {}).get("cost_credits") or 0) for r in ok),
           "cancel": [r for r in recs if r.get("phase") == "cancel"],
           "run_state": json.loads((run_dir / "run_state.json").read_text())}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--phase", choices=["council", "fallback", "cancel"])
    ap.add_argument("--out", type=Path, default=Path("results/e2e_vllm"))
    ap.add_argument("--authorize-paid-calls", action="store_true")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-spend-usd", type=float)
    ap.add_argument("--gpu-campaign", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    if a.report:
        print(json.dumps(analyse(a.report), indent=2))
        return
    cfg = yaml.safe_load(a.config.read_text())
    if a.max_requests is None or a.max_spend_usd is None or a.gpu_campaign is None:
        raise SystemExit("--max-requests, --max-spend-usd and --gpu-campaign are required")
    ep = None
    if os.environ.get(INFERENCE_KEY_ENV) and os.environ.get(ENDPOINT_ENV):
        ep = Endpoint(os.environ[ENDPOINT_ENV], os.environ[INFERENCE_KEY_ENV])
    campaign = load_campaign(a.gpu_campaign)
    budget = GpuSession(campaign, f"e2e_{a.phase}")
    ro = os.environ.get(READONLY_KEY_ENV) or ""
    if not ro or ep is None:
        raise SystemExit(f"{READONLY_KEY_ENV}, {ENDPOINT_ENV} and {INFERENCE_KEY_ENV} are required: live spend monitoring is mandatory")
    monitor = SpendMonitor(ro, session_worst_case_usd(campaign, budget.max_wall_s), ep.health)
    bench_cfg = load_bench_config(Path(cfg.get("endpoint_config", "config/experiments/runpod_phi4_bench.yaml")))
    d = run(cfg, a.phase, a.out, a.authorize_paid_calls, a.max_requests, a.max_spend_usd, ep=ep, budget=budget, monitor=monitor,
            placement=lambda: serverless_placement(bench_cfg, ro), quiet=quiet_check(ep, ro))
    print(f"results: {d}")
    print(json.dumps(analyse(d), indent=2)[:4000])


if __name__ == "__main__":
    main()
