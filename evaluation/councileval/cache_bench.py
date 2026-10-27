"""Paired real-model benchmark: cold council vs warm L1 exact cache vs direct.

For each of N seeded, document-grounded questions (same user, document,
top_k and pinned models):

  A  cold    strategy=council, must be uncached (cache_hit false, inference
             executed) and must have run candidates, reviews and chairman;
  B  warm    the identical request repeated `warm_repeats` times, only if the
             cold answer was stored in L1; every repeat must be an l1_exact
             hit with inference_executed=false, and the Prometheus model-call
             counter must not move. Any warm request that runs inference is
             marked failed and no further repeats are sent for that question;
  C  direct  strategy=direct (a different cache key, so it is uncached);
  D  deep    strategy=council_deep for the predetermined `deep_questions`.

L2 semantic reuse must be disabled (checked on every response).

Safety (mode=real): /health must report llm.mode == "live" AND exactly the
pinned `expected_models` (checked before every question; a change aborts the
run); --authorize-paid-calls, --max-requests and --max-spend-usd are
required; every model needs a verified price. A BudgetGuard refuses any
request whose worst case could exceed the caps, charging each finished
request its provider-reported cost when every call reported one, otherwise
its worst-case estimate (fail closed). Ingestion (one summarizer call per
document) is charged at its worst case. mode=offline requires a mock backend
and validates the procedure without any provider call.

Records are appended to records.jsonl as a checkpoint; --resume continues.

  python -m councileval.cache_bench --config config/experiments/real_cache_bench.yaml --dry-run
  python -m councileval.cache_bench --config ... [--authorize-paid-calls --max-requests N --max-spend-usd X]
  python -m councileval.cache_bench --report results/cache_bench/<id>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .client import APIError, Client, redact
from .cost import BudgetExceeded, BudgetGuard, RequestEstimate, estimate_tokens, load_pricing, require_verified, worst_case_request
from .metrics import exact_match, token_f1
from .runner import ModeError, check_mode
from .schema import load_dataset
from .scoring import is_correct, retrieval_status, taxonomy

PHASES = ("cold", "warm", "direct", "deep")


class IdentityError(RuntimeError):
    pass


class WarmInferenceError(RuntimeError):
    pass


# ── configuration and question selection ─────────────────────────────

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("name", "mode", "api_base", "dataset", "split", "per_category", "warm_repeats", "expected_models"):
        if k not in cfg:
            raise ValueError(f"config missing {k}")
    if cfg["mode"] not in ("offline", "real"):
        raise ValueError("mode must be offline or real")
    cfg.setdefault("top_k", 4)
    cfg.setdefault("seed", 11)
    cfg.setdefault("deep_questions", 4)
    cfg.setdefault("max_output_tokens_estimate", 2048)
    cfg.setdefault("metrics_url", None)
    cfg.setdefault("jaeger_url", None)
    cfg.setdefault("likely", {})
    return cfg


def select_questions(cfg: dict):
    items, docs = load_dataset(Path(cfg["dataset"]))
    pool = sorted((it for it in items if it.split == cfg["split"] and it.answerable), key=lambda i: i.id)
    rng = random.Random(f"{cfg['seed']}:cache-bench")
    picked = []
    for cat, n in sorted(cfg["per_category"].items()):
        cand = [it for it in pool if it.category == cat]
        if len(cand) < n:
            raise ValueError(f"only {len(cand)} {cat} items available")
        picked += rng.sample(cand, n)
    order = list(picked)
    rng.shuffle(order)  # interleave categories
    deep = set()
    for cat in sorted(cfg["per_category"]):  # predetermined: first item of each category, until deep_questions
        if len(deep) >= cfg["deep_questions"]:
            break
        deep.add(next(it.id for it in order if it.category == cat))
    return order, deep, docs


# ── identity, metrics and verification helpers ──────────────────────

def check_identity(llm: dict, expected: dict) -> None:
    got = {"members": list(llm.get("members", [])), "chairman": llm.get("chairman")}
    want = {"members": list(expected.get("members", [])), "chairman": expected.get("chairman")}
    if got != want:
        raise IdentityError(f"backend models {got} differ from the pinned {want}; refusing to continue")


def served_ok(requested: str, served: Optional[str]) -> bool:
    """A provider may append a version/date suffix to the requested id."""
    if not served:
        return True
    r = requested.split(":", 1)[-1] if requested.startswith(("gemini:", "nim:", "local:")) else requested
    return served == r or served.startswith(r) or served.startswith(r.split("/")[-1])


def llm_calls_total(prom_text: Optional[str]) -> Optional[float]:
    if prom_text is None:
        return None
    return sum(float(l.rsplit(" ", 1)[1]) for l in prom_text.splitlines() if l.startswith("councilai_llm_request_seconds_count"))


def classify(resp: Optional[dict]) -> dict:
    """Cache/inference classification of one /query response."""
    if not resp:
        return {"valid": False, "tier": None, "inference": None, "stored_l1": False, "degraded": None, "l2": None}
    c = resp.get("cache") or {}
    return {"valid": True, "tier": c.get("tier"), "inference": c.get("inference_executed"),
            "stored_l1": "l1_exact" in (c.get("stored") or []), "degraded": bool(resp.get("degraded")),
            "l2": c.get("l2"), "cache_hit": resp.get("cache_hit")}


def council_ran(resp: dict) -> bool:
    cands = resp.get("candidates") or []
    return (len(cands) >= 2 and not any(c.get("error") for c in cands) and len(resp.get("reviews") or []) >= 1
            and (resp.get("synthesis") or {}).get("status") == "synthesized")


def model_calls(resp: Optional[dict]) -> int:
    if not resp or not (resp.get("cache") or {}).get("inference_executed"):
        return 0
    if resp.get("strategy") == "direct":
        return 1
    refl = resp.get("reflection") or {}
    return (len(resp.get("candidates") or []) + len(resp.get("reviews") or []) + (1 if resp.get("synthesis") else 0)
            + (1 if refl else 0) + (1 if refl.get("revision") else 0))


def reported_cost(resp: Optional[dict]) -> Optional[float]:
    """Provider-reported cost of a response, only if every model call reported usage."""
    if not resp:
        return None
    u = resp.get("token_usage") or {}
    calls = (resp.get("candidates") or []) + (resp.get("reviews") or [])
    if not u or any(not (c.get("usage") or {}).get("total_tokens") for c in calls):
        return None
    if model_calls(resp) and not u.get("cost_credits"):
        return None
    return float(u.get("cost_credits") or 0.0)


# ── estimates (dry run and budget) ───────────────────────────────────

@dataclass
class Prices:
    worst_in: float
    worst_out: float
    likely: dict  # model -> (in, out)


def request_estimate(kind: str, question_chars: int, cfg: dict) -> RequestEstimate:
    ev = cfg["top_k"] * 1500
    strategy = {"cold": "council", "warm": "council", "direct": "direct", "deep": "council_deep"}.get(kind)
    if kind == "ingest":
        return RequestEstimate(1, 1200 + estimate_tokens(3 * 1500), 200)  # summarizer: BME preview, max 200 output tokens
    return worst_case_request(strategy, question_chars, ev, cfg["max_output_tokens_estimate"], routed=False)


def likely_cost(cfg: dict, prices: dict, question_chars: int, evidence_chars: int) -> dict:
    """Likely (not worst) cost per request kind with stated output-length
    assumptions (cfg['likely']) and each role's primary price."""
    L = {"candidate_out": 350, "review_out": 250, "chairman_out": 450, "direct_out": 350, "reflection_out": 200,
         "revision_out": 450, "summary_in": 900, "summary_out": 150, "chars_per_token": 4, "system_tokens": 350}
    L.update(cfg.get("likely") or {})
    base = (question_chars + evidence_chars) / L["chars_per_token"] + L["system_tokens"]
    m = cfg["expected_models"]

    def c(model, tin, tout):
        pin, pout = prices[model]
        return tin / 1e6 * pin + tout / 1e6 * pout

    cand = sum(c(x, base, L["candidate_out"]) for x in m["members"])
    review_in = base + 3 * L["candidate_out"]
    rev = sum(c(x, review_in, L["review_out"]) for x in m["members"])
    chair = c(m["chairman"], base + 3 * L["candidate_out"] + 3 * L["review_out"], L["chairman_out"])
    council = cand + rev + chair
    direct = c(m["chairman"], base, L["direct_out"])
    deep = council + c(m["chairman"], base + L["chairman_out"], L["reflection_out"]) + c(m["chairman"], base + L["chairman_out"] + L["reflection_out"], L["revision_out"])
    ingest = c(m.get("ingestion", m["chairman"]), L["summary_in"], L["summary_out"])
    return {"council": council, "direct": direct, "deep_with_revision": deep, "ingest": ingest, "assumptions": L}


def dry_run(cfg: dict, pricing_path: Path, evidence_chars_per_question: Optional[dict] = None) -> dict:
    order, deep, docs = select_questions(cfg)
    pricing = load_pricing(pricing_path)
    m = cfg["expected_models"]
    models = sorted(set(m["members"]) | {m["chairman"], m.get("ingestion", m["chairman"])})
    missing = [x for x in models if x not in pricing or not pricing[x].verified]
    worst_in = max((pricing[x].input_usd_per_mtok for x in models if x in pricing), default=0.0)
    worst_out = max((pricing[x].output_usd_per_mtok for x in models if x in pricing), default=0.0)
    likely_prices = {x: tuple(cfg.get("likely_prices", {}).get(x, (pricing[x].input_usd_per_mtok, pricing[x].output_usd_per_mtok)))
                     for x in models if x in pricing}
    n, w = len(order), cfg["warm_repeats"]
    ndocs = len({it.doc_id for it in order})
    calls = {"ingest_summarizer": ndocs, "candidates": 3 * n + 3 * len(deep), "reviews": 3 * n + 3 * len(deep),
             "chairman_synthesis": n + len(deep), "direct": n, "reflection": len(deep), "revision_max": len(deep), "warm_expected": 0}
    worst = likely = 0.0
    rows = []
    for it in order:
        q = len(it.question)
        evc = (evidence_chars_per_question or {}).get(it.id, cfg["top_k"] * 1000)
        lk = likely_cost(cfg, likely_prices, q, evc) if len(likely_prices) == len(models) else None
        kinds = ["cold", "direct"] + (["deep"] if it.id in deep else [])
        wc = sum(request_estimate(k, q, cfg).cost(worst_in, worst_out) for k in kinds)
        lc = (lk["council"] + lk["direct"] + (lk["deep_with_revision"] if it.id in deep else 0)) if lk else None
        worst += wc
        likely += lc or 0
        rows.append({"id": it.id, "category": it.category, "doc_id": it.doc_id, "question": it.question,
                     "deep": it.id in deep, "worst_usd": round(wc, 4), "likely_usd": round(lc, 5) if lc is not None else None})
    ingest_worst = ndocs * request_estimate("ingest", 0, cfg).cost(worst_in, worst_out)
    ingest_likely = ndocs * (likely_cost(cfg, likely_prices, 0, 0)["ingest"] if len(likely_prices) == len(models) else 0)
    requests = {"ingest": ndocs, "cold_council": n, "warm": n * w, "direct": n, "deep": len(deep)}
    return {
        "questions": rows, "requests": requests, "http_requests_total": sum(requests.values()),
        "billable_requests_cap_suggestion": requests["cold_council"] + requests["direct"] + requests["deep"] + requests["warm"],
        "model_calls_expected": calls, "model_calls_total_max": sum(v for k, v in calls.items() if k != "warm_expected"),
        "models": m, "prices_missing_or_unverified": missing,
        "worst_case_price_usd_per_mtok": {"input": worst_in, "output": worst_out},
        "worst_case_usd": round(worst + ingest_worst, 4), "likely_usd": round(likely + ingest_likely, 4),
        "worst_case_assumes": f"every call priced at the most expensive configured endpoint, {cfg['max_output_tokens_estimate']} output tokens per call, inputs at 3 chars/token",
        "likely_assumes": likely_cost(cfg, likely_prices, 0, 0)["assumptions"] if len(likely_prices) == len(models) else None,
        "warm_requests_note": "expected to cost 0; if a warm request unexpectedly runs inference it is charged at the cold-council worst case and the question's remaining repeats are cancelled",
    }


# ── live / offline execution ──────────────────────────────────────────

def _key(qid, phase, idx=0):
    return f"{qid}|{phase}|{idx}"


def run(cfg: dict, out_root: Path, resume: Optional[Path] = None, authorize_paid: bool = False,
        max_requests: Optional[int] = None, max_spend_usd: Optional[float] = None, pricing_path: Path = Path("config/pricing.yaml"),
        client: Optional[Client] = None) -> Path:
    client = client or Client(cfg["api_base"])
    llm = check_mode(client.health(), cfg["mode"], authorize_paid)
    guard = None
    if cfg["mode"] == "real":
        check_identity(llm, cfg["expected_models"])
        if not max_requests or max_requests < 1 or not max_spend_usd or max_spend_usd <= 0:
            raise ModeError("real mode needs --max-requests >= 1 and --max-spend-usd > 0")
        pricing = load_pricing(pricing_path)
        m = cfg["expected_models"]
        models = sorted(set(m["members"]) | {m["chairman"], m.get("ingestion", m["chairman"])})
        require_verified(pricing, models)
        guard = BudgetGuard(max_requests, max_spend_usd, max(pricing[x].input_usd_per_mtok for x in models),
                            max(pricing[x].output_usd_per_mtok for x in models))
    order, deep, docs = select_questions(cfg)
    run_dir = resume or out_root / f"{cfg['name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    if not (run_dir / "metadata.json").exists():
        (run_dir / "metadata.json").write_text(json.dumps({
            "config": cfg, "git_commit": os.environ.get("GIT_COMMIT", "unknown"), "backend_llm": llm,
            "questions": [it.id for it in order], "deep_questions": sorted(deep),
            "host": os.environ.get("HOST_DESCRIPTION", "unknown"), "started_utc": datetime.now(timezone.utc).isoformat(),
            "external_model_calls_enabled": cfg["mode"] == "real"}, indent=2) + "\n")

    def write(rec):
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def metrics():
        return client.metrics_text(cfg["metrics_url"]) if cfg.get("metrics_url") else None

    resumed = bool(done)
    client.register()  # a resumed run gets a new user (tokens are never written to disk)
    docmap: dict[str, str] = {}
    status = "completed"
    try:
        for it in order:
            if cfg["mode"] == "real":
                check_identity(client.health().get("llm") or {}, cfg["expected_models"])
            if it.doc_id not in docmap:
                if guard:
                    est = request_estimate("ingest", 0, cfg)
                    guard.check(est)
                info = client.ingest(f"{it.doc_id}.md", docs[it.doc_id].text.encode())
                if guard:
                    guard.record(est, None)  # summarizer usage is not returned by /ingest: charge the worst case
                    want = cfg["expected_models"].get("ingestion")
                    if want and info.get("summary_model") and info["summary_model"] != want:
                        raise IdentityError(f"ingestion model {info['summary_model']} is not the pinned {want}")
                docmap[it.doc_id] = info["doc_id"]
            body = {"question": it.question, "doc_id": docmap[it.doc_id], "top_k": cfg["top_k"]}

            def send(phase, idx, strategy):
                est = request_estimate("cold" if phase == "warm" else phase, len(it.question), cfg)
                if guard:
                    guard.check(est)
                before = metrics()
                out = client.query(dict(body, strategy=strategy))
                after = metrics()
                cost = reported_cost(out.body)
                if guard:
                    c = classify(out.body)
                    verified_hit = c["tier"] == "l1_exact" and not c["inference"] and model_calls(out.body) == 0
                    guard.record(est, 0.0 if verified_hit else cost)  # None -> worst case (fail closed)
                b, a = llm_calls_total(before), llm_calls_total(after)
                rec = {"key": _key(it.id, phase, idx), "item_id": it.id, "phase": phase, "idx": idx, "strategy": strategy,
                       "status": out.status, "latency_ms": round(out.latency_ms, 3), "trace_id": out.trace_id, "error": out.error,
                       "classification": classify(out.body), "model_calls": model_calls(out.body),
                       "prometheus_llm_calls_delta": (a - b) if (a is not None and b is not None) else None,
                       "reported_cost_credits": cost, "response": out.body, "utc": datetime.now(timezone.utc).isoformat()}
                if guard:
                    rec["budget"] = guard.state()
                write(rec)
                return rec

            cold_key = _key(it.id, "cold", 0)
            cold = None
            if cold_key not in done:
                cold = send("cold", 0, "council")
            else:
                cold = next(json.loads(l) for l in rec_path.read_text().splitlines() if json.loads(l)["key"] == cold_key)
            cc = cold["classification"]
            eligible = (cold["status"] == 200 and cc["tier"] == "none" and cc["inference"] and cc["stored_l1"]
                        and not cc["degraded"] and council_ran(cold["response"] or {}))
            warm_pending = [i for i in range(cfg["warm_repeats"]) if _key(it.id, "warm", i) not in done]
            if eligible and resumed and cold_key in done and warm_pending:
                # The new user's cache is empty: a "warm" request would run a paid council.
                write({"key": _key(it.id, "warm_skipped", 0), "item_id": it.id, "phase": "warm_skipped", "idx": 0,
                       "reason": "resumed run: cold answer belongs to the previous user's cache"})
                eligible = False
            if eligible:
                for i in range(cfg["warm_repeats"]):
                    if _key(it.id, "warm", i) in done:
                        continue
                    w = send("warm", i, "council")
                    wc = w["classification"]
                    bad = (w["status"] != 200 or wc["tier"] != "l1_exact" or wc["inference"] or w["model_calls"]
                           or (w["prometheus_llm_calls_delta"] or 0) > 0)
                    if bad:
                        write({"key": _key(it.id, "warm_failed", i), "item_id": it.id, "phase": "warm_failed", "idx": i,
                               "reason": "warm request did not come from L1 or executed inference"})
                        break
            elif not resumed and _key(it.id, "not_cacheable", 0) not in done:
                write({"key": _key(it.id, "not_cacheable", 0), "item_id": it.id, "phase": "not_cacheable", "idx": 0,
                       "reason": f"cold answer not eligible for warm repeats: {cc}"})
            if _key(it.id, "direct", 0) not in done:
                send("direct", 0, "direct")
            if it.id in deep and _key(it.id, "deep", 0) not in done:
                send("deep", 0, "council_deep")
    except BudgetExceeded as exc:
        status = f"stopped: {exc}"
    except IdentityError as exc:
        status = f"stopped: {exc}"
    except APIError as exc:
        status = f"stopped: {redact(str(exc))}"
    (run_dir / "run_state.json").write_text(json.dumps({"status": status, "budget": guard.state() if guard else None,
                                                        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    return run_dir


# ── analysis ──────────────────────────────────────────────────────────

def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def speedups(pairs: list[tuple[float, list[float]]]) -> dict:
    """pairs: (cold_ms, [warm_ms...]) per question with >=1 valid warm hit."""
    per_q = [cold / statistics.median(w) for cold, w in pairs if w]
    colds = [c for c, w in pairs if w]
    warms = [x for _, w in pairs for x in w]
    return {"questions": len(per_q), "per_question": per_q,
            "median_paired_speedup": statistics.median(per_q) if per_q else None,
            "min_paired_speedup": min(per_q) if per_q else None, "max_paired_speedup": max(per_q) if per_q else None,
            "ratio_of_medians": (statistics.median(colds) / statistics.median(warms)) if colds and warms else None}


def jaeger_stages(jaeger_url: Optional[str], trace_id: str) -> Optional[dict]:
    if not jaeger_url or not trace_id:
        return None
    try:
        data = json.load(urllib.request.urlopen(f"{jaeger_url.rstrip('/')}/api/traces/{trace_id}", timeout=10))["data"]
    except Exception:
        return None
    if not data:
        return None
    spans = data[0]["spans"]
    out = {}
    for s in spans:
        name = s["operationName"]
        tags = {t["key"]: t["value"] for t in s["tags"]}
        out.setdefault(name, []).append({"ms": s["duration"] / 1000, "tokens_in": tags.get("llm.usage.prompt_tokens"),
                                         "tokens_out": tags.get("llm.usage.completion_tokens"), "result": tags.get("llm.result")})
    return out


def analyse(run_dir: Path, jaeger_url: Optional[str] = None) -> dict:
    meta = json.loads((run_dir / "metadata.json").read_text())
    cfg = meta["config"]
    items, _ = load_dataset(Path(cfg["dataset"]))
    by_id = {i.id: i for i in items}
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines() if l.strip()]
    q = {}
    for r in recs:
        q.setdefault(r["item_id"], {}).setdefault(r["phase"], []).append(r)
    pairs, per_question, quality = [], [], []
    calls_avoided = tokens_avoided_in = tokens_avoided_out = 0
    cost_avoided = 0.0
    cost_avoided_complete = True
    l1_hits = l1_lookups = 0
    for qid in meta["questions"]:
        ph = q.get(qid, {})
        cold = (ph.get("cold") or [None])[0]
        warms = [w for w in ph.get("warm", []) if w["status"] == 200 and w["classification"]["tier"] == "l1_exact"
                 and not w["classification"]["inference"] and not w["model_calls"] and not (w["prometheus_llm_calls_delta"] or 0)]
        n_warm_sent = len(ph.get("warm", []))
        if cold and cold["status"] == 200:
            l1_lookups += 1 + n_warm_sent
            l1_hits += len(warms)
        valid_cold = cold and cold["status"] == 200 and cold["classification"]["stored_l1"] and not cold["classification"]["degraded"]
        if valid_cold and warms:
            pairs.append((cold["latency_ms"], [w["latency_ms"] for w in warms]))
            u = (cold["response"] or {}).get("token_usage") or {}
            calls_avoided += cold["model_calls"] * len(warms)
            tokens_avoided_in += (u.get("prompt_tokens") or 0) * len(warms)
            tokens_avoided_out += (u.get("completion_tokens") or 0) * len(warms)
            if cold["reported_cost_credits"] is None:
                cost_avoided_complete = False
            else:
                cost_avoided += cold["reported_cost_credits"] * len(warms)
        it = by_id[qid]
        row = {"id": qid, "category": it.category, "cold_ms": cold and cold["latency_ms"],
               "warm_ms": [w["latency_ms"] for w in warms], "warm_failed": bool(ph.get("warm_failed")),
               "not_cacheable": bool(ph.get("not_cacheable"))}
        if valid_cold and warms:
            row["paired_speedup"] = cold["latency_ms"] / statistics.median(row["warm_ms"])
        per_question.append(row)
        # Quality: council (cold) vs direct on the same question
        direct = (ph.get("direct") or [None])[0]
        res = {"id": qid, "category": it.category, "answer_type": it.answer_type}
        for name, r in (("council", cold), ("direct", direct), ("deep", (ph.get("deep") or [None])[0])):
            if not r or r["status"] != 200 or not r["response"]:
                res[name] = None
                continue
            ans = r["response"].get("answer", "")
            corr = is_correct(it, ans)
            ret = retrieval_status(it, (r["response"].get("retrieval") or {}).get("chunks") or [])
            res[name] = {"correct": corr, "taxonomy": taxonomy(it, ans, corr, ret), "em": exact_match(ans, it.answers),
                         "f1": token_f1(ans, it.answers), "latency_ms": r["latency_ms"], "model_calls": r["model_calls"],
                         "tokens": (r["response"].get("token_usage") or {}), "cost_credits": r["reported_cost_credits"],
                         "evidence": "|".join(str(c.get("chunk_index")) for c in (r["response"].get("retrieval") or {}).get("chunks") or []),
                         "served_models": [c.get("served_model") for c in r["response"].get("candidates") or []],
                         "reflection": (r["response"].get("reflection") or {}).get("quality"),
                         "revision": ((r["response"].get("reflection") or {}).get("revision") or {}).get("status")}
        if res.get("council") and res.get("direct"):
            c, d = res["council"]["correct"], res["direct"]["correct"]
            res["outcome"] = "tie" if c == d else ("council_win" if c else "direct_win")
            res["same_evidence"] = res["council"]["evidence"] == res["direct"]["evidence"]
        quality.append(res)
    colds = [r["cold_ms"] for r in per_question if r["cold_ms"]]
    warm_all = [x for r in per_question for x in r["warm_ms"]]
    direct_ms = [x["direct"]["latency_ms"] for x in quality if x.get("direct")]

    def dist(xs):
        return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "min": min(xs) if xs else None, "max": max(xs) if xs else None}

    outcomes = [x.get("outcome") for x in quality if x.get("outcome")]
    def agg(name):
        rows = [x[name] for x in quality if x.get(name)]
        correct = sum(1 for r in rows if r["correct"])
        cost = [r["cost_credits"] for r in rows]
        total_cost = sum(cost) if rows and all(c is not None for c in cost) else None
        return {"n": len(rows), "correct": correct, "calls": sum(r["model_calls"] for r in rows),
                "prompt_tokens": sum((r["tokens"] or {}).get("prompt_tokens", 0) for r in rows),
                "completion_tokens": sum((r["tokens"] or {}).get("completion_tokens", 0) for r in rows),
                "reported_cost_credits": total_cost, "cost_per_correct": (total_cost / correct) if (total_cost is not None and correct) else None,
                "latency": dist([r["latency_ms"] for r in rows]),
                "taxonomy": {t: sum(1 for r in rows if r["taxonomy"] == t) for t in sorted({r["taxonomy"] for r in rows})}}
    traces = {}
    if jaeger_url:
        for r in recs:
            if r.get("phase") in ("cold", "direct", "deep") and r.get("trace_id"):
                traces[r["key"]] = jaeger_stages(jaeger_url, r["trace_id"])
    summary = {
        "meta": meta, "run_state": json.loads((run_dir / "run_state.json").read_text()) if (run_dir / "run_state.json").exists() else None,
        "cold_council_ms": dist(colds), "warm_l1_ms": dist(warm_all), "direct_ms": dist(direct_ms),
        "speedup": speedups(pairs), "per_question": per_question,
        "cache_workload": {"requests": l1_lookups, "l1_hits": l1_hits, "l1_hit_rate": l1_hits / l1_lookups if l1_lookups else None,
                           "repeat_share_by_construction": (cfg["warm_repeats"] / (cfg["warm_repeats"] + 1)),
                           "l2_semantic": "disabled (not measured)", "valid_cacheable_cold_answers": len(pairs)},
        "avoided": {"model_calls": calls_avoided, "prompt_tokens": tokens_avoided_in, "completion_tokens": tokens_avoided_out,
                    "reported_cost_credits": cost_avoided if cost_avoided_complete else None},
        "quality": {"per_question": quality, "outcomes": {o: outcomes.count(o) for o in ("council_win", "direct_win", "tie")},
                    "council": agg("council"), "direct": agg("direct"), "deep": agg("deep")},
        "warm_failures": sum(1 for r in recs if r.get("phase") == "warm_failed"),
        "not_cacheable": sum(1 for r in recs if r.get("phase") == "not_cacheable"),
        "traces": traces,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evidence-from", type=Path, help="offline run dir whose retrieved chunks size the likely-cost estimate")
    ap.add_argument("--out", type=Path, default=Path("results/cache_bench"))
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--jaeger-url")
    ap.add_argument("--authorize-paid-calls", action="store_true")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-spend-usd", type=float)
    ap.add_argument("--pricing", type=Path, default=Path("config/pricing.yaml"))
    a = ap.parse_args()
    if a.report:
        s = analyse(a.report, a.jaeger_url)
        print(json.dumps({k: s[k] for k in ("cold_council_ms", "warm_l1_ms", "direct_ms", "speedup", "cache_workload", "avoided")}, indent=2, default=str))
        return
    cfg = load_config(a.config)
    if a.dry_run:
        ev = None
        if a.evidence_from:
            ev = {}
            for l in (a.evidence_from / "records.jsonl").read_text().splitlines():
                r = json.loads(l)
                if r.get("response"):
                    ev[r["item_id"]] = sum(len(c.get("content", "")) for c in (r["response"].get("retrieval") or {}).get("chunks") or [])
        plan = dry_run(cfg, a.pricing, ev)
        print(json.dumps(plan, indent=2))
        return
    try:
        d = run(cfg, a.out, a.resume, a.authorize_paid_calls, a.max_requests, a.max_spend_usd, a.pricing)
    except (ModeError, IdentityError, APIError) as exc:
        raise SystemExit(f"refused: {redact(str(exc))}")
    print((d / "run_state.json").read_text())
    print(f"results: {d}")


if __name__ == "__main__":
    main()
