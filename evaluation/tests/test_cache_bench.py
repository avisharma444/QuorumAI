import json
from pathlib import Path

import pytest
import yaml

from councileval import cache_bench as CB
from councileval.client import QueryOutcome
from councileval.cost import BudgetExceeded
from councileval.schema import Document, EvalItem

MEMBERS = ["openai/gpt-4.1-nano", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-small-3.2-24b-instruct"]
CHAIR = "openai/gpt-4o-mini"


def dataset(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for cat, n in (("extractive", 3), ("numeric", 3)):
        for i in range(n):
            items.append(EvalItem(id=f"{cat}:{i}", question=f"{cat} question {i}?", source="t", source_id=str(i), category=cat,
                                  answer_type="number" if cat == "numeric" else "span", answers=["42"] if cat == "numeric" else ["Ilse Varga"],
                                  answerable=True, doc_id=f"d{i}", evidence=[], split="heldout", license="t", metadata={}))
    (d / "items.jsonl").write_text("".join(it.to_json() + "\n" for it in items))
    (d / "documents.jsonl").write_text("".join(json.dumps({"doc_id": f"d{i}", "title": "t", "text": "Ilse Varga built it. 42.",
                                                          "source": "t", "license": "t"}) + "\n" for i in range(3)))
    return d


def config(tmp_path, mode="offline", **kw):
    c = {"name": "t", "mode": mode, "api_base": "http://x", "dataset": str(dataset(tmp_path / "data")), "split": "heldout",
         "per_category": {"extractive": 2, "numeric": 2}, "warm_repeats": 3, "deep_questions": 2, "seed": 11,
         "metrics_url": "http://m", "expected_models": {"members": MEMBERS, "chairman": CHAIR, "ingestion": CHAIR}}
    c.update(kw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    return CB.load_config(p)


def pricing(tmp_path) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"models": [{"model": m, "input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0, "verified": True,
                                             "source": "https://x.test", "effective_date": "2026-10-27"} for m in MEMBERS + [CHAIR]]}))
    return p


class FakeCouncilAPI:
    """In-memory stand-in with a real per-request L1 rule."""

    def __init__(self, mode="mock", usage=True, warm_misses=False, degraded=False, swap_after=None, cost=0.001,
                 chair_unparsed=False, long_output=False):
        self.mode, self.usage, self.warm_misses, self.degraded = mode, usage, warm_misses, degraded
        self.chair_unparsed, self.long_output = chair_unparsed, long_output
        self.swap_after, self.cost = swap_after, cost
        self.l1, self.calls, self.queries, self.health_checks = {}, 0, [], 0

    def fresh(self):
        return self

    def health(self):
        self.health_checks += 1
        members = MEMBERS if self.mode == "live" else ["mock:member-1", "mock:member-2", "mock:member-3"]
        if self.swap_after is not None and self.health_checks > self.swap_after:
            members = ["openrouter/free"] + members[1:]
        return {"llm": {"mode": self.mode, "members": members, "chairman": CHAIR if self.mode == "live" else "mock:chairman"}}

    def register(self):
        self.l1 = {}  # a new user has an empty cache

    def ingest(self, name, data):
        self.calls += 1
        return {"doc_id": "app-" + name, "summary_model": CHAIR}

    def metrics_text(self, url):
        return f"councilai_llm_request_seconds_count{{stage=\"x\"}} {self.calls}\n"

    def query(self, body):
        self.queries.append(body)
        key = (body["question"], body["strategy"])
        if key in self.l1 and not self.warm_misses:
            return QueryOutcome(200, {"answer": self.l1[key], "strategy": body["strategy"], "cache_hit": True,
                                      "cache": {"tier": "l1_exact", "inference_executed": False, "l2": "disabled_by_policy"}}, 2.0, "", None, 0)
        u = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150} if self.usage else None
        if u and self.cost:
            u["cost_credits"] = self.cost
        if body["strategy"] == "direct":
            self.calls += 1
            resp = {"answer": "It was Ilse Varga, 42.", "strategy": "direct", "token_usage": u,
                    "cache": {"tier": "none", "inference_executed": True, "stored": ["l1_exact"], "l2": "disabled_by_policy"}}
            return QueryOutcome(200, resp, 300.0, "t", None, 0)
        if u and self.long_output:
            u = dict(u, completion_tokens=5000, total_tokens=5100)
        cands = [{"model": m, "served_model": m, "label": "ABC"[i], "answer": "Ilse Varga 42", "usage": u} for i, m in enumerate(MEMBERS)]
        reviews = [{"reviewer": m, "served_model": m, "usage": u} for m in MEMBERS]
        n = 7 + (2 if body["strategy"] == "council_deep" else 0)
        self.calls += n
        total = {k: v * n for k, v in u.items()} if u else None  # request total = every call's usage
        syn = {"status": "failed", "model": CHAIR, "error": "invalid JSON"} if self.chair_unparsed else {"status": "synthesized", "model": CHAIR}
        resp = {"answer": "Ilse Varga built it; 42.", "strategy": body["strategy"], "candidates": cands, "reviews": reviews,
                "synthesis": syn, "selected": "A", "token_usage": total,
                "degraded": ["chairman-failed-fallback"] if self.degraded else [],
                "cache": {"tier": "none", "inference_executed": True, "stored": [] if self.degraded else ["l1_exact"], "l2": "disabled_by_policy"}}
        if body["strategy"] == "council_deep":
            resp["reflection"] = {"quality": "needs_revision", "revision": {"status": "succeeded"}}
        if not self.degraded:
            self.l1[key] = resp["answer"]
        return QueryOutcome(200, resp, 4000.0, "t", None, 0)


def records(d):
    return [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines()]


# ── calculations ──────────────────────────────────────────────────────

def test_speedup_calculations():
    s = CB.speedups([(4000.0, [2.0, 4.0, 3.0]), (1000.0, [1.0, 1.0, 1.0]), (500.0, [])])
    assert s["questions"] == 2
    assert s["per_question"] == [4000 / 3.0, 1000.0]
    assert s["median_paired_speedup"] == pytest.approx((4000 / 3 + 1000) / 2)
    assert s["ratio_of_medians"] == pytest.approx(2500.0 / 1.5), "median cold of paired questions / median of all warm"
    assert CB.pct([1, 2, 3, 4], 0.5) == 2.5 and CB.pct([5], 0.95) == 5


def test_classification_and_call_counting():
    hit = {"cache": {"tier": "l1_exact", "inference_executed": False}}
    assert CB.classify(hit)["tier"] == "l1_exact" and CB.model_calls(hit) == 0
    council = {"strategy": "council", "candidates": [{}] * 3, "reviews": [{}] * 3, "synthesis": {"status": "synthesized"},
               "cache": {"inference_executed": True}}
    assert CB.model_calls(council) == 7
    deep = dict(council, strategy="council_deep", reflection={"revision": {"status": "succeeded"}})
    assert CB.model_calls(deep) == 9
    assert CB.model_calls({"strategy": "direct", "cache": {"inference_executed": True}}) == 1
    assert CB.classify(None)["valid"] is False


def test_reported_cost_needs_complete_usage():
    u = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost_credits": 0.01}
    full = {"strategy": "council", "candidates": [{"usage": u}], "reviews": [{"usage": u}], "token_usage": u, "cache": {"inference_executed": True}}
    assert CB.reported_cost(full) == 0.01
    missing = dict(full, candidates=[{"usage": None}])
    assert CB.reported_cost(missing) is None, "a call without usage makes the request's cost unknown"
    no_cost = dict(full, token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    assert CB.reported_cost(no_cost) is None


def test_served_model_check():
    assert CB.served_ok("openai/gpt-4.1-nano", "openai/gpt-4.1-nano-2025-04-14")
    assert CB.served_ok("openai/gpt-4.1-nano", None)
    assert not CB.served_ok("openai/gpt-4.1-nano", "anthropic/claude-x")


# ── procedure ─────────────────────────────────────────────────────────

def test_offline_procedure_cold_warm_direct_deep(tmp_path):
    api = FakeCouncilAPI()
    d = CB.run(config(tmp_path), tmp_path / "o", client=api)
    rs = records(d)
    phases = [r["phase"] for r in rs]
    assert phases.count("cold") == 4 and phases.count("warm") == 12 and phases.count("direct") == 4 and phases.count("deep") == 2
    for r in rs:
        if r["phase"] == "warm":
            assert r["classification"]["tier"] == "l1_exact" and r["model_calls"] == 0 and r["prometheus_llm_calls_delta"] == 0
    s = CB.analyse(d)
    assert s["cache_workload"]["l1_hit_rate"] == pytest.approx(12 / 16) and s["avoided"]["model_calls"] == 12 * 7
    assert s["avoided"]["prompt_tokens"] == 12 * 7 * 100 and s["speedup"]["median_paired_speedup"] == pytest.approx(2000.0)
    assert s["quality"]["outcomes"]["tie"] == 4


def test_unexpected_warm_inference_stops_repeats(tmp_path):
    api = FakeCouncilAPI(warm_misses=True)
    d = CB.run(config(tmp_path), tmp_path / "o", client=api)
    rs = records(d)
    assert sum(1 for r in rs if r["phase"] == "warm") == 4, "one failed warm per question, then no more repeats"
    assert sum(1 for r in rs if r["phase"] == "warm_failed") == 4
    s = CB.analyse(d)
    assert s["cache_workload"]["l1_hits"] == 0 and s["avoided"]["model_calls"] == 0 and s["speedup"]["questions"] == 0


def test_degraded_cold_answer_is_not_repeated(tmp_path):
    api = FakeCouncilAPI(degraded=True)
    d = CB.run(config(tmp_path), tmp_path / "o", client=api)
    rs = records(d)
    assert not any(r["phase"] == "warm" for r in rs) and sum(1 for r in rs if r["phase"] == "not_cacheable") == 4


def test_model_identity_change_aborts_real_run(tmp_path):
    api = FakeCouncilAPI(mode="live", swap_after=2)
    d = CB.run(config(tmp_path, mode="real"), tmp_path / "o", client=api, authorize_paid=True, max_requests=100,
               max_spend_usd=100.0, pricing_path=pricing(tmp_path))
    state = json.loads((d / "run_state.json").read_text())
    assert state["status"].startswith("stopped") and "differ" in state["status"]
    assert len({r["item_id"] for r in records(d)}) == 1, "stopped before the second question"


def test_real_run_refuses_wrong_models_before_any_query(tmp_path):
    api = FakeCouncilAPI(mode="live", swap_after=0)
    with pytest.raises(CB.IdentityError):
        CB.run(config(tmp_path, mode="real"), tmp_path / "o", client=api, authorize_paid=True, max_requests=10,
               max_spend_usd=10.0, pricing_path=pricing(tmp_path))
    assert api.queries == []


def test_budget_cap_refuses_a_request_whose_worst_case_does_not_fit(tmp_path):
    api = FakeCouncilAPI(mode="live")
    c = config(tmp_path, mode="real")
    est_cold = CB.request_estimate("cold", 20, c).cost(1.0, 2.0)
    est_ing = CB.request_estimate("ingest", 0, c).cost(1.0, 2.0)
    d = CB.run(c, tmp_path / "o", client=api, authorize_paid=True, max_requests=1000,
               max_spend_usd=est_ing + est_cold * 0.5, pricing_path=pricing(tmp_path))
    state = json.loads((d / "run_state.json").read_text())
    assert state["status"].startswith("stopped") and api.queries == [], "no council request may start when its worst case exceeds the cap"
    d2 = CB.run(c, tmp_path / "o2", client=FakeCouncilAPI(mode="live"), authorize_paid=True, max_requests=4,
                max_spend_usd=100.0, pricing_path=pricing(tmp_path))
    assert "request limit" in json.loads((d2 / "run_state.json").read_text())["status"]


def test_verified_warm_hits_are_charged_nothing(tmp_path):
    d = CB.run(config(tmp_path, mode="real"), tmp_path / "o", client=FakeCouncilAPI(mode="live"), authorize_paid=True,
               max_requests=1000, max_spend_usd=100.0, pricing_path=pricing(tmp_path))
    rs = records(d)
    for i, r in enumerate(rs):
        if r["phase"] == "warm":
            prev = next(p for p in reversed(rs[:i]) if "budget" in p)
            assert r["budget"]["charged_usd"] == prev["budget"]["charged_usd"]
    cold = [r for r in rs if r["phase"] == "cold"]
    assert all(r["reported_cost_credits"] == pytest.approx(0.007) for r in cold), "observed provider cost is recorded, not the estimate"


def test_missing_usage_is_charged_at_worst_case(tmp_path):
    api = FakeCouncilAPI(mode="live", usage=False)
    c = config(tmp_path, mode="real", per_category={"extractive": 1}, deep_questions=0)
    d = CB.run(c, tmp_path / "o", client=api, authorize_paid=True, max_requests=100, max_spend_usd=100.0, pricing_path=pricing(tmp_path))
    cold = next(r for r in records(d) if r["phase"] == "cold")
    worst = CB.request_estimate("ingest", 0, c).cost(1.0, 2.0) + CB.request_estimate("cold", len("extractive question 0?"), c).cost(1.0, 2.0)
    assert cold["reported_cost_credits"] is None and cold["budget"]["charged_usd"] == pytest.approx(worst, rel=1e-6)


def test_resume_does_not_send_paid_warm_requests_for_a_new_user(tmp_path):
    api = FakeCouncilAPI()
    c = config(tmp_path)
    d = CB.run(c, tmp_path / "o", client=api)
    # Simulate an interruption right after the first cold request: keep only that record.
    first = (d / "records.jsonl").read_text().splitlines()[0]
    (d / "records.jsonl").write_text(first + "\n")
    api2 = FakeCouncilAPI()
    CB.run(c, tmp_path / "o", resume=d, client=api2)
    rs = records(d)
    q0 = json.loads(first)["item_id"]
    assert sum(1 for r in rs if r["item_id"] == q0 and r["phase"] == "cold") == 1, "cold not repeated"
    assert any(r["item_id"] == q0 and r["phase"] == "warm_skipped" for r in rs)
    assert not any(r["item_id"] == q0 and r["phase"] == "warm" for r in rs), "no 'warm' request against an empty cache"


def test_dry_run_plans_without_any_http(tmp_path):
    c = config(tmp_path, mode="real", likely_prices={m: [0.1, 0.4] for m in MEMBERS + [CHAIR]})
    plan = CB.dry_run(c, pricing(tmp_path))
    assert plan["requests"] == {"ingest": 3, "cold_council": 4, "warm": 12, "direct": 4, "deep": 2}
    assert plan["model_calls_expected"]["candidates"] == 3 * 4 + 3 * 2
    assert 0 < plan["likely_usd"] < plan["worst_case_usd"]
    assert plan["prices_missing_or_unverified"] == []


# ── live preflight (first cold council + first warm repeat) ─────────

def real_run(tmp_path, api):
    return CB.run(config(tmp_path, mode="real"), tmp_path / "o", client=api, authorize_paid=True, max_requests=1000,
                  max_spend_usd=100.0, pricing_path=pricing(tmp_path))


def test_preflight_passes_once_then_the_run_continues(tmp_path):
    d = real_run(tmp_path, FakeCouncilAPI(mode="live"))
    rs = records(d)
    pf = [r for r in rs if r["phase"].startswith("preflight")]
    assert [r["phase"] for r in pf] == ["preflight"] and pf[0]["status"] == "passed"
    i = rs.index(pf[0])
    assert [r["phase"] for r in rs[:i]] == ["cold", "warm"], "preflight is decided after the first cold and first warm"
    assert json.loads((d / "run_state.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("kw,needle", [
    ({"chair_unparsed": True}, "chairman"),
    ({"degraded": True}, "degraded"),
    ({"usage": False}, "usage"),
    ({"cost": 0}, "cost"),
    ({"long_output": True}, "cap"),
    ({"warm_misses": True}, "warm repeat"),
])
def test_preflight_failure_stops_the_whole_run(tmp_path, kw, needle):
    api = FakeCouncilAPI(mode="live", **kw)
    d = real_run(tmp_path, api)
    rs = records(d)
    fail = [r for r in rs if r["phase"] == "preflight_failed"]
    assert len(fail) == 1 and any(needle in f for f in fail[0]["failures"]), fail
    state = json.loads((d / "run_state.json").read_text())["status"]
    assert state.startswith("stopped: preflight failed")
    assert len({r["item_id"] for r in rs}) == 1, "no further question after a failed preflight"
    assert not any(r["phase"] in ("direct", "deep") for r in rs)
    assert len(api.queries) <= 2


def test_resumed_real_run_without_passed_preflight_is_refused(tmp_path):
    api = FakeCouncilAPI(mode="live", chair_unparsed=True)
    d = real_run(tmp_path, api)
    with pytest.raises(CB.PreflightError):
        CB.run(config(tmp_path, mode="real"), tmp_path / "o", resume=d, client=FakeCouncilAPI(mode="live"), authorize_paid=True,
               max_requests=1000, max_spend_usd=100.0, pricing_path=pricing(tmp_path))
