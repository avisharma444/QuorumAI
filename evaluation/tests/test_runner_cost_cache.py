import json
from pathlib import Path

import pytest
import yaml

from councileval import cache_analysis as CA
from councileval import guards as G
from councileval.client import QueryOutcome, redact
from councileval.cost import (BudgetExceeded, BudgetGuard, PricingError, UsageLedger, load_pricing, require_verified,
                              worst_case_request)
from councileval.runner import ModeError, check_mode, done_keys, load_config, plan, run
from councileval.schema import Document, EvalItem


# ── fixtures ───────────────────────────────────────────────────────────

def write_dataset(d: Path, n: int = 3) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    items = [EvalItem(id=f"t:{i}", question=f"Who built bridge {i}?", source="test", source_id=str(i), category="extractive",
                      answer_type="span", answers=["Ilse Varga"], answerable=True, doc_id="d1",
                      evidence=["It was built by Ilse Varga."], split="dev", license="test", metadata={}) for i in range(n)]
    (d / "items.jsonl").write_text("".join(i.to_json() + "\n" for i in items))
    (d / "documents.jsonl").write_text(json.dumps({"doc_id": "d1", "title": "t", "text": "It was built by Ilse Varga.",
                                                   "source": "test", "license": "test"}) + "\n")
    return d


class FakeAPI:
    """Stands in for the HTTP API; records what the runner sends."""

    def __init__(self, mode="mock", usage=None, fail_status=None):
        self.mode, self.usage, self.fail_status = mode, usage, fail_status
        self.queries, self.registered, self.ingested = [], 0, 0

    def fresh(self):
        return self

    def health(self):
        return {"llm": {"mode": self.mode, "members": ["mock:a" if self.mode == "mock" else "openrouter/x"],
                        "chairman": "mock:c" if self.mode == "mock" else "gemini:g"}}

    def register(self):
        self.registered += 1
        self.token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJlc2ln"

    def ingest(self, name, data):
        self.ingested += 1
        return {"doc_id": "app-" + name}

    def metrics_text(self, url):
        return None

    def query(self, body):
        self.queries.append(body)
        if self.fail_status:
            return QueryOutcome(self.fail_status, None, 1.0, "", "boom", 0)
        resp = {"answer": "It was built by Ilse Varga.", "strategy": body.get("strategy", "council"),
                "candidates": [{"model": "mock:a", "label": "A", "answer": "Ilse Varga", "usage": self.usage}],
                "reviews": [], "selected": "A", "token_usage": self.usage, "grounding": "document",
                "retrieval": {"chunks": [{"chunk_index": 0, "content": "It was built by Ilse Varga."}]},
                "cache": {"tier": "none"}, "source": "chairman:mock:c", "synthesis": {"status": "synthesized"}}
        return QueryOutcome(200, resp, 5.0, "abc", None, 0)


def cfg(tmp_path, **kw):
    c = {"name": "t", "mode": "offline", "api_base": "http://x", "dataset": str(write_dataset(tmp_path / "data")), "split": "dev",
         "strategies": ["direct", "council"]}
    c.update(kw)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(c))
    return load_config(p)


# ── mode safeguards ───────────────────────────────────────────────────

def test_offline_refuses_live_backend_before_any_query(tmp_path):
    api = FakeAPI(mode="live")
    with pytest.raises(ModeError):
        run(cfg(tmp_path), tmp_path / "out", client=api)
    assert api.queries == [] and api.registered == 0


def test_real_mode_requires_flag_live_backend_budget_and_verified_prices(tmp_path):
    c = cfg(tmp_path, mode="real")
    api = FakeAPI(mode="live")
    with pytest.raises(ModeError, match="authorize"):
        run(c, tmp_path / "o", client=api)
    with pytest.raises(ModeError, match="max-requests"):
        run(c, tmp_path / "o", client=api, authorize_paid=True)
    pricing = tmp_path / "p.yaml"
    pricing.write_text("models: []\n")
    with pytest.raises(PricingError):
        run(c, tmp_path / "o", client=api, authorize_paid=True, max_requests=5, max_spend_usd=1.0, pricing_path=pricing)
    with pytest.raises(ModeError, match="live backend"):
        check_mode({"llm": {"mode": "mock"}}, "real", True)
    assert api.queries == []


def test_pricing_requires_source_and_date_for_verified_entries(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"models": [{"model": "m", "input_usd_per_mtok": 1, "output_usd_per_mtok": 2, "verified": True}]}))
    with pytest.raises(PricingError):
        load_pricing(p)
    p.write_text(yaml.safe_dump({"models": [{"model": "m", "input_usd_per_mtok": 1, "output_usd_per_mtok": 2, "verified": False,
                                             "source": "https://x", "effective_date": "2026-10-27"}]}))
    with pytest.raises(PricingError):
        require_verified(load_pricing(p), ["m"])


# ── budget and cost accounting ───────────────────────────────────────

def test_budget_guard_stops_before_exceeding_limits():
    est = worst_case_request("council", 100, 6000, 1024, routed=False)
    g = BudgetGuard(max_requests=2, max_spend_usd=1.0, price_in=1.0, price_out=2.0)
    one = est.cost(1.0, 2.0)
    assert 0 < one < 1.0
    g.check(est)
    g.record(est, None)
    g.check(est)
    g.record(est, None)
    with pytest.raises(BudgetExceeded, match="request limit"):
        g.check(est)
    tight = BudgetGuard(10, max_spend_usd=one * 1.5, price_in=1.0, price_out=2.0)
    tight.check(est)
    tight.record(est, None)
    with pytest.raises(BudgetExceeded, match="could cost"):
        tight.check(est)


def test_real_run_stops_on_budget(tmp_path):
    pricing = tmp_path / "p.yaml"
    pricing.write_text(yaml.safe_dump({"models": [
        {"model": m, "input_usd_per_mtok": 10.0, "output_usd_per_mtok": 30.0, "verified": True, "source": "https://example.test/pricing",
         "effective_date": "2026-10-27"} for m in ("openrouter/x", "gemini:g")]}))
    api = FakeAPI(mode="live", usage=None)
    est = worst_case_request("council", 30, 4 * 1500, 1024, routed=False).cost(10.0, 30.0)
    d = run(cfg(tmp_path, mode="real", strategies=["council"]), tmp_path / "o", client=api, authorize_paid=True,
            max_requests=100, max_spend_usd=est * 2.5, pricing_path=pricing)
    state = json.loads((d / "run_state.json").read_text())
    assert state["status"].startswith("stopped") and len(api.queries) == 2, "third request could exceed the spend limit"


def test_ledger_separates_reported_missing_and_unpriced_usage():
    led = UsageLedger()
    led.add_call("mock:a", None, {})
    led.add_call("mock:a", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, {})
    s = led.summary(mock=True)
    assert s["actual_api_spend_usd"] == 0.0 and s["estimated_cost_usd"] is None and s["calls_without_reported_usage"] == 2
    assert s["hypothetical_real_model_cost"] == "unmeasured"
    led2 = UsageLedger()
    led2.add_call("x", {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}, {})
    s2 = led2.summary(mock=False)
    assert s2["reported_prompt_tokens"] == 1000 and s2["estimated_cost_usd"] == 0.0 and s2["cost_unmeasured_calls"] == 1


# ── checkpoints, resume and cancellation ─────────────────────────────

def test_resume_continues_without_duplicates(tmp_path):
    c = cfg(tmp_path)
    api = FakeAPI()
    d = run(c, tmp_path / "o", client=api, stop_after=4)
    assert json.loads((d / "run_state.json").read_text())["status"] == "interrupted"
    assert len(api.queries) == 4
    run(c, tmp_path / "o", resume=d, client=api)
    keys = [json.loads(l)["key"] for l in (d / "records.jsonl").read_text().splitlines()]
    assert len(keys) == 6 and len(set(keys)) == 6, "3 items x 2 strategies, each exactly once"
    assert len(api.queries) == 6


def test_torn_checkpoint_line_is_redone(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text(json.dumps({"item_id": "a", "strategy": "direct", "repeat": 0}) + "\n{\"item_id\": \"b\", \"str")
    assert done_keys(p) == {"a|direct|0"}


def test_plan_is_seeded_and_complete():
    items = [type("I", (), {"id": f"i{n}"})() for n in range(5)]
    p1 = [(i.id, s, r) for i, s, r in plan(items, ["direct", "council", "auto"], 2, seed=7)]
    p2 = [(i.id, s, r) for i, s, r in plan(items, ["direct", "council", "auto"], 2, seed=7)]
    assert p1 == p2 and len(set(p1)) == 30


def test_metadata_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc1234def")
    d = run(cfg(tmp_path), tmp_path / "o", client=FakeAPI())
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["git_commit"] == "abc1234def" and meta["external_model_calls_enabled"] is False
    assert meta["backend_llm"]["mode"] == "mock" and meta["items"] == 3 and len(meta["item_ids_sha256"]) == 64
    rec = json.loads((d / "records.jsonl").read_text().splitlines()[0])
    assert "eyJ" not in json.dumps(rec), "tokens never reach result files"


def test_failed_requests_are_recorded_not_retried_forever(tmp_path):
    api = FakeAPI(fail_status=502)
    d = run(cfg(tmp_path), tmp_path / "o", client=api)
    recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines()]
    assert all(r["status"] == 502 and r["retries_5xx"] == 1 for r in recs) and len(api.queries) == 12


# ── redaction ─────────────────────────────────────────────────────────

def test_secret_redaction():
    txt = ('Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJlc2ln key sk-or-v1-abcdefghijklmnopqrstuv '
           'AIzaSyA1234567890abcdefghijklmnop {"password": "hunter22", "api_key": "zzz"}')
    out = redact(txt)
    for secret in ("eyJhbGciOiJIUzI1NiJ9", "sk-or-v1-abc", "AIzaSyA123", "hunter22", "zzz"):
        assert secret not in out
    assert redact("ordinary error: connection refused") == "ordinary error: connection refused"


# ── cache analysis and guards ─────────────────────────────────────────

def dec(id_, eq, sim, numbers=True, cat="paraphrase", split="heldout", q1="a", q2="b"):
    hits = {f"{t:.3f}": (sim >= t and numbers) for t in (0.80, 0.85, 0.90)}
    return {"id": id_, "equivalent": eq, "similarity": sim, "redis_similarity": sim, "numbers_match": numbers,
            "production_hit": hits, "category": cat, "source": "authored", "split": split, "q1": q1, "q2": q2}


def test_cache_confusion_and_threshold_boundary():
    rows = [dec("p1", True, 0.95), dec("p2", True, 0.84), dec("n1", False, 0.86), dec("n2", False, 0.70)]
    c = CA.table(rows, 0.85)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert CA.decide(dec("b", True, 0.85), 0.85) is True, "similarity equal to the threshold is a hit"
    assert CA.decide(dec("b", True, 0.8499999), 0.85) is False
    assert CA.decide(dec("n", False, 0.99, numbers=False), 0.85) is False, "number guard vetoes"


def test_verification_catches_inconsistent_decisions():
    good = dec("g", True, 0.9)
    bad = dec("b", False, 0.95)
    bad["production_hit"]["0.900"] = False  # recorded miss although similarity >= threshold
    v = CA.verify([good, bad])
    assert v["rule_mismatches"] == 1
    bad["redis_similarity"] = 0.5
    assert CA.verify([bad])["redis_similarity_mismatches"] == 1


def test_threshold_selection_uses_explore_only():
    explore = [dec("e1", True, 0.95, split="explore"), dec("e2", False, 0.88, split="explore")]
    ch = CA.choose_threshold(explore, [0.80, 0.85, 0.90])
    assert ch["threshold"] == 0.90 and ch["explore"]["fpr"] == 0.0


def test_guards_veto_known_meaning_changes_but_not_paraphrases():
    assert G.polarity_differs("Which plans include support?", "Which plans do not include support?")
    assert not G.polarity_differs("Which plans include support?", "What plans include support?")
    assert G.opposite_terms("What is the minimum age?", "What is the maximum age?")
    assert not G.opposite_terms("What is the minimum age?", "What is the lowest age allowed, at minimum?")
    assert G.entities_differ("When was Marie Curie born?", "When was Pierre Curie born?")
    assert not G.entities_differ("When was Marie Curie born?", "In what year was Marie Curie born?")
    assert G.role_swap("Is A faster than B?", "Is B faster than A?")
    assert not G.role_swap("Is A faster than B?", "Is A faster than B?")


def test_per_category_sample_is_seeded_and_stratified(tmp_path):
    c = cfg(tmp_path, per_category=2, strategies=["direct"])
    a, b = FakeAPI(), FakeAPI()
    run(c, tmp_path / "o1", client=a)
    run(c, tmp_path / "o2", client=b)
    assert [q["question"] for q in a.queries] == [q["question"] for q in b.queries] and len(a.queries) == 2
