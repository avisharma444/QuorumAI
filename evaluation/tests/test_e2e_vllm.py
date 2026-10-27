"""e2e_vllm against an in-memory fake backend (TEST-ONLY, no network)."""
import json

import pytest
import yaml

from pathlib import Path

from councileval import e2e_vllm as E
from councileval import gpu_bench as G
from councileval.client import QueryOutcome
from councileval.gpu_budget import GpuSession, load_campaign
from test_cache_bench import dataset, pricing, MEMBERS, CHAIR

WANT = ["openai/gpt-4.1-nano", "meta-llama/llama-3.3-70b-instruct", E.PHI]
CAMPAIGN = Path(__file__).resolve().parents[1] / "config/experiments/runpod_campaign.yaml"


def cfg(tmp_path, **kw):
    c = {"name": "t", "api_base": "http://x", "dataset": str(dataset(tmp_path / "data")), "split": "heldout", "seed": 23,
         "per_category": {"extractive": 2, "numeric": 2}, "deep_questions": 0, "questions": 3, "top_k": 4,
         "max_output_tokens_estimate": 1024, "cancel_after_s": 1, "served_model": "microsoft/Phi-4-mini-instruct",
         "client_timeout_s": 180, "warmup_allocation_timeout_s": 480, "warmup_init_timeout_s": 600, "expected_models": {"members": WANT, "chairman": CHAIR}}
    c.update(kw)
    return c


class FakeBackend:
    def __init__(self, members=WANT, phi_ok=True):
        self.members, self.phi_ok, self.queries = members, phi_ok, []
        self.token = None

    def health(self):
        return {"llm": {"mode": "live", "members": self.members, "chairman": CHAIR}}

    def register(self):
        self.token = "t"

    def ingest(self, name, data):
        return {"doc_id": "d-" + name}

    def query(self, body):
        self.queries.append(body)
        u = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
        ok = [m for m in WANT if self.phi_ok or m != E.PHI]
        cands = [{"model": m, "served_model": m.removeprefix("local:"), "latency_ms": 900 if m == E.PHI else 1500, "usage": u} for m in ok]
        failed = [] if self.phi_ok else [{"model": E.PHI, "latency_ms": 120, "error": "vLLM API error (status 401)"}]
        # Like the backend: only members whose candidate succeeded review.
        revs = [{"reviewer": m, "latency_ms": 800, "usage": u, "error": None} for m in ok]
        return QueryOutcome(200, {"answer": "Ilse Varga built it; 42.", "candidates": cands, "failed_candidates": failed,
                                  "reviews": revs, "degraded": [],
                                  "synthesis": {"status": "synthesized"}, "duration_ms": 5000,
                                  "token_usage": {**u, "cost_credits": 0.0004}, "cache": {"tier": "none", "l2": "disabled_by_policy"}},
                            5100.0, "trace", None, 0)


class FakeEndpoint:
    """TEST-ONLY stand-in for the RunPod endpoint: async jobs and health."""

    def __init__(self, job_error=None, left_in_queue=0):
        self.jobs, self.purged, self.job_error, self.left = [], 0, job_error, left_in_queue

    def job(self, body, phase, idx, prompt_id, timeout_s, poll_s=1.0, sleep=None, on_poll=None, alloc_timeout_s=None, allocated=None):
        self.alloc_timeout = alloc_timeout_s
        self.jobs.append((body, phase, timeout_s))
        if on_poll:
            on_poll()
        r = G.JobResult(phase, idx, prompt_id, final_status="COMPLETED", client_total_s=42.0, delay_ms=40000, execution_ms=300)
        if self.job_error:
            r.error_class, r.final_status = self.job_error, "IN_QUEUE"
        return r

    def health(self):
        return {"jobs": {"inQueue": self.left, "inProgress": 0}, "workers": {"idle": 0, "running": 0}}

    def purge_queue(self):
        self.purged += 1
        return 200


def campaign(tmp_path, **over):
    c = load_campaign(CAMPAIGN)
    c["ledger"] = str(tmp_path / "gpu-ledger.json")
    c.update(over)
    return c


def session(tmp_path, phase, **over):
    return GpuSession(campaign(tmp_path, **over), f"e2e_{phase}")


def go(tmp_path, phase, api=None, ep="default", budget=None, **kw):
    ep = FakeEndpoint() if ep == "default" else ep
    return E.run(cfg(tmp_path, **kw), phase, tmp_path / "o", True, 10, 1.0, pricing(tmp_path), client=api or FakeBackend(),
                 ep=ep, budget=budget or session(tmp_path, phase), sleep=lambda s: None)


def test_refuses_without_authorisation_budget_endpoint_or_with_wrong_models(tmp_path):
    with pytest.raises(SystemExit, match="authorize"):
        E.run(cfg(tmp_path), "council", tmp_path / "o", False, 10, 1.0, pricing(tmp_path), client=FakeBackend(),
              ep=FakeEndpoint(), budget=session(tmp_path, "council"))
    with pytest.raises(SystemExit, match="GPU campaign budget"):
        E.run(cfg(tmp_path), "council", tmp_path / "o", True, 10, 1.0, pricing(tmp_path), client=FakeBackend(), ep=FakeEndpoint())
    with pytest.raises(SystemExit, match="GPU campaign budget"):  # a session for another phase does not count
        go(tmp_path, "council", budget=session(tmp_path, "fallback"))
    with pytest.raises(SystemExit, match="warm-up job"):
        go(tmp_path, "council", ep=None)
    with pytest.raises(SystemExit, match="not the approved"):
        go(tmp_path, "council", api=FakeBackend(members=MEMBERS))
    assert not (tmp_path / "gpu-ledger.json").exists()


def test_council_phase_warms_up_async_and_counts_every_gpu_call(tmp_path):
    api, ep = FakeBackend(), FakeEndpoint()
    d = go(tmp_path, "council", api=api, ep=ep)
    s = E.analyse(d)
    assert s["requests"] == 3 and s["ok"] == 3 and all(q["strategy"] == "council" for q in api.queries)
    assert s["per_model"][E.PHI]["candidate_ms"]["p50"] == 900 and s["slowest_candidate_counts"] == {"openai/gpt-4.1-nano": 3}
    assert s["hosted_cost_credits"] == pytest.approx(0.0012) and s["phi_candidate_missing"] == [] and s["correct"] == 3
    # One async warm-up job (not a synchronous request), recorded with RunPod's timings.
    assert len(ep.jobs) == 1 and ep.jobs[0][1] == "warmup" and ep.jobs[0][2] == 600 and ep.alloc_timeout == 480
    assert "stream" not in ep.jobs[0][0]
    warm = json.loads((d / "records.jsonl").read_text().splitlines()[0])
    assert warm["phase"] == "warmup" and warm["delay_ms"] == 40000
    st = json.loads((d / "run_state.json").read_text())
    assert st["gpu_session"]["requests"] == 1 + 3 * 2 and st["scaled_to_zero"] is True


def test_failed_warm_up_stops_before_any_council(tmp_path):
    api = FakeBackend()
    d = go(tmp_path, "council", api=api, ep=FakeEndpoint(job_error="job_timeout"))
    st = json.loads((d / "run_state.json").read_text())
    assert st["status"].startswith("stopped: warm-up failed") and api.queries == []


def test_fallback_phase_reports_failed_phi_and_no_phi_review(tmp_path):
    ep = FakeEndpoint()
    d = go(tmp_path, "fallback", api=FakeBackend(phi_ok=False), ep=ep)
    s = E.analyse(d)
    assert s["ok"] == 1 and len(s["phi_candidate_missing"]) == 1 and "401" in s["phi_candidate_errors"][0]
    assert s["phi_review_errors"] == [] and s["phi_reviewed_after_failure"] == []
    assert ep.jobs == [], "no warm-up with a wrong key"
    assert json.loads((d / "run_state.json").read_text())["gpu_session"]["requests"] == 2


def test_cancel_step_runs_in_the_council_session_and_purges(tmp_path, monkeypatch):
    class GiveUp:
        def __init__(self, base, timeout):
            self.token = None

        def query(self, body):
            return QueryOutcome(None, None, 3000.0, None, "client timeout", 0)
    monkeypatch.setattr(E, "Client", GiveUp)
    ep, api = FakeEndpoint(left_in_queue=1), FakeBackend()
    d = go(tmp_path, "council", api=api, ep=ep, cancel_observe_samples=2, cancel_after_councils=True)
    rec = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if "spend_monitor" not in l]
    assert [r["phase"] for r in rec] == ["warmup", "council", "council", "council", "cancel"] and len(ep.jobs) == 1
    assert rec[-1]["purge_queue_status"] == 200 and ep.purged == 1
    assert rec[-1]["item_id"] not in {r.get("item_id") for r in rec[1:4]}, "the cancel question is a fresh one"
    assert json.loads((d / "run_state.json").read_text())["gpu_session"]["requests"] == 1 + 3 * 2 + 2


def test_council_that_could_outrun_the_wall_cap_is_refused(tmp_path):
    # Session wall 1200 s: the warm-up may take 480 + 600 s, so it fits. On a
    # fake clock it takes 1050 s, so a council that may run 180 s more is refused.
    now = [0.0]

    class SlowStart(FakeEndpoint):
        def job(self, *a, **kw):
            now[0] += 1050
            return super().job(*a, **kw)
    api = FakeBackend()
    c = campaign(tmp_path)
    c["sessions"] = {**c["sessions"], "e2e_council": {"gpu_requests": 17, "max_wall_s": 1200}}
    d = go(tmp_path, "council", api=api, ep=SlowStart(), budget=GpuSession(c, "e2e_council", clock=lambda: now[0]))
    st = json.loads((d / "run_state.json").read_text())
    assert "GPU wall cap" in st["status"] and api.queries == [] and st["gpu_session"]["requests"] == 1


def test_ledger_is_shared_across_phases_and_enforces_the_campaign_cap(tmp_path):
    go(tmp_path, "council")                              # 7 GPU requests
    go(tmp_path, "fallback", api=FakeBackend(phi_ok=False))  # 2 more
    led = json.loads((tmp_path / "gpu-ledger.json").read_text())
    assert [x["session"] for x in led["sessions"]] == ["e2e_council", "e2e_fallback"]
    assert GpuSession.totals(led)["gpu_requests"] == 9 and all(x["status"] == "completed" for x in led["sessions"])
    # Campaign request cap of 10: another fallback session may send only 1 more, so its council (2 calls) is refused.
    api = FakeBackend(phi_ok=False)
    d = go(tmp_path, "fallback", api=api, budget=session(tmp_path, "fallback", max_gpu_requests=10))
    st = json.loads((d / "run_state.json").read_text())
    assert "GPU request cap" in st["status"] and api.queries == []


def test_tripped_spend_monitor_stops_councils(tmp_path):
    from test_gpu_bench import BalanceSession
    api, ep = FakeBackend(), FakeEndpoint()
    m = G.SpendMonitor("ro", 0.5, ep.health, every_s=0, session=BalanceSession([10.0, 10.0, 10.0, 9.0]))
    d = E.run(cfg(tmp_path), "council", tmp_path / "o", True, 10, 1.0, pricing(tmp_path), client=api, ep=ep,
              budget=session(tmp_path, "council"), sleep=lambda s: None, monitor=m)
    st = json.loads((d / "run_state.json").read_text())
    assert "provider-reported spend" in st["status"] and len(api.queries) == 1
    assert st["spend_monitor"]["spent_usd"] == pytest.approx(1.0)


def test_warm_up_is_not_submitted_without_a_placement_match(tmp_path):
    api, ep = FakeBackend(), FakeEndpoint()
    d = E.run(cfg(tmp_path), "council", tmp_path / "o", True, 10, 1.0, pricing(tmp_path), client=api, ep=ep,
              budget=session(tmp_path, "council"), sleep=lambda s: None, placement=lambda: {"ok": False, "gpus": {}})
    st = json.loads((d / "run_state.json").read_text())
    assert "no Serverless host matches" in st["status"] and ep.jobs == [] and api.queries == []
    assert st["gpu_session"]["requests"] == 0
