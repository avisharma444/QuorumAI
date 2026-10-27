"""GPU benchmark harness against a local fake of the RunPod Serverless API.
TEST-ONLY: the fake streams scripted tokens; no GPU or provider is involved."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from councileval import gpu_bench as G

KEY = "rpa_TESTKEY0123456789abcdef"
EP = "ep-fake"
CFG_PATH = Path(__file__).resolve().parents[1] / "config/experiments/runpod_phi4_bench.yaml"


class FakeRunPod:
    """Health, purge-queue, async jobs (/run, /status, /cancel) and a
    streaming OpenAI chat route, like the RunPod gateway. Async jobs stay
    IN_QUEUE for cold_s (a cold start), then IN_PROGRESS for exec_s."""

    def __init__(self, tokens=8, token_delay=0.004, buffered=False, http11=False, cold_s=0.3, exec_s=0.02, job_never_ends=False):
        self.tokens, self.delay, self.buffered = tokens, token_delay, buffered
        self.cold_s, self.exec_s, self.job_never_ends = cold_s, exec_s, job_never_ends
        self.active = 0
        self.purged = 0
        self.lock = threading.Lock()
        self.bodies = []      # streamed chat bodies
        self.jobs = {}        # id -> {"input", "t0", "cancelled"}
        self.cancelled = []
        self.order = []       # "job" / "stream" in arrival order
        self.warm = False
        self.errors_as_200 = False
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if http11 else "HTTP/1.0"

            def log_message(self, *a):
                pass

            def out(self, data: bytes):
                if http11:  # chunked transfer encoding, like a streaming gateway
                    data = b"%x\r\n%s\r\n" % (len(data), data)
                self.wfile.write(data)
                self.wfile.flush()

            def plain(self, code, body=b""):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    self.plain(401, b'{"error":"Unauthorized"}')
                    return False
                return True

            def do_GET(self):
                if not self._auth():
                    return
                if self.path == f"/v2/{EP}/health":
                    with fake.lock:
                        busy = fake.active
                    body = {"jobs": {"inProgress": busy, "inQueue": 0}, "workers": {"idle": 0, "running": 1 if busy else 0}}
                    self.plain(200, json.dumps(body).encode())
                elif self.path.startswith(f"/v2/{EP}/status/"):
                    self.plain(200, json.dumps(fake.job_status(self.path.rsplit("/", 1)[1])).encode())
                else:
                    self.plain(404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")  # read first: keep-alive connections are reused
                if not self._auth():
                    return
                if self.path == f"/v2/{EP}/purge-queue":
                    fake.purged += 1
                    self.plain(200)
                    return
                if self.path == f"/v2/{EP}/run":
                    jid = f"job-{len(fake.jobs)}"
                    fake.jobs[jid] = {"input": body["input"], "t0": time.monotonic(), "cancelled": False}
                    fake.order.append("job")
                    self.plain(200, json.dumps({"id": jid, "status": "IN_QUEUE"}).encode())
                    return
                if self.path.startswith(f"/v2/{EP}/cancel/"):
                    jid = self.path.rsplit("/", 1)[1]
                    fake.jobs[jid]["cancelled"] = True
                    fake.cancelled.append(jid)
                    self.plain(200, json.dumps({"id": jid, "status": "CANCELLED"}).encode())
                    return
                fake.bodies.append(body)
                fake.order.append("stream")
                if body.get("model") != "microsoft/Phi-4-mini-instruct":
                    if fake.errors_as_200:  # like the live gateway: HTTP 200 carrying the worker's error JSON
                        self.plain(200, b'{"error":{"message":"vLLM returned HTTP 404: model not found","type":"worker_error"}}')
                        return
                    self.plain(404, b'{"error":{"message":"model not found"}}')
                    return
                if len(body["messages"][0]["content"]) > 20000:
                    self.plain(400, b'{"error":{"message":"maximum context length is 4096 tokens"}}')
                    return
                with fake.lock:
                    fake.active += 1
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    if http11:
                        self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    chunks = []
                    for i in range(fake.tokens):
                        ev = {"model": "microsoft/Phi-4-mini-instruct", "choices": [{"delta": {"content": f"t{i} "}, "finish_reason": None}]}
                        chunks.append(f"data: {json.dumps(ev)}\n\n".encode())
                        if not fake.buffered:
                            time.sleep(fake.delay)
                            self.out(chunks[-1])
                    if fake.buffered:
                        time.sleep(fake.delay * fake.tokens)
                        self.out(b"".join(chunks))
                    done = {"choices": [{"delta": {}, "finish_reason": "length"}],
                            "usage": {"prompt_tokens": 40, "completion_tokens": fake.tokens, "total_tokens": 40 + fake.tokens}}
                    self.out(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
                    if http11:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with fake.lock:
                        fake.active -= 1

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.H = H
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_port}"

    def job_status(self, jid):
        j = self.jobs[jid]
        if j["cancelled"]:
            return {"id": jid, "status": "CANCELLED"}
        age = time.monotonic() - j["t0"]
        cold = 0 if self.warm else self.cold_s
        if self.job_never_ends or age < cold:
            return {"id": jid, "status": "IN_QUEUE"}
        if age < cold + self.exec_s:
            return {"id": jid, "status": "IN_PROGRESS"}
        self.warm = True
        oi = j["input"]["openai_input"]
        if j["input"].get("openai_route") != "/v1/chat/completions" or oi.get("stream") is not False:
            return {"id": jid, "status": "FAILED", "error": "bad job input"}
        if oi.get("model") != "microsoft/Phi-4-mini-instruct":
            out = [{"error": {"message": "vLLM returned HTTP 404: model not found", "type": "worker_error"}}]
        else:
            n = oi["max_tokens"]
            out = [{"model": oi["model"], "choices": [{"message": {"content": "x " * n}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": n, "total_tokens": 40 + n}}]
        return {"id": jid, "status": "COMPLETED", "delayTime": int(cold * 1000), "executionTime": int(self.exec_s * 1000),
                "workerId": "w-1", "output": out}

    def close(self):
        self.srv.shutdown()


@pytest.fixture(params=["http10", "http11-chunked"])
def fake(request):
    # token gaps dominate connection overhead, as on a real GPU
    f = FakeRunPod(token_delay=0.02, http11=request.param == "http11-chunked")
    yield f
    f.close()


CAMPAIGN_PATH = CFG_PATH.parent / "runpod_campaign.yaml"


def campaign(tmp_path, **over):
    c = G.load_campaign(CAMPAIGN_PATH)
    c["ledger"] = str(tmp_path / "ledger" / "gpu.json")
    c.update(over)
    return c


def small_cfg(**over):
    cfg = G.load_config(CFG_PATH)
    cfg["endpoint"]["idle_timeout_s"] = 1
    cfg["phases"]["startup"].update(allocation_timeout_s=5, init_timeout_s=5, poll_s=0.01)
    cfg["phases"]["worker_timing"].update(requests=2, poll_s=0.01)
    cfg["phases"]["warm"]["requests"] = 3
    cfg["phases"]["concurrency"] = {"levels": [1, 2, 4], "requests_per_level": 4}
    cfg["phases"]["failures"] = [{"kind": "invalid_key"}, {"kind": "wrong_model"}, {"kind": "prompt_too_long", "repeat": 900},
                                 {"kind": "client_cancel", "after_chunks": 2, "observe_s": 0.2}]
    cfg.update(over)
    return cfg


_NO_BUDGET = object()


PLACEABLE = lambda: {"ok": True, "match": [{"gpu": "NVIDIA GeForce RTX 4090", "cuda": ["13.2"]}]}


def go(tmp_path, fake, cfg=None, budget=None, monitor=None, placement=PLACEABLE, **kw):
    cfg = cfg or small_cfg()
    args = dict(authorize_paid=True, max_requests=500, max_wall_s=600, max_spend_usd=5.0)
    args.update(kw)
    if budget is None:
        budget = G.GpuSession(campaign(tmp_path), "gpu_bench")
    elif budget is _NO_BUDGET:
        budget = None
    return G.run(cfg, tmp_path, G.Endpoint(EP, KEY, fake.base), budget=budget, sleep=lambda s: time.sleep(0.01),
                 health_every_s=0.05, monitor=monitor, placement=placement, **args)


def test_config_and_plan_are_pinned():
    cfg = G.load_config(CFG_PATH)
    p = G.plan(cfg)
    assert p["requests"] == {"startup": 1, "worker_timing": 8, "warm": 20, "concurrency": {"1": 16, "2": 16, "4": 16, "8": 16},
                             "failures": 5, "total": 98}
    assert cfg["model"]["revision"] and cfg["endpoint"]["max_workers"] == 1 and cfg["endpoint"]["min_workers"] == 0
    assert cfg["endpoint"]["image"] == "runpod/worker-v1-vllm:v2.28.0" and cfg["endpoint"]["image_digest"].startswith("sha256:")
    assert cfg["endpoint"]["gpu_type_ids"] == ["NVIDIA GeForce RTX 4090"] and cfg["price_usd_per_hour"]["value"] == 1.10
    st = cfg["phases"]["startup"]
    assert (st["allocation_timeout_s"], st["init_timeout_s"]) == (480, 600) == (480, cfg["endpoint"]["cold_start_allowance_s"])
    assert cfg["phases"]["request_timeout_s"] == cfg["endpoint"]["execution_timeout_s"] == 120
    assert G.worst_case_usd(cfg, 1680) == pytest.approx((1680 + 600 + 120 + 30) * 1.10 / 3600)


def test_campaign_plan_covers_every_gpu_phase_within_caps():
    c = G.load_campaign(CAMPAIGN_PATH)
    cp = G.campaign_plan(c)
    assert set(cp["sessions"]) == {"gpu_bench", "e2e_council", "e2e_fallback"}
    # 1 request was used by the first (cancelled) attempt; the cap of 120 includes it.
    assert cp["total"]["gpu_requests"] == 98 + (1 + 6 * 2 + 2) + 2 == 115 <= c["max_gpu_requests"] - 1
    assert cp["total"]["max_wall_s"] == c["max_gpu_wall_s"] == 1680 + 2360 + 200
    assert cp["total"]["worst_case_usd"] == pytest.approx((1680 + 2360 + 200 + 3 * 750) * 1.10 / 3600, abs=1e-3)
    assert cp["total"]["worst_case_usd"] <= c["max_gpu_spend_usd"] == 2.0 and cp["problems"] == []
    G.check_campaign(c, G.load_config(CFG_PATH))  # same price and timeouts as the benchmark
    with pytest.raises(SystemExit, match="price"):
        G.check_campaign({**c, "price_usd_per_hour": 0.69}, G.load_config(CFG_PATH))


def test_refusals_happen_before_any_request(tmp_path, fake):
    with pytest.raises(G.Abort, match="authorize"):
        go(tmp_path, fake, authorize_paid=False)
    with pytest.raises(G.Abort, match="campaign budget"):
        go(tmp_path, fake, budget=_NO_BUDGET)
    with pytest.raises(G.Abort, match="exceeds --max-spend-usd"):
        go(tmp_path, fake, max_spend_usd=0.01)
    with pytest.raises(G.Abort, match="below the planned"):
        go(tmp_path, fake, max_requests=3)
    with pytest.raises(G.Abort, match="exceeds the campaign"):
        go(tmp_path, fake, max_wall_s=99999, max_spend_usd=99)
    # Campaign already spent: refused before anything is sent.
    spent = G.GpuSession(campaign(tmp_path, max_gpu_spend_usd=0.9), "gpu_bench").open()
    spent.close("completed")
    with pytest.raises(G.Abort, match="GPU spend"):
        go(tmp_path, fake, budget=G.GpuSession(campaign(tmp_path, max_gpu_spend_usd=0.9), "gpu_bench"))
    assert fake.bodies == [] and fake.jobs == {}


def test_full_run_measures_streaming_concurrency_and_failures(tmp_path, fake):
    d = go(tmp_path, fake)
    s = G.analyse(d)
    assert s["run_state"]["status"] == "completed" and s["run_state"]["scaled_to_zero"] is True
    # Startup is an async job on the cold endpoint, measured apart from warm inference.
    su = s["startup"]
    assert su["phase"] == "startup" and su["final_status"] == "COMPLETED" and su["delay_ms"] == 300 and su["worker_id"] == "w-1"
    assert [st for _, st in su["status_trace"]][:1] == ["IN_QUEUE"] and su["status_trace"][-1][1] == "COMPLETED"
    assert fake.order[:3] == ["job", "job", "job"] and fake.order[3] == "stream", "no streamed request before startup is done"
    first = fake.jobs["job-0"]["input"]
    assert first["openai_route"] == "/v1/chat/completions" and first["openai_input"]["max_tokens"] == 8
    assert first["openai_input"]["stream"] is False and "stream_options" not in first["openai_input"]
    # Worker-side timing (1-token jobs) vs gateway TTFT are reported side by side, labelled.
    assert s["worker_timing"]["execution_ms"]["n"] == 2 and s["worker_timing"]["completion_tokens"]["p50"] == 1
    t = s["ttft"]
    assert t["gateway_ttft_p50_s"] > 0 and t["worker_1token_execution_p50_s"] == pytest.approx(0.02)
    assert "upper bound" in t["definitions"]["worker_1token_execution"] and "not measured" in t["definitions"]["in_engine_ttft"]
    assert s["totals"]["async_jobs"] == 3
    # Every request was counted in the shared GPU ledger.
    gs = s["run_state"]["gpu_session"]
    assert gs["requests"] == s["run_state"]["requests_sent"] == 3 + 3 + 12 + 4
    assert gs["campaign_totals"]["gpu_requests"] == 22
    assert s["warm"]["latency_s"]["n"] == 3 and s["warm"]["ttft_s"]["p50"] < s["warm"]["latency_s"]["p50"]
    assert s["warm"]["decode_tok_s"]["p50"] > 0 and s["warm"]["completion_tokens"]["p50"] == 8
    assert set(s["concurrency"]) == {"1", "2", "4"}
    c1, c4 = s["concurrency"]["1"], s["concurrency"]["4"]
    assert c1["ok"] == 4 and c1["output_tokens"] == 32 and c1["throughput_vs_c1"] == pytest.approx(1.0)
    assert c4["aggregate_output_tok_s"] > c1["aggregate_output_tok_s"], "the fake serves requests in parallel"
    assert s["streaming"]["median_content_chunks"] == 8 and s["streaming"]["median_first_to_last_chunk_share"] > 0.5
    fails = {f["phase"]: f["error_class"] for f in s["failures"]}
    assert fails["fail_invalid_key"] == "http_401" and fails["fail_wrong_model"] == "http_404"
    assert fails["fail_prompt_too_long"] == "http_400" and fails["fail_client_cancel"] == "client_cancelled"
    # Every request used the fixed generation settings.
    b = fake.bodies[0]
    assert b["max_tokens"] == 256 and b["temperature"] == 0.0 and b["seed"] == 1234 and b["ignore_eos"] is True
    assert b["stream"] is True and b["stream_options"] == {"include_usage": True}
    # Keys never reach disk.
    for f in d.iterdir():
        assert KEY not in f.read_text() and "invalid-key-for-test" not in f.read_text()


def test_buffered_proxy_is_detected_not_reported_as_streaming(tmp_path):
    f = FakeRunPod(buffered=True, token_delay=0.01)
    try:
        cfg = small_cfg()
        cfg["phases"]["failures"] = []
        s = G.analyse(go(tmp_path, f, cfg=cfg))
    finally:
        f.close()
    # All chunks arrive together at the end: TTFT ~ total latency.
    assert s["streaming"]["median_first_to_last_chunk_share"] < 0.2
    assert s["warm"]["ttft_s"]["p50"] > 0.8 * s["warm"]["latency_s"]["p50"]


def test_client_timeout_is_recorded(tmp_path):
    f = FakeRunPod(tokens=40, token_delay=0.05)  # 2 s of streaming
    try:
        cfg = small_cfg()
        cfg["phases"]["warm"]["requests"] = 0
        cfg["phases"]["concurrency"] = {"levels": [1], "requests_per_level": 1}
        cfg["phases"]["failures"] = [{"kind": "client_timeout", "timeout_s": 0.02}]
        s = G.analyse(go(tmp_path, f, cfg=cfg))
    finally:
        f.close()
    assert {x["phase"]: x["error_class"] for x in s["failures"]}["fail_client_timeout"] in ("client_timeout", "transport")


def test_no_worker_allocated_cancels_after_the_allocation_wait(tmp_path):
    f = FakeRunPod(job_never_ends=True)  # IN_QUEUE forever, health shows no workers
    try:
        cfg = small_cfg()
        cfg["phases"]["startup"].update(allocation_timeout_s=0.3, init_timeout_s=60)
        d = go(tmp_path, f, cfg=cfg)
    finally:
        f.close()
    state = json.loads((d / "run_state.json").read_text())
    assert state["status"].startswith("aborted: startup job failed: allocation_timeout: no worker allocated after 0s")
    assert f.cancelled == ["job-0"] and f.purged == 1 and f.bodies == []
    assert state["gpu_session"]["requests"] == 1
    rec = next(json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if '"type": "job"' in l)
    assert rec["allocated_s"] is None and rec["client_total_s"] < 30, "init timeout must not apply before allocation"


def _health_override(f, workers):
    orig = f.H.do_GET

    def do_GET(h):
        if h.path == f"/v2/{EP}/health":
            w = workers if (f.jobs and not f.cancelled) else {}  # a worker exists only while a job is waiting
            body = json.dumps({"jobs": {"inQueue": 0 if f.cancelled else 1, "inProgress": 0}, "workers": w}).encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        orig(h)
    f.H.do_GET = do_GET


def test_allocated_worker_gets_the_separate_init_timeout(tmp_path):
    f = FakeRunPod(job_never_ends=True)
    _health_override(f, {"initializing": 1})  # a worker holds a GPU but the model never finishes loading
    try:
        cfg = small_cfg()
        cfg["phases"]["startup"].update(allocation_timeout_s=60, init_timeout_s=0.5)
        d = go(tmp_path, f, cfg=cfg)
    finally:
        f.close()
    state = json.loads((d / "run_state.json").read_text())
    assert state["status"].startswith("aborted: startup job failed: job_timeout: no result after 0s from worker allocation")
    rec = next(json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if '"type": "job"' in l)
    assert rec["allocated_s"] is not None and rec["client_total_s"] < 30 and f.cancelled == ["job-0"]


def test_request_that_could_outrun_the_wall_cap_is_refused(tmp_path, fake):
    # 200 s wall: the startup (5 + 5 s) and worker-timing (120 s) jobs fit, but a
    # streamed request (300 s timeout here) could overrun, so none is sent.
    cfg = small_cfg()
    cfg["phases"]["request_timeout_s"] = 300
    d = go(tmp_path, fake, cfg=cfg, max_wall_s=200)
    state = json.loads((d / "run_state.json").read_text())
    assert "warm could run past 200s" in state["status"] and fake.bodies == [] and len(fake.jobs) == 3


def test_wall_cap_aborts_and_purges_the_queue(tmp_path, fake):
    cfg = small_cfg()
    d = go(tmp_path, fake, cfg=cfg, max_wall_s=0.0001, max_spend_usd=5.0)
    state = json.loads((d / "run_state.json").read_text())
    recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines()]
    assert state["status"].startswith("aborted") and fake.purged == 1
    assert any(r["type"] == "abort" for r in recs) and state["scaled_to_zero"] is True


def test_endpoint_settings_check():
    cfg = G.load_config(CFG_PATH)
    good = {"workersMin": 0, "workersMax": 1, "idleTimeout": 30, "gpuTypeIds": ["NVIDIA GeForce RTX 4090"], "gpuCount": 1,
            "executionTimeoutMs": 120000, "allowedCudaVersions": ["13.0"],
            "template": {"imageName": cfg["endpoint"]["image"], "containerDiskInGb": 30,
                         "env": {k: str(v) for k, v in cfg["endpoint"]["env"].items()}}}
    good["template"]["env"]["GPU_MEMORY_UTILIZATION"] = "0.90"
    assert G.check_settings(good, cfg) == []
    bad = json.loads(json.dumps(good))
    bad.update(workersMin=1, workersMax=3, idleTimeout=600, gpuTypeIds=["NVIDIA H100 80GB HBM3"],
               executionTimeoutMs=600000, allowedCudaVersions=["12.4", "13.0"], networkVolumeId="nv-1")
    bad["template"]["containerDiskInGb"] = 10
    bad["template"]["env"]["MODEL_NAME"] = "other/model"
    bad["template"]["imageName"] = "runpod/worker-v1-vllm:latest"
    probs = " | ".join(G.check_settings(bad, cfg))
    for needle in ("workersMin", "workersMax", "idleTimeout", "gpuTypeIds", "MODEL_NAME", "image", "executionTimeoutMs", "allowedCudaVersions",
                   "network volume", "container disk"):
        assert needle in probs


def test_billing_sums_provider_amounts(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"endpoint_id": EP}))
    (tmp_path / "run_state.json").write_text(json.dumps({"started_utc": "2026-10-28T10:05:00+00:00", "finished_utc": "2026-10-28T10:40:00+00:00"}))

    class Sess:
        def get(self, url, params, headers, timeout):
            assert url.endswith("/billing/endpoints") and params["endpointId"] == EP and params["bucketSize"] == "hour"
            assert headers["Authorization"] == "Bearer ro-key"

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return [{"amount": 0.12, "timeBilledMs": 600000, "endpointId": EP}, {"amount": 0.03, "timeBilledMs": 150000}]
            return R()
    b = G.fetch_billing(tmp_path, "ro-key", session=Sess())
    assert b["amount_usd"] == pytest.approx(0.15) and b["time_billed_s"] == 750
    assert "ro-key" not in (tmp_path / "billing.json").read_text()


def test_endpoint_settings_fills_the_image_from_the_template_route():
    class Sess:
        def __init__(self):
            self.urls = []

        def get(self, url, params=None, headers=None, timeout=None):
            self.urls.append(url)
            body = ({"id": EP, "templateId": "tpl1", "template": {"id": "tpl1", "containerDiskInGb": 30}}
                    if url.endswith(f"/endpoints/{EP}") else {"id": "tpl1", "imageName": "runpod/worker-v1-vllm:v2.28.0"})

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return body
            return R()
    s = Sess()
    ep = G.endpoint_settings(EP, "ro-key", session=s)
    assert ep["template"] == {"id": "tpl1", "containerDiskInGb": 30, "imageName": "runpod/worker-v1-vllm:v2.28.0"}
    assert s.urls[-1].endswith("/templates/tpl1")


class BalanceSession:
    """TEST-ONLY: scripted GraphQL balance answers (a float, or an exception)."""

    def __init__(self, answers):
        self.answers = list(answers)

    def post(self, url, json=None, timeout=None, headers=None):
        assert "graphql" in url and headers["User-Agent"] == G.USER_AGENT
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": {"myself": {"clientBalance": a, "currentSpendPerHr": 0.69}}}
        return R()


def test_spend_monitor_trips_on_provider_spend_unreadable_billing_and_retries():
    import requests as rq
    jobs = {"retried": 0}
    health = lambda: {"jobs": dict(jobs)}
    m = G.SpendMonitor("ro", 1.00, health, every_s=0, session=BalanceSession([10.0, 9.6, 8.9]))
    m.start()
    m.check()  # spent 0.40: fine
    assert m.last["spent_usd"] == pytest.approx(0.40)
    with pytest.raises(G.Abort, match="provider-reported spend \\$1.1000 exceeds \\$1.00"):
        m.check()
    with pytest.raises(G.Abort):  # stays tripped
        m.check()
    m = G.SpendMonitor("ro", 1.00, health, every_s=0, max_failures=3,
                       session=BalanceSession([10.0, rq.ConnectionError("x"), rq.ConnectionError("x"), rq.ConnectionError("x")]))
    m.start()
    m.check()
    m.check()
    with pytest.raises(G.Abort, match="billing cannot be monitored"):
        m.check()
    m = G.SpendMonitor("ro", 1.00, health, every_s=0, session=BalanceSession([10.0]))
    m.start()
    jobs["retried"] = 1
    with pytest.raises(G.Abort, match="unexpected retries"):
        m.check()


def test_spend_monitor_reads_at_most_every_interval():
    t = [0.0]
    s = BalanceSession([10.0, 9.9, 9.8])
    m = G.SpendMonitor("ro", 1.0, lambda: {"jobs": {"retried": 0}}, every_s=20, session=s, clock=lambda: t[0])
    m.start()
    m.check()
    assert m.last["spent_usd"] == 0.0, "no read before the interval"
    t[0] = 21
    m.check()
    assert m.last["spent_usd"] == pytest.approx(0.1)


def test_startup_validation():
    cfg = G.load_config(CFG_PATH)
    ok = G.JobResult("startup", 0, "p1", final_status="COMPLETED", served_model="microsoft/Phi-4-mini-instruct",
                     prompt_tokens=40, completion_tokens=8, text="A message queue ...")
    assert G.validate_startup(ok, cfg, 8) == []
    bad = G.JobResult("startup", 0, "p1", final_status="COMPLETED", served_model="other", prompt_tokens=None,
                      completion_tokens=None, text=" ")
    probs = " | ".join(G.validate_startup(bad, cfg, 8))
    assert "served model" in probs and "no token usage" in probs and "empty answer" in probs


def test_monitor_and_startup_validation_run_inside_the_benchmark(tmp_path, fake):
    m = G.SpendMonitor("ro", 5.0, G.Endpoint(EP, KEY, fake.base).health, every_s=0, session=BalanceSession([10.0, 9.99]))
    d = go(tmp_path, fake, monitor=m)
    recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines()]
    val = next(r for r in recs if r["type"] == "startup_validation")
    assert val["problems"] == []
    spend = [r for r in recs if r["type"] == "spend"]
    assert spend[0]["event"] == "start" and len(spend) > 10, "checked before requests and while polling jobs"
    st = json.loads((d / "run_state.json").read_text())
    assert st["status"] == "completed" and st["spend_monitor"]["spent_usd"] == pytest.approx(0.01)


def test_tripped_monitor_stops_the_run_and_purges(tmp_path, fake):
    m = G.SpendMonitor("ro", 0.5, G.Endpoint(EP, KEY, fake.base).health, every_s=0, session=BalanceSession([10.0, 10.0, 9.0]))
    d = go(tmp_path, fake, monitor=m)
    st = json.loads((d / "run_state.json").read_text())
    assert "provider-reported spend" in st["status"] and fake.purged == 1
    assert len(fake.bodies) < 3 + 12 + 4


def test_unhealthy_workers_during_startup_cancel_the_job(tmp_path):
    f = FakeRunPod(job_never_ends=True)
    orig = f.H.do_GET

    def do_GET(h):
        if h.path == f"/v2/{EP}/health":
            body = json.dumps({"jobs": {"inQueue": 1, "inProgress": 0}, "workers": {"unhealthy": 1}}).encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        orig(h)
    f.H.do_GET = do_GET
    try:
        cfg = small_cfg()
        cfg["phases"]["startup"].update(timeout_s=30, poll_s=0.01)
        watch_every = G.UnhealthyWatch.__init__.__defaults__
        G.UnhealthyWatch.__init__.__defaults__ = (3, 0.0, watch_every[2])
        try:
            d = go(tmp_path, f, cfg=cfg)
        finally:
            G.UnhealthyWatch.__init__.__defaults__ = watch_every
    finally:
        f.close()
    st = json.loads((d / "run_state.json").read_text())
    assert "unhealthy worker" in st["status"] and f.cancelled == ["job-0"] and f.bodies == []


def test_release_reservation_appends_an_audited_credit(tmp_path):
    from councileval.gpu_budget import release_reservation
    c = campaign(tmp_path)
    sess = G.GpuSession(c, "gpu_bench").open()
    sess.take(1, "startup", 10)
    sess.close("aborted: allocation")
    path = Path(c["ledger"])
    original = json.loads(path.read_text())["sessions"][0]
    entry = release_reservation(path, sess.key, "provider billed $0", {"billing_records": []}, "owner")
    led = json.loads(path.read_text())
    assert led["sessions"][0] == original, "the original record is never rewritten"
    assert entry["usd_charged"] == -original["usd_charged"] and entry["gpu_requests"] == 0
    t = G.GpuSession.totals(led)
    assert t["usd"] == pytest.approx(0) and t["wall_s"] == pytest.approx(0) and t["gpu_requests"] == 1
    with pytest.raises(Exception, match="already released"):
        release_reservation(path, sess.key, "again", {}, "owner")


def _catalog_session(gpus, fail=None):
    class S:
        def get(self, url, params=None, timeout=None, headers=None):
            assert url == G.CATALOG and params == {"include": "AVAILABILITY", "product": "SERVERLESS"}
            assert headers["User-Agent"] == G.USER_AGENT
            if fail:
                raise fail

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"gpus": gpus}
            return R()
    return S()


def test_serverless_placement_needs_an_exact_gpu_and_cuda_match():
    cfg = G.load_config(CFG_PATH)
    assert cfg["endpoint"]["allowed_cuda_versions"] == ["13.0", "13.1", "13.2", "13.3", "13.4"]
    live = [{"id": "NVIDIA GeForce RTX 4090", "availability": "HIGH",  # today's catalog (2026-10-27)
             "cudaVersions": [{"available": False, "version": "12.8"}, {"available": False, "version": "13.0"},
                              {"available": True, "version": "13.2"}]},
            {"id": "NVIDIA L4", "availability": "NONE", "cudaVersions": [{"available": False, "version": "13.2"}]}]
    pl = G.serverless_placement(cfg, "ro", session=_catalog_session(live))
    assert pl["ok"] and pl["match"] == [{"gpu": "NVIDIA GeForce RTX 4090", "cuda": ["13.2"]}]
    # The configuration that failed twice: CUDA "13.0" only matches nothing.
    old = json.loads(json.dumps(cfg))
    old["endpoint"]["allowed_cuda_versions"] = ["13.0"]
    pl = G.serverless_placement(old, "ro", session=_catalog_session(live))
    assert not pl["ok"] and pl["gpus"]["NVIDIA GeForce RTX 4090"]["cuda_available"] == ["13.2"]
    # Fail closed: errors, missing GPUs, missing fields, no availability.
    import requests as rq
    assert not G.serverless_placement(cfg, "ro", session=_catalog_session([], fail=rq.ConnectionError("x")))["ok"]
    assert not G.serverless_placement(cfg, "ro", session=_catalog_session([]))["ok"]
    assert not G.serverless_placement(cfg, "ro", session=_catalog_session([{"id": "NVIDIA GeForce RTX 4090", "availability": "HIGH"}]))["ok"]
    none = [{**live[0], "availability": "NONE"}]
    assert not G.serverless_placement(cfg, "ro", session=_catalog_session(none))["ok"]


def test_no_placement_match_stops_before_any_job(tmp_path, fake):
    d = go(tmp_path, fake, placement=lambda: {"ok": False, "gpus": {"NVIDIA GeForce RTX 4090": {"cuda_available": ["13.2"]}}})
    st = json.loads((d / "run_state.json").read_text())
    assert st["status"].startswith("aborted: no Serverless host matches") and fake.jobs == {} and fake.bodies == []
    assert st["gpu_session"]["requests"] == 0, "nothing was sent, nothing counted"
    d = go(tmp_path, fake, placement=None)  # and no Read Only key: refuse to submit blind
    assert "placement preflight needs the Read Only key" in json.loads((d / "run_state.json").read_text())["status"]


def test_quiet_check_accepts_only_a_stopped_unbilled_standby_worker(monkeypatch):
    state = {"health": {"workers": {}, "jobs": {}}, "listed": [], "rate": 0}

    class Ep:
        id = EP

        def health(self, timeout=10):
            return state["health"]
    monkeypatch.setattr(G, "endpoint_settings", lambda eid, key, session=None: {"workers": [{"id": "w", "desiredStatus": s} for s in state["listed"]]})
    monkeypatch.setattr(G, "read_balance", lambda key, session=None: {"balance": 10.0, "spend_per_hr": state["rate"]})
    q = G.quiet_check(Ep(), "ro")
    assert q()["kind"] == "zero" and q()["quiet"]
    # What RunPod showed on 2026-10-27 after the CUDA fix: idle/ready standby, container EXITED, $0/h.
    state.update(health={"workers": {"idle": 1, "ready": 1}, "jobs": {}}, listed=["EXITED"], rate=0)
    assert q()["kind"] == "standby" and q()["quiet"]
    for change in ({"rate": 0.89}, {"listed": ["RUNNING"]}, {"listed": []},
                   {"health": {"workers": {"idle": 1, "running": 1}, "jobs": {}}},
                   {"health": {"workers": {"idle": 1}, "jobs": {"inQueue": 1}}}):
        saved = dict(state)
        state.update(change)
        assert not q()["quiet"], change
        state.clear(); state.update(saved)
    assert not G.quiet_check(Ep(), None)()["quiet"], "without the Read Only key only zero counts"


def test_standby_start_is_labelled_and_accepted(tmp_path, fake, monkeypatch):
    orig = fake.H.do_GET

    def do_GET(h):
        if h.path == f"/v2/{EP}/health":
            body = json.dumps({"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"idle": 1, "ready": 1}}).encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        orig(h)
    fake.H.do_GET = do_GET
    monkeypatch.setattr(G, "endpoint_settings", lambda eid, key, session=None: {"workers": [{"id": "w", "desiredStatus": "EXITED"}]})
    monkeypatch.setattr(G, "read_balance", lambda key, session=None: {"balance": 10.0, "spend_per_hr": 0})
    monkeypatch.setattr(G, "WorkerSampler", lambda *a, **k: type("N", (), {"start": lambda s: None, "stop": lambda s: None})())
    cfg = small_cfg()
    cfg["phases"]["failures"] = []
    d = go(tmp_path, fake, cfg=cfg, readonly_key="ro")
    st = json.loads((d / "run_state.json").read_text())
    s = G.analyse(d)
    assert st["status"] == "completed" and st["scaled_to_zero"] is True and st["end_condition"] == "standby"
    assert s["startup"]["phase"] == "startup_standby"


def test_http_200_without_tokens_is_a_failure_with_the_body_kept(tmp_path, fake):
    fake.errors_as_200 = True
    cfg = small_cfg()
    cfg["phases"]["failures"] = [{"kind": "wrong_model"}]
    s = G.analyse(go(tmp_path, fake, cfg=cfg))
    f = next(x for x in s["failures"] if x["phase"] == "fail_wrong_model")
    assert f["status"] == 200 and f["error_class"] == "empty_stream" and "model not found" in f["error"]
    assert s["totals"]["ok"] == s["totals"]["requests"] - 1
