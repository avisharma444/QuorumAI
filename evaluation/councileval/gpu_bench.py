"""Direct GPU inference benchmark for a RunPod Serverless vLLM endpoint.

Talks to the endpoint's OpenAI-compatible route with streaming, so time to
first token (TTFT) is measured from real token chunks, not from QuorumAI's
SSE candidate events. Phases, all with fixed prompts and generation settings:

  startup      one small ASYNC job (/run + /status) while the endpoint is
               scaled to zero. RunPod's synchronous routes give up after about
               90 s, which a cold start (weights download + load) can exceed,
               so startup is measured on its own and never mixed with warm
               inference. Records RunPod's delayTime/executionTime.
  worker_timing  1-token async jobs: RunPod's executionTime is time inside the
               worker (vLLM prefill + first token + the worker's local proxy).
               It bounds in-engine TTFT from above and excludes the gateway,
               network and queue, unlike the streamed (gateway) TTFT below.
  warm         sequential streamed requests on a warm worker (p50/p95
               latency, gateway TTFT measured from the first token chunk)
  concurrency  the same request mix at concurrency 1, 2, 4, 8 (aggregate
               output tokens/s, i.e. whether vLLM batching helps)
  failures     client timeout, over-long prompt, wrong model name, invalid
               key, and a client cancel mid-stream (each recorded as observed)

Safety: real mode needs --authorize-paid-calls, --max-requests,
--max-wall-seconds, --max-spend-usd and --gpu-campaign (the shared GPU
ledger in gpu_budget.py, which also covers the end-to-end phases). A request
is refused if it could still be running at the wall cap. The worst case is bounded because
the endpoint runs at most one worker: GPU cost <= (run wall time + idle
timeout + cold-start allowance) x price per second, and the run refuses to
start if that bound exceeds --max-spend-usd. On abort it purges the
endpoint queue. At the end it waits until the endpoint reports zero
workers. Keys come from the environment only and are never written.

  python -m councileval.gpu_bench --config config/experiments/runpod_phi4_bench.yaml --dry-run
  python -m councileval.gpu_bench --config ... --gpu-campaign config/experiments/runpod_campaign.yaml \
      --authorize-paid-calls --max-requests 98 --max-wall-seconds 2400 --max-spend-usd 1.00
  python -m councileval.gpu_bench --report results/gpu_bench/<run>
  python -m councileval.gpu_bench --billing results/gpu_bench/<run>     (needs RUNPOD_READONLY_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import requests
import urllib3
import yaml

from .client import redact
from .gpu_budget import GpuBudgetExceeded, GpuSession, campaign_plan, load_campaign

INFERENCE_KEY_ENV = "RUNPOD_API_KEY"            # restricted to the benchmark endpoint
READONLY_KEY_ENV = "RUNPOD_READONLY_API_KEY"    # optional: endpoint settings, workers, billing
ENDPOINT_ENV = "RUNPOD_ENDPOINT_ID"
REST = "https://rest.runpod.io/v1"
USER_AGENT = "quorumai-bench/1.0"


class Abort(RuntimeError):
    pass


# ── configuration ─────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("name", "model", "gpu", "price_usd_per_hour", "generation", "prompts", "phases", "endpoint"):
        if k not in cfg:
            raise ValueError(f"config missing {k}")
    if not cfg["price_usd_per_hour"].get("verified"):
        raise ValueError("GPU price must be verified (source and date) before a paid run")
    cfg.setdefault("api_base", "https://api.runpod.ai")
    return cfg


def plan(cfg: dict) -> dict:
    """Every request this benchmark can send to the endpoint (no retries)."""
    ph = cfg["phases"]
    conc = {str(c): ph["concurrency"]["requests_per_level"] for c in ph["concurrency"]["levels"]}
    fails = len(ph.get("failures", []))
    wt = ph["worker_timing"]["requests"]
    reqs = 1 + wt + ph["warm"]["requests"] + sum(conc.values()) + fails
    max_out = cfg["generation"]["max_tokens"]
    gen_tokens = (ph["startup"]["max_tokens"] + wt * ph["worker_timing"]["max_tokens"]
                  + (ph["warm"]["requests"] + sum(conc.values())) * max_out)
    price_s = cfg["price_usd_per_hour"]["value"] / 3600
    return {"requests": {"startup": 1, "worker_timing": wt, "warm": ph["warm"]["requests"], "concurrency": conc,
                         "failures": fails, "total": reqs},
            "max_generated_tokens": gen_tokens, "price_usd_per_second": price_s,
            "endpoint": cfg["endpoint"], "model": cfg["model"], "gpu": cfg["gpu"], "generation": cfg["generation"]}


def worst_case_usd(cfg: dict, max_wall_s: float) -> float:
    """At most one worker: billed time <= wall time + cold-start allowance
    (a worker may still be loading after an abort) + one execution timeout
    (a job the gateway did not cancel) + idle timeout."""
    e = cfg["endpoint"]
    secs = max_wall_s + e["cold_start_allowance_s"] + e["execution_timeout_s"] + e["idle_timeout_s"]
    return secs * cfg["price_usd_per_hour"]["value"] / 3600


# ── one streamed request ──────────────────────────────────────────────

@dataclass
class Result:
    phase: str
    idx: int
    prompt_id: str
    concurrency: int = 1
    status: Optional[int] = None
    error: Optional[str] = None
    error_class: Optional[str] = None
    t_start_utc: str = ""
    latency_s: Optional[float] = None
    headers_s: Optional[float] = None
    ttft_s: Optional[float] = None
    content_chunks: int = 0
    first_to_last_chunk_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    served_model: Optional[str] = None
    decode_tok_s: Optional[float] = None
    e2e_tok_s: Optional[float] = None
    text_chars: int = 0
    max_gap_s: Optional[float] = None


@dataclass
class JobResult:
    """One asynchronous job (/run, then /status polling). delay_ms and
    execution_ms are RunPod's own numbers: delayTime is time from submission
    to a worker picking the job up (queue + any cold start), executionTime is
    time inside the worker handler."""
    phase: str
    idx: int
    prompt_id: str
    job_id: Optional[str] = None
    t_start_utc: str = ""
    submit_status: Optional[int] = None
    final_status: Optional[str] = None
    client_total_s: Optional[float] = None
    first_seen_in_progress_s: Optional[float] = None   # resolution = poll interval
    allocated_s: Optional[float] = None                # first sign of a worker for this endpoint (health) or job leaving the queue
    delay_ms: Optional[float] = None
    execution_ms: Optional[float] = None
    worker_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    served_model: Optional[str] = None
    text: str = ""                                      # first 200 characters of the answer (validity evidence)
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    error_class: Optional[str] = None
    polls: int = 0
    status_trace: list = field(default_factory=list)   # [(seconds since submit, status)] on change


def _job_output(out) -> tuple[Optional[dict], Optional[str]]:
    """The worker yields the chat completion once; RunPod aggregates generator
    output into a list. Returns (completion, error message)."""
    items = out if isinstance(out, list) else [out]
    for it in items:
        if isinstance(it, dict) and it.get("error"):
            e = it["error"]
            return None, json.dumps(e)[:400] if not isinstance(e, str) else e[:400]
        if isinstance(it, dict) and ("choices" in it or "usage" in it):
            return it, None
    return None, None


def sse_lines(resp: requests.Response):
    """Yield SSE lines as soon as their bytes arrive. requests' iter_lines()
    buffers 512-byte blocks (or, without chunked encoding, the whole body),
    which would delay early tokens and inflate TTFT."""
    raw = resp.raw
    raw.decode_content = True
    buf = b""
    while True:
        data = raw.read1(65536)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line:
                yield line.decode("utf-8", "replace")
    if buf.strip():
        yield buf.strip().decode("utf-8", "replace")


class Endpoint:
    def __init__(self, endpoint_id: str, key: str, api_base: str = "https://api.runpod.ai",
                 session: Optional[requests.Session] = None):
        self.id, self._key, self.base = endpoint_id, key, api_base.rstrip("/")
        self.s = session or requests.Session()

    @property
    def chat_url(self) -> str:
        return f"{self.base}/v2/{self.id}/openai/v1/chat/completions"

    def _h(self, key: Optional[str] = None) -> dict:
        return {"Authorization": f"Bearer {self._key if key is None else key}", "Content-Type": "application/json",
                "User-Agent": USER_AGENT}  # RunPod's edge rejects some default library agents (HTTP 403, code 1010)

    def health(self, timeout: float = 10) -> dict:
        r = self.s.get(f"{self.base}/v2/{self.id}/health", headers=self._h(), timeout=timeout)
        r.raise_for_status()
        return r.json()

    def purge_queue(self) -> Optional[int]:
        try:
            return self.s.post(f"{self.base}/v2/{self.id}/purge-queue", headers=self._h(), timeout=10).status_code
        except requests.RequestException:
            return None

    def submit(self, job_input: dict, timeout: float = 30) -> requests.Response:
        return self.s.post(f"{self.base}/v2/{self.id}/run", headers=self._h(), json={"input": job_input}, timeout=timeout)

    def status(self, job_id: str, timeout: float = 15) -> dict:
        r = self.s.get(f"{self.base}/v2/{self.id}/status/{job_id}", headers=self._h(), timeout=timeout)
        r.raise_for_status()
        return r.json()

    def cancel(self, job_id: str) -> Optional[int]:
        try:
            return self.s.post(f"{self.base}/v2/{self.id}/cancel/{job_id}", headers=self._h(), timeout=10).status_code
        except requests.RequestException:
            return None

    def job(self, body: dict, phase: str, idx: int, prompt_id: str, timeout_s: float, poll_s: float = 1.0,
            sleep: Callable[[float], None] = time.sleep, on_poll: Optional[Callable[[], None]] = None,
            alloc_timeout_s: Optional[float] = None, allocated: Optional[Callable[[], bool]] = None) -> JobResult:
        """Run one OpenAI chat request as an async job and poll it to the end.
        body must be non-streaming. on_poll runs after every poll and may raise
        to stop: the job is then cancelled before the exception propagates.

        Timeouts (the job is cancelled when either expires):
        - without alloc_timeout_s: timeout_s from submission;
        - with alloc_timeout_s: up to alloc_timeout_s waiting for a worker to
          be allocated (allocated() true, or the job leaving the queue), then
          timeout_s from allocation for model initialization and the answer."""
        res = JobResult(phase, idx, prompt_id, t_start_utc=datetime.now(timezone.utc).isoformat())
        t0 = time.perf_counter()
        try:
            r = self.submit({"openai_route": "/v1/chat/completions", "openai_input": {**body, "stream": False}})
            res.submit_status = r.status_code
            if r.status_code != 200:
                res.error, res.error_class = redact(r.text[:400]), f"http_{r.status_code}"
                res.client_total_s = time.perf_counter() - t0
                return res
            res.job_id = r.json().get("id")
            last = None
            while True:
                st = self.status(res.job_id)
                res.polls += 1
                now = time.perf_counter() - t0
                cur = st.get("status")
                if cur != last:
                    res.status_trace.append((round(now, 2), cur))
                    last = cur
                if cur == "IN_PROGRESS" and res.first_seen_in_progress_s is None:
                    res.first_seen_in_progress_s = now
                if res.allocated_s is None and (cur not in ("IN_QUEUE", None) or (allocated is not None and allocated())):
                    res.allocated_s = now
                if cur in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                    res.final_status = cur
                    res.delay_ms, res.execution_ms = st.get("delayTime"), st.get("executionTime")
                    res.worker_id = st.get("workerId")
                    comp, err = _job_output(st.get("output"))
                    if comp:
                        u = comp.get("usage") or {}
                        res.prompt_tokens, res.completion_tokens = u.get("prompt_tokens"), u.get("completion_tokens")
                        res.served_model = comp.get("model")
                        ch = (comp.get("choices") or [{}])[0]
                        res.text = ((ch.get("message") or {}).get("content") or "")[:200]
                        res.finish_reason = ch.get("finish_reason")
                    if err or cur != "COMPLETED":
                        res.error = redact(err or json.dumps(st.get("error") or cur))[:400]
                        res.error_class = "job_error" if cur == "COMPLETED" else f"job_{cur.lower()}"
                    break
                if alloc_timeout_s is not None and res.allocated_s is None and now > alloc_timeout_s:
                    res.error_class = "allocation_timeout"
                    res.error = f"no worker allocated after {alloc_timeout_s:.0f}s; cancel -> {self.cancel(res.job_id)}"
                    res.final_status = cur
                    break
                since = now - (res.allocated_s or 0.0) if alloc_timeout_s is not None else now
                if (alloc_timeout_s is None or res.allocated_s is not None) and since > timeout_s:
                    what = " from worker allocation" if alloc_timeout_s is not None else ""
                    res.error_class, res.error = "job_timeout", f"no result after {timeout_s:.0f}s{what}; cancel -> {self.cancel(res.job_id)}"
                    res.final_status = cur
                    break
                if on_poll is not None:
                    try:
                        on_poll()
                    except Exception:
                        self.cancel(res.job_id)
                        raise
                sleep(poll_s)
        except (requests.RequestException, ValueError) as exc:
            res.error_class, res.error = "transport", redact(str(exc))[:300]
            if res.job_id:
                self.cancel(res.job_id)
        res.client_total_s = time.perf_counter() - t0
        return res

    def stream(self, body: dict, phase: str, idx: int, prompt_id: str, concurrency: int = 1, timeout: float = 300,
               key: Optional[str] = None, cancel_after_chunks: Optional[int] = None) -> Result:
        res = Result(phase, idx, prompt_id, concurrency, t_start_utc=datetime.now(timezone.utc).isoformat())
        t0 = time.perf_counter()
        chunk_times: list[float] = []
        try:
            with self.s.post(self.chat_url, headers=self._h(key), json=body, stream=True, timeout=timeout) as r:
                res.status, res.headers_s = r.status_code, time.perf_counter() - t0
                if r.status_code != 200:
                    res.error, res.error_class = redact(r.text[:400]), f"http_{r.status_code}"
                    res.latency_s = time.perf_counter() - t0
                    return res
                other: list[str] = []  # non-SSE lines: the worker relays vLLM errors as a plain JSON body
                for raw in sse_lines(r):
                    if not raw.startswith("data:"):
                        if len(other) < 20:
                            other.append(raw)
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    if ev.get("model"):
                        res.served_model = ev["model"]
                    for ch in ev.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece:
                            now = time.perf_counter() - t0
                            chunk_times.append(now)
                            res.text_chars += len(piece)
                            if res.ttft_s is None:
                                res.ttft_s = now
                        if ch.get("finish_reason"):
                            res.finish_reason = ch["finish_reason"]
                    if ev.get("usage"):
                        res.prompt_tokens = ev["usage"].get("prompt_tokens")
                        res.completion_tokens = ev["usage"].get("completion_tokens")
                    if ev.get("error"):
                        res.error, res.error_class = redact(json.dumps(ev["error"])[:400]), "stream_error"
                    if time.perf_counter() - t0 > timeout:  # read timeouts are per read; also cap the total
                        res.error_class, res.error = "client_timeout", f"stream exceeded {timeout:.0f}s in total"
                        break
                    if cancel_after_chunks and len(chunk_times) >= cancel_after_chunks:
                        res.error_class, res.error = "client_cancelled", f"client closed the stream after {len(chunk_times)} chunks"
                        break
                if res.error_class is None and not chunk_times and not res.completion_tokens:
                    # HTTP 200 without a single token or usage: a failure, not a success.
                    res.error_class = "empty_stream"
                    res.error = redact(" ".join(other))[:400] or "HTTP 200 with no tokens and no usage"
        except (requests.Timeout, urllib3.exceptions.ReadTimeoutError, TimeoutError) as exc:
            res.error_class, res.error = "client_timeout", redact(str(exc))[:300]
        except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as exc:
            res.error_class, res.error = "transport", redact(str(exc))[:300]
        res.latency_s = time.perf_counter() - t0
        res.content_chunks = len(chunk_times)
        if chunk_times:
            res.first_to_last_chunk_s = chunk_times[-1] - chunk_times[0]
            gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
            res.max_gap_s = max(gaps) if gaps else 0.0
        if res.completion_tokens and res.error_class is None:
            if res.first_to_last_chunk_s and res.completion_tokens > 1 and res.content_chunks > 1:
                res.decode_tok_s = (res.completion_tokens - 1) / res.first_to_last_chunk_s
            res.e2e_tok_s = res.completion_tokens / res.latency_s
        return res


# ── health sampler (evidence of worker state; never starts a worker) ──

class HealthSampler(threading.Thread):
    def __init__(self, ep: Endpoint, out: Callable[[dict], None], every_s: float = 2.0):
        super().__init__(daemon=True)
        self.ep, self.out, self.every = ep, out, every_s
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            rec = {"type": "health", "utc": datetime.now(timezone.utc).isoformat(), "mono": time.perf_counter()}
            try:
                rec["health"] = self.ep.health(timeout=8)
            except requests.RequestException as exc:
                rec["error"] = redact(str(exc))[:200]
            self.out(rec)
            self.stop_evt.wait(self.every)

    def stop(self):
        self.stop_evt.set()


_GQL = "https://api.runpod.io/graphql"
_GPU_QUERY = "query($id: String!) { pod(input: {podId: $id}) { id machine { gpuDisplayName } runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } } }"


def worker_snapshot(endpoint_id: str, key: str, session: Optional[requests.Session] = None) -> dict:
    """Read-only: which GPU the worker runs on and, if RunPod exposes it for
    serverless workers, GPU utilisation and memory. Missing data is recorded
    as missing, never estimated."""
    s = session or requests.Session()
    snap = {"utc": datetime.now(timezone.utc).isoformat(), "workers": [], "gpu_metrics": "not attempted"}
    try:
        ep = endpoint_settings(endpoint_id, key, s)
    except requests.RequestException as exc:
        snap["error"] = redact(str(exc))[:200]
        return snap
    keep = ("id", "gpuTypeId", "gpuCount", "costPerHr", "adjustedCostPerHr", "desiredStatus", "machineId", "lastStatusChange")
    for w in ep.get("workers") or []:
        rec = {k: w.get(k) for k in keep if k in w}
        rec["gpu"] = (w.get("machine") or {}).get("gpuDisplayName") or (w.get("gpu") or {}).get("displayName")
        try:
            r = s.post(_GQL, json={"query": _GPU_QUERY, "variables": {"id": w.get("id")}},
                       headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}, timeout=15)
            pod = ((r.json() or {}).get("data") or {}).get("pod") if r.ok else None
            runtime = (pod or {}).get("runtime")
            rec["gpus"] = (runtime or {}).get("gpus")
            rec["gpu"] = rec["gpu"] or ((pod or {}).get("machine") or {}).get("gpuDisplayName")
            snap["gpu_metrics"] = "reported" if rec["gpus"] else "not exposed"
        except (requests.RequestException, ValueError):
            snap["gpu_metrics"] = "not exposed"
        snap["workers"].append(rec)
    return snap


class WorkerSampler(threading.Thread):
    def __init__(self, endpoint_id: str, key: str, out: Callable[[dict], None], every_s: float = 5.0):
        super().__init__(daemon=True)
        self.args, self.out, self.every = (endpoint_id, key), out, every_s
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            self.out({"type": "worker", **worker_snapshot(*self.args)})
            self.stop_evt.wait(self.every)

    def stop(self):
        self.stop_evt.set()


def workers_up(h: dict) -> int:
    w = (h or {}).get("workers") or {}
    return sum(int(w.get(k) or 0) for k in ("idle", "running", "initializing", "ready"))


def quiet_check(ep: "Endpoint", readonly_key: Optional[str]) -> Callable[[], dict]:
    """Is nothing running or billing? Returns {"quiet": bool, "kind": ...}.

    "zero": health reports no workers at all.
    "standby": RunPod keeps a standby worker (the endpoint's workersStandby)
      that health reports as idle/ready while its container is stopped. It is
      accepted only when health shows nothing running or initializing and no
      jobs, every listed worker has desiredStatus EXITED, and the account
      spends $0/h. Owner-approved on 2026-10-27, after 7+ minutes of $0/h
      observation.
    Without a Read Only key only "zero" counts (strict)."""
    def check() -> dict:
        h = ep.health()
        w, j = (h.get("workers") or {}), (h.get("jobs") or {})
        if workers_up(h) == 0:
            return {"quiet": True, "kind": "zero", "health": h}
        out = {"quiet": False, "kind": "active", "health": h}
        if not readonly_key or int(w.get("running") or 0) or int(w.get("initializing") or 0) \
                or int(j.get("inQueue") or 0) or int(j.get("inProgress") or 0):
            return out
        try:
            live = endpoint_settings(ep.id, readonly_key)
            rate = read_balance(readonly_key).get("spend_per_hr")
        except (requests.RequestException, ValueError):
            return out
        listed = [(x.get("id"), x.get("desiredStatus")) for x in live.get("workers") or []]
        out.update(listed_workers=listed, spend_per_hr=rate)
        if listed and all(st == "EXITED" for _, st in listed) and rate in (0, 0.0):
            out.update(quiet=True, kind="standby")
        return out
    return check


# ── Serverless placement preflight ────────────────────────────────────

CATALOG = "https://api.runpod.io/v2/catalog/gpus"


def serverless_placement(cfg: dict, key: str, session: Optional[requests.Session] = None) -> dict:
    """Fail closed: is there, right now, a Serverless host whose GPU type is
    in the endpoint's gpu_type_ids AND that advertises an available CUDA
    version in its allowed_cuda_versions (an exact set)? Any error, missing
    field or empty answer counts as no match. Read Only key."""
    e = cfg["endpoint"]
    want_gpus, want_cuda = list(e["gpu_type_ids"]), [str(v) for v in e["allowed_cuda_versions"]]
    out = {"utc": datetime.now(timezone.utc).isoformat(), "gpus": {}, "match": [], "ok": False}
    s = session or requests.Session()
    try:
        r = s.get(CATALOG, params={"include": "AVAILABILITY", "product": "SERVERLESS"}, timeout=20,
                  headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})
        r.raise_for_status()
        gpus = {g.get("id"): g for g in (r.json() or {}).get("gpus") or []}
    except (requests.RequestException, ValueError, AttributeError) as exc:
        out["error"] = redact(str(exc))[:200]
        return out
    for gid in want_gpus:
        g = gpus.get(gid)
        if not g:
            out["gpus"][gid] = {"error": "not in the Serverless catalog"}
            continue
        cuda = [c for c in (g.get("cudaVersions") or []) if isinstance(c, dict)]
        avail = [str(c.get("version")) for c in cuda if c.get("available") is True]
        hit = [v for v in avail if v in want_cuda]
        out["gpus"][gid] = {"availability": g.get("availability"), "cuda_available": avail, "cuda_matching": hit}
        if g.get("availability") not in (None, "NONE") and hit:
            out["match"].append({"gpu": gid, "cuda": hit})
    out["ok"] = bool(out["match"])
    return out


# ── live provider-side monitoring ─────────────────────────────────────

_GQL_BALANCE = "query { myself { clientBalance currentSpendPerHr } }"


def read_balance(key: str, session: Optional[requests.Session] = None) -> dict:
    """Account balance and current spend rate (works with a Read Only key)."""
    s = session or requests.Session()
    r = s.post(_GQL, json={"query": _GQL_BALANCE}, timeout=15,
               headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})
    r.raise_for_status()
    me = ((r.json() or {}).get("data") or {}).get("myself")
    if not me or me.get("clientBalance") is None:
        raise ValueError("no balance in response")
    return {"balance": float(me["clientBalance"]), "spend_per_hr": me.get("currentSpendPerHr")}


class SpendMonitor:
    """Stops a paid run on provider-side evidence, independent of our own
    worst-case bound:
      - the account balance fell by more than max_spend_usd since start
        (RunPod's own accounting; it updates with some delay);
      - the balance could not be read max_failures times in a row
        ("billing cannot be monitored reliably");
      - the endpoint's retried-job counter rose (no retries are expected).
    check() is cheap to call often; it reads the provider at most every
    every_s seconds."""

    def __init__(self, readonly_key: str, max_spend_usd: float, health: Callable[[], dict],
                 every_s: float = 20.0, max_failures: int = 3, session=None, clock=time.monotonic,
                 out: Optional[Callable[[dict], None]] = None):
        self.key, self.max_spend, self.health = readonly_key, max_spend_usd, health
        self.every, self.max_failures, self.s, self.clock, self.out = every_s, max_failures, session, clock, out
        self.lock = threading.Lock()
        self.start_balance: Optional[float] = None
        self.start_retried: Optional[int] = None
        self.failures, self.last_t, self.tripped = 0, None, None
        self.last: dict = {}

    def start(self) -> dict:
        b = read_balance(self.key, self.s)
        h = self.health()
        self.start_balance, self.start_retried = b["balance"], int(((h or {}).get("jobs") or {}).get("retried") or 0)
        self.last_t = self.clock()
        self.last = {"balance": b["balance"], "spent_usd": 0.0, "retried": self.start_retried}
        self._emit("start", b)
        return self.last

    def _emit(self, what, extra):
        if self.out:
            self.out({"type": "spend", "event": what, "utc": datetime.now(timezone.utc).isoformat(), **extra, **self.last})

    def check(self, force: bool = False) -> None:
        with self.lock:
            if self.tripped:
                raise Abort(self.tripped)
            if not force and self.last_t is not None and self.clock() - self.last_t < self.every:
                return
            self.last_t = self.clock()
            try:
                b = read_balance(self.key, self.s)
                self.failures = 0
                spent = round(self.start_balance - b["balance"], 6)
                self.last.update(balance=b["balance"], spent_usd=spent, spend_per_hr=b["spend_per_hr"])
                if spent > self.max_spend:
                    self.tripped = f"provider-reported spend ${spent:.4f} exceeds ${self.max_spend:.2f}"
            except (requests.RequestException, ValueError) as exc:
                self.failures += 1
                self.last.update(balance_error=redact(str(exc))[:120])
                if self.failures >= self.max_failures:
                    self.tripped = f"billing cannot be monitored: balance unreadable {self.failures} times in a row"
            try:
                retried = int(((self.health() or {}).get("jobs") or {}).get("retried") or 0)
                self.last.update(retried=retried)
                if retried > self.start_retried:
                    self.tripped = self.tripped or f"unexpected retries: endpoint retried {retried - self.start_retried} job(s)"
            except requests.RequestException:
                pass
            self._emit("check", {})
            if self.tripped:
                raise Abort(self.tripped)


def validate_startup(r: "JobResult", cfg: dict, max_tokens: int) -> list[str]:
    """The first (cold) job must prove the model loaded and generated."""
    probs = []
    if r.final_status != "COMPLETED" or r.error_class:
        probs.append(f"status {r.final_status} {r.error_class or ''} {r.error or ''}".strip())
    if r.served_model != cfg["model"]["served_name"]:
        probs.append(f"served model {r.served_model!r}, expected {cfg['model']['served_name']!r}")
    if not r.completion_tokens or not r.prompt_tokens:
        probs.append(f"no token usage reported (prompt {r.prompt_tokens}, completion {r.completion_tokens})")
    elif r.completion_tokens != max_tokens:
        probs.append(f"completion_tokens {r.completion_tokens}, expected {max_tokens} (ignore_eos)")
    if not r.text.strip():
        probs.append("empty answer text")
    return probs


ALLOCATED_STATES = ("initializing", "ready", "running", "idle", "unhealthy")  # "throttled" = waiting for capacity


class AllocationProbe:
    """True once endpoint health shows any worker that holds a GPU slot
    (not "throttled", which means waiting for capacity). Reads health at
    most every every_s seconds and remembers a positive answer."""

    def __init__(self, ep: "Endpoint", every_s: float = 5.0, clock=time.monotonic):
        self.ep, self.every, self.clock = ep, every_s, clock
        self.last_t, self.seen = None, False

    def __call__(self) -> bool:
        if self.seen:
            return True
        if self.last_t is not None and self.clock() - self.last_t < self.every:
            return False
        self.last_t = self.clock()
        try:
            w = (self.ep.health(timeout=8) or {}).get("workers") or {}
        except requests.RequestException:
            return False
        self.seen = any(int(w.get(k) or 0) > 0 for k in ALLOCATED_STATES)
        return self.seen


class UnhealthyWatch:
    """During startup: stop if RunPod reports unhealthy workers in
    `limit` consecutive health reads (a crash loop, e.g. CUDA mismatch)."""

    def __init__(self, ep: "Endpoint", limit: int = 3, every_s: float = 10.0, clock=time.monotonic):
        self.ep, self.limit, self.every, self.clock = ep, limit, every_s, clock
        self.streak, self.last_t, self.seen = 0, None, []

    def __call__(self):
        if self.last_t is not None and self.clock() - self.last_t < self.every:
            return
        self.last_t = self.clock()
        try:
            w = (self.ep.health(timeout=8) or {}).get("workers") or {}
        except requests.RequestException:
            return
        bad = int(w.get("unhealthy") or 0)
        self.seen.append(bad)
        self.streak = self.streak + 1 if bad else 0
        if self.streak >= self.limit:
            raise Abort(f"startup failing: {bad} unhealthy worker(s) in {self.streak} consecutive health reads "
                        "(repeated startup failure; check the worker log for CUDA or model errors)")


# ── run ───────────────────────────────────────────────────────────────

def body_for(cfg: dict, prompt: dict, **over) -> dict:
    g = cfg["generation"]
    b = {"model": cfg["model"]["served_name"], "messages": [{"role": "user", "content": prompt["text"]}],
         "max_tokens": g["max_tokens"], "temperature": g["temperature"], "seed": g["seed"], "stream": True,
         "stream_options": {"include_usage": True}}
    if g.get("ignore_eos"):
        b["ignore_eos"] = True  # vLLM extension: fixed output length, so throughput is comparable
    b.update(over)
    return b


def run(cfg: dict, out_root: Path, ep: Endpoint, authorize_paid: bool, max_requests: int, max_wall_s: float,
        max_spend_usd: float, budget: Optional[GpuSession] = None, sleep: Callable[[float], None] = time.sleep,
        health_every_s: float = 2.0, readonly_key: Optional[str] = None, monitor: Optional[SpendMonitor] = None,
        placement: Optional[Callable[[], dict]] = None) -> Path:
    if not authorize_paid:
        raise Abort("refusing: real GPU inference needs --authorize-paid-calls")
    if budget is None:
        raise Abort("refusing: a GPU campaign budget (--gpu-campaign) is required")
    p = plan(cfg)
    if max_requests < p["requests"]["total"]:
        raise Abort(f"--max-requests {max_requests} is below the planned {p['requests']['total']}")
    wc = worst_case_usd(cfg, max_wall_s)
    if wc > max_spend_usd:
        raise Abort(f"worst case ${wc:.2f} (wall {max_wall_s:.0f}s at one worker) exceeds --max-spend-usd {max_spend_usd}")
    if budget.max_wall_s < max_wall_s:
        raise Abort(f"--max-wall-seconds {max_wall_s:.0f} exceeds the campaign's {budget.max_wall_s:.0f}s for {budget.name}")
    if budget.max_requests < p["requests"]["total"]:
        raise Abort(f"campaign allows {budget.max_requests} requests for {budget.name}, plan needs {p['requests']['total']}")
    try:
        budget.open()
    except GpuBudgetExceeded as exc:
        raise Abort(f"GPU campaign budget: {exc}")
    run_dir = out_root / f"{cfg['name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    rec_path = run_dir / "records.jsonl"

    def write(rec: dict):
        with lock, open(rec_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc)
    (run_dir / "metadata.json").write_text(json.dumps({
        "config": cfg, "plan": p, "worst_case_usd": wc, "max_wall_s": max_wall_s, "max_requests": max_requests,
        "gpu_session": budget.state(), "endpoint_id": ep.id, "started_utc": start_utc.isoformat(),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown")}, indent=2) + "\n")
    sent = {"n": 0}
    status = "completed"
    ph = cfg["phases"]
    req_timeout = ph.get("request_timeout_s", 300)

    def guard(what: str, may_run_s: float):
        """Count one GPU request; refuse it if it could end after the wall cap
        or if the provider-side spend monitor has tripped."""
        if monitor is not None:
            monitor.check()
        with lock:
            if sent["n"] >= max_requests:
                raise Abort("request cap reached")
            if time.perf_counter() - started + may_run_s > max_wall_s:
                raise Abort(f"wall-clock cap: {what} could run past {max_wall_s:.0f}s")
            try:
                budget.take(1, what, may_run_s)
            except GpuBudgetExceeded as exc:
                raise Abort(str(exc))
            sent["n"] += 1

    def one(prompt, phase, idx, conc=1, **kw):
        guard(phase, req_timeout)
        r = ep.stream(body_for(cfg, prompt), phase, idx, prompt["id"], conc, timeout=req_timeout, **kw)
        write({"type": "request", **asdict(r)})
        return r

    def job(prompt, phase, idx, max_tokens, timeout_s, poll_s, on_poll=None, alloc_timeout_s=None):
        guard(phase, timeout_s + (alloc_timeout_s or 0))
        body = body_for(cfg, prompt, max_tokens=max_tokens)
        body.pop("stream_options", None)

        def poll_checks():
            if monitor is not None:
                monitor.check()
            if on_poll is not None:
                on_poll()
        r = ep.job(body, phase, idx, prompt["id"], timeout_s=timeout_s, poll_s=poll_s, sleep=sleep, on_poll=poll_checks,
                   alloc_timeout_s=alloc_timeout_s, allocated=AllocationProbe(ep) if alloc_timeout_s is not None else None)
        write({"type": "job", **asdict(r)})
        return r

    prompts = cfg["prompts"]
    sampler = HealthSampler(ep, write, health_every_s)
    wsampler = WorkerSampler(ep.id, readonly_key, write) if readonly_key else None
    try:
        quiet = quiet_check(ep, readonly_key)
        q = quiet()
        write({"type": "preflight", "health": q["health"], "quiet": {k: v for k, v in q.items() if k != "health"},
               "utc": datetime.now(timezone.utc).isoformat()})
        # Startup needs an endpoint with nothing running: wait (bounded) for it.
        deadline = time.perf_counter() + cfg["endpoint"]["idle_timeout_s"] + 120
        while not q["quiet"] and time.perf_counter() < deadline:
            sleep(5)
            q = quiet()
        cold_ok = q["quiet"]
        # "startup": nothing on the endpoint; "startup_standby": only RunPod's
        # stopped standby worker (image already on its host, model not loaded).
        startup_phase = {"zero": "startup", "standby": "startup_standby"}.get(q["kind"], "startup_not_cold")
        write({"type": "startup_condition", "kind": q["kind"], "phase": startup_phase})
        # Never submit a job that RunPod cannot place (fail closed).
        if placement is None and not readonly_key:
            raise Abort("placement preflight needs the Read Only key; refusing to submit blind")
        pl = placement() if placement is not None else serverless_placement(cfg, readonly_key)
        write({"type": "placement", **pl})
        if not pl.get("ok"):
            raise Abort(f"no Serverless host matches GPU {cfg['endpoint']['gpu_type_ids']} with CUDA "
                        f"{cfg['endpoint']['allowed_cuda_versions']}: {pl.get('error') or pl.get('gpus')}")
        if monitor is not None:
            monitor.out = monitor.out or write
            monitor.start()
        sampler.start()
        if wsampler:
            wsampler.start()
        st = ph["startup"]
        watch = UnhealthyWatch(ep, clock=time.monotonic)
        r = job(prompts[0], startup_phase, 0, st["max_tokens"], st["init_timeout_s"],
                st.get("poll_s", 2), on_poll=watch, alloc_timeout_s=st["allocation_timeout_s"])
        if r.error_class:
            raise Abort(f"startup job failed: {r.error_class}: {r.error}")
        bad = validate_startup(r, cfg, st["max_tokens"])
        write({"type": "startup_validation", "problems": bad, "unhealthy_reads": watch.seen})
        if bad:
            raise Abort("startup output invalid: " + "; ".join(bad))
        wt = ph["worker_timing"]
        for i in range(wt["requests"]):
            job(prompts[i % len(prompts)], "worker_timing", i, wt["max_tokens"], wt.get("timeout_s", 120), wt.get("poll_s", 0.5))
        # Warm sequential
        for i in range(ph["warm"]["requests"]):
            one(prompts[i % len(prompts)], "warm", i)
        # Concurrency sweep
        cc = ph["concurrency"]
        for level in cc["levels"]:
            t_level = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                futs = [pool.submit(one, prompts[i % len(prompts)], f"conc_{level}", i, level) for i in range(cc["requests_per_level"])]
                rs = [f.result() for f in futs]
            write({"type": "level", "concurrency": level, "wall_s": time.perf_counter() - t_level,
                   "ok": sum(1 for x in rs if not x.error_class)})
        # Failure behaviour (each observed and recorded, never retried)
        for f in ph.get("failures", []):
            pr = prompts[0]
            kind = f["kind"]
            if kind == "client_timeout":
                guard("fail_client_timeout", f["timeout_s"])
                r = ep.stream(body_for(cfg, pr), "fail_client_timeout", 0, pr["id"], timeout=f["timeout_s"])
            elif kind == "prompt_too_long":
                guard("fail_prompt_too_long", req_timeout)
                long = {"id": "too_long", "text": pr["text"] + "\n" + ("lorem ipsum dolor sit amet " * f["repeat"])}
                r = ep.stream(body_for(cfg, long), "fail_prompt_too_long", 0, "too_long", timeout=req_timeout)
            elif kind == "wrong_model":
                guard("fail_wrong_model", req_timeout)
                r = ep.stream(body_for(cfg, pr, model="not-the-served-model"), "fail_wrong_model", 0, pr["id"], timeout=req_timeout)
            elif kind == "invalid_key":
                guard("fail_invalid_key", req_timeout)
                r = ep.stream(body_for(cfg, pr), "fail_invalid_key", 0, pr["id"], key="invalid-key-for-test", timeout=req_timeout)
            elif kind == "client_cancel":
                guard("fail_client_cancel", req_timeout + f.get("observe_s", 10))
                r = ep.stream(body_for(cfg, pr), "fail_client_cancel", 0, pr["id"], cancel_after_chunks=f["after_chunks"], timeout=req_timeout)
                sleep(f.get("observe_s", 10))  # sampler records whether the job keeps running
            else:
                continue
            write({"type": "request", **asdict(r)})
    except Abort as exc:
        status = f"aborted: {exc}"
        write({"type": "abort", "reason": str(exc), "purge_queue_status": ep.purge_queue()})
    except requests.RequestException as exc:
        status = f"aborted: {redact(str(exc))[:200]}"
        write({"type": "abort", "reason": status, "purge_queue_status": ep.purge_queue()})
    finally:
        # Wait for scale-down: no running or billing worker may remain
        # (a stopped, unbilled standby worker is allowed; see quiet_check).
        deadline = time.perf_counter() + cfg["endpoint"]["idle_timeout_s"] + 180
        final, zero, end_kind = None, False, None
        quiet = quiet_check(ep, readonly_key)
        while time.perf_counter() < deadline:
            try:
                fq = quiet()
                final, end_kind = fq["health"], fq["kind"]
                if fq["quiet"]:
                    zero = True
                    break
            except requests.RequestException:
                final = None
            sleep(5)
        sampler.stop()
        if wsampler:
            wsampler.stop()
        if monitor is not None and monitor.start_balance is not None:
            try:
                monitor.check(force=True)
            except Abort as exc:
                status = status if status != "completed" else f"completed; monitor: {exc}"
        end_utc = datetime.now(timezone.utc)
        gpu_state = budget.close(status)
        (run_dir / "run_state.json").write_text(json.dumps({
            "status": status, "requests_sent": sent["n"], "wall_s": time.perf_counter() - started,
            "started_utc": start_utc.isoformat(), "finished_utc": end_utc.isoformat(),
            "scaled_to_zero": zero, "end_condition": end_kind, "final_health": final, "gpu_session": gpu_state,
            "spend_monitor": monitor.last if monitor is not None else None}, indent=2) + "\n")
    return run_dir


# ── analysis ──────────────────────────────────────────────────────────

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "min": min(xs) if xs else None, "max": max(xs) if xs else None}


def analyse(run_dir: Path) -> dict:
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines()]
    meta = json.loads((run_dir / "metadata.json").read_text())
    state = json.loads((run_dir / "run_state.json").read_text()) if (run_dir / "run_state.json").exists() else {}
    price_s = meta["config"]["price_usd_per_hour"]["value"] / 3600
    reqs = [r for r in recs if r["type"] == "request"]
    for r in reqs:  # records written before the parser fix: 200 with no tokens and no usage is a failure
        if r["error_class"] is None and r["status"] == 200 and not r.get("content_chunks") and not r.get("completion_tokens"):
            r["error_class"], r["error"] = "empty_stream", r.get("error") or "HTTP 200 with no tokens and no usage (body not recorded)"
    ok = lambda r: r["error_class"] is None and r["status"] == 200
    warm = [r for r in reqs if r["phase"] == "warm" and ok(r)]
    out = {"run_state": state, "model": meta["config"]["model"], "gpu": meta["config"]["gpu"],
           "generation": meta["config"]["generation"], "price_usd_per_hour": meta["config"]["price_usd_per_hour"]}
    jobs = [r for r in recs if r["type"] == "job"]
    su = next((r for r in jobs if r["phase"] in ("startup", "startup_standby", "startup_not_cold")), None)
    out["startup"] = su and {
        **{k: su.get(k) for k in ("phase", "final_status", "client_total_s", "allocated_s", "first_seen_in_progress_s", "delay_ms",
                              "execution_ms", "worker_id", "completion_tokens", "error_class", "status_trace")},
        "note": "async /run job on a scaled-to-zero endpoint. delay_ms (RunPod) = submission until a worker took the "
                "job: queueing, image pull if any, container start and model load. Reported apart from warm inference."}
    wt = [r for r in jobs if r["phase"] == "worker_timing" and not r["error_class"]]
    out["worker_timing"] = {"execution_ms": dist([r["execution_ms"] for r in wt]), "delay_ms": dist([r["delay_ms"] for r in wt]),
                            "client_total_s": dist([r["client_total_s"] for r in wt]),
                            "completion_tokens": dist([r["completion_tokens"] for r in wt]),
                            "failed": sum(1 for r in jobs if r["phase"] == "worker_timing" and r["error_class"])}
    out["warm"] = {"latency_s": dist([r["latency_s"] for r in warm]), "ttft_s": dist([r["ttft_s"] for r in warm]),
                   "decode_tok_s": dist([r["decode_tok_s"] for r in warm]), "e2e_tok_s": dist([r["e2e_tok_s"] for r in warm]),
                   "completion_tokens": dist([r["completion_tokens"] for r in warm]), "failed": sum(1 for r in reqs if r["phase"] == "warm" and not ok(r))}
    levels = {}
    walls = {r["concurrency"]: r["wall_s"] for r in recs if r["type"] == "level"}
    for c in sorted(walls):
        rs = [r for r in reqs if r["phase"] == f"conc_{c}"]
        good = [r for r in rs if ok(r)]
        toks = sum(r["completion_tokens"] or 0 for r in good)
        wall = walls[c]
        levels[str(c)] = {"requests": len(rs), "ok": len(good), "wall_s": wall, "output_tokens": toks,
                          "aggregate_output_tok_s": toks / wall if wall else None, "requests_per_s": len(good) / wall if wall else None,
                          "latency_s": dist([r["latency_s"] for r in good]), "ttft_s": dist([r["ttft_s"] for r in good]),
                          "per_request_decode_tok_s": dist([r["decode_tok_s"] for r in good]),
                          "usd_per_1m_output_tokens_busy_basis": (wall * price_s) / toks * 1e6 if toks else None}
    base = levels.get("1", {}).get("aggregate_output_tok_s")
    for c, v in levels.items():
        v["throughput_vs_c1"] = (v["aggregate_output_tok_s"] / base) if base and v["aggregate_output_tok_s"] else None
        v["scaling_efficiency"] = (v["throughput_vs_c1"] / int(c)) if v["throughput_vs_c1"] else None
    out["concurrency"] = levels
    g = out["warm"]["ttft_s"]["p50"]
    w = out["worker_timing"]["execution_ms"]["p50"]
    out["ttft"] = {
        "gateway_ttft_p50_s": g,
        "worker_1token_execution_p50_s": w / 1000 if w is not None else None,
        "gateway_minus_worker_p50_s": (g - w / 1000) if g is not None and w is not None else None,
        "definitions": {
            "gateway_ttft": "client-measured: request sent to the RunPod OpenAI route until the first streamed content "
                            "chunk arrives (gateway, network, queue, worker, vLLM prefill + first token). Same prompts, warm worker.",
            "worker_1token_execution": "RunPod executionTime of an async job asking for max_tokens=1: time inside the worker "
                                       "(its local proxy plus vLLM prefill and one token). An upper bound on in-engine TTFT.",
            "in_engine_ttft": "not measured: vLLM's own TTFT histogram is on its /metrics route, which the worker does not "
                              "expose (the vLLM server listens on loopback and the worker proxy only relays JSON)."}}
    streamed = [r for r in reqs if ok(r) and r["completion_tokens"]]
    out["streaming"] = {
        "median_content_chunks": pct([r["content_chunks"] for r in streamed], 0.5),
        "median_completion_tokens": pct([r["completion_tokens"] for r in streamed], 0.5),
        "median_first_to_last_chunk_share": pct([(r["first_to_last_chunk_s"] or 0) / r["latency_s"] for r in streamed if r["latency_s"]], 0.5),
        "note": "token-level streaming if chunks ~ completion tokens and first-to-last spans most of the latency; "
                "a buffered proxy shows few chunks arriving together at the end"}
    out["failures"] = [{k: r[k] for k in ("phase", "status", "error_class", "latency_s", "ttft_s", "content_chunks", "error")}
                       for r in reqs if r["phase"].startswith("fail_") or not ok(r)]
    health = [r for r in recs if r["type"] == "health" and r.get("health")]
    workers = [r for r in recs if r["type"] == "worker"]
    gpu_names = sorted({w.get("gpu") or w.get("gpuTypeId") for r in workers for w in r["workers"] if (w.get("gpu") or w.get("gpuTypeId"))})
    utils = [g for r in workers for w in r["workers"] for g in (w.get("gpus") or [])]
    out["gpu"] = {"observed_types": gpu_names, "samples": len(workers),
                  "utilization_pct": dist([g.get("gpuUtilPercent") for g in utils]) if utils else "not exposed",
                  "memory_util_pct": dist([g.get("memoryUtilPercent") for g in utils]) if utils else "not exposed"}
    out["health_samples"] = len(health)
    out["max_workers_seen"] = max((workers_up(r["health"]) for r in health), default=None)
    # Cancel: did the job keep running after the client closed the stream?
    cancel = next((r for r in reqs if r["phase"] == "fail_client_cancel"), None)
    if cancel:
        t = datetime.fromisoformat(cancel["t_start_utc"]) + timedelta(seconds=cancel["latency_s"])
        after = [r for r in health if datetime.fromisoformat(r["utc"]) > t]
        out["cancel_observation"] = [{"s_after_close": round((datetime.fromisoformat(r["utc"]) - t).total_seconds(), 1),
                                      "jobs_in_progress": ((r["health"].get("jobs") or {}).get("inProgress"))} for r in after[:8]]
    gen_tokens = sum(r["completion_tokens"] or 0 for r in reqs if ok(r))
    gen_tokens += sum(r["completion_tokens"] or 0 for r in jobs if not r["error_class"])
    out["totals"] = {"requests": len(reqs) + len(jobs), "streamed": len(reqs), "async_jobs": len(jobs),
                     "ok": sum(1 for r in reqs if ok(r)) + sum(1 for r in jobs if not r["error_class"]),
                     "generated_tokens": gen_tokens, "wall_s": state.get("wall_s")}
    if (run_dir / "billing.json").exists():
        b = json.loads((run_dir / "billing.json").read_text())
        out["billing"] = b
        if b.get("amount_usd") is not None and gen_tokens:
            out["billing"]["usd_per_1m_generated_tokens_all_in"] = b["amount_usd"] / gen_tokens * 1e6
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def fetch_billing(run_dir: Path, key: str, session: Optional[requests.Session] = None) -> dict:
    """Provider-side charges for the endpoint over the run window (hour buckets)."""
    s = session or requests.Session()
    meta = json.loads((run_dir / "metadata.json").read_text())
    state = json.loads((run_dir / "run_state.json").read_text())
    start = datetime.fromisoformat(state["started_utc"]).replace(minute=0, second=0, microsecond=0)
    end = datetime.fromisoformat(state["finished_utc"]) + timedelta(hours=1)
    params = {"endpointId": meta["endpoint_id"], "bucketSize": "hour", "grouping": "endpointId",
              "startTime": start.isoformat().replace("+00:00", "Z"), "endTime": end.isoformat().replace("+00:00", "Z")}
    r = s.get(f"{REST}/billing/endpoints", params=params, headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    rows = r.json()
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    out = {"window": params, "records": rows, "fetched_utc": datetime.now(timezone.utc).isoformat(),
           "amount_usd": sum(float(x.get("amount") or 0) for x in rows),
           "time_billed_s": sum(float(x.get("timeBilledMs") or 0) for x in rows) / 1000,
           "note": "hour buckets: may include other use of this endpoint in the same hours; billing can lag"}
    (run_dir / "billing.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def endpoint_settings(endpoint_id: str, key: str, session: Optional[requests.Session] = None) -> dict:
    s = session or requests.Session()
    r = s.get(f"{REST}/endpoints/{endpoint_id}", params={"includeTemplate": "true", "includeWorkers": "true"},
              headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    ep = r.json()
    t = ep.get("template") or {}
    if ep.get("templateId") and not t.get("imageName"):
        # The endpoint's embedded template omits imageName (seen 2026-10-27);
        # the template route has it.
        rt = s.get(f"{REST}/templates/{ep['templateId']}", headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}, timeout=20)
        rt.raise_for_status()
        ep["template"] = {**t, **rt.json()}
    return ep


def _same(got, want) -> bool:
    if got is None:
        return False
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return str(got).strip().lower() == str(want).strip().lower()


def check_settings(ep: dict, cfg: dict) -> list[str]:
    """Compare the live endpoint with the approved deployment; returns problems."""
    want, probs = cfg["endpoint"], []
    if ep.get("workersMin") != 0:
        probs.append(f"workersMin is {ep.get('workersMin')}, must be 0")
    if ep.get("workersMax") != want["max_workers"]:
        probs.append(f"workersMax is {ep.get('workersMax')}, approved {want['max_workers']}")
    if ep.get("idleTimeout") is not None and ep["idleTimeout"] > want["idle_timeout_s"]:
        probs.append(f"idleTimeout {ep['idleTimeout']}s exceeds the approved {want['idle_timeout_s']}s")
    gpus = set(ep.get("gpuTypeIds") or [])
    if not gpus or not gpus <= set(want["gpu_type_ids"]):
        probs.append(f"gpuTypeIds {sorted(gpus)} not within the approved {want['gpu_type_ids']}")
    if (ep.get("gpuCount") or 1) != 1:
        probs.append(f"gpuCount {ep.get('gpuCount')}, must be 1")
    env = {**((ep.get("template") or {}).get("env") or {}), **(ep.get("env") or {})}
    for k, v in want["env"].items():
        if not _same(env.get(k), v):
            probs.append(f"env {k}={env.get(k)!r}, expected {v!r}")
    image = (ep.get("template") or {}).get("imageName")
    if image != want["image"]:
        probs.append(f"image {image}, expected {want['image']}")
    ex = ep.get("executionTimeoutMs")
    if ex is None or ex > want["execution_timeout_s"] * 1000:
        probs.append(f"executionTimeoutMs {ex}, must be <= {want['execution_timeout_s'] * 1000}")
    # The image is built on CUDA 13.0: a host with an older driver cannot start it.
    cuda = ep.get("allowedCudaVersions")
    if not cuda or not set(cuda) <= set(want["allowed_cuda_versions"]):
        probs.append(f"allowedCudaVersions {cuda}, must be within {want['allowed_cuda_versions']}")
    if ep.get("networkVolumeId") or ep.get("networkVolumeIds"):
        probs.append("a network volume is attached; the approved deployment has none")
    if (ep.get("computeType") or "GPU") != "GPU":
        probs.append(f"computeType {ep.get('computeType')}, must be GPU")
    disk = (ep.get("template") or {}).get("containerDiskInGb")
    if "container_disk_gb" in want and disk != want["container_disk_gb"]:
        probs.append(f"container disk {disk} GB, expected {want['container_disk_gb']}")
    return probs


def check_campaign(campaign: dict, cfg: dict) -> None:
    """The campaign must price and time GPU use exactly like this benchmark."""
    e = cfg["endpoint"]
    for ck, want in (("price_usd_per_hour", cfg["price_usd_per_hour"]["value"]), ("idle_timeout_s", e["idle_timeout_s"]),
                     ("cold_start_allowance_s", e["cold_start_allowance_s"]), ("execution_timeout_s", e["execution_timeout_s"])):
        if float(campaign[ck]) != float(want):
            raise SystemExit(f"campaign {ck}={campaign[ck]} differs from the benchmark config ({want})")
    cp = campaign_plan(campaign)
    if cp["sessions"].get("gpu_bench", {}).get("gpu_requests") != plan(cfg)["requests"]["total"]:
        raise SystemExit("campaign plan for gpu_bench does not match this benchmark's planned requests")
    if cp["problems"]:
        raise SystemExit("campaign plan exceeds its caps: " + "; ".join(cp["problems"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/gpu_bench"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-endpoint", action="store_true", help="read-only: compare live endpoint settings with the config")
    ap.add_argument("--authorize-paid-calls", action="store_true")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-wall-seconds", type=float)
    ap.add_argument("--max-spend-usd", type=float)
    ap.add_argument("--gpu-campaign", type=Path, help="campaign caps shared with e2e_vllm (config/experiments/runpod_campaign.yaml)")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--billing", type=Path)
    a = ap.parse_args()
    if a.report:
        print(json.dumps(analyse(a.report), indent=2))
        return
    if a.billing:
        key = os.environ.get(READONLY_KEY_ENV, "")
        if not key:
            raise SystemExit(f"{READONLY_KEY_ENV} is required for billing")
        b = fetch_billing(a.billing, key)
        print(json.dumps({k: b[k] for k in ("amount_usd", "time_billed_s", "note")}, indent=2))
        analyse(a.billing)
        return
    cfg = load_config(a.config)
    if a.dry_run:
        p = plan(cfg)
        wall = a.max_wall_seconds or cfg["phases"]["suggested_max_wall_s"]
        print(json.dumps({**p, "suggested_max_wall_s": wall, "worst_case_usd_at_that_wall": round(worst_case_usd(cfg, wall), 4)}, indent=2))
        if a.gpu_campaign:
            campaign = load_campaign(a.gpu_campaign)
            print(json.dumps({"gpu_campaign": campaign_plan(campaign)}, indent=2))
            check_campaign(campaign, cfg)
        return
    endpoint_id = os.environ.get(ENDPOINT_ENV, "").strip()
    if a.check_endpoint:
        key = os.environ.get(READONLY_KEY_ENV, "")
        if not (key and endpoint_id):
            raise SystemExit(f"{READONLY_KEY_ENV} and {ENDPOINT_ENV} are required")
        live = endpoint_settings(endpoint_id, key)
        probs = check_settings(live, cfg)
        shown = {k: live.get(k) for k in ("workersMin", "workersMax", "idleTimeout", "gpuTypeIds", "gpuCount",
                                          "allowedCudaVersions", "minCudaVersion", "executionTimeoutMs", "scalerType", "scalerValue")}
        shown["workers_now"] = len(live.get("workers") or [])
        print(json.dumps(shown, indent=2))
        print("OK: settings match the approved deployment" if not probs else "\n".join("FAIL: " + p for p in probs))
        raise SystemExit(1 if probs else 0)
    key = os.environ.get(INFERENCE_KEY_ENV, "")
    if not (key and endpoint_id):
        raise SystemExit(f"{INFERENCE_KEY_ENV} and {ENDPOINT_ENV} are required")
    if a.max_requests is None or a.max_wall_seconds is None or a.max_spend_usd is None or a.gpu_campaign is None:
        raise SystemExit("--max-requests, --max-wall-seconds, --max-spend-usd and --gpu-campaign are required")
    campaign = load_campaign(a.gpu_campaign)
    check_campaign(campaign, cfg)
    try:
        ro = os.environ.get(READONLY_KEY_ENV) or ""
        if not ro:
            raise SystemExit(f"{READONLY_KEY_ENV} is required: live spend monitoring is mandatory for paid runs")
        ep = Endpoint(endpoint_id, key, cfg["api_base"])
        d = run(cfg, a.out, ep, a.authorize_paid_calls,
                a.max_requests, a.max_wall_seconds, a.max_spend_usd, budget=GpuSession(campaign, "gpu_bench"),
                readonly_key=ro, monitor=SpendMonitor(ro, a.max_spend_usd, ep.health))
    except Abort as exc:
        raise SystemExit(f"refused: {exc}")
    print(f"results: {d}")
    print(json.dumps(json.loads((d / "run_state.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
