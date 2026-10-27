"""One GPU budget for every phase that can reach the RunPod endpoint.

The direct benchmark (startup probe, worker-timing jobs, warm, concurrency,
failure probes) and the QuorumAI end-to-end phases (warm-up, councils,
fallback, cancel) run as separate processes. Each process is a *session*
in one ledger file, so the campaign-wide caps hold across all of them:

  max_gpu_requests   every request sent to the endpoint's inference routes,
                     counted when it is sent (failure probes and requests the
                     gateway rejects count too). Nothing in the harness or the
                     backend retries; if anything did, each attempt would need
                     its own take().
  max_gpu_wall_s     sum of the sessions' wall-clock caps
  max_gpu_spend_usd  sum of the sessions' worst-case GPU cost

With max workers = 1, billed GPU time in a session is at most
(session wall + cold-start allowance + execution timeout + idle timeout): a
worker can still be loading the model after an abort, a job the gateway did
not cancel can run until the execution timeout, then the worker idles. A session
reserves that worst case before it sends anything. When it finishes, the
reservation is replaced by the same formula on its real wall time. A session
that crashes keeps its full reservation (fail closed). This is a bound, not a
measurement: billed amounts come from RunPod's billing API.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


class GpuBudgetExceeded(RuntimeError):
    pass


def load_campaign(path: Path) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    for k in ("name", "ledger", "price_usd_per_hour", "idle_timeout_s", "cold_start_allowance_s", "execution_timeout_s",
              "max_gpu_requests", "max_gpu_wall_s", "max_gpu_spend_usd", "sessions"):
        if k not in c:
            raise ValueError(f"campaign config missing {k}")
    return c


def session_worst_case_usd(c: dict, max_wall_s: float) -> float:
    secs = max_wall_s + c["cold_start_allowance_s"] + c["execution_timeout_s"] + c["idle_timeout_s"]
    return secs * c["price_usd_per_hour"] / 3600


def campaign_plan(c: dict) -> dict:
    """Planned GPU requests, wall time and worst-case cost of every session."""
    rows = {name: {"gpu_requests": s["gpu_requests"], "max_wall_s": s["max_wall_s"],
                   "worst_case_usd": round(session_worst_case_usd(c, s["max_wall_s"]), 4)}
            for name, s in c["sessions"].items()}
    tot = {"gpu_requests": sum(r["gpu_requests"] for r in rows.values()),
           "max_wall_s": sum(r["max_wall_s"] for r in rows.values()),
           "worst_case_usd": round(sum(r["worst_case_usd"] for r in rows.values()), 4)}
    probs = []
    if tot["gpu_requests"] > c["max_gpu_requests"]:
        probs.append(f"planned {tot['gpu_requests']} GPU requests > cap {c['max_gpu_requests']}")
    if tot["max_wall_s"] > c["max_gpu_wall_s"]:
        probs.append(f"planned wall {tot['max_wall_s']}s > cap {c['max_gpu_wall_s']}s")
    if tot["worst_case_usd"] > c["max_gpu_spend_usd"]:
        probs.append(f"worst case ${tot['worst_case_usd']} > cap ${c['max_gpu_spend_usd']}")
    return {"sessions": rows, "total": tot, "caps": {k: c[k] for k in ("max_gpu_requests", "max_gpu_wall_s", "max_gpu_spend_usd")},
            "problems": probs}


class GpuSession:
    """A reservation in the campaign ledger for one process (thread-safe)."""

    def __init__(self, campaign: dict, session: str, ledger_path: Optional[Path] = None,
                 clock=time.perf_counter):
        if session not in campaign["sessions"]:
            raise GpuBudgetExceeded(f"session {session!r} is not in the campaign plan")
        self.c, self.name, self.clock = campaign, session, clock
        self.spec = campaign["sessions"][session]
        self.path = Path(ledger_path or campaign["ledger"])
        self.max_wall_s = float(self.spec["max_wall_s"])
        self.max_requests = int(self.spec["gpu_requests"])
        self.requests = 0
        self.lock = threading.Lock()
        self.started: Optional[float] = None
        self.key: Optional[str] = None

    # ledger file (single writer at a time is enough: sessions run one after another)
    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"campaign": self.c["name"], "sessions": []}

    def _save(self, led: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(led, indent=2) + "\n")
        os.replace(tmp, self.path)

    @staticmethod
    def totals(led: dict) -> dict:
        s = led["sessions"]
        return {"gpu_requests": sum(x["gpu_requests"] for x in s), "wall_s": sum(x["wall_charged_s"] for x in s),
                "usd": sum(x["usd_charged"] for x in s)}

    def open(self) -> "GpuSession":
        led = self._load()
        if led.get("campaign") != self.c["name"]:
            raise GpuBudgetExceeded(f"ledger {self.path} belongs to campaign {led.get('campaign')!r}")
        t = self.totals(led)
        reserve = session_worst_case_usd(self.c, self.max_wall_s)
        if t["usd"] + reserve > self.c["max_gpu_spend_usd"] + 1e-9:
            raise GpuBudgetExceeded(f"GPU spend: ${t['usd']:.4f} already charged + ${reserve:.4f} reserved for {self.name} "
                                    f"> cap ${self.c['max_gpu_spend_usd']}")
        if t["wall_s"] + self.max_wall_s > self.c["max_gpu_wall_s"]:
            raise GpuBudgetExceeded(f"GPU wall: {t['wall_s']:.0f}s used + {self.max_wall_s:.0f}s for {self.name} "
                                    f"> cap {self.c['max_gpu_wall_s']}s")
        if t["gpu_requests"] >= self.c["max_gpu_requests"]:
            raise GpuBudgetExceeded(f"GPU requests: campaign cap {self.c['max_gpu_requests']} already reached")
        # Requests this session may send: its plan, but never past the campaign cap.
        self.max_requests = min(self.max_requests, self.c["max_gpu_requests"] - t["gpu_requests"])
        self.key = f"{self.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
        led["sessions"].append({"key": self.key, "session": self.name, "status": "open", "gpu_requests": 0,
                                "wall_charged_s": self.max_wall_s, "usd_charged": reserve,
                                "opened_utc": datetime.now(timezone.utc).isoformat()})
        self._save(led)
        self.started = self.clock()
        return self

    def elapsed(self) -> float:
        return self.clock() - self.started if self.started is not None else 0.0

    def take(self, n: int = 1, what: str = "", may_run_s: float = 0.0) -> None:
        """Count n GPU requests about to be sent. Refuses if that passes the
        request cap or if they could still be running at the wall cap."""
        with self.lock:
            if self.started is None:
                raise GpuBudgetExceeded("GPU session not opened")
            if self.requests + n > self.max_requests:
                raise GpuBudgetExceeded(f"GPU request cap reached ({self.requests}/{self.max_requests}) before {what or 'request'}")
            if self.elapsed() + may_run_s > self.max_wall_s:
                raise GpuBudgetExceeded(f"GPU wall cap: {self.elapsed():.0f}s + up to {may_run_s:.0f}s for {what or 'request'} "
                                        f"> {self.max_wall_s:.0f}s")
            self.requests += n
            self._update(status="open")

    def _update(self, **kw) -> None:
        led = self._load()
        for x in led["sessions"]:
            if x["key"] == self.key:
                x["gpu_requests"] = self.requests
                x.update(kw)
        self._save(led)

    def close(self, status: str) -> dict:
        wall = self.elapsed()  # real wall time, including any wait for scale-down
        with self.lock:
            self._update(status=status, wall_charged_s=round(wall, 3), usd_charged=session_worst_case_usd(self.c, wall),
                         closed_utc=datetime.now(timezone.utc).isoformat())
        return self.state()

    def state(self) -> dict:
        led = self._load()
        return {"session": self.name, "requests": self.requests, "max_requests": self.max_requests,
                "elapsed_s": round(self.elapsed(), 1), "max_wall_s": self.max_wall_s, "campaign_totals": self.totals(led),
                "campaign_caps": {k: self.c[k] for k in ("max_gpu_requests", "max_gpu_wall_s", "max_gpu_spend_usd")}}


def release_reservation(ledger_path: Path, session_key: str, reason: str, evidence: dict, approved_by: str) -> dict:
    """Append an audited credit that cancels a closed session's charged wall
    time and spend (never its request count). The original session record is
    left exactly as it was; totals() simply adds the negative entry. Only for
    a session whose provider-side billing is confirmed $0."""
    path = Path(ledger_path)
    led = json.loads(path.read_text())
    target = next((x for x in led["sessions"] if x.get("key") == session_key), None)
    if target is None:
        raise GpuBudgetExceeded(f"no session {session_key!r} in {path}")
    if target.get("status") == "open":
        raise GpuBudgetExceeded("cannot release an open session")
    if any(x.get("releases") == session_key for x in led["sessions"]):
        raise GpuBudgetExceeded(f"{session_key} was already released")
    entry = {"key": f"release-{session_key}", "session": "adjustment", "status": "reservation released",
             "releases": session_key, "gpu_requests": 0, "wall_charged_s": -target["wall_charged_s"],
             "usd_charged": -target["usd_charged"], "reason": reason, "evidence": evidence,
             "approved_by": approved_by, "recorded_utc": datetime.now(timezone.utc).isoformat()}
    led["sessions"].append(entry)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=2) + "\n")
    os.replace(tmp, path)
    return entry
