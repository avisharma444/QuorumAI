"""Minimal client for the CouncilAI HTTP API used by the evaluation runner."""
from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional

import requests

_REDACT = [
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[redacted-jwt]"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|nvapi-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{16,})"), "[redacted-key]"),
    (re.compile(r'("?(?:password|api[_-]?key|token|secret)"?\s*[:=]\s*)"?[^",\s}]+', re.I), r"\1[redacted]"),
]


def redact(text: str) -> str:
    """Remove tokens, API keys and passwords from error text before it is stored."""
    for rx, repl in _REDACT:
        text = rx.sub(repl, text)
    return text


class APIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {redact(message)[:300]}")
        self.status = status


@dataclass
class QueryOutcome:
    status: int
    body: Optional[dict]
    latency_ms: float
    trace_id: str
    error: Optional[str]
    retries_429: int


class Client:
    def __init__(self, base: str, timeout: float = 180.0, session: Optional[requests.Session] = None):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.http = session or requests.Session()
        self.token: Optional[str] = None

    def fresh(self) -> "Client":
        """A new client (new user session) against the same API."""
        return Client(self.base, self.timeout)

    def health(self) -> dict:
        r = self.http.get(f"{self.base}/health", timeout=10)
        if r.status_code != 200:
            raise APIError(r.status_code, r.text)
        return r.json()

    def _post(self, path: str, max_429: int = 60, **kw) -> tuple[requests.Response, int]:
        waits = 0
        while True:
            r = self.http.post(f"{self.base}{path}", timeout=self.timeout, **kw)
            if r.status_code != 429 or waits >= max_429:
                return r, waits
            waits += 1
            time.sleep(min(30, int(r.headers.get("Retry-After", "1") or 1)))

    def register(self) -> str:
        """A fresh user with random credentials (never stored)."""
        username = "eval" + secrets.token_hex(6)
        r, _ = self._post("/api/v1/register", json={"username": username, "password": secrets.token_urlsafe(18)})
        if r.status_code != 201:
            raise APIError(r.status_code, r.text)
        self.token = r.json()["token"]
        return username

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def ingest(self, filename: str, data: bytes) -> dict:
        r, _ = self._post("/api/v1/ingest", files={"file": (filename, data, "text/markdown")}, headers=self._auth())
        if r.status_code != 200:
            raise APIError(r.status_code, r.text)
        return r.json()

    def query(self, body: dict) -> QueryOutcome:
        t = time.perf_counter()
        try:
            r, waits = self._post("/api/v1/query", json=body, headers=self._auth())
        except requests.RequestException as exc:
            return QueryOutcome(0, None, (time.perf_counter() - t) * 1000, "", redact(str(exc))[:300], 0)
        ms = (time.perf_counter() - t) * 1000
        trace = next((v for k, v in r.headers.items() if k.lower() == "x-trace-id"), "")
        try:
            data = r.json()
        except ValueError:
            data = None
        err = None if r.status_code == 200 else redact((data or {}).get("error", r.text) if isinstance(data, dict) else r.text)[:300]
        return QueryOutcome(r.status_code, data if r.status_code == 200 else None, ms, trace, err, waits)

    def metrics_text(self, url: str) -> Optional[str]:
        try:
            r = self.http.get(url, timeout=10)
            return r.text if r.status_code == 200 else None
        except requests.RequestException:
            return None
