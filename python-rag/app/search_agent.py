"""Web search grounding: DuckDuckGo search plus HTML paragraph scraping.

Reproduces the reference WebSearchAgent (DuckDuckGo results, then the first
three substantive <p> paragraphs of each page, falling back to the search
snippet). Added safeguards the reference lacks:

* Only http(s) URLs whose host resolves exclusively to public addresses are
  fetched, and every redirect hop is re-checked, so a search result cannot
  make this service read internal resources (SSRF).
* Connect/read timeouts, a per-page byte limit, an HTML content-type check,
  and an overall time budget for scraping.
* Failures are reported in `status`/`error` instead of being swallowed, so the
  caller can tell "no grounding" from "grounded".

Known limit: the host is resolved for the check and again by `requests` when
connecting, so a DNS-rebinding attacker could still race the two lookups.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; CouncilAI-reproduction/0.1)"
MAX_QUERY_CHARS = 400

SearchFn = Callable[[str, int], list[dict]]
Resolver = Callable[[str], list[str]]


def ddg_search(query: str, max_results: int) -> list[dict]:
    """Live DuckDuckGo text search (returns dicts with title, href, body)."""
    from duckduckgo_search import DDGS  # imported lazily so tests need no network library state

    with DDGS(timeout=10) as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def resolve_host(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


def is_public_url(url: str, resolver: Resolver = resolve_host) -> bool:
    """True only for http(s) URLs whose every resolved address is public."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        addresses = [parts.hostname] if _is_ip(parts.hostname) else resolver(parts.hostname)
    except (OSError, UnicodeError):
        return False
    if not addresses:
        return False
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if not ip.is_global or ip.is_multicast:
            return False
    return True


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_paragraphs(html: str, limit: int = 3, min_len: int = 40) -> str | None:
    """First `limit` <p> texts longer than `min_len`, scripts/styles removed."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()
    paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
    substantive = [p for p in paragraphs if len(p) > min_len]
    return "\n".join(substantive[:limit]) if substantive else None


@dataclass
class SearchOutcome:
    status: str  # "ok" | "no_results" | "search_failed"
    results: list[str] = field(default_factory=list)  # reference format, sent to the LLMs
    sources: list[dict] = field(default_factory=list)  # attribution metadata
    error: str | None = None

    def as_dict(self) -> dict:
        return {"status": self.status, "results": self.results, "sources": self.sources, "error": self.error}


class WebSearchAgent:
    def __init__(
        self,
        search_fn: SearchFn = ddg_search,
        session: requests.Session | None = None,
        resolver: Resolver = resolve_host,
        connect_timeout: float = 3.0,
        read_timeout: float = 5.0,
        max_page_bytes: int = 1_000_000,
        max_redirects: int = 3,
        scrape_budget: float = 12.0,
    ):
        self.search_fn = search_fn
        self.session = session or requests.Session()
        self.resolver = resolver
        self.timeout = (connect_timeout, read_timeout)
        self.max_page_bytes = max_page_bytes
        self.max_redirects = max_redirects
        self.scrape_budget = scrape_budget

    def search(self, query: str, max_results: int = 3) -> SearchOutcome:
        query = query.strip()[:MAX_QUERY_CHARS]
        try:
            hits = self.search_fn(query, max_results)
        except Exception as exc:  # network errors, rate limits, library errors
            logger.warning("DuckDuckGo search failed: %s", exc)
            return SearchOutcome(status="search_failed", error=f"{type(exc).__name__}: {exc}"[:300])

        outcome = SearchOutcome(status="ok")
        started = time.monotonic()
        for hit in hits[:max_results]:
            title = str(hit.get("title") or "").strip()
            url = str(hit.get("href") or "").strip()
            snippet = str(hit.get("body") or "").strip()
            page_text = None
            if url and time.monotonic() - started < self.scrape_budget:
                page_text = self._scrape(url)
            content = page_text or snippet
            if not content:
                continue  # nothing to ground on
            rank = len(outcome.results) + 1
            outcome.results.append(f"Source [{rank}]: {title}\nURL: {url}\nContent: {content}\n")
            outcome.sources.append({"rank": rank, "title": title, "url": url, "content_source": "page" if page_text else "search_snippet"})
        if not outcome.results:
            outcome.status = "no_results"
        return outcome

    def search_and_scrape(self, query: str, max_results: int = 3) -> list[str]:
        """Reference-compatible API: just the formatted snippets."""
        return self.search(query, max_results).results

    def _scrape(self, url: str) -> str | None:
        try:
            html = self._fetch_html(url)
            return extract_paragraphs(html) if html else None
        except Exception as exc:  # malformed HTML, decoding, network
            logger.info("Scrape failed for %s: %s", url, exc)
            return None

    def _fetch_html(self, url: str) -> str | None:
        for _ in range(self.max_redirects + 1):
            if not is_public_url(url, self.resolver):
                logger.info("Refusing to fetch non-public URL %s", url)
                return None
            resp = self.session.get(url, timeout=self.timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=False, stream=True)
            try:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        return None
                    url = urljoin(url, location)
                    continue
                if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
                    return None
                body = bytearray()
                for block in resp.iter_content(16_384):
                    body.extend(block)
                    if len(body) >= self.max_page_bytes:
                        del body[self.max_page_bytes:]
                        break
                return body.decode(resp.encoding or "utf-8", errors="replace")
            finally:
                resp.close()
        return None  # too many redirects
