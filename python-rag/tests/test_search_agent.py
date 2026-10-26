import requests

from app.search_agent import WebSearchAgent, extract_paragraphs, is_public_url

PARA = "This paragraph is long enough to count as substantive page content for grounding."


class FakeResponse:
    def __init__(self, status=200, body=b"", ctype="text/html; charset=utf-8", location=None):
        self.status_code = status
        self.headers = {"content-type": ctype}
        if location:
            self.headers["location"] = location
        self.is_redirect = location is not None
        self._body = body
        self.encoding = "utf-8"
        self.closed = False

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        assert kwargs["allow_redirects"] is False and kwargs["stream"] is True and kwargs["timeout"]
        page = self.pages.get(url)
        if isinstance(page, Exception):
            raise page
        return page or FakeResponse(status=404)


def public_resolver(host):
    return {"internal.example": ["10.0.0.5"], "localhost": ["127.0.0.1"]}.get(host, ["93.184.216.34"])


def html(*paras):
    return ("<html><script>var x = 'ignored script text that is quite long indeed';</script><body>"
            + "".join(f"<p>{p}</p>" for p in paras) + "</body></html>").encode()


def agent(hits, pages, **kw):
    return WebSearchAgent(search_fn=lambda q, n: hits, session=FakeSession(pages), resolver=public_resolver, **kw)


def test_results_ordered_formatted_and_attributed():
    hits = [
        {"title": "One", "href": "https://a.example/1", "body": "snippet one"},
        {"title": "Two", "href": "https://b.example/2", "body": "snippet two"},
    ]
    pages = {"https://a.example/1": FakeResponse(body=html("short", PARA, PARA + " 2", PARA + " 3", PARA + " 4"))}
    out = agent(hits, pages).search("q", 3)
    assert out.status == "ok"
    assert out.results[0] == f"Source [1]: One\nURL: https://a.example/1\nContent: {PARA}\n{PARA} 2\n{PARA} 3\n"
    # Page 2 is missing (404), so the search snippet is used and attributed as such.
    assert out.results[1] == "Source [2]: Two\nURL: https://b.example/2\nContent: snippet two\n"
    assert out.sources == [
        {"rank": 1, "title": "One", "url": "https://a.example/1", "content_source": "page"},
        {"rank": 2, "title": "Two", "url": "https://b.example/2", "content_source": "search_snippet"},
    ]


def test_max_results_respected():
    hits = [{"title": f"T{i}", "href": "", "body": f"b{i}"} for i in range(5)]
    assert len(agent(hits, {}).search("q", 2).results) == 2


def test_search_failure_is_reported_not_swallowed():
    def boom(q, n):
        raise requests.ConnectionError("rate limited")
    out = WebSearchAgent(search_fn=boom, session=FakeSession({}), resolver=public_resolver).search("q")
    assert out.status == "search_failed" and out.results == [] and "rate limited" in out.error


def test_no_usable_results():
    out = agent([{"title": "x", "href": "", "body": ""}], {}).search("q")
    assert out.status == "no_results" and out.results == []


def test_scrape_exceptions_fall_back_to_snippet():
    hits = [{"title": "T", "href": "https://a.example/x", "body": "snippet"}]
    out = agent(hits, {"https://a.example/x": requests.Timeout("slow")}).search("q")
    assert out.results[0].endswith("Content: snippet\n") and out.sources[0]["content_source"] == "search_snippet"


def test_malformed_html_and_non_html():
    assert extract_paragraphs("<p>unclosed " + PARA) == "unclosed " + PARA
    assert extract_paragraphs("<<<>>> not html at all") is None
    hits = [{"title": "PDF", "href": "https://a.example/f.pdf", "body": "pdf snippet"}]
    out = agent(hits, {"https://a.example/f.pdf": FakeResponse(body=b"%PDF", ctype="application/pdf")}).search("q")
    assert out.sources[0]["content_source"] == "search_snippet"


def test_unsafe_urls_are_never_fetched():
    unsafe = ["http://127.0.0.1/admin", "http://localhost:8000/health", "http://169.254.169.254/latest/meta-data",
              "http://internal.example/secret", "http://[::1]/", "file:///etc/passwd", "ftp://a.example/x", "http:///nohost"]
    for url in unsafe:
        assert not is_public_url(url, public_resolver), url
    hits = [{"title": "T", "href": u, "body": "snippet"} for u in unsafe]
    session = FakeSession({})
    out = WebSearchAgent(search_fn=lambda q, n: hits, session=session, resolver=public_resolver).search("q", 10)
    assert session.requested == []
    assert all(s["content_source"] == "search_snippet" for s in out.sources)


def test_redirect_to_private_address_is_blocked():
    pages = {"https://a.example/r": FakeResponse(status=302, location="http://10.0.0.5/internal")}
    session = FakeSession(pages)
    hits = [{"title": "T", "href": "https://a.example/r", "body": "snippet"}]
    out = WebSearchAgent(search_fn=lambda q, n: hits, session=session, resolver=public_resolver).search("q")
    assert session.requested == ["https://a.example/r"]
    assert out.sources[0]["content_source"] == "search_snippet"


def test_safe_redirect_is_followed_with_limit():
    pages = {
        "https://a.example/r": FakeResponse(status=301, location="/final"),
        "https://a.example/final": FakeResponse(body=html(PARA)),
    }
    out = agent([{"title": "T", "href": "https://a.example/r", "body": ""}], pages).search("q")
    assert out.sources[0]["content_source"] == "page"
    loop = {"https://a.example/l": FakeResponse(status=302, location="https://a.example/l")}
    session = FakeSession(loop)
    WebSearchAgent(search_fn=lambda q, n: [{"title": "T", "href": "https://a.example/l", "body": "s"}],
                   session=session, resolver=public_resolver, max_redirects=3).search("q")
    assert len(session.requested) == 4


def test_page_size_is_capped():
    big = html(PARA) + b"<p>" + b"x" * 5_000_000 + b"</p>"
    out = agent([{"title": "T", "href": "https://a.example/big", "body": "s"}],
                {"https://a.example/big": FakeResponse(body=big)}, max_page_bytes=10_000).search("q")
    assert out.results[0].count("x") < 10_000


def test_scrape_budget_stops_fetching():
    hits = [{"title": f"T{i}", "href": f"https://a.example/{i}", "body": f"s{i}"} for i in range(3)]
    session = FakeSession({})
    WebSearchAgent(search_fn=lambda q, n: hits, session=session, resolver=public_resolver, scrape_budget=0).search("q")
    assert session.requested == []


def test_query_is_bounded():
    seen = {}
    WebSearchAgent(search_fn=lambda q, n: seen.setdefault("q", q) and [], session=FakeSession({}), resolver=public_resolver).search("x" * 5000)
    assert len(seen["q"]) == 400


def test_reference_compatible_api():
    hits = [{"title": "T", "href": "", "body": "snippet"}]
    assert agent(hits, {}).search_and_scrape("q") == ["Source [1]: T\nURL: \nContent: snippet\n"]
