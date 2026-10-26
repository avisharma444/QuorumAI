package api

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"example.com/councilai-starter/internal/cache/cachetest"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm/llmtest"
)

func testCouncil(f *llmtest.Fake) *council.Council {
	return council.New(f.Clients("a", "b", "c"), f.Client("chair"), 2*time.Second)
}

var errProvider = errors.New("provider down")

// newTestServer disables web search: these tests have no RAG service, and web
// grounding has its own tests with a fake one.
func newTestServer(c *council.Council, ragURL string) *Server {
	s := NewServer(c, ragURL)
	s.WebSearch = false
	s.Cache = cachetest.NewExact()
	return s
}

// ── HTTP integration ────────────────────────────────────────────────

func postQuery(t *testing.T, srv *Server, body string, stream bool) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(body))
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)
	return rec
}

type sseEvent struct {
	name string
	data string
}

func parseSSE(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	var name string
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			name = line[7:]
		case strings.HasPrefix(line, "data: "):
			if !json.Valid([]byte(line[6:])) {
				t.Fatalf("invalid JSON in %s event: %s", name, line)
			}
			events = append(events, sseEvent{name, line[6:]})
		}
	}
	return events
}

func TestHTTPJSONResponse(t *testing.T) {
	srv := newTestServer(testCouncil(llmtest.New(nil)), "http://unused.invalid")
	rec := postQuery(t, srv, `{"question":"Compare A and B","strategy":"council"}`, false)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var out council.QueryResult
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Reviews) != 3 || !reflect.DeepEqual(out.Reviews[0].Ranking, []string{"A", "B", "C"}) || !out.PeerReviewed || out.Selected != "A" {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "best_index") {
		t.Fatal("old best_index format still present")
	}
}

func TestSSEEventOrder(t *testing.T) {
	srv := newTestServer(testCouncil(llmtest.New(nil)), "http://unused.invalid")
	events := parseSSE(t, postQuery(t, srv, `{"question":"compare A vs B","strategy":"council"}`, true).Body.String())
	var names []string
	for _, e := range events {
		names = append(names, e.name)
	}
	want := []string{"start", "routing", "retrieved", "candidate", "candidate", "candidate", "review", "review", "review", "synthesis", "final"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("got events %v, want %v", names, want)
	}
	var final council.QueryResult
	_ = json.Unmarshal([]byte(events[len(events)-1].data), &final)
	if !final.PeerReviewed || final.Source != "chairman:chair" {
		t.Fatalf("final payload: %s", events[len(events)-1].data)
	}
}

func TestSSEReviewErrorAndFallbackEvents(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{"review/a": {Err: errProvider}, "chair/chair": {Err: errProvider}})
	srv := newTestServer(testCouncil(f), "http://unused.invalid")
	events := parseSSE(t, postQuery(t, srv, `{"question":"compare","strategy":"council"}`, true).Body.String())
	seen := map[string]int{}
	for _, e := range events {
		seen[e.name]++
	}
	if seen["review_error"] != 1 || seen["review"] != 2 || seen["fallback"] != 1 || seen["final"] != 1 {
		t.Fatalf("events %v", seen)
	}
	if events[len(events)-1].name != "final" {
		t.Fatal("final must be last")
	}
}

func TestHTTPCaching(t *testing.T) {
	f := llmtest.New(nil)
	srv := newTestServer(testCouncil(f), "http://unused.invalid")
	body := `{"question":"Compare concurrency and parallelism","strategy":"council"}`
	var first, second council.QueryResult
	_ = json.Unmarshal(postQuery(t, srv, body, false).Body.Bytes(), &first)
	_ = json.Unmarshal(postQuery(t, srv, body, false).Body.Bytes(), &second)
	if first.CacheHit || !second.CacheHit || srv.Hits.Load() != 1 {
		t.Fatal("second identical request must be an exact cache hit")
	}
	if f.Count("member") != 3 || f.Count("review") != 3 {
		t.Fatalf("cache hit must not call models again: %v", f.Calls())
	}
	if !reflect.DeepEqual(first.Reviews, second.Reviews) {
		t.Fatal("cached reviews differ")
	}
}

func TestDirectModeBypassesPeerReview(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{"direct/chair": {Text: "direct answer"}})
	srv := newTestServer(testCouncil(f), "http://unused.invalid")
	var out council.QueryResult
	_ = json.Unmarshal(postQuery(t, srv, `{"question":"hi","strategy":"direct"}`, false).Body.Bytes(), &out)
	if out.Answer != "direct answer" || out.Source != "direct:chair" || len(out.Reviews) != 0 {
		t.Fatalf("unexpected: %+v", out)
	}
	if f.Count("member")+f.Count("review")+f.Count("chair") != 0 {
		t.Fatalf("direct mode called council stages: %v", f.Calls())
	}
}

func TestDocumentRetrievalFeedsTheCouncil(t *testing.T) {
	var gotDoc string
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotDoc, _ = req["doc_id"].(string)
		fmt.Fprint(w, `{"chunks":[{"content":"The sky is green on Zog.","chunk_type":"paragraph","page_number":2,"chunk_index":3,"metadata":{"ocr_method":"tesseract","ocr_confidence":0.91,"vector_distance":3.1,"rerank_score":2.5}}],"reranked":true,"candidates":5,"distance_metric":"l2"}`)
	}))
	defer rag.Close()
	f := llmtest.New(nil)
	srv := newTestServer(testCouncil(f), rag.URL)
	var out council.QueryResult
	rec := postQuery(t, srv, `{"question":"What colour is the sky?","doc_id":"doc-1","strategy":"council"}`, false)
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if gotDoc != "doc-1" || len(out.Sources) != 1 || !strings.Contains(out.Sources[0], "page 2") {
		t.Fatalf("retrieval not wired: doc=%q body=%s", gotDoc, rec.Body.String())
	}
	if !strings.Contains(f.Prompt("member", "a"), "The sky is green on Zog.") || !strings.Contains(f.Prompt("chair", "chair"), "The sky is green on Zog.") {
		t.Fatal("evidence missing from member or chairman prompt")
	}
}
