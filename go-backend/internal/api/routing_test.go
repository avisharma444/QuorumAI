package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"example.com/councilai-starter/internal/agent"
	"example.com/councilai-starter/internal/cache/cachetest"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/llm/llmtest"
)

// fakeRAG imitates the Python service. Behaviour is set per test.
type fakeRAG struct {
	*httptest.Server
	mu             sync.Mutex
	calls          map[string]int
	ingestN        int
	preview        string
	retrieveStatus int
	searchStatus   int
	searchBody     string
	// embed maps a question to (vector, model); nil means a fixed vector.
	embed func(text string) ([]float32, string)
}

const okSearch = `{"status":"ok","results":["Source [1]: Mirevale\nURL: https://example.org/m\nContent: Mirevale is the capital.\n"],"sources":[{"rank":1,"title":"Mirevale","url":"https://example.org/m","content_source":"page"}],"error":null}`

func newFakeRAG(t *testing.T) *fakeRAG {
	t.Helper()
	f := &fakeRAG{calls: map[string]int{}, preview: "Zogland facts. The capital of Zogland is Mirevale.", retrieveStatus: 200, searchStatus: 200, searchBody: okSearch}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.calls[r.URL.Path]++
		f.mu.Unlock()
		switch r.URL.Path {
		case "/ingest":
			f.mu.Lock()
			f.ingestN++
			n := f.ingestN
			f.mu.Unlock()
			preview, _ := json.Marshal(f.preview)
			fmt.Fprintf(w, `{"doc_id":"doc-%d","filename":"z.md","pages":1,"chunks":1,"chunk_count":1,"preview_text":%s}`, n, preview)
		case "/retrieve":
			w.WriteHeader(f.retrieveStatus)
			fmt.Fprint(w, `{"chunks":[{"content":"The capital of Zogland is Mirevale.","chunk_type":"paragraph","page_number":1,"chunk_index":0,"metadata":{"ocr_method":"direct_text","vector_distance":4.2,"rerank_score":7.5}}],"reranked":true,"candidates":3,"distance_metric":"l2"}`)
		case "/search":
			w.WriteHeader(f.searchStatus)
			fmt.Fprint(w, f.searchBody)
		case "/embed":
			var req struct {
				Text string `json:"text"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			vec, model := unitVector(0), "BAAI/bge-small-en-v1.5"
			if f.embed != nil {
				vec, model = f.embed(req.Text)
			}
			out, _ := json.Marshal(map[string]any{"embedding": vec, "model": model, "dimension": len(vec)})
			_, _ = w.Write(out)
		default:
			w.WriteHeader(404)
		}
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeRAG) count(path string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[path]
}

func route(strategy string, needsDoc bool) llmtest.Reply {
	return llmtest.Reply{Text: fmt.Sprintf(`{"strategy":%q,"reasoning":"test plan","needs_doc":%t}`, strategy, needsDoc)}
}

type env struct {
	srv *Server
	rag *fakeRAG
	llm *llmtest.Fake
}

func newEnv(t *testing.T, replies map[string]llmtest.Reply) *env {
	t.Helper()
	rag := newFakeRAG(t)
	f := llmtest.New(replies)
	srv := NewServer(council.New(f.Clients("a", "b", "c"), f.Client("chair"), 2*time.Second), rag.URL)
	srv.Cache = cachetest.NewExact()
	withTestAuth(srv, true)
	srv.Router = agent.NewRouter(f.Client("router"))
	srv.IngestAgent = agent.NewIngestAgent(f.Client("ingest"))
	return &env{srv, rag, f}
}

func (e *env) query(t *testing.T, body string) (int, council.QueryResult, string) {
	t.Helper()
	rec := postQuery(t, e.srv, body, false)
	var out council.QueryResult
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out, rec.Body.String()
}

func (e *env) ingest(t *testing.T) map[string]any {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, _ := mw.CreateFormFile("file", "z.md")
	_, _ = part.Write([]byte("Zogland facts. The capital of Zogland is Mirevale."))
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/ingest", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	serveAuthed(e.srv, rec, req)
	if rec.Code != 200 {
		t.Fatalf("ingest status %d: %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return out
}

// ── Ingestion summaries ─────────────────────────────────────────────

func TestIngestGeneratesAndStoresSummary(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Text: "A summary about Zogland and its capital."}})
	out := e.ingest(t)
	if out["summary_status"] != "generated" || out["summary"] != "A summary about Zogland and its capital." || out["summary_model"] != "ingest" {
		t.Fatalf("ingest response %v", out)
	}
	if out["preview_text"] == "" || out["chunk_count"] != 1.0 || out["doc_id"] != "doc-1" {
		t.Fatalf("RAG fields must pass through: %v", out)
	}
	if got := e.llm.Prompt("summarize", "ingest"); got != "Document Sample:\nZogland facts. The capital of Zogland is Mirevale." {
		t.Fatalf("summarizer got %q", got)
	}
	if s, ok := e.srv.getSummary(context.Background(), "doc-1"); !ok || s != "A summary about Zogland and its capital." {
		t.Fatal("summary not stored under the doc_id")
	}
	// Queries reuse the stored summary; no second summarization.
	e.query(t, `{"question":"Capital? #1","doc_id":"doc-1"}`)
	if e.llm.Count("summarize") != 1 {
		t.Fatal("summary must not be regenerated per query")
	}
}

func TestDuplicateIngestionKeepsSummariesSeparate(t *testing.T) {
	e := newEnv(t, nil)
	first, second := e.ingest(t), e.ingest(t)
	if first["doc_id"] == second["doc_id"] || e.llm.Count("summarize") != 2 {
		t.Fatalf("docs %v %v", first["doc_id"], second["doc_id"])
	}
	for _, id := range []string{"doc-1", "doc-2"} {
		if s, ok := e.srv.getSummary(context.Background(), id); !ok || !strings.HasPrefix(s, "[MOCK summary]") {
			t.Fatalf("%s: %q", id, s)
		}
	}
}

func TestIngestSummaryFailuresDoNotFailIngestion(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Err: fmt.Errorf("provider HTTP 503")}})
	out := e.ingest(t)
	if out["summary_status"] != "failed" || !strings.Contains(out["summary_error"].(string), "503") || out["summary"] != nil {
		t.Fatalf("%v", out)
	}
	if _, ok := e.srv.getSummary(context.Background(), "doc-1"); ok {
		t.Fatal("no summary may be stored after a failure")
	}

	e = newEnv(t, nil)
	e.rag.preview = ""
	if out := e.ingest(t); out["summary_status"] != "skipped" || e.llm.Count("summarize") != 0 {
		t.Fatalf("empty preview: %v", out)
	}

	e = newEnv(t, nil)
	e.srv.IngestAgent = nil
	if out := e.ingest(t); out["summary_status"] != "disabled" {
		t.Fatalf("no agent: %v", out)
	}
}

func TestSummarizationCancelled(t *testing.T) {
	e := newEnv(t, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	status, summary, _ := e.srv.summarizeDocument(ctx, "doc-x", "text")
	if status != "cancelled" || summary != "" {
		t.Fatalf("%s %q", status, summary)
	}
	if _, ok := e.srv.getSummary(context.Background(), "doc-x"); ok {
		t.Fatal("cancelled summary stored")
	}
}

func TestIngestRAGErrorPassesThroughWithoutSummarizing(t *testing.T) {
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		fmt.Fprint(w, `{"detail":"No readable text found"}`)
	}))
	defer rag.Close()
	e := newEnv(t, nil)
	e.srv.RAGURL = rag.URL
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, _ := mw.CreateFormFile("file", "empty.txt")
	_, _ = part.Write([]byte("  "))
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/ingest", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	serveAuthed(e.srv, rec, req)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "No readable text") || e.llm.Count("summarize") != 0 {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
}

// ── Context routing ─────────────────────────────────────────────────

func TestNoDocumentSimpleQuestionDirectWithWebGrounding(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("direct", true), "direct/chair": {Text: "Hello!"}})
	code, res, _ := e.query(t, `{"question":"hi there"}`)
	if code != 200 || res.Answer != "Hello!" || res.Strategy != "direct" {
		t.Fatalf("%d %+v", code, res)
	}
	r := res.Routing
	if r.Mode != "auto" || r.Strategy != "direct" || r.NeedsDoc || r.DocumentAttached || r.RouterModel != "router" || r.Fallback {
		t.Fatalf("routing %+v", r)
	}
	if res.Grounding != "web" || len(r.WebSources) != 1 || r.WebSources[0].URL != "https://example.org/m" || e.rag.count("/search") != 1 || e.rag.count("/retrieve") != 0 {
		t.Fatalf("web grounding %+v", res)
	}
	if e.llm.Count("member")+e.llm.Count("review")+e.llm.Count("chair") != 0 || e.llm.Count("direct") != 1 {
		t.Fatalf("direct must call one model only: %v", e.llm.Calls())
	}
	if p := e.llm.Prompt("direct", "chair"); !strings.Contains(p, "Web search results (unverified") || strings.Contains(p, "Document excerpts") {
		t.Fatalf("direct prompt: %s", p)
	}
	if !strings.Contains(e.llm.Prompt("route", "router"), "No document attached.") {
		t.Fatal("router must be told there is no document")
	}
}

func TestNoDocumentComplexQuestionDeepCouncil(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council_deep", false)})
	_, res, _ := e.query(t, `{"question":"Should a hospital adopt AI triage?"}`)
	if res.Strategy != "council_deep" || len(res.Candidates) != 3 || len(res.Reviews) != 3 || res.Reflection == nil || res.Reflection.Status != "completed" || e.llm.Count("reflect") != 1 {
		t.Fatalf("%+v", res)
	}
}

func TestRelevantDocumentIsRetrieved(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Text: "Zogland geography."}, "route/router": route("council", true)})
	e.ingest(t)
	_, res, _ := e.query(t, `{"question":"What is the capital of Zogland?","doc_id":"doc-1"}`)
	if !strings.Contains(e.llm.Prompt("route", "router"), "Attached Document Summary:\nZogland geography.") {
		t.Fatalf("router prompt: %s", e.llm.Prompt("route", "router"))
	}
	r := res.Routing
	if !r.DocumentAttached || !r.NeedsDoc || !r.DocumentUsed || r.DocumentSummary != "available" || res.Grounding != "document" {
		t.Fatalf("routing %+v grounding %s", r, res.Grounding)
	}
	if e.rag.count("/retrieve") != 1 || e.rag.count("/search") != 0 {
		t.Fatal("document questions must not use web search")
	}
	if !strings.Contains(e.llm.Prompt("member", "a"), "Document excerpts from the attached document") {
		t.Fatal("member prompt lacks document evidence")
	}
}

func TestIrrelevantDocumentIsNotRetrieved(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Text: "Tax law."}, "route/router": route("direct", false)})
	e.ingest(t)
	_, res, _ := e.query(t, `{"question":"good morning!","doc_id":"doc-1"}`)
	if res.Routing.NeedsDoc || res.Routing.DocumentUsed || res.Grounding != "none" || e.rag.count("/retrieve")+e.rag.count("/search") != 0 {
		t.Fatalf("%+v", res.Routing)
	}
}

func TestMissingSummaryKeepsDocument(t *testing.T) {
	// The model, seeing no summary, says needs_doc=false; the router keeps the document anyway.
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("direct", false)})
	_, res, _ := e.query(t, `{"question":"What does it say?","doc_id":"never-summarized"}`)
	r := res.Routing
	if r.DocumentSummary != "missing" || !r.NeedsDoc || !r.DocumentUsed || res.Grounding != "document" || r.Strategy != "direct" {
		t.Fatalf("%+v", r)
	}
	if !strings.Contains(e.llm.Prompt("route", "router"), "summary is unavailable") {
		t.Fatal("router must be told the summary is missing, not that there is no document")
	}
}

func TestSummaryFailureThenQueryKeepsDocument(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Err: fmt.Errorf("down")}, "route/router": route("council", false)})
	e.ingest(t)
	_, res, _ := e.query(t, `{"question":"Summarize section 2","doc_id":"doc-1"}`)
	if !res.Routing.DocumentUsed || res.Grounding != "document" {
		t.Fatalf("%+v", res.Routing)
	}
}

func TestRouterFailureFallsBackToCouncilAndIsNotCached(t *testing.T) {
	for name, reply := range map[string]llmtest.Reply{"error": {Err: fmt.Errorf("router down")}, "invalid": {Text: "use the council"}} {
		e := newEnv(t, map[string]llmtest.Reply{"route/router": reply})
		_, res, _ := e.query(t, `{"question":"Explain X","doc_id":"doc-9"}`)
		r := res.Routing
		if !r.Fallback || r.Strategy != "council" || !r.NeedsDoc || !r.DocumentUsed || len(res.Candidates) != 3 {
			t.Fatalf("%s: %+v", name, r)
		}
		_, again, _ := e.query(t, `{"question":"Explain X","doc_id":"doc-9"}`)
		if again.CacheHit {
			t.Fatalf("%s: a fallback-routed answer must not be cached", name)
		}
	}
}

func TestExplicitOverrideSkipsRouter(t *testing.T) {
	e := newEnv(t, nil)
	_, res, _ := e.query(t, `{"question":"Capital?","doc_id":"doc-1","strategy":"direct"}`)
	r := res.Routing
	if e.llm.Count("route") != 0 || r.Mode != "override" || r.Strategy != "direct" || !r.DocumentUsed || r.RouterModel != "" {
		t.Fatalf("%+v", r)
	}
}

func TestCacheIdentitiesAreSeparated(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	q := func(extra string) bool {
		_, res, _ := e.query(t, `{"question":"Same question"`+extra+`}`)
		return res.CacheHit
	}
	if q(``) || !q(``) || !q(`,"strategy":"auto"`) {
		t.Fatal("automatic routing must cache under one identity (\"\" and \"auto\" are the same)")
	}
	if q(`,"strategy":"direct"`) || q(`,"strategy":"council"`) || !q(`,"strategy":"direct"`) {
		t.Fatal("each explicit override needs its own identity, separate from auto")
	}
	if e.llm.Count("route") != 1 {
		t.Fatalf("router calls %d", e.llm.Count("route"))
	}
	// A different model configuration never reuses an entry.
	q1 := council.QueryRequest{Question: "x"}
	k1 := e.srv.l1Key(testUserID, q1, ModeAuto, "")
	e.srv.Router = agent.NewRouter(e.llm.Client("other-router"))
	if e.srv.l1Key(testUserID, q1, ModeAuto, "") == k1 {
		t.Fatal("router model must be part of the cache identity")
	}
}

func TestUnknownDocumentIs404(t *testing.T) {
	e := newEnv(t, nil)
	e.rag.retrieveStatus = 404
	code, _, body := e.query(t, `{"question":"What is in it?","doc_id":"nope"}`)
	if code != 404 || !strings.Contains(body, "document not found") || e.llm.Count("member") != 0 {
		t.Fatalf("%d %s", code, body)
	}
}

func TestGroundingFailuresDegradeHonestly(t *testing.T) {
	cases := map[string]func(*fakeRAG){
		"search HTTP 500": func(f *fakeRAG) { f.searchStatus = 500 },
		"search failed": func(f *fakeRAG) {
			f.searchBody = `{"status":"search_failed","results":[],"sources":[],"error":"rate limited"}`
		},
		"no results":       func(f *fakeRAG) { f.searchBody = `{"status":"no_results","results":[],"sources":[],"error":null}` },
		"malformed search": func(f *fakeRAG) { f.searchBody = `not json` },
	}
	for name, setup := range cases {
		e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
		setup(e.rag)
		code, res, _ := e.query(t, `{"question":"General question"}`)
		if code != 200 || res.Grounding != "none" || res.Routing.GroundingError == "" || len(res.Sources) != 0 || res.Answer == "" {
			t.Fatalf("%s: %d %+v", name, code, res.Routing)
		}
		if strings.Contains(e.llm.Prompt("member", "a"), "Web search results") {
			t.Fatalf("%s: ungrounded answer labelled as web-grounded", name)
		}
		if _, again, _ := e.query(t, `{"question":"General question"}`); again.CacheHit {
			t.Fatalf("%s: ungrounded fallback must not be cached", name)
		}
	}
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	e.rag.retrieveStatus = 500
	code, res, _ := e.query(t, `{"question":"Doc question","doc_id":"doc-1"}`)
	if code != 200 || res.Grounding != "none" || !strings.Contains(res.Routing.GroundingError, "document retrieval failed") || res.Routing.DocumentUsed {
		t.Fatalf("retrieval failure: %d %+v", code, res.Routing)
	}
	e.srv.WebSearch = false
	_, res, _ = e.query(t, `{"question":"No search configured"}`)
	if res.Grounding != "none" || res.Routing.GroundingError != "" || e.rag.count("/search") != 0 {
		t.Fatalf("web search disabled: %+v", res.Routing)
	}
}

func TestCancellationDuringRoutingStopsTheRequest(t *testing.T) {
	e := newEnv(t, nil)
	started := make(chan struct{})
	e.llm.Hook = func(ctx context.Context, stage, model string) error {
		if stage == "route" {
			close(started)
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-started; cancel() }()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(`{"question":"q"}`)).WithContext(ctx)
	rec := httptest.NewRecorder()
	serveAuthed(e.srv, rec, req)
	if rec.Code != http.StatusGatewayTimeout || e.llm.Count("member")+e.llm.Count("direct") != 0 || e.rag.count("/search") != 0 {
		t.Fatalf("%d %s calls=%v", rec.Code, rec.Body.String(), e.llm.Calls())
	}
}

func TestRouterUsageCountsTowardTokenUsage(t *testing.T) {
	r := route("direct", false)
	r.Usage = llm.Usage{PromptTokens: 90, CompletionTokens: 10, TotalTokens: 100}
	e := newEnv(t, map[string]llmtest.Reply{"route/router": r, "direct/chair": {Text: "ok", Usage: llm.Usage{TotalTokens: 20, PromptTokens: 15, CompletionTokens: 5}}})
	_, res, _ := e.query(t, `{"question":"hi"}`)
	if res.TokenUsage == nil || *res.TokenUsage != (llm.Usage{PromptTokens: 105, CompletionTokens: 15, TotalTokens: 120}) {
		t.Fatalf("%+v", res.TokenUsage)
	}
}

func TestSSEDirectModeHasNoCouncilEvents(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("direct", false)})
	events := parseSSE(t, postQuery(t, e.srv, `{"question":"hello sse"}`, true).Body.String())
	var names []string
	for _, ev := range events {
		names = append(names, ev.name)
	}
	if strings.Join(names, ",") != "start,routing,retrieved,final" {
		t.Fatalf("events %v", names)
	}
	var routing council.RoutingInfo
	_ = json.Unmarshal([]byte(events[1].data), &routing)
	var final council.QueryResult
	_ = json.Unmarshal([]byte(events[3].data), &final)
	if routing.Strategy != "direct" || final.Routing == nil || final.Routing.Strategy != "direct" || final.Grounding != "web" {
		t.Fatalf("routing %+v final %+v", routing, final.Routing)
	}
}

func TestRequestValidation(t *testing.T) {
	e := newEnv(t, nil)
	for body, want := range map[string]int{
		`{"question":"q","doc_id":"../../etc/passwd"}`:                400,
		`{"question":"q","doc_id":"` + strings.Repeat("a", 65) + `"}`: 400,
		`{"question":"q","strategy":"fast"}`:                          400,
		`{"question":"   "}`:                                          400,
	} {
		if code, _, _ := e.query(t, body); code != want {
			t.Fatalf("%s: %d", body, code)
		}
	}
	if e.llm.Count("route") != 0 {
		t.Fatal("invalid requests must not reach the router")
	}
}

func TestConcurrentRoutedRequests(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			rec := postQuery(t, e.srv, fmt.Sprintf(`{"question":"parallel question %d"}`, i), false)
			if rec.Code != 200 {
				t.Errorf("status %d", rec.Code)
			}
		}(i)
	}
	wg.Wait()
	if e.llm.Count("route") != 8 || e.llm.Count("member") != 24 {
		t.Fatalf("calls %v", e.llm.Calls())
	}
}
