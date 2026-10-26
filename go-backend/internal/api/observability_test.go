package api

// Metrics and traces for real request executions (fake models and fake
// RAG service, real HTTP router, middleware and instrumentation).

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"example.com/councilai-starter/internal/cache"
	"example.com/councilai-starter/internal/cache/cachetest"
	"example.com/councilai-starter/internal/llm/llmtest"
	"example.com/councilai-starter/internal/metrics"
	"example.com/councilai-starter/internal/telemetry"
)

// traced installs an in-memory tracer for one test and restores noop after.
func traced(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	_, exp := telemetry.NewTestTracer()
	t.Cleanup(func() { _, _ = telemetry.NewTracerProvider(telemetry.DefaultConfig()) })
	return exp
}

func counter(c prometheus.Counter) float64 { return testutil.ToFloat64(c) }

func histCount(o prometheus.Observer) uint64 {
	var m dto.Metric
	_ = o.(prometheus.Metric).Write(&m)
	return m.GetHistogram().GetSampleCount()
}

type spanIndex struct {
	all    tracetest.SpanStubs
	byName map[string][]tracetest.SpanStub
}

func index(exp *tracetest.InMemoryExporter) spanIndex {
	ix := spanIndex{all: exp.GetSpans(), byName: map[string][]tracetest.SpanStub{}}
	for _, s := range ix.all {
		ix.byName[s.Name] = append(ix.byName[s.Name], s)
	}
	return ix
}

func (ix spanIndex) one(t *testing.T, name string) tracetest.SpanStub {
	t.Helper()
	if len(ix.byName[name]) != 1 {
		t.Fatalf("want exactly one %q span, got %d (spans: %v)", name, len(ix.byName[name]), ix.names())
	}
	return ix.byName[name][0]
}

func (ix spanIndex) names() []string {
	var n []string
	for _, s := range ix.all {
		n = append(n, s.Name)
	}
	return n
}

func childOf(t *testing.T, child, parent tracetest.SpanStub) {
	t.Helper()
	if child.Parent.SpanID() != parent.SpanContext.SpanID() {
		t.Fatalf("%q is not a child of %q", child.Name, parent.Name)
	}
}

func attr(s tracetest.SpanStub, key string) string {
	for _, a := range s.Attributes {
		if string(a.Key) == key {
			return a.Value.Emit()
		}
	}
	return ""
}

func TestFullCouncilTraceHierarchyAndPropagation(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	doc := e.ingest(t)["doc_id"].(string)
	exp := traced(t)
	const question = "What is the capital of Zogland, secretly?"
	const incomingTrace = "4bf92f3577b34da6a3ce929d0e0e4736"

	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(`{"question":"`+question+`","doc_id":"`+doc+`"}`))
	req.Header.Set("traceparent", "00-"+incomingTrace+"-00f067aa0ba902b7-01")
	token := tokenFor(e.srv, testUserID)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	e.srv.Routes().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if rec.Header().Get("X-Trace-ID") != incomingTrace {
		t.Fatalf("X-Trace-ID %q", rec.Header().Get("X-Trace-ID"))
	}

	ix := index(exp)
	for _, s := range ix.all {
		if s.SpanContext.TraceID().String() != incomingTrace {
			t.Fatalf("span %q is in another trace", s.Name)
		}
	}
	root := ix.one(t, "POST /api/v1/query")
	if root.SpanKind != trace.SpanKindServer || !root.Parent.IsRemote() || attr(root, "http.route") != "/api/v1/query" ||
		attr(root, "http.response.status_code") != "200" || attr(root, "query.strategy") != "council" || attr(root, "cache.tier") != "none" {
		t.Fatalf("root %+v", root.Attributes)
	}
	for _, name := range []string{"auth.document_access", "cache.l1_lookup", "router.plan", "rag POST /retrieve", "council.run", "cache.store"} {
		childOf(t, ix.one(t, name), root)
	}
	childOf(t, ix.one(t, "llm.router"), ix.one(t, "router.plan"))
	run := ix.one(t, "council.run")
	cands, review, chair := ix.one(t, "council.candidates"), ix.one(t, "council.peer_review"), ix.one(t, "council.chairman")
	for _, st := range []tracetest.SpanStub{cands, review, chair} {
		childOf(t, st, run)
	}
	if len(ix.byName["llm.candidate"]) != 3 || len(ix.byName["llm.review"]) != 3 {
		t.Fatalf("model spans: %v", ix.names())
	}
	seen := map[trace.SpanID]bool{}
	for _, s := range ix.byName["llm.candidate"] {
		childOf(t, s, cands) // concurrent goroutines keep the parent: siblings, not new traces
		seen[s.SpanContext.SpanID()] = true
	}
	for _, s := range ix.byName["llm.review"] {
		childOf(t, s, review)
	}
	childOf(t, ix.one(t, "llm.chairman"), chair)
	if len(seen) != 3 {
		t.Fatal("candidate spans must be distinct siblings")
	}
	if len(ix.byName["council.reflection"])+len(ix.byName["council.revision"])+len(ix.byName["cache.l2_lookup"]) != 0 {
		t.Fatalf("spans for stages that did not run: %v", ix.names())
	}

	// The Python service received this trace with the client span as parent.
	e.rag.mu.Lock()
	tp := e.rag.traceparents["/retrieve"]
	e.rag.mu.Unlock()
	ragSpan := ix.one(t, "rag POST /retrieve")
	if want := "00-" + incomingTrace + "-" + ragSpan.SpanContext.SpanID().String() + "-01"; tp != want {
		t.Fatalf("traceparent to RAG %q, want %q", tp, want)
	}

	// Nothing sensitive in any attribute.
	for _, s := range ix.all {
		for _, a := range s.Attributes {
			v := a.Value.Emit()
			for _, secret := range []string{"secretly", testUserID, token, "Zogland", doc} {
				if strings.Contains(v, secret) {
					t.Fatalf("span %q attribute %s leaks %q", s.Name, a.Key, secret)
				}
			}
		}
	}
}

func TestCacheHitHasLookupSpanButNoInference(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	body := `{"question":"Compare A and B"}`
	e.query(t, body)
	exp := traced(t)
	hits := counter(metrics.CacheOperations.WithLabelValues("l1_exact", "hit"))
	chair := counter(metrics.ChairmanSynthesis.WithLabelValues("success"))
	cand := histCount(metrics.LLMRequestDuration.WithLabelValues("candidate", "other", "success"))
	answered := counter(metrics.QueryOutcomes.WithLabelValues("council", "l1_exact", "answered"))
	_, res, _ := e.query(t, body)
	if !res.CacheHit {
		t.Fatal("expected L1 hit")
	}
	ix := index(exp)
	if attr(ix.one(t, "cache.l1_lookup"), "cache.result") != "hit" {
		t.Fatal("lookup span result")
	}
	for _, s := range ix.all {
		if strings.HasPrefix(s.Name, "llm.") || strings.HasPrefix(s.Name, "council.") || strings.HasPrefix(s.Name, "rag ") {
			t.Fatalf("cache hit produced %q span", s.Name)
		}
	}
	if counter(metrics.CacheOperations.WithLabelValues("l1_exact", "hit"))-hits != 1 ||
		counter(metrics.ChairmanSynthesis.WithLabelValues("success")) != chair ||
		histCount(metrics.LLMRequestDuration.WithLabelValues("candidate", "other", "success")) != cand ||
		counter(metrics.QueryOutcomes.WithLabelValues("council", "l1_exact", "answered"))-answered != 1 {
		t.Fatal("cache hit metrics")
	}
}

func TestL2HitMetricsAndSpan(t *testing.T) {
	e, _, _ := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	e.query(t, docQ("What is the capital?", ""))
	exp := traced(t)
	l2hits := counter(metrics.CacheOperations.WithLabelValues("l2_semantic", "hit"))
	l1miss := counter(metrics.CacheOperations.WithLabelValues("l1_exact", "miss"))
	lookups := histCount(metrics.CacheLookupDuration.WithLabelValues("l2_semantic"))
	_, res, _ := e.query(t, docQ("paraphrase capital", ""))
	if res.Cache.Tier != "l2_semantic" {
		t.Fatalf("%+v", res.Cache)
	}
	ix := index(exp)
	if attr(ix.one(t, "cache.l2_lookup"), "cache.result") != "hit" || len(ix.byName["council.run"]) != 0 {
		t.Fatalf("spans %v", ix.names())
	}
	childOf(t, ix.one(t, "rag POST /embed"), ix.one(t, "cache.l2_lookup"))
	if counter(metrics.CacheOperations.WithLabelValues("l2_semantic", "hit"))-l2hits != 1 ||
		counter(metrics.CacheOperations.WithLabelValues("l1_exact", "miss"))-l1miss != 1 ||
		histCount(metrics.CacheLookupDuration.WithLabelValues("l2_semantic"))-lookups != 1 {
		t.Fatal("L2 hit metrics")
	}
}

func TestDeepCouncilReflectionAndRevisionSpans(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{
		"chair/chair":   {Text: `{"answer":"draft","reasoning":"r","source":"chairman-synthesis","confidence":0.5}`},
		"reflect/chair": {Text: `{"quality":"needs_revision","issues":"x","suggestion":"y","faithful":false}`},
		"revise/chair":  {Text: `{"answer":"revised","reasoning":"r2","source":"chairman-synthesis","confidence":0.7}`},
	})
	exp := traced(t)
	refl := histCount(metrics.StageDuration.WithLabelValues("reflection"))
	rev := histCount(metrics.StageDuration.WithLabelValues("revision"))
	deep := histCount(metrics.CouncilResponse.WithLabelValues("council_deep"))
	_, res, _ := e.query(t, `{"question":"q","strategy":"council_deep"}`)
	if res.Answer != "revised" {
		t.Fatalf("%+v", res)
	}
	ix := index(exp)
	run := ix.one(t, "council.run")
	childOf(t, ix.one(t, "council.reflection"), run)
	childOf(t, ix.one(t, "council.revision"), run)
	childOf(t, ix.one(t, "llm.reflection"), ix.one(t, "council.reflection"))
	childOf(t, ix.one(t, "llm.revision"), ix.one(t, "council.revision"))
	if attr(run, "council.source_kind") != "chairman_revised" {
		t.Fatal(attr(run, "council.source_kind"))
	}
	if histCount(metrics.StageDuration.WithLabelValues("reflection"))-refl != 1 || histCount(metrics.StageDuration.WithLabelValues("revision"))-rev != 1 ||
		histCount(metrics.CouncilResponse.WithLabelValues("council_deep"))-deep != 1 {
		t.Fatal("stage metrics")
	}
}

func TestPartialMemberFailureAndChairmanFallback(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{
		"member/b":    {Err: errors.New("provider 500")},
		"chair/chair": {Err: errors.New("chairman down")},
	})
	exp := traced(t)
	memberFail := counter(metrics.LLMFailures.WithLabelValues("candidate", "other"))
	chairFail := counter(metrics.LLMFailures.WithLabelValues("chairman", "other"))
	synthFail := counter(metrics.ChairmanSynthesis.WithLabelValues("failure"))
	code, res, _ := e.query(t, `{"question":"q","strategy":"council"}`)
	if code != 200 || len(res.Degraded) == 0 {
		t.Fatalf("%d %+v", code, res.Degraded)
	}
	if counter(metrics.LLMFailures.WithLabelValues("candidate", "other"))-memberFail != 1 ||
		counter(metrics.LLMFailures.WithLabelValues("chairman", "other"))-chairFail != 1 ||
		counter(metrics.ChairmanSynthesis.WithLabelValues("failure"))-synthFail != 1 {
		t.Fatal("failure metrics")
	}
	ix := index(exp)
	if attr(ix.one(t, "council.candidates"), "council.stage_outcome") != "partial" {
		t.Fatal("candidates outcome")
	}
	failed := 0
	for _, s := range ix.byName["llm.candidate"] {
		if s.Status.Code == codes.Error {
			failed++
		}
	}
	if failed != 1 || ix.one(t, "council.chairman").Status.Code != codes.Error {
		t.Fatalf("span statuses: failed candidates %d", failed)
	}
	if attr(ix.one(t, "council.run"), "council.source_kind") != "fallback" {
		t.Fatal("fallback not visible in trace")
	}
}

func TestRedisCircuitOpenIsCounted(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	before := counter(metrics.CacheOperations.WithLabelValues("l1_exact", "circuit_open"))
	e.srv.Cache.(*cachetest.Exact).GetErr = cache.ErrCircuitOpen
	exp := traced(t)
	code, _, _ := e.query(t, `{"question":"q"}`)
	if code != 200 || counter(metrics.CacheOperations.WithLabelValues("l1_exact", "circuit_open"))-before != 1 {
		t.Fatal("circuit-open lookup not counted or request failed")
	}
	if index(exp).one(t, "cache.l1_lookup").Status.Code != codes.Error {
		t.Fatal("lookup span not marked")
	}
}

func TestPostgresOutageIsMeasuredWithoutInference(t *testing.T) {
	e := newEnv(t, nil)
	repo := withTestAuth(e.srv, false)
	repo.Err = errors.New("connection refused")
	exp := traced(t)
	dep := counter(metrics.DependencyErrors.WithLabelValues("postgres", "document_access"))
	code, _, _ := e.query(t, `{"question":"q","doc_id":"doc-1"}`)
	if code != 503 || counter(metrics.DependencyErrors.WithLabelValues("postgres", "document_access"))-dep != 1 || len(e.llm.Calls()) != 0 {
		t.Fatalf("%d calls=%v", code, e.llm.Calls())
	}
	ix := index(exp)
	if ix.one(t, "auth.document_access").Status.Code != codes.Error || attr(ix.one(t, "POST /api/v1/query"), "http.response.status_code") != "503" {
		t.Fatal("outage not visible in trace")
	}
	if ix.one(t, "POST /api/v1/query").Status.Code != codes.Error {
		t.Fatal("5xx root span must be an error")
	}
}

func TestAuthRejectionsAreMeasured(t *testing.T) {
	e := newEnv(t, nil)
	exp := traced(t)
	before := counter(metrics.RequestCount.WithLabelValues("POST", "/api/v1/query", "401"))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(`{"question":"q"}`))
	req.Header.Set("Authorization", "Bearer forged.token.value")
	rec := httptest.NewRecorder()
	e.srv.Routes().ServeHTTP(rec, req)
	if rec.Code != 401 || counter(metrics.RequestCount.WithLabelValues("POST", "/api/v1/query", "401"))-before != 1 {
		t.Fatal("401 not counted")
	}
	ix := index(exp)
	root := ix.one(t, "POST /api/v1/query")
	if attr(root, "auth.failure") != "invalid_token" || len(ix.all) != 1 || len(e.llm.Calls()) != 0 {
		t.Fatalf("spans %v", ix.names())
	}
	for _, a := range root.Attributes {
		if strings.Contains(a.Value.Emit(), "forged") {
			t.Fatal("token in span")
		}
	}
}

func TestRouteLabelsAreBounded(t *testing.T) {
	e := newEnv(t, nil)
	for i := 0; i < 20; i++ {
		rec := httptest.NewRecorder()
		e.srv.Routes().ServeHTTP(rec, httptest.NewRequest("GET", fmt.Sprintf("/api/v1/query/%d?token=x%d", i, i), nil))
		rec = httptest.NewRecorder()
		e.srv.Routes().ServeHTTP(rec, httptest.NewRequest("GET", fmt.Sprintf("/user-%d/doc-%d", i, i), nil))
	}
	ch := make(chan prometheus.Metric, 1024)
	metrics.RequestCount.Collect(ch)
	close(ch)
	allowed := map[string]bool{"unmatched": true, "/health": true, "/api/v1/query": true, "/api/v1/ingest": true, "/api/v1/register": true,
		"/api/v1/login": true, "/api/v1/conversation/clear": true, "/api/v1/conversation/history": true}
	for m := range ch {
		var d dto.Metric
		_ = m.Write(&d)
		for _, l := range d.GetLabel() {
			if l.GetName() == "route" && !allowed[l.GetValue()] {
				t.Fatalf("unbounded route label %q", l.GetValue())
			}
		}
	}
	if rec := httptest.NewRecorder(); true {
		e.srv.Routes().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
		if rec.Code != 404 {
			t.Fatal("/metrics must not be served on the public API port")
		}
	}
}

func TestSSEStreamsThroughInstrumentedWriter(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	release := make(chan struct{})
	e.llm.Hook = func(ctx context.Context, stage, model string) error {
		if stage == "review" {
			select {
			case <-release:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	}
	ts := httptest.NewServer(e.srv.Routes())
	defer ts.Close()
	lat := histCount(metrics.RequestLatency.WithLabelValues("POST", "/api/v1/query"))
	req, _ := http.NewRequest("POST", ts.URL+"/api/v1/query", strings.NewReader(`{"question":"q","strategy":"council"}`))
	req.Header.Set("Authorization", "Bearer "+tokenFor(e.srv, testUserID))
	req.Header.Set("Accept", "text/event-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	candidates := 0
	for candidates < 3 && sc.Scan() {
		if sc.Text() == "event: candidate" {
			candidates++
		}
	}
	// All three candidates arrived while reviews are still blocked: the
	// middleware's writer flushes, it does not buffer the stream.
	if candidates != 3 {
		t.Fatal("candidate events were not streamed before the request finished")
	}
	close(release)
	final := false
	for sc.Scan() {
		if sc.Text() == "event: final" {
			final = true
		}
	}
	if !final {
		t.Fatal("no final event")
	}
	deadline := time.Now().Add(2 * time.Second)
	for histCount(metrics.RequestLatency.WithLabelValues("POST", "/api/v1/query"))-lat != 1 {
		if time.Now().After(deadline) {
			t.Fatal("SSE request latency not observed after the stream ended")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestClientDisconnectIsRecorded(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	started := make(chan struct{}, 3)
	e.llm.Hook = func(ctx context.Context, stage, model string) error {
		if stage == "member" {
			started <- struct{}{}
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}
	ts := httptest.NewServer(e.srv.Routes())
	defer ts.Close()
	disc := counter(metrics.ClientDisconnects.WithLabelValues("/api/v1/query"))
	closed := counter(metrics.RequestCount.WithLabelValues("POST", "/api/v1/query", "499"))
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/api/v1/query", strings.NewReader(`{"question":"q","strategy":"council"}`))
	req.Header.Set("Authorization", "Bearer "+tokenFor(e.srv, testUserID))
	go func() { <-started; cancel() }()
	if _, err := http.DefaultClient.Do(req); err == nil {
		t.Fatal("request should have been cancelled")
	}
	deadline := time.Now().Add(3 * time.Second)
	for counter(metrics.ClientDisconnects.WithLabelValues("/api/v1/query"))-disc != 1 {
		if time.Now().After(deadline) {
			t.Fatal("disconnect not recorded")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if counter(metrics.RequestCount.WithLabelValues("POST", "/api/v1/query", "499"))-closed != 1 {
		t.Fatal("disconnected request must be labelled 499, not a server error")
	}
}

func TestBoundedSeriesExistAtZero(t *testing.T) {
	metrics.Init()
	if n := testutil.CollectAndCount(metrics.LLMFailures); n < len(metrics.Stages)*6 {
		t.Fatalf("only %d failure series pre-created", n)
	}
	if testutil.CollectAndCount(metrics.DependencyErrors) < 11 || testutil.CollectAndCount(metrics.ClientDisconnects) < len(metrics.Routes) {
		t.Fatal("dependency / disconnect series not pre-created")
	}
}
