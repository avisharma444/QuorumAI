package api

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"example.com/councilai-starter/internal/agent"
	"example.com/councilai-starter/internal/cache"
	"example.com/councilai-starter/internal/cache/cachetest"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/llm/llmtest"
)

func unitVector(i int) []float32 {
	v := make([]float32, cache.VectorDim)
	v[i] = 1
	return v
}

// rotatedVector is a unit vector at cosine similarity cos to unitVector(0).
func rotatedVector(cos float64) []float32 {
	v := make([]float32, cache.VectorDim)
	v[0], v[1] = float32(cos), float32(math.Sqrt(1-cos*cos))
	return v
}

// semEnv is an env with both cache tiers and scripted embeddings:
// questions containing "capital" embed near e0; "paraphrase" at cos 0.95;
// anything else is orthogonal.
func semEnv(t *testing.T, replies map[string]llmtest.Reply) (*env, *cachetest.Exact, *cachetest.Semantic) {
	t.Helper()
	e := newEnv(t, replies)
	l1, l2 := cachetest.NewExact(), cachetest.NewSemantic()
	e.srv.Cache, e.srv.Semantic, e.srv.SemanticThreshold = l1, l2, 0.85
	e.rag.embed = func(text string) ([]float32, string) {
		switch {
		case strings.Contains(text, "paraphrase"):
			return rotatedVector(0.95), "BAAI/bge-small-en-v1.5"
		case strings.Contains(text, "capital"):
			return unitVector(0), "BAAI/bge-small-en-v1.5"
		}
		return unitVector(7), "BAAI/bge-small-en-v1.5"
	}
	return e, l1, l2
}

func docQ(question string, extra string) string {
	return `{"question":"` + question + `","doc_id":"doc-1"` + extra + `}`
}

func TestL1HitIsTruthfulAboutExecution(t *testing.T) {
	usage := llm.Usage{PromptTokens: 40, CompletionTokens: 10, TotalTokens: 50}
	e, l1, _ := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true), "chair/chair": {Text: `{"answer":"A","reasoning":"R","source":"chairman-synthesis"}`, Usage: usage}})
	_, first, _ := e.query(t, docQ("What is the capital?", ""))
	if first.CacheHit || first.Cache.Tier != "none" || first.Cache.L1 != "miss" || !first.Cache.InferenceExecuted {
		t.Fatalf("first: %+v", first.Cache)
	}
	if strings.Join(first.Cache.Stored, ",") != "l1_exact,l2_semantic" {
		t.Fatalf("stored %v", first.Cache.Stored)
	}
	calls := e.llm.Calls()
	_, second, _ := e.query(t, docQ("  what IS the   capital? ", ""))
	c := second.Cache
	if !second.CacheHit || c.Tier != "l1_exact" || c.InferenceExecuted || c.RouterCalled || second.TokenUsage != nil {
		t.Fatalf("L1 hit metadata %+v usage %+v", c, second.TokenUsage)
	}
	if c.OriginalTokenUsage == nil || c.OriginalTokenUsage.TotalTokens != first.TokenUsage.TotalTokens {
		t.Fatalf("original usage lost: %+v", c.OriginalTokenUsage)
	}
	if second.Answer != "A" || second.Routing == nil || len(second.Candidates) != 3 {
		t.Fatal("cached answer and provenance must be returned intact")
	}
	for stage, n := range e.llm.Calls() {
		if calls[stage] != n {
			t.Fatalf("L1 hit made model calls: %v → %v", calls, e.llm.Calls())
		}
	}
	for _, k := range l1.Keys() {
		if strings.Contains(strings.ToLower(k), "capital") {
			t.Fatalf("question text in key name: %s", k)
		}
	}
}

func TestL1HitOverSSEIsASingleFinalEvent(t *testing.T) {
	e, _, _ := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	e.query(t, docQ("capital question sse", ""))
	events := parseSSE(t, postQuery(t, e.srv, docQ("capital question sse", ""), true).Body.String())
	if len(events) != 1 || events[0].name != "final" {
		t.Fatalf("events %v", events)
	}
	var res council.QueryResult
	_ = json.Unmarshal([]byte(events[0].data), &res)
	if !res.CacheHit || res.Cache.Tier != "l1_exact" {
		t.Fatalf("%+v", res.Cache)
	}
}

func TestL1KeyDimensions(t *testing.T) {
	e, _, _ := semEnv(t, nil)
	base := council.QueryRequest{Question: "What is X?", DocID: "doc-1", TopK: 4}
	k := e.srv.l1Key(base, ModeAuto, "")
	same := base
	same.Question, same.TopK = "  what is   x? ", 0 // normalisation and effective top_k
	if e.srv.l1Key(same, ModeAuto, "") != k {
		t.Fatal("case/whitespace and default top_k must share a key")
	}
	other := []string{}
	q := base
	q.DocID = "doc-2"
	other = append(other, e.srv.l1Key(q, ModeAuto, ""))
	q = base
	q.TopK = 6
	other = append(other, e.srv.l1Key(q, ModeAuto, ""))
	other = append(other, e.srv.l1Key(base, ModeOverride, "direct"), e.srv.l1Key(base, ModeOverride, "council"))
	e.srv.Tenant = "org-42"
	other = append(other, e.srv.l1Key(base, ModeAuto, ""))
	e.srv.Tenant = ""
	e.srv.Router = agent.NewRouter(e.llm.Client("other-router"))
	other = append(other, e.srv.l1Key(base, ModeAuto, ""))
	seen := map[string]bool{k: true}
	for i, o := range other {
		if seen[o] {
			t.Fatalf("dimension %d did not change the key", i)
		}
		seen[o] = true
	}
	if !strings.HasPrefix(k, "councilai:v3:l1:") {
		t.Fatalf("unversioned key %s", k)
	}
}

func TestL2SemanticHitAfterRouting(t *testing.T) {
	e, _, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	_, first, _ := e.query(t, docQ("What is the capital?", ""))
	if l2.Len() != 1 || first.Cache.L2 != "miss" {
		t.Fatalf("L2 not stored: %+v", first.Cache)
	}
	members := e.llm.Count("member")
	routes := e.llm.Count("route")
	_, hit, _ := e.query(t, docQ("paraphrase of the capital question", ""))
	c := hit.Cache
	if !hit.CacheHit || c.Tier != "l2_semantic" || c.L1 != "miss" || c.InferenceExecuted || !c.RouterCalled {
		t.Fatalf("L2 hit metadata %+v", c)
	}
	if math.Abs(*c.Similarity-0.95) > 1e-6 || math.Abs(*c.Distance-0.05) > 1e-6 || *c.Threshold != 0.85 {
		t.Fatalf("similarity %v distance %v", *c.Similarity, *c.Distance)
	}
	if e.llm.Count("member") != members || e.llm.Count("route") != routes+1 {
		t.Fatal("L2 hit must run the router only")
	}
	if hit.Answer != first.Answer {
		t.Fatal("wrong cached answer")
	}
}

func TestL2HitOverSSEEmitsNoCouncilEvents(t *testing.T) {
	e, _, _ := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	e.query(t, docQ("What is the capital?", ""))
	events := parseSSE(t, postQuery(t, e.srv, docQ("paraphrase sse capital", ""), true).Body.String())
	var names []string
	for _, ev := range events {
		names = append(names, ev.name)
	}
	if strings.Join(names, ",") != "start,routing,final" {
		t.Fatalf("events %v", names)
	}
}

func TestL2ScopeIsolation(t *testing.T) {
	cases := map[string]func(e *env) string{
		"different document": func(e *env) string { return `{"question":"paraphrase capital","doc_id":"doc-2"}` },
		"override vs auto":   func(e *env) string { return docQ("paraphrase capital", `,"strategy":"council"`) },
		"different top_k":    func(e *env) string { return docQ("paraphrase capital", `,"top_k":7`) },
		"different strategy chosen by router": func(e *env) string {
			e.llm = llmtest.New(map[string]llmtest.Reply{"route/router": route("council_deep", true)})
			e.srv.Router = agent.NewRouter(e.llm.Client("router"))
			return docQ("paraphrase capital", "")
		},
		"different embedding model": func(e *env) string {
			old := e.rag.embed
			e.rag.embed = func(text string) ([]float32, string) { v, _ := old(text); return v, "other/embedder" }
			return docQ("paraphrase capital", "")
		},
	}
	for name, mutate := range cases {
		e, _, _ := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
		e.query(t, docQ("What is the capital?", ""))
		_, res, _ := e.query(t, mutate(e))
		if res.CacheHit || res.Cache.L2 != "miss" {
			t.Fatalf("%s: reused across scopes: %+v", name, res.Cache)
		}
	}
}

func TestL2EligibilityRules(t *testing.T) {
	// Web-grounded (no document): never looked up or stored in L2.
	e, _, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	_, res, _ := e.query(t, `{"question":"general capital question"}`)
	if res.Cache.L2 != "ineligible" || l2.Len() != 0 || l2.Gets != 0 || res.Grounding != "web" {
		t.Fatalf("web answer: %+v", res.Cache)
	}
	// Document attached but judged irrelevant.
	e, _, l2 = semEnv(t, map[string]llmtest.Reply{"route/router": route("direct", false), "summarize/ingest": {Text: "Tax law."}})
	e.ingest(t) // the router can only judge relevance with a summary (else the document is kept)
	_, res, _ = e.query(t, docQ("hello capital", ""))
	if res.Cache.L2 != "ineligible" || l2.Len() != 0 {
		t.Fatalf("irrelevant doc: %+v", res.Cache)
	}
	// Router fallback: no lookup, nothing stored anywhere.
	e, l1, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": {Text: "not json"}})
	_, res, _ = e.query(t, docQ("capital fallback", ""))
	if res.Cache.L2 != "ineligible" || l2.Len() != 0 || len(l1.Keys()) != 0 {
		t.Fatalf("fallback cached: %+v keys %v", res.Cache, l1.Keys())
	}
	// Degraded council (chairman invalid): not stored in either tier.
	e, l1, l2 = semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true), "chair/chair": {Text: "nope"}})
	_, res, _ = e.query(t, docQ("capital degraded", ""))
	if len(res.Degraded) == 0 || l2.Len() != 0 || len(res.Cache.Stored) != 0 {
		t.Fatalf("degraded cached: %+v", res.Cache)
	}
	for _, k := range l1.Keys() {
		if strings.Contains(k, ":l1:") {
			t.Fatal("degraded answer in L1")
		}
	}
}

func TestDirectDocumentAnswersUseTheirOwnScope(t *testing.T) {
	e, _, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("direct", true)})
	e.query(t, docQ("What is the capital?", ""))
	if l2.Len() != 1 {
		t.Fatal("document-grounded direct answers are L2-eligible")
	}
	e.llm = llmtest.New(map[string]llmtest.Reply{"route/router": route("council", true)})
	e.srv.Router = agent.NewRouter(e.llm.Client("router"))
	if _, res, _ := e.query(t, docQ("paraphrase capital", "")); res.CacheHit {
		t.Fatal("a council request must not reuse a direct answer")
	}
}

func TestCacheFailuresDegradeGracefully(t *testing.T) {
	type tc struct {
		setup func(l1 *cachetest.Exact, l2 *cachetest.Semantic, e *env)
		check func(t *testing.T, res council.QueryResult)
	}
	cases := map[string]tc{
		"L1 get error": {func(l1 *cachetest.Exact, _ *cachetest.Semantic, _ *env) { l1.GetErr = errors.New("i/o timeout") },
			func(t *testing.T, r council.QueryResult) {
				if r.Cache.L1 != "error" || !r.Cache.InferenceExecuted {
					t.Fatalf("%+v", r.Cache)
				}
			}},
		"L1 set error (OOM)": {func(l1 *cachetest.Exact, _ *cachetest.Semantic, _ *env) {
			l1.SetErr = errors.New("OOM command not allowed when used memory > 'maxmemory'")
		},
			func(t *testing.T, r council.QueryResult) {
				if !strings.Contains(r.Cache.WriteError, "OOM") || r.Answer == "" {
					t.Fatalf("%+v", r.Cache)
				}
			}},
		"circuit open": {func(l1 *cachetest.Exact, l2 *cachetest.Semantic, _ *env) {
			l1.GetErr, l2.GetErr = cache.ErrCircuitOpen, cache.ErrCircuitOpen
		},
			func(t *testing.T, r council.QueryResult) {
				if r.Cache.L1 != "circuit_open" || r.Cache.L2 != "circuit_open" || r.Answer == "" {
					t.Fatalf("%+v", r.Cache)
				}
			}},
		"L2 KNN error": {func(_ *cachetest.Exact, l2 *cachetest.Semantic, _ *env) { l2.GetErr = errors.New("ft.search failed") },
			func(t *testing.T, r council.QueryResult) {
				if r.Cache.L2 != "error" || !r.Cache.InferenceExecuted {
					t.Fatalf("%+v", r.Cache)
				}
			}},
		"embed failure": {func(_ *cachetest.Exact, _ *cachetest.Semantic, e *env) { e.rag.Close() },
			func(t *testing.T, r council.QueryResult) {
				// RAG down: embedding fails (L2 skipped) and retrieval fails (ungrounded, not cached).
				if r.Cache.L2 != "embed_failed" || len(r.Cache.Stored) != 0 {
					t.Fatalf("%+v", r.Cache)
				}
			}},
		"invalid embedding dimension": {func(_ *cachetest.Exact, _ *cachetest.Semantic, e *env) {
			e.rag.embed = func(string) ([]float32, string) { return make([]float32, 12), "m" }
		},
			func(t *testing.T, r council.QueryResult) {
				if r.Cache.L2 != "embed_failed" || r.Cache.Stored[0] != "l1_exact" || len(r.Cache.Stored) != 1 {
					t.Fatalf("%+v", r.Cache)
				}
			}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			e, l1, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
			c.setup(l1, l2, e)
			code, res, body := e.query(t, docQ("capital "+name, ""))
			if code != 200 {
				t.Fatalf("%d %s", code, body)
			}
			c.check(t, res)
		})
	}
}

func TestCorruptL1EntryIsRecomputed(t *testing.T) {
	e, l1, _ := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	q := council.QueryRequest{Question: "capital corrupt", DocID: "doc-1"}
	l1.PutRaw(e.srv.l1Key(q, ModeAuto, ""), []byte("{not json"))
	_, res, _ := e.query(t, docQ("capital corrupt", ""))
	if res.CacheHit || res.Cache.L1 != "corrupt_entry" || !res.Cache.InferenceExecuted {
		t.Fatalf("%+v", res.Cache)
	}
}

func TestSummariesLiveInTheCache(t *testing.T) {
	e, l1, _ := semEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Text: "Zogland summary."}})
	e.ingest(t)
	key := "councilai:v3:doc_summary:doc-1"
	if l1.TTL(key) != time.Hour {
		t.Fatalf("summary key/TTL: %v %s", l1.Keys(), l1.TTL(key))
	}
	// A new server (Go restart) with the same cache still routes with the summary.
	srv2 := NewServer(e.srv.Council, e.rag.URL)
	srv2.Cache = l1
	if s, ok := srv2.getSummary(context.Background(), "doc-1"); !ok || s != "Zogland summary." {
		t.Fatal("summary lost across server restart")
	}
	// Redis rejecting the write: ingestion succeeds, summary reported not stored.
	e2, l1b, _ := semEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Text: "S"}})
	l1b.SetErr = errors.New("connection refused")
	if out := e2.ingest(t); out["summary_status"] != "not_stored" || out["summary"] != "S" {
		t.Fatalf("%v", out)
	}
	// Without any cache, summaries cannot be kept; the document is still used.
	e3 := newEnv(t, map[string]llmtest.Reply{"summarize/ingest": {Text: "S"}, "route/router": route("direct", false)})
	e3.srv.Cache = nil
	e3.ingest(t)
	if _, res, _ := e3.query(t, docQ("anything", "")); res.Routing.DocumentSummary != "missing" || !res.Routing.DocumentUsed {
		t.Fatalf("%+v", res.Routing)
	}
}

func TestCancelledRequestIsNotCached(t *testing.T) {
	e, l1, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	e.llm.Hook = func(ctx context.Context, stage, model string) error {
		if stage == "chair" {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}
	e.srv.RequestTimeout = 100 * time.Millisecond
	code, _, _ := e.query(t, docQ("capital slow", ""))
	if code == 200 || l2.Len() != 0 || len(l1.Keys()) != 0 {
		t.Fatalf("cancelled request cached: %d %v", code, l1.Keys())
	}
}
