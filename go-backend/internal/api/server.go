// Package api serves the HTTP and SSE endpoints and the exact-match cache.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"example.com/councilai-starter/internal/agent"
	"example.com/councilai-starter/internal/cache"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm"
)

// Routing modes.
const (
	ModeAuto     = "auto"     // the LLM router chooses (normal workflow)
	ModeOverride = "override" // explicit strategy in the request (diagnostic)
)

type Server struct {
	Council     *council.Council
	Router      *agent.Router      // nil: automatic routing uses the default council plan
	IngestAgent *agent.IngestAgent // nil: documents are not summarized
	RAGURL      string
	// WebSearch enables web grounding for questions without a document
	// (reference behaviour). Tests that have no RAG service turn it off.
	WebSearch bool
	// RequestTimeout bounds one query, including routing, retrieval and every stage.
	RequestTimeout time.Duration
	HTTP           *http.Client
	// Cache is the L1 exact cache, also holding document summaries. In
	// production it is Redis; nil disables it (no in-process fallback).
	Cache ExactCache
	// Semantic is the L2 RediSearch cache; nil disables it.
	Semantic          cache.SemanticCache
	SemanticThreshold float64
	// Breaker is the shared Redis breaker, reported by /health and /metrics.
	Breaker cache.CircuitBreaker
	// Tenant is the authorization namespace in cache keys ("public" until
	// authentication exists).
	Tenant   string
	Requests atomic.Uint64
	Hits     atomic.Uint64 // L1 hits
	L2Hits   atomic.Uint64
	Failures atomic.Uint64
}

func NewServer(c *council.Council, ragURL string) *Server {
	return &Server{
		Council:           c,
		RAGURL:            strings.TrimRight(ragURL, "/"),
		WebSearch:         true,
		RequestTimeout:    120 * time.Second,
		HTTP:              &http.Client{Timeout: 25 * time.Second},
		SemanticThreshold: cache.DefaultThreshold,
	}
}

func respond(w http.ResponseWriter, code int, val any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(val)
}
func respondError(w http.ResponseWriter, code int, msg string) {
	respond(w, code, map[string]string{"error": msg})
}

// docIDPattern bounds document IDs (the RAG service issues 16 hex chars;
// the reference's filename-based IDs also fit).
var docIDPattern = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

// modelFingerprint identifies the model configuration so cached answers are
// never reused across a different council, chairman or router.
func (s *Server) modelFingerprint() string {
	var b strings.Builder
	for _, m := range s.Council.Members {
		b.WriteString(m.ModelName() + ",")
	}
	if s.Council.Chairman != nil {
		b.WriteString("|chair=" + s.Council.Chairman.ModelName())
	}
	if s.Router != nil {
		b.WriteString("|router=" + s.Router.ModelName())
	}
	b.WriteString("|web=" + strconv.FormatBool(s.WebSearch))
	return b.String()
}

// requestError is an error with an HTTP status for the client.
type requestError struct {
	status int
	msg    string
}

func (e *requestError) Error() string { return e.msg }

// plan decides the strategy and whether the document is needed.
func (s *Server) plan(ctx context.Context, q council.QueryRequest, mode, override string) (council.RoutingInfo, *llm.Usage, error) {
	info := council.RoutingInfo{Mode: mode, DocumentAttached: q.DocID != ""}
	var summary string
	if info.DocumentAttached {
		var ok bool
		summary, ok = s.getSummary(ctx, q.DocID)
		info.DocumentSummary = map[bool]string{true: "available", false: "missing"}[ok]
	}
	if mode == ModeOverride {
		info.Strategy, info.NeedsDoc = override, info.DocumentAttached
		info.Reasoning = "Explicit strategy override (diagnostic); router not consulted."
		return info, nil, nil
	}
	if s.Router == nil {
		info.Strategy, info.NeedsDoc, info.Fallback = agent.StrategyCouncil, info.DocumentAttached, true
		info.Reasoning = "No router configured, using default council strategy."
		return info, nil, nil
	}
	p, err := s.Router.PlanWithDocument(ctx, q.Question, agent.DocumentContext{Attached: info.DocumentAttached, Summary: summary})
	if err != nil {
		return info, nil, err
	}
	info.Strategy, info.Reasoning, info.NeedsDoc, info.Fallback = p.Strategy, p.Reasoning, p.NeedsDoc, p.Fallback
	info.RouterModel = s.Router.ModelName()
	return info, p.Usage, nil
}

// evidence gathers grounding for the plan (reference: fetchQueryContext):
//   - document needed: retrieve its chunks;
//   - no document: web search;
//   - document attached but judged irrelevant: no external evidence.
//
// Retrieval and search failures degrade to an ungrounded answer and are
// recorded in info.GroundingError; an unknown document is a client error.
func (s *Server) evidence(ctx context.Context, q council.QueryRequest, info *council.RoutingInfo) (council.Evidence, *council.RetrievalInfo, error) {
	switch {
	case info.DocumentAttached && info.NeedsDoc:
		chunks, retrieval, err := s.Retrieve(ctx, q)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return council.Evidence{}, nil, ctxErr
		}
		if errors.Is(err, errDocumentNotFound) {
			return council.Evidence{}, nil, &requestError{http.StatusNotFound, "document not found: " + q.DocID}
		}
		if err != nil {
			log.Printf("[Query] Retrieval failed: %v, answering without document evidence", err)
			info.GroundingError = "document retrieval failed: " + err.Error()
			return council.Evidence{}, nil, nil
		}
		info.DocumentUsed = len(chunks) > 0
		return council.Evidence{Kind: council.EvidenceDocument, Passages: chunks}, retrieval, nil
	case !info.DocumentAttached && s.WebSearch:
		res, err := s.SearchWeb(ctx, q.Question)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return council.Evidence{}, nil, ctxErr
		}
		switch {
		case err != nil:
			info.GroundingError = "web search failed: " + err.Error()
		case res.Status != "ok" || len(res.Results) == 0:
			info.GroundingError = "web search returned no usable results (" + res.Status + ")"
			if res.Error != nil {
				info.GroundingError += ": " + *res.Error
			}
		default:
			info.WebSources = res.Sources
			return council.Evidence{Kind: council.EvidenceWeb, Passages: res.Results}, nil, nil
		}
		log.Printf("[Query] %s; answering without web grounding", info.GroundingError)
	}
	return council.Evidence{}, nil, nil
}

func addUsage(a, b *llm.Usage) *llm.Usage {
	if b == nil {
		return a
	}
	out := *b
	if a != nil {
		out.PromptTokens += a.PromptTokens
		out.CompletionTokens += a.CompletionTokens
		out.TotalTokens += a.TotalTokens
	}
	return &out
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		respondError(w, 405, "POST required")
		return
	}
	s.Requests.Add(1)
	r.Body = http.MaxBytesReader(w, r.Body, 32<<10)
	var q council.QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
		respondError(w, 400, "invalid JSON query")
		return
	}
	q.Question = strings.TrimSpace(q.Question)
	if q.Question == "" || len(q.Question) > 10000 {
		respondError(w, 400, "question must contain 1–10000 characters")
		return
	}
	q.DocID = strings.TrimSpace(q.DocID)
	if q.DocID != "" && !docIDPattern.MatchString(q.DocID) {
		respondError(w, 400, "invalid doc_id")
		return
	}
	// An explicit strategy is a diagnostic override; "" or "auto" uses the router.
	mode, override := ModeAuto, strings.TrimSpace(strings.ToLower(q.Strategy))
	if override != "" && override != "auto" {
		if !council.ValidStrategy(override) {
			respondError(w, 400, fmt.Sprintf("unknown strategy %q (use auto, direct, council or council_deep)", override))
			return
		}
		mode = ModeOverride
	}
	q.TopK = effectiveTopK(q.TopK)
	wantsStream := strings.Contains(r.Header.Get("Accept"), "text/event-stream")
	started := time.Now()
	ctx, cancel := context.WithTimeout(r.Context(), s.RequestTimeout)
	defer cancel()

	ci := &council.CacheInfo{Tier: "none", L1: "disabled", L2: "disabled"}
	if s.Semantic != nil {
		ci.L2 = "not_checked" // becomes ineligible / miss / hit / ... once L2 is considered
	}
	var lookup time.Duration

	// 1. L1 exact lookup: no model or embedding call is needed on a hit.
	key := s.l1Key(q, mode, override)
	if s.Cache != nil {
		t := time.Now()
		var prev council.QueryResult
		hit, err := s.Cache.Get(ctx, key, &prev)
		lookup += time.Since(t)
		ci.L1 = cacheStatus(err)
		if err != nil && ci.L1 != "cancelled" {
			log.Printf("[Cache] L1 lookup degraded (%s): %v", ci.L1, err)
		}
		if hit {
			s.Hits.Add(1)
			ci.Tier, ci.L1, ci.LookupMS = "l1_exact", "hit", ms(lookup)
			s.serveCached(w, wantsStream, nil, prev, ci, nil, started)
			return
		}
	}

	var emit council.EventFn = func(string, any) {}
	if wantsStream {
		emit = newEventWriter(w).Emit
		emit("start", map[string]string{"mode": mode, "strategy": override})
	}
	fail := func(status int, msg string) {
		s.Failures.Add(1)
		if wantsStream {
			emit("error", map[string]any{"error": msg, "status": status})
		} else {
			respondError(w, status, msg)
		}
	}

	// 2. Routing decides the strategy and whether the document is needed.
	info, routerUsage, err := s.plan(ctx, q, mode, override)
	if err != nil {
		fail(http.StatusGatewayTimeout, "routing cancelled: "+err.Error())
		return
	}
	ci.RouterCalled = mode == ModeAuto && s.Router != nil
	emit("routing", info)

	// 3. L2 semantic lookup, only for document-grounded plans and only
	// within the exact scope of this request (after routing, so the
	// resolved strategy is part of the scope). Web-grounded and ungrounded
	// answers are never semantically cached.
	var vector []float32
	var scope cache.Scope
	if s.Semantic != nil {
		ci.L2 = "ineligible"
		if info.DocumentAttached && info.NeedsDoc && !info.Fallback {
			t := time.Now()
			vec, embModel, err := s.embedQuery(ctx, q.Question)
			if err != nil {
				if ctx.Err() != nil {
					fail(http.StatusGatewayTimeout, "cancelled: "+ctx.Err().Error())
					return
				}
				ci.L2 = "embed_failed"
				log.Printf("[Cache] L2 skipped, embedding failed: %v", err)
			} else {
				vector = vec
				scope = s.semanticScope(q, mode, override, info.Strategy, embModel)
				var prev council.QueryResult
				hit, found, err := s.Semantic.Get(ctx, scope, normalizeQuestion(q.Question), vector, s.SemanticThreshold, &prev)
				ci.L2 = cacheStatus(err)
				if err != nil && ci.L2 != "cancelled" {
					log.Printf("[Cache] L2 lookup degraded (%s): %v", ci.L2, err)
				}
				if found {
					lookup += time.Since(t)
					s.L2Hits.Add(1)
					dist, sim, thr := hit.Distance, hit.Similarity, s.SemanticThreshold
					ci.Tier, ci.L2, ci.LookupMS = "l2_semantic", "hit", ms(lookup)
					ci.Distance, ci.Similarity, ci.Threshold = &dist, &sim, &thr
					s.serveCached(w, wantsStream, emit, prev, ci, routerUsage, started)
					return
				}
			}
			lookup += time.Since(t)
		}
	}
	ci.LookupMS = ms(lookup)

	// 4. Normal execution.
	ev, retrieval, err := s.evidence(ctx, q, &info)
	var reqErr *requestError
	if errors.As(err, &reqErr) {
		fail(reqErr.status, reqErr.msg)
		return
	}
	if err != nil {
		fail(http.StatusGatewayTimeout, "evidence retrieval cancelled: "+err.Error())
		return
	}
	grounding := council.EvidenceNone
	if len(ev.Passages) > 0 {
		grounding = ev.Kind
	}
	emit("retrieved", map[string]any{"chunks": len(ev.Passages), "grounding": grounding})

	// The final result is decorated, stored (if eligible) and then emitted,
	// so the streamed "final" event reports what was actually cached.
	var final *council.QueryResult
	finalize := func(res council.QueryResult) council.QueryResult {
		infoCopy := info
		res.Routing = &infoCopy
		res.Retrieval = retrieval
		res.TokenUsage = addUsage(res.TokenUsage, routerUsage)
		ci.InferenceExecuted = true
		res.Cache = ci
		if cacheableResult(info, res) {
			s.storeResult(ctx, key, scope, q.Question, vector, res, ci)
		}
		final = &res
		return res
	}
	runEmit := func(name string, v any) {
		if res, ok := v.(council.QueryResult); ok && name == "final" {
			v = finalize(res)
		}
		emit(name, v)
	}
	q.Strategy = info.Strategy
	result, err := s.Council.Run(ctx, q, ev, runEmit)
	if err != nil {
		fail(502, err.Error())
		return
	}
	if final == nil {
		result = finalize(result)
	} else {
		result = *final
	}
	if !wantsStream {
		respond(w, 200, result)
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// storeResult writes an eligible result to L1 and, for document-grounded
// answers with a lookup embedding, to L2. Writes use a detached context
// (the answer is complete even if the client has gone) with a short
// deadline; failures are reported, never fatal.
func (s *Server) storeResult(ctx context.Context, key string, scope cache.Scope, question string, vector []float32, res council.QueryResult, ci *council.CacheInfo) {
	stored := res
	stored.Cache, stored.CacheHit = nil, false
	wctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Second)
	defer cancel()
	var errs []string
	if s.Cache != nil {
		if err := s.Cache.Set(wctx, key, stored); err != nil {
			errs = append(errs, "l1: "+err.Error())
		} else {
			ci.Stored = append(ci.Stored, "l1_exact")
		}
	}
	if s.Semantic != nil && vector != nil && res.Grounding == council.EvidenceDocument {
		if err := s.Semantic.Put(wctx, scope, normalizeQuestion(question), vector, stored); err != nil {
			errs = append(errs, "l2: "+err.Error())
		} else {
			ci.Stored = append(ci.Stored, "l2_semantic")
		}
	}
	if len(errs) > 0 {
		ci.WriteError = strings.Join(errs, "; ")
		log.Printf("[Cache] write degraded: %s", ci.WriteError)
	}
}

// serveCached returns a cached answer. TokenUsage and DurationMS are
// replaced with this request's figures (router tokens for an L2 hit,
// nothing for L1); the original generation's are kept in ci. No candidate,
// review or synthesis events are emitted, because none ran.
func (s *Server) serveCached(w http.ResponseWriter, wantsStream bool, emit council.EventFn, prev council.QueryResult, ci *council.CacheInfo, currentUsage *llm.Usage, started time.Time) {
	ci.InferenceExecuted = false
	ci.OriginalTokenUsage, ci.OriginalDurationMS = prev.TokenUsage, prev.DurationMS
	prev.CacheHit = true
	prev.TokenUsage = currentUsage
	prev.DurationMS = time.Since(started).Milliseconds()
	prev.Cache = ci
	switch {
	case !wantsStream:
		respond(w, 200, prev)
	case emit != nil:
		emit("final", prev)
	default:
		newEventWriter(w).Emit("final", prev)
	}
}

type eventWriter struct {
	w     http.ResponseWriter
	mu    sync.Mutex
	flush http.Flusher
}

func newEventWriter(w http.ResponseWriter) *eventWriter {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	f, _ := w.(http.Flusher)
	return &eventWriter{w: w, flush: f}
}
func (e *eventWriter) Emit(event string, data any) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b, err := json.Marshal(data)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(e.w, "event: %s\ndata: %s\n\n", event, b)
	if e.flush != nil {
		e.flush.Flush()
	}
}

// redisHealth reports Redis reachability without affecting liveness: the
// service answers questions without Redis.
func (s *Server) redisHealth(ctx context.Context) string {
	p, ok := s.Cache.(interface{ Ping(context.Context) error })
	if s.Cache == nil || !ok {
		return "disabled"
	}
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	if err := p.Ping(ctx); err != nil {
		if errors.Is(err, cache.ErrCircuitOpen) {
			return "circuit_open"
		}
		return "unhealthy"
	}
	return "healthy"
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		body := map[string]any{"status": "ok", "service": "go-council", "redis": s.redisHealth(r.Context())}
		if s.Breaker != nil {
			body["redis_breaker"] = s.Breaker.State().String()
		}
		respond(w, 200, body)
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		body := map[string]any{"requests": s.Requests.Load(), "exact_cache_hits": s.Hits.Load(), "semantic_cache_hits": s.L2Hits.Load(), "failures": s.Failures.Load()}
		if s.Breaker != nil {
			cs, cf, ts, tf := s.Breaker.Counts()
			body["redis_breaker"] = map[string]any{"state": s.Breaker.State().String(), "consecutive_successes": cs, "consecutive_failures": cf, "total_successes": ts, "total_failures": tf}
		}
		respond(w, 200, body)
	})
	mux.HandleFunc("/api/v1/query", s.handleQuery)
	mux.HandleFunc("/api/v1/ingest", s.handleIngest)
	return mux
}
