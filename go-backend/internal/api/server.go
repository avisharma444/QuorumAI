// Package api serves the HTTP and SSE endpoints and the exact-match cache.
package api

import (
	"context"
	"crypto/sha256"
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
	Cache          *MemoryCache[council.QueryResult]
	Summaries      *MemoryCache[string] // doc_id → summary (reference: Redis "doc_summary:<id>")
	Requests       atomic.Uint64
	Hits           atomic.Uint64
	Failures       atomic.Uint64
}

func NewServer(c *council.Council, ragURL string) *Server {
	return &Server{
		Council:        c,
		RAGURL:         strings.TrimRight(ragURL, "/"),
		WebSearch:      true,
		RequestTimeout: 120 * time.Second,
		HTTP:           &http.Client{Timeout: 25 * time.Second},
		Cache:          NewMemoryCache[council.QueryResult](time.Hour),
		Summaries:      NewMemoryCache[string](time.Hour),
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

// cacheKey separates automatic routing from each explicit override, so a
// diagnostic "direct" answer is never served to an automatic request.
func (s *Server) cacheKey(q council.QueryRequest, mode, strategy string) string {
	route := mode
	if mode == ModeOverride {
		route += ":" + strategy
	}
	raw := strings.Join([]string{"v2", s.modelFingerprint(), q.DocID, strings.ToLower(q.Question), route, strconv.Itoa(q.TopK)}, "\x00")
	return fmt.Sprintf("%x", sha256.Sum256([]byte(raw)))
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
		summary, ok = s.Summaries.Get(q.DocID)
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
func (s *Server) evidence(ctx context.Context, q council.QueryRequest, info *council.RoutingInfo) (council.Evidence, error) {
	switch {
	case info.DocumentAttached && info.NeedsDoc:
		chunks, err := s.Retrieve(ctx, q)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return council.Evidence{}, ctxErr
		}
		if errors.Is(err, errDocumentNotFound) {
			return council.Evidence{}, &requestError{http.StatusNotFound, "document not found: " + q.DocID}
		}
		if err != nil {
			log.Printf("[Query] Retrieval failed: %v, answering without document evidence", err)
			info.GroundingError = "document retrieval failed: " + err.Error()
			return council.Evidence{}, nil
		}
		info.DocumentUsed = len(chunks) > 0
		return council.Evidence{Kind: council.EvidenceDocument, Passages: chunks}, nil
	case !info.DocumentAttached && s.WebSearch:
		res, err := s.SearchWeb(ctx, q.Question)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return council.Evidence{}, ctxErr
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
			return council.Evidence{Kind: council.EvidenceWeb, Passages: res.Results}, nil
		}
		log.Printf("[Query] %s; answering without web grounding", info.GroundingError)
	}
	return council.Evidence{}, nil
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
	key := s.cacheKey(q, mode, override)
	wantsStream := strings.Contains(r.Header.Get("Accept"), "text/event-stream")
	if prev, ok := s.Cache.Get(key); ok {
		s.Hits.Add(1)
		prev.CacheHit = true
		prev.DurationMS = 0
		if wantsStream {
			newEventWriter(w).Emit("final", prev)
		} else {
			respond(w, 200, prev)
		}
		return
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
	ctx, cancel := context.WithTimeout(r.Context(), s.RequestTimeout)
	defer cancel()

	info, routerUsage, err := s.plan(ctx, q, mode, override)
	if err != nil {
		fail(http.StatusGatewayTimeout, "routing cancelled: "+err.Error())
		return
	}
	emit("routing", info)

	ev, err := s.evidence(ctx, q, &info)
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

	// Attach routing metadata and the router's token usage to the final
	// result, including the copy streamed in the "final" event.
	decorate := func(res *council.QueryResult) {
		info := info
		res.Routing = &info
		res.TokenUsage = addUsage(res.TokenUsage, routerUsage)
	}
	runEmit := func(name string, v any) {
		if res, ok := v.(council.QueryResult); ok && name == "final" {
			decorate(&res)
			v = res
		}
		emit(name, v)
	}
	q.Strategy = info.Strategy
	result, err := s.Council.Run(ctx, q, ev, runEmit)
	if err != nil {
		fail(502, err.Error())
		return
	}
	decorate(&result)
	// Only fully completed executions are cached. Degraded routing,
	// grounding, or council stages (fallback answers, failed reflection or
	// revision) are not, so a later request can get the complete answer.
	if !info.Fallback && info.GroundingError == "" && len(result.Degraded) == 0 {
		s.Cache.Set(key, result)
	}
	if !wantsStream {
		respond(w, 200, result)
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

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		respond(w, 200, map[string]string{"status": "ok", "service": "go-council"})
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		respond(w, 200, map[string]uint64{"requests": s.Requests.Load(), "exact_cache_hits": s.Hits.Load(), "failures": s.Failures.Load()})
	})
	mux.HandleFunc("/api/v1/query", s.handleQuery)
	mux.HandleFunc("/api/v1/ingest", s.handleIngest)
	return mux
}
