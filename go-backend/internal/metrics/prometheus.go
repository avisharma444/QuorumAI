// Package metrics defines the Prometheus metrics (reference:
// internal/metrics/prometheus.go). The six reference families keep their
// names; labels are restricted to small, fixed value sets. User IDs,
// document IDs, session IDs, questions, model output and raw URLs are never
// used as label values.
package metrics

import (
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Buckets in seconds. HTTP covers sub-millisecond cache hits up to the
// 2-minute request timeout; LLM calls start at 10 ms (mock models are fast).
var (
	httpBuckets  = []float64{.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120}
	stageBuckets = []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120}
	fastBuckets  = []float64{.0001, .00025, .0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1}
)

// ── Reference families ──────────────────────────────────────────────

var (
	// RequestCount: one per completed HTTP request. route is the chi route
	// pattern ("unmatched" for 404s), never the raw path.
	RequestCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_request_count_total",
		Help: "HTTP requests by method, route pattern and status code.",
	}, []string{"method", "route", "status"})

	RequestLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "councilai_request_latency_seconds",
		Help:    "HTTP request duration in seconds (SSE: until the stream ends).",
		Buckets: httpBuckets,
	}, []string{"method", "route"})

	// ChairmanSynthesis counts chairman synthesis invocations (one per
	// council run that reaches the chairman; a repair retry is not a second
	// invocation). The reference declared this counter but never incremented it.
	ChairmanSynthesis = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_chairman_synthesis_count_total",
		Help: "Chairman synthesis invocations by result (success, failure).",
	}, []string{"result"})

	// LLMFailures counts failed model requests of every stage (the
	// reference counted only candidate failures).
	LLMFailures = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_llm_failure_count_total",
		Help: "Failed model requests by stage and provider.",
	}, []string{"stage", "provider"})

	// CacheOperations: exactly one observation per cache tier per query
	// request (lookups only; writes are councilai_cache_writes_total).
	CacheOperations = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_cache_operations_total",
		Help: "Cache lookups per query by level (l1_exact, l2_semantic) and result.",
	}, []string{"level", "result"})

	// CouncilResponse: one observation per successful orchestration run
	// (the reference observed it twice per request).
	CouncilResponse = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "councilai_council_response_seconds",
		Help:    "Orchestration duration (candidates through final answer) by executed strategy.",
		Buckets: stageBuckets,
	}, []string{"strategy"})
)

// ── Extensions ─────────────────────────────────────────────────────

var (
	InFlight = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "councilai_http_requests_in_flight",
		Help: "HTTP requests currently being served.",
	})

	ClientDisconnects = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_http_client_disconnects_total",
		Help: "Requests whose client went away before the handler finished.",
	}, []string{"route"})

	// QueryOutcomes: one per /query request that reached the handler.
	QueryOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_query_total",
		Help: "Query requests by executed strategy, answer tier (none, l1_exact, l2_semantic) and outcome.",
	}, []string{"strategy", "tier", "outcome"})

	StageDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "councilai_council_stage_seconds",
		Help:    "Council stage wall time: candidates, peer_review, chairman, reflection, revision, direct.",
		Buckets: stageBuckets,
	}, []string{"stage"})

	LLMRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "councilai_llm_request_seconds",
		Help:    "Individual model request duration by stage, provider and result.",
		Buckets: stageBuckets,
	}, []string{"stage", "provider", "result"})

	// LLMTokens is added only from provider-reported usage; nothing is estimated.
	LLMTokens = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_llm_tokens_total",
		Help: "Provider-reported tokens by stage, provider and type (prompt, completion).",
	}, []string{"stage", "provider", "type"})

	RouterDecisions = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_router_decisions_total",
		Help: "LLM router plans by chosen strategy and whether the default plan was used after a router failure.",
	}, []string{"strategy", "fallback"})

	CacheLookupDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "councilai_cache_lookup_seconds",
		Help:    "Cache lookup duration by level (l2 includes the query embedding).",
		Buckets: fastBuckets,
	}, []string{"level"})

	CacheWrites = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_cache_writes_total",
		Help: "Cache insertions by level and result (ok, error).",
	}, []string{"level", "result"})

	BreakerTransitions = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_circuit_breaker_transitions_total",
		Help: "Circuit breaker state changes by breaker and new state.",
	}, []string{"breaker", "to"})

	RAGRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "councilai_rag_request_seconds",
		Help:    "Requests to the Python RAG service by endpoint and result.",
		Buckets: stageBuckets,
	}, []string{"endpoint", "result"})

	DependencyErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_dependency_errors_total",
		Help: "Failed calls to PostgreSQL, Redis or the RAG service by operation.",
	}, []string{"dependency", "operation"})

	RateLimitRejections = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_rate_limit_rejections_total",
		Help: "Requests refused by a rate limiter: reason=limited (429) or unavailable (503, fail-closed).",
	}, []string{"limiter", "reason"})

	ConversationOps = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "councilai_conversation_operations_total",
		Help: "Conversation memory loads and saves by result.",
	}, []string{"operation", "result"})

	TelemetryExportErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "councilai_telemetry_errors_total",
		Help: "OpenTelemetry SDK errors (for example an unreachable trace exporter).",
	})
)

// BreakerState exposes a breaker's state as 0 closed, 1 half-open, 2 open.
func BreakerState(name string, state func() float64) {
	prometheus.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name:        "councilai_circuit_breaker_state",
		Help:        "Circuit breaker state: 0 closed, 1 half-open, 2 open.",
		ConstLabels: prometheus.Labels{"breaker": name},
	}, state))
}

// Provider maps a model name ("openrouter:openrouter/free", "mock:member-1")
// to a fixed provider label, so model names never become label values.
func Provider(modelName string) string {
	p, _, found := strings.Cut(modelName, ":")
	if !found {
		// OpenRouter model names are unprefixed "vendor/model" ids.
		if strings.Contains(modelName, "/") {
			return "openrouter"
		}
		return "other"
	}
	switch p {
	case "gemini", "local", "mock":
		return p
	case "nim":
		return "nvidia-nim"
	}
	return "other"
}

// CacheResult maps a CacheInfo status to a bounded label value.
func CacheResult(status string) string {
	switch status {
	case "hit", "miss", "circuit_open", "error", "cancelled", "disabled", "disabled_by_policy",
		"bypassed_conversation", "ineligible", "embed_failed", "corrupt_entry", "not_checked":
		return status
	}
	return "other"
}

// Known label values, used to create every bounded series at zero.
var (
	Stages    = []string{"candidate", "review", "chairman", "reflection", "revision", "direct", "router", "ingest_summary"}
	providers = []string{"gemini", "openrouter", "nvidia-nim", "local", "mock", "other"}
	Routes    = []string{"/health", "/api/v1/register", "/api/v1/login", "/api/v1/query", "/api/v1/ingest",
		"/api/v1/conversation/clear", "/api/v1/conversation/history"}
)

// Init creates the counter series for all known label combinations at 0.
// A counter series that first appears with value 1 is invisible to
// increase()/rate() (they need two samples), so without this the first
// failure of each kind would never show on the dashboard.
func Init() {
	for _, st := range Stages {
		for _, p := range providers {
			LLMFailures.WithLabelValues(st, p)
		}
	}
	for _, r := range []string{"success", "failure"} {
		ChairmanSynthesis.WithLabelValues(r)
	}
	for dep, ops := range map[string][]string{
		"postgres": {"document_access", "users", "grant_document"},
		"redis":    {"rate_limit", "conversation_load", "conversation_save", "conversation"},
		"rag":      {"retrieve", "search", "embed", "ingest"},
	} {
		for _, op := range ops {
			DependencyErrors.WithLabelValues(dep, op)
		}
	}
	for _, l := range []string{"user", "auth"} {
		for _, r := range []string{"limited", "unavailable"} {
			RateLimitRejections.WithLabelValues(l, r)
		}
	}
	for _, to := range []string{"open", "half_open", "closed"} {
		BreakerTransitions.WithLabelValues("redis", to)
	}
	for _, r := range Routes {
		ClientDisconnects.WithLabelValues(r)
	}
	for _, op := range []string{"load", "save"} {
		for _, r := range []string{"ok", "error"} {
			ConversationOps.WithLabelValues(op, r)
		}
	}
}
