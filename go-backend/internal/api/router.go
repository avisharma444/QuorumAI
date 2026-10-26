package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"

	"example.com/councilai-starter/internal/api/middleware"
	"example.com/councilai-starter/internal/cache"
)

// Routes builds the Chi router (reference internal/api/router.go).
//
// Public: /health, /metrics (JSON counters, no user data), /api/v1/register,
// /api/v1/login (rate-limited per client IP when AuthLimiter is set).
// Protected (JWT, then per-user rate limit): query, ingest and conversation
// endpoints. Unauthenticated requests are rejected before any cache, RAG
// or model call.
//
// CORS is off unless CORSOrigins lists explicit origins; credentials are
// never allowed (tokens travel in the Authorization header). The reference
// allowed "*" with credentials, which go-chi/cors answers by reflecting
// any Origin.
func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(chimw.RequestID)
	r.Use(middleware.Logging)
	r.Use(chimw.Recoverer)
	if len(s.CORSOrigins) > 0 {
		r.Use(cors.Handler(cors.Options{
			AllowedOrigins:   s.CORSOrigins,
			AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
			AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type"},
			AllowCredentials: false,
			MaxAge:           300,
		}))
	}

	r.Get("/health", s.handleHealth)
	r.Get("/metrics", s.handleMetrics)

	r.Group(func(r chi.Router) {
		if s.AuthLimiter != nil {
			r.Use(s.AuthLimiter.Middleware(middleware.ByIP))
		}
		r.Post("/api/v1/register", s.handleRegister)
		r.Post("/api/v1/login", s.handleLogin)
	})

	r.Group(func(r chi.Router) {
		r.Use(middleware.Auth(s.JWT))
		if s.UserLimiter != nil {
			r.Use(s.UserLimiter.Middleware(middleware.ByUser))
		}
		r.Use(chimw.Timeout(s.RequestTimeout + 10*time.Second))
		r.Post("/api/v1/query", s.handleQuery)
		r.Post("/api/v1/ingest", s.handleIngest)
		r.Post("/api/v1/conversation/clear", s.handleClearConversation)
		r.Post("/api/v1/conversation/history", s.handleConversationHistory)
	})
	return r
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	body := map[string]any{"status": "ok", "service": "go-council", "redis": s.redisHealth(r.Context()), "postgres": s.postgresHealth(r.Context())}
	if s.Breaker != nil {
		body["redis_breaker"] = s.Breaker.State().String()
	}
	respond(w, 200, body)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	body := map[string]any{"requests": s.Requests.Load(), "exact_cache_hits": s.Hits.Load(), "semantic_cache_hits": s.L2Hits.Load(), "failures": s.Failures.Load()}
	if s.Breaker != nil {
		cs, cf, ts, tf := s.Breaker.Counts()
		body["redis_breaker"] = map[string]any{"state": s.Breaker.State().String(), "consecutive_successes": cs, "consecutive_failures": cf, "total_successes": ts, "total_failures": tf}
	}
	respond(w, 200, body)
}

func (s *Server) postgresHealth(ctx context.Context) string {
	if s.Users == nil {
		return "disabled"
	}
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	if err := s.Users.Ping(ctx); err != nil {
		return "unhealthy"
	}
	return "healthy"
}

// redisHealth reports Redis reachability without affecting liveness.
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
