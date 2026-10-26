// Command server runs the CouncilAI Go backend.
package main

import (
	"example.com/councilai-starter/internal/metrics"
	"example.com/councilai-starter/internal/telemetry"

	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"example.com/councilai-starter/internal/agent"
	"example.com/councilai-starter/internal/api"
	"example.com/councilai-starter/internal/api/middleware"
	"example.com/councilai-starter/internal/auth"
	"example.com/councilai-starter/internal/cache"
	"example.com/councilai-starter/internal/config"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/memory"
)

// Per-role HTTP client timeouts, as in the reference: members use
// LLM_TIMEOUT, the chairman STAGE_TIMEOUT, the router 15s, ingestion 30s.
const (
	routerTimeout    = 15 * time.Second
	ingestionTimeout = 30 * time.Second
)

// pipelineClients holds one independently configured client per role.
type pipelineClients struct {
	Members   []llm.LLMClient
	Chairman  llm.LLMClient
	Router    llm.LLMClient
	Ingestion llm.LLMClient
}

// newClient is the factory; tests can replace it.
var newClient = llm.NewClientFromProvider

// buildClients creates every client and reports all configuration problems
// together. With MOCK_LLM=true it builds offline mock clients only; it never
// falls back to mocks because a real provider is misconfigured.
func buildClients(cfg *config.Config) (pipelineClients, error) {
	var out pipelineClients
	if cfg.MockLLM {
		for i, slot := range cfg.CouncilSlots {
			if slot.Model != "" {
				out.Members = append(out.Members, &llm.MockClient{Name: fmt.Sprintf("member-%d", i+1)})
			}
		}
		out.Chairman = &llm.MockClient{Name: "chairman"}
		out.Router = &llm.MockClient{Name: "router"}
		out.Ingestion = &llm.MockClient{Name: "ingestion"}
		return out, nil
	}

	keys, urls := cfg.Keys(), cfg.URLs()
	var errs []error
	build := func(role string, slot config.ModelSlot, timeout time.Duration) llm.LLMClient {
		c, err := newClient(slot.Provider, slot.Model, keys, urls, timeout)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", role, err))
		}
		return c
	}
	for i, slot := range cfg.CouncilSlots {
		if slot.Model == "" {
			log.Printf("Council member %d: skipped (no model configured)", i+1)
			continue
		}
		if c := build(fmt.Sprintf("council member %d", i+1), slot, cfg.LLMTimeout); c != nil {
			out.Members = append(out.Members, c)
		}
	}
	out.Chairman = build("chairman", cfg.ChairmanSlot, cfg.StageTimeout)
	out.Router = build("router", cfg.RouterSlot, routerTimeout)
	out.Ingestion = build("ingestion", cfg.IngestionSlot, ingestionTimeout)
	if len(errs) > 0 {
		return pipelineClients{}, errors.Join(errs...)
	}
	return out, nil
}

// startupSummary describes the configuration without credentials.
func startupSummary(cfg *config.Config, clients pipelineClients) []string {
	lines := []string{fmt.Sprintf("Config file: %q | RAG: %s | stage timeout %s, request timeout %s", cfg.ConfigFile, cfg.RAGServiceURL, cfg.StageTimeout, cfg.RequestTimeout)}
	if cfg.MockLLM {
		lines = append(lines, "MOCK_LLM=true: offline mock clients; configured providers are not contacted")
	}
	for i, m := range clients.Members {
		lines = append(lines, fmt.Sprintf("  Member %d: %s", i+1, m.ModelName()))
	}
	lines = append(lines,
		"Chairman: "+clients.Chairman.ModelName(),
		"Router: "+clients.Router.ModelName(),
		"Ingestion summarizer: "+clients.Ingestion.ModelName(),
		fmt.Sprintf("Web search grounding for questions without a document: %t", cfg.WebSearch),
	)
	return lines
}

// setupRedis wires the Redis Stack caches. Redis is an optimisation: if it
// is down at startup the server still starts, the shared breaker opens on
// failures, and caching resumes when Redis recovers. There is no
// in-process fallback cache.
func setupRedis(cfg *config.Config, srv *api.Server) func() {
	r := cfg.Redis
	if !r.Enabled {
		log.Println("WARNING: Redis disabled (REDIS_ENABLED=false): no caching, no conversation memory and NO RATE LIMITING")
		return func() {}
	}
	client := cache.NewClient(r.Addr, r.Password, r.DB)
	breaker := cache.NewCircuitBreaker("redis", cache.Config{
		FailureThreshold: r.FailureThreshold, SuccessThreshold: r.SuccessThreshold,
		Timeout: r.BreakerTimeout, HalfOpenMaxCalls: 1,
	})
	l1 := cache.NewRedisCacheWithBreaker(client, breaker, r.ExactTTL)
	l1.SetOpTimeout(r.OpTimeout)
	l2 := cache.NewRedisSemanticCacheWithBreaker(client, breaker, r.SemanticTTL)
	l2.SetOpTimeout(r.OpTimeout)
	srv.Cache, srv.Semantic, srv.Breaker, srv.SemanticThreshold = l1, l2, breaker, r.SemanticThreshold
	srv.SemanticReuse = r.SemanticReuse
	if !r.SemanticReuse {
		log.Println("[Redis] semantic (L2) answer reuse is OFF (SEMANTIC_CACHE_ENABLED=false); exact L1 caching, summaries, memory and rate limits still use Redis")
	}
	srv.Memory = memory.NewConversationStore(client, breaker, cfg.Conversation.MaxTurns, cfg.Conversation.TTL)
	policy := middleware.FailurePolicy(cfg.RateLimit.FailurePolicy)
	userLimiter, err := middleware.NewRateLimiter(client, breaker, cfg.RateLimit.RPS, cfg.RateLimit.Burst, "user", policy)
	if err != nil {
		log.Fatalf("rate limiter: %v", err)
	}
	authLimiter, err := middleware.NewRateLimiter(client, breaker, cfg.RateLimit.AuthRPS, cfg.RateLimit.AuthBurst, "auth", policy)
	if err != nil {
		log.Fatalf("rate limiter: %v", err)
	}
	srv.UserLimiter, srv.AuthLimiter = userLimiter, authLimiter
	log.Printf("[RateLimit] per user: %.2f req/s, burst %d; register/login per IP: %.2f req/s, burst %d; Redis failure policy: %s",
		cfg.RateLimit.RPS, cfg.RateLimit.Burst, cfg.RateLimit.AuthRPS, cfg.RateLimit.AuthBurst, policy)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := l1.Ping(ctx); err != nil {
		log.Printf("[Redis] not reachable at %s at startup (%v); answering without cache until it recovers", r.Addr, err)
	} else if err := l2.EnsureIndex(ctx); err != nil {
		log.Printf("[Redis] semantic index not ready: %v (retried on first use)", err)
	} else {
		log.Printf("[Redis] connected to %s; L1 TTL %s, L2 TTL %s, threshold %.2f, index %s", r.Addr, r.ExactTTL, r.SemanticTTL, r.SemanticThreshold, cache.SemanticIndexName)
	}
	return func() { _ = client.Close() }
}

// connectPostgres opens the user database, retrying for up to 30 s while
// the container starts, then applies migrations. There is no fallback: the
// reference switched to an in-memory repository with a demo user.
func connectPostgres(cfg *config.Config) (*auth.PostgresUserRepository, error) {
	d := cfg.Database
	pc := auth.PostgresConfig{URL: d.URL, Host: d.Host, Port: d.Port, User: d.User, Password: d.Password, Database: d.Name, SSLMode: d.SSLMode,
		MaxConns: int32(d.MaxOpenConns), MinConns: int32(d.MaxIdleConns), MaxConnLifetime: d.ConnMaxLifetime, MaxConnIdleTime: d.ConnMaxIdleTime}
	var repo *auth.PostgresUserRepository
	var err error
	deadline := time.Now().Add(30 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		repo, err = auth.NewPostgresUserRepository(ctx, pc)
		cancel()
		if err == nil || time.Now().After(deadline) {
			break
		}
		log.Printf("[Postgres] not ready (%v); retrying", err)
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := repo.Migrate(ctx); err != nil {
		repo.Close()
		return nil, err
	}
	if cfg.Auth.DemoUser {
		if err := repo.SeedUser(ctx, "demo", cfg.Auth.DemoPassword, cfg.Auth.BcryptCost); err != nil {
			log.Printf("[Auth] development demo user not created: %v", err)
		} else {
			log.Println("[Auth] APP_ENV=development: demo user ensured (password from AUTH_DEMO_PASSWORD)")
		}
	}
	log.Printf("[Postgres] connected to %s/%s; migrations applied", d.Host, d.Name)
	return repo, nil
}

// initTracer configures OpenTelemetry (reference: initTracer, which
// hard-coded the noop exporter). Tracing problems never stop the server:
// an invalid setup falls back to noop with a warning, and export errors
// are logged (rate-limited) and counted in councilai_telemetry_errors_total.
func initTracer(cfg *config.Config) *telemetry.Tracer {
	telemetry.InstallErrorHandler(metrics.TelemetryExportErrors.Inc, 30*time.Second)
	ob := cfg.Observability
	tc := telemetry.Config{ServiceName: telemetry.InstrumentationName, ServiceVersion: "2.0.0", Environment: cfg.Auth.AppEnv,
		ExporterType: telemetry.ExporterType(ob.TraceExporter), SampleRatio: ob.SampleRatio,
		OTLPEndpoint: ob.OTLPEndpoint, OTLPInsecure: true}
	t, err := telemetry.NewTracerProvider(tc)
	if err != nil {
		log.Printf("[Telemetry] WARNING: tracing disabled: %v", err)
		tc.ExporterType = telemetry.ExporterNoop
		t, _ = telemetry.NewTracerProvider(tc)
	}
	switch t.Exporter() {
	case telemetry.ExporterOTLP:
		log.Printf("[Telemetry] tracing: OTLP/HTTP to %s, sampling %.2f of new traces", ob.OTLPEndpoint, ob.SampleRatio)
	case telemetry.ExporterStdout:
		log.Printf("[Telemetry] tracing: stdout exporter")
	default:
		log.Printf("[Telemetry] tracing off (noop); incoming traceparent is still forwarded to the RAG service")
	}
	return t
}

// startMetricsServer serves Prometheus metrics on an internal-only address.
func startMetricsServer(addr string) *http.Server {
	if addr == "" {
		log.Println("[Metrics] METRICS_ADDR empty: Prometheus endpoint disabled")
		return nil
	}
	ms := &http.Server{Addr: addr, Handler: api.MetricsHandler(), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if err := ms.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("[Metrics] endpoint failed: %v", err)
		}
	}()
	log.Printf("[Metrics] Prometheus endpoint on %s/metrics (internal network only)", addr)
	return ms
}

func main() {
	log.Println("Starting CouncilAI Go backend")
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Configuration error:\n%v", err)
	}
	clients, err := buildClients(cfg)
	if err != nil {
		log.Fatalf("Configuration error:\n%v", err)
	}
	metrics.Init()
	tracer := initTracer(cfg)
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := tracer.Shutdown(ctx); err != nil { // flushes queued spans
			log.Printf("[Telemetry] shutdown: %v", err)
		}
	}()

	srv := api.NewServer(council.New(clients.Members, clients.Chairman, cfg.StageTimeout), cfg.RAGServiceURL)
	srv.RequestTimeout = cfg.RequestTimeout
	srv.Router = agent.NewRouter(clients.Router)
	srv.IngestAgent = agent.NewIngestAgent(clients.Ingestion)
	srv.WebSearch = cfg.WebSearch
	srv.CORSOrigins = cfg.CORSOrigins
	srv.BcryptCost = cfg.Auth.BcryptCost
	srv.HistoryExchanges = cfg.Conversation.HistoryExchanges

	jwtManager, err := auth.NewJWTManager(cfg.Auth.JWTSecret, cfg.Auth.JWTExpiration)
	if err != nil {
		log.Fatalf("Configuration error: JWT_SECRET: %v (set a random secret of at least %d bytes, e.g. openssl rand -base64 48)", err, auth.MinSecretBytes)
	}
	srv.JWT = jwtManager
	users, err := connectPostgres(cfg)
	if err != nil {
		log.Fatalf("PostgreSQL is required for authentication and is unavailable: %v", err)
	}
	defer users.Close()
	srv.Users, srv.Docs = users, users

	closeRedis := setupRedis(cfg, srv)
	defer closeRedis()
	if srv.Breaker != nil {
		b := srv.Breaker
		metrics.BreakerState("redis", func() float64 {
			switch b.State() {
			case cache.StateOpen:
				return 2
			case cache.StateHalfOpen:
				return 1
			}
			return 0
		})
	}
	metricsServer := startMetricsServer(cfg.Observability.MetricsAddr)
	httpServer := &http.Server{
		Addr:         ":" + cfg.ServerPort,
		Handler:      srv.Routes(),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: cfg.RequestTimeout + 5*time.Second, // SSE streams last the whole request
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("Shutting down gracefully...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
		if metricsServer != nil {
			_ = metricsServer.Shutdown(ctx)
		}
	}()

	for _, line := range startupSummary(cfg, clients) {
		log.Println(line)
	}
	log.Printf("Listening on :%s", cfg.ServerPort)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("Server failed: %v", err)
	}
}
