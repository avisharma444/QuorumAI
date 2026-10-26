// Command server runs the CouncilAI Go backend.
package main

import (
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
	"example.com/councilai-starter/internal/config"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm"
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

	srv := api.NewServer(council.New(clients.Members, clients.Chairman, cfg.StageTimeout), cfg.RAGServiceURL)
	srv.RequestTimeout = cfg.RequestTimeout
	srv.Router = agent.NewRouter(clients.Router)
	srv.IngestAgent = agent.NewIngestAgent(clients.Ingestion)
	srv.WebSearch = cfg.WebSearch
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
	}()

	for _, line := range startupSummary(cfg, clients) {
		log.Println(line)
	}
	log.Printf("Listening on :%s", cfg.ServerPort)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("Server failed: %v", err)
	}
}
