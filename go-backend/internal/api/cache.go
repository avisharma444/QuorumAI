package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"example.com/councilai-starter/internal/cache"
	"example.com/councilai-starter/internal/council"
)

// ExactCache is the L1 exact-match cache (cache.RedisCache in production).
type ExactCache interface {
	Get(ctx context.Context, key string, dest any) (bool, error)
	Set(ctx context.Context, key string, value any) error
	SetWithTTL(ctx context.Context, key string, value any, ttl time.Duration) error
}

const (
	// keyNamespace versions every key this service writes. Bump it when the
	// cached response format or key layout changes, so old entries are
	// never read with new code.
	keyNamespace = "councilai:v3"
	// promptVersion changes whenever a router, council, chairman or
	// reflection prompt changes, invalidating answers built from old prompts.
	promptVersion = "prompts-2026-10-cp5"
	// DefaultTenant is the authorization namespace until authentication
	// exists. Every key and semantic scope includes it, so per-user or
	// per-organisation isolation only needs this value set per request.
	DefaultTenant = "public"
	// summaryTTL matches the reference (summaries share the L1 TTL).
	summaryTTL = time.Hour
)

func digest(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// normalizeQuestion makes trivially different spellings share an entry:
// case and runs of whitespace do not change an answer.
func normalizeQuestion(q string) string {
	return strings.Join(strings.Fields(strings.ToLower(q)), " ")
}

// effectiveTopK is the top_k the RAG request actually uses (see Retrieve).
func effectiveTopK(k int) int {
	if k <= 0 || k > 10 {
		return 4
	}
	return k
}

func (s *Server) tenant() string {
	if s.Tenant == "" {
		return DefaultTenant
	}
	return s.Tenant
}

// l1Key is the exact-match identity of a request. It covers everything
// known before routing that changes the answer: tenant, document, the
// normalised question, auto vs. a specific override strategy, the
// retrieval top_k (reranking is always on), the model configuration
// (members, chairman, router, web search) and the prompt version. The key
// is a SHA-256 digest, so no question text or document content appears in
// Redis key names (the reference used "cache:<doc>:<raw question>").
func (s *Server) l1Key(q council.QueryRequest, mode, override string) string {
	route := mode
	if mode == ModeOverride {
		route += ":" + override
	}
	return keyNamespace + ":l1:" + digest(s.tenant(), q.DocID, normalizeQuestion(q.Question), route,
		"top_k="+strconv.Itoa(effectiveTopK(q.TopK)), "rerank=true", s.modelFingerprint(), promptVersion)
}

// semanticScope is the L2 isolation scope, evaluated after routing. On top
// of the L1 dimensions (except the question) it includes the RESOLVED
// strategy, so a paraphrase routed differently never reuses an answer, and
// the embedding model that produced the vector.
func (s *Server) semanticScope(q council.QueryRequest, mode, override, strategy, embeddingModel string) cache.Scope {
	route := mode
	if mode == ModeOverride {
		route += ":" + override
	}
	return cache.Scope{
		DocID: q.DocID,
		Key: digest(s.tenant(), q.DocID, route, "strategy="+strategy, "grounding=document",
			"top_k="+strconv.Itoa(effectiveTopK(q.TopK)), "rerank=true", s.modelFingerprint(), promptVersion, "embedding="+embeddingModel),
	}
}

func summaryKey(docID string) string { return keyNamespace + ":doc_summary:" + docID }

// getSummary reads a document summary from Redis. Any failure counts as
// "missing", which keeps the document (conservative routing rule).
func (s *Server) getSummary(ctx context.Context, docID string) (string, bool) {
	if s.Cache == nil {
		return "", false
	}
	var summary string
	ok, err := s.Cache.Get(ctx, summaryKey(docID), &summary)
	if err != nil {
		log.Printf("[Cache] summary lookup for %s failed: %v", docID, err)
		return "", false
	}
	return summary, ok && summary != ""
}

func (s *Server) setSummary(ctx context.Context, docID, summary string) error {
	if s.Cache == nil {
		return errors.New("no cache configured for document summaries")
	}
	return s.Cache.SetWithTTL(ctx, summaryKey(docID), summary, summaryTTL)
}

// cacheStatus turns a cache error into the short status reported to clients.
func cacheStatus(err error) string {
	switch {
	case err == nil:
		return "miss"
	case errors.Is(err, cache.ErrCircuitOpen):
		return "circuit_open"
	case errors.Is(err, cache.ErrCorruptEntry):
		return "corrupt_entry"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return "cancelled"
	}
	return "error"
}

// embedQuery fetches the question's embedding from the RAG service.
func (s *Server) embedQuery(ctx context.Context, text string) ([]float32, string, error) {
	var out struct {
		Embedding []float32 `json:"embedding"`
		Model     string    `json:"model"`
	}
	if _, err := s.postRAG(ctx, "/embed", map[string]any{"text": text}, &out); err != nil {
		return nil, "", err
	}
	if err := cache.ValidateVector(out.Embedding); err != nil {
		return nil, "", fmt.Errorf("embedding rejected: %w", err)
	}
	if out.Model == "" {
		return nil, "", errors.New("embedding response has no model name")
	}
	return out.Embedding, out.Model, nil
}

// cacheableResult reports whether a finished result may be stored: routing
// did not fall back, grounding did not fail, and no council stage degraded.
func cacheableResult(info council.RoutingInfo, res council.QueryResult) bool {
	return !info.Fallback && info.GroundingError == "" && len(res.Degraded) == 0
}
