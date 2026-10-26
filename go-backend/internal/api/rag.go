package api

import (
	"go.opentelemetry.io/otel/attribute"
	"time"

	"example.com/councilai-starter/internal/metrics"
	"example.com/councilai-starter/internal/telemetry"

	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"example.com/councilai-starter/internal/council"
)

// errDocumentNotFound means the RAG service has no document with that ID.
var errDocumentNotFound = errors.New("document not found")

// ragChunk is the RAG service's Chunk model (reference app/models.py).
type ragChunk struct {
	Content    string         `json:"content"`
	ChunkType  string         `json:"chunk_type"`
	PageNumber int            `json:"page_number"`
	ChunkIndex int            `json:"chunk_index"`
	Metadata   map[string]any `json:"metadata"`
}

func metaFloat(m map[string]any, key string) *float64 {
	if v, ok := m[key].(float64); ok {
		return &v
	}
	return nil
}

func metaInt(m map[string]any, key string) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return 0
}

// excerptLabel cites only provenance the RAG service actually reported.
func excerptLabel(docID string, c council.DocumentChunk) string {
	parts := []string{"Document " + docID}
	switch {
	case c.PageNumber > 0 && c.PageEnd > c.PageNumber:
		parts = append(parts, fmt.Sprintf("pages %d-%d", c.PageNumber, c.PageEnd))
	case c.PageNumber > 0:
		parts = append(parts, fmt.Sprintf("page %d", c.PageNumber))
	}
	if c.ChunkType == "table" {
		parts = append(parts, "table")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Retrieve fetches the top-K chunks of q.DocID relevant to the question
// (vector search plus cross-encoder rerank in the RAG service).
func (s *Server) Retrieve(ctx context.Context, q council.QueryRequest) ([]string, *council.RetrievalInfo, error) {
	if q.DocID == "" {
		return nil, nil, nil
	}
	topK := q.TopK
	if topK <= 0 || topK > 10 {
		topK = 4
	}
	var data struct {
		Chunks         []ragChunk `json:"chunks"`
		Reranked       bool       `json:"reranked"`
		RerankError    *string    `json:"rerank_error"`
		Candidates     int        `json:"candidates"`
		DistanceMetric string     `json:"distance_metric"`
	}
	status, err := s.postRAG(ctx, "/retrieve", map[string]any{"doc_id": q.DocID, "question": q.Question, "top_k": topK, "rerank": true}, &data)
	if status == http.StatusNotFound {
		return nil, nil, errDocumentNotFound
	}
	if err != nil {
		return nil, nil, err
	}
	info := &council.RetrievalInfo{Reranked: data.Reranked, Candidates: data.Candidates, DistanceMetric: data.DistanceMetric}
	if data.RerankError != nil {
		info.RerankError = *data.RerankError
	}
	excerpts := make([]string, 0, len(data.Chunks))
	for _, c := range data.Chunks {
		m := c.Metadata
		dc := council.DocumentChunk{
			Content: c.Content, ChunkType: c.ChunkType, PageNumber: c.PageNumber, ChunkIndex: c.ChunkIndex,
			PageEnd: metaInt(m, "page_end"), OCRConfidence: metaFloat(m, "ocr_confidence"),
			VectorDistance: metaFloat(m, "vector_distance"), RerankScore: metaFloat(m, "rerank_score"),
		}
		dc.OCRMethod, _ = m["ocr_method"].(string)
		info.Chunks = append(info.Chunks, dc)
		excerpts = append(excerpts, excerptLabel(q.DocID, dc)+" "+c.Content)
	}
	return excerpts, info, nil
}

// webSearchResult is the Python /search response.
type webSearchResult struct {
	Status  string              `json:"status"` // "ok" | "no_results" | "search_failed"
	Results []string            `json:"results"`
	Sources []council.WebSource `json:"sources"`
	Error   *string             `json:"error"`
}

// webSearchMaxResults matches the reference handler.
const webSearchMaxResults = 3

// SearchWeb asks the RAG service for live web grounding (reference:
// retrieveWebSearch → Python /search).
func (s *Server) SearchWeb(ctx context.Context, query string) (webSearchResult, error) {
	var out webSearchResult
	_, err := s.postRAG(ctx, "/search", map[string]any{"query": query, "max_results": webSearchMaxResults}, &out)
	return out, err
}

// postRAG POSTs JSON to the RAG service and decodes a 200 response into out.
// It returns the HTTP status (0 on transport errors).
func (s *Server) postRAG(ctx context.Context, path string, body any, out any) (int, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.RAGURL+path, bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	req, done := startRAG(ctx, path, req)
	res, err := s.HTTP.Do(req)
	if err != nil {
		done(0, err)
		return 0, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		done(res.StatusCode, nil)
		return res.StatusCode, fmt.Errorf("RAG service %s returned %d", path, res.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(res.Body, 4<<20)).Decode(out); err != nil {
		done(res.StatusCode, err)
		return res.StatusCode, fmt.Errorf("RAG service %s: invalid response: %w", path, err)
	}
	done(res.StatusCode, nil)
	return res.StatusCode, nil
}

// startRAG opens a client span for one call to the Python RAG service and
// injects the W3C traceparent (and baggage) into the request, so the
// Python server span joins this trace. The returned function records the
// outcome (span status, duration histogram, dependency error counter).
// endpoint is a fixed path such as "/retrieve", never a URL with data.
func startRAG(ctx context.Context, endpoint string, req *http.Request) (*http.Request, func(status int, err error)) {
	ctx, span := telemetry.StartClient(ctx, "rag POST "+endpoint,
		attribute.String("rag.endpoint", endpoint), attribute.String("http.request.method", req.Method))
	req = req.WithContext(ctx)
	telemetry.Inject(ctx, req)
	start := time.Now()
	return req, func(status int, err error) {
		result := "ok"
		switch {
		case err != nil && ctx.Err() != nil:
			result = "cancelled"
		case err != nil:
			result = "error"
		case status >= 500:
			result = "http_5xx"
		case status >= 400:
			result = "http_4xx"
		}
		metrics.RAGRequestDuration.WithLabelValues(endpoint, result).Observe(time.Since(start).Seconds())
		if result == "error" || result == "http_5xx" {
			metrics.DependencyErrors.WithLabelValues("rag", strings.TrimPrefix(endpoint, "/")).Inc()
		}
		if status != 0 {
			span.SetAttributes(attribute.Int("http.response.status_code", status))
		}
		if result != "ok" {
			telemetry.Fail(span, result)
		}
		span.End()
	}
}
