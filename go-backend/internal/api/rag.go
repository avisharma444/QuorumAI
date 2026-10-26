package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"example.com/councilai-starter/internal/council"
)

// errDocumentNotFound means the RAG service has no document with that ID.
var errDocumentNotFound = errors.New("document not found")

// Retrieve fetches the top-K chunks of q.DocID relevant to the question.
func (s *Server) Retrieve(ctx context.Context, q council.QueryRequest) ([]string, error) {
	if q.DocID == "" {
		return nil, nil
	}
	topK := q.TopK
	if topK <= 0 || topK > 10 {
		topK = 4
	}
	var data struct {
		Chunks []struct {
			Text  string  `json:"text"`
			Page  int     `json:"page"`
			Score float64 `json:"score"`
		} `json:"chunks"`
	}
	status, err := s.postRAG(ctx, "/retrieve", map[string]any{"doc_id": q.DocID, "question": q.Question, "top_k": topK}, &data)
	if status == http.StatusNotFound {
		return nil, errDocumentNotFound
	}
	if err != nil {
		return nil, err
	}
	excerpts := make([]string, 0, len(data.Chunks))
	for _, chunk := range data.Chunks {
		excerpts = append(excerpts, fmt.Sprintf("[Document %s, page %d] %s", q.DocID, chunk.Page, chunk.Text))
	}
	return excerpts, nil
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
	res, err := s.HTTP.Do(req)
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return res.StatusCode, fmt.Errorf("RAG service %s returned %d", path, res.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(res.Body, 4<<20)).Decode(out); err != nil {
		return res.StatusCode, fmt.Errorf("RAG service %s: invalid response: %w", path, err)
	}
	return res.StatusCode, nil
}
