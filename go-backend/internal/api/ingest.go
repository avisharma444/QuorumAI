package api

import (
	"example.com/councilai-starter/internal/metrics"

	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime/multipart"
	"net/http"

	"example.com/councilai-starter/internal/agent"
	"example.com/councilai-starter/internal/api/middleware"
	"example.com/councilai-starter/internal/auth"
)

// Summary statuses reported in the ingestion response.
const (
	summaryGenerated = "generated"
	summaryFailed    = "failed"     // model error; ingestion still succeeds
	summarySkipped   = "skipped"    // empty preview text
	summaryDisabled  = "disabled"   // no ingestion agent configured
	summaryCancelled = "cancelled"  // client went away during summarization
	summaryNotStored = "not_stored" // generated, but Redis rejected the write
)

func multipartWriter(buf *bytes.Buffer, filename string, source io.Reader) string {
	writer := multipart.NewWriter(buf)
	part, err := writer.CreateFormFile("file", filename)
	if err == nil {
		_, _ = io.Copy(part, source)
	}
	_ = writer.Close()
	return writer.FormDataContentType()
}

// handleIngest forwards the upload to the RAG service, then summarizes the
// returned BME preview with the ingestion model and stores the summary
// under the document ID for the router (reference: HandleIngest).
//
// Policy, as in the reference: the document is already indexed when
// summarization runs, so a summarization failure does not fail ingestion.
// The response reports summary_status, and queries on that document keep
// its context through the router's missing-summary rule.
func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		respondError(w, 405, "POST required")
		return
	}
	userID, ok := middleware.UserID(r.Context())
	if !ok {
		respondError(w, http.StatusUnauthorized, "unauthenticated")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 12<<20)
	if err := r.ParseMultipartForm(12 << 20); err != nil {
		respondError(w, 400, "invalid upload: "+err.Error())
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		respondError(w, 400, "file field missing")
		return
	}
	defer file.Close()
	if header.Size > 10<<20 {
		respondError(w, 413, "file exceeds 10 MiB")
		return
	}
	// Preserve the original filename so the RAG service can pick a parser.
	var buf bytes.Buffer
	contentType := multipartWriter(&buf, header.Filename, file)
	req, err := http.NewRequestWithContext(r.Context(), "POST", s.RAGURL+"/ingest", &buf)
	if err != nil {
		respondError(w, 500, err.Error())
		return
	}
	req.Header.Set("Content-Type", contentType)
	req, done := startRAG(r.Context(), "/ingest", req)
	// Ingestion runs OCR and embeddings, so it gets the request timeout
	// rather than the short client timeout used for retrieval.
	res, err := (&http.Client{Timeout: s.RequestTimeout, Transport: s.HTTP.Transport}).Do(req)
	if err != nil {
		done(0, err)
		respondError(w, 502, "RAG service unavailable: "+err.Error())
		return
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 2<<20))
	done(res.StatusCode, err)
	if err != nil {
		respondError(w, 502, "failed to read ingestion response")
		return
	}
	if res.StatusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(res.StatusCode)
		_, _ = w.Write(body)
		return
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		respondError(w, 502, "invalid ingestion response from RAG service")
		return
	}
	docID, _ := parsed["doc_id"].(string)
	preview, _ := parsed["preview_text"].(string)
	if docID == "" {
		respondError(w, 502, "RAG service returned no doc_id")
		return
	}
	// Record the uploader's access only after the RAG service has the
	// document. If recording fails, no access is granted and the client is
	// told to retry; re-uploading is idempotent (same bytes, same doc_id).
	chunks, _ := parsed["chunk_count"].(float64)
	method, _ := parsed["ocr_method"].(string)
	if s.Docs == nil {
		respondError(w, http.StatusServiceUnavailable, "document authorization unavailable")
		return
	}
	if err := s.Docs.GrantDocument(r.Context(), auth.DocumentGrant{UserID: userID, DocID: docID, Filename: header.Filename, ChunkCount: int(chunks), OCRMethod: method}); err != nil {
		metrics.DependencyErrors.WithLabelValues("postgres", "grant_document").Inc()
		log.Printf("[Ingest] access grant failed for doc %s: %v", docID, err)
		respondError(w, http.StatusServiceUnavailable, "document was processed but access could not be recorded; please retry the upload")
		return
	}
	status, summary, summaryErr := s.summarizeDocument(r.Context(), docID, preview)
	parsed["summary_status"] = status
	if summary != "" {
		parsed["summary"] = summary
		if s.IngestAgent != nil {
			parsed["summary_model"] = s.IngestAgent.ModelName()
		}
	}
	if summaryErr != "" {
		parsed["summary_error"] = summaryErr
	}
	respond(w, http.StatusOK, parsed)
}

// summarizeDocument runs the ingestion agent and caches the summary.
func (s *Server) summarizeDocument(ctx context.Context, docID, preview string) (status, summary, errMsg string) {
	if s.IngestAgent == nil {
		return summaryDisabled, "", ""
	}
	summary, err := s.IngestAgent.SummarizeDocument(ctx, preview)
	switch {
	case errors.Is(err, agent.ErrEmptyPreview):
		return summarySkipped, "", err.Error()
	case err != nil && ctx.Err() != nil:
		return summaryCancelled, "", ctx.Err().Error()
	case err != nil:
		log.Printf("[Ingest] Summarization failed for doc %s: %v", docID, err)
		return summaryFailed, "", err.Error()
	}
	if err := s.setSummary(context.WithoutCancel(ctx), docID, summary); err != nil {
		// The summary exists but could not be stored; queries will use the
		// missing-summary rule (document kept).
		log.Printf("[Ingest] Summary for doc %s not stored: %v", docID, err)
		return summaryNotStored, summary, err.Error()
	}
	log.Printf("[Ingest] Generated and cached summary for doc %s", docID)
	return summaryGenerated, summary, ""
}
