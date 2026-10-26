package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func env(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}

type CacheEntry struct {
	Data    QueryResult
	Expires time.Time
}
type MemoryCache struct {
	mu     sync.RWMutex
	values map[string]CacheEntry
	ttl    time.Duration
}

func NewMemoryCache(ttl time.Duration) *MemoryCache {
	return &MemoryCache{values: make(map[string]CacheEntry), ttl: ttl}
}
func (c *MemoryCache) Get(key string) (QueryResult, bool) {
	c.mu.RLock()
	value, ok := c.values[key]
	c.mu.RUnlock()
	if !ok || time.Now().After(value.Expires) {
		return QueryResult{}, false
	}
	return value.Data, true
}
func (c *MemoryCache) Set(key string, value QueryResult) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values[key] = CacheEntry{Data: value, Expires: time.Now().Add(c.ttl)}
}

type Server struct {
	Council  *Council
	RAGURL   string
	HTTP     *http.Client
	Cache    *MemoryCache
	Requests atomic.Uint64
	Hits     atomic.Uint64
	Failures atomic.Uint64
}

func NewServer(c *Council, ragURL string) *Server {
	return &Server{Council: c, RAGURL: strings.TrimRight(ragURL, "/"), HTTP: &http.Client{Timeout: 25 * time.Second}, Cache: NewMemoryCache(time.Hour)}
}

func respond(w http.ResponseWriter, code int, val any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(val)
}
func respondError(w http.ResponseWriter, code int, msg string) {
	respond(w, code, map[string]string{"error": msg})
}

func (s *Server) Retrieve(ctx context.Context, q QueryRequest) ([]string, error) {
	if q.DocID == "" {
		return nil, nil
	}
	topK := q.TopK
	if topK <= 0 || topK > 10 {
		topK = 4
	}
	payload, _ := json.Marshal(map[string]any{"doc_id": q.DocID, "question": q.Question, "top_k": topK})
	req, err := http.NewRequestWithContext(ctx, "POST", s.RAGURL+"/retrieve", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("RAG service returned %d", res.StatusCode)
	}
	var data struct {
		Chunks []struct {
			Text  string  `json:"text"`
			Page  int     `json:"page"`
			Score float64 `json:"score"`
		} `json:"chunks"`
	}
	if err := json.NewDecoder(io.LimitReader(res.Body, 2<<20)).Decode(&data); err != nil {
		return nil, err
	}
	excerpts := make([]string, 0, len(data.Chunks))
	for _, chunk := range data.Chunks {
		excerpts = append(excerpts, fmt.Sprintf("[Document %s, page %d] %s", q.DocID, chunk.Page, chunk.Text))
	}
	return excerpts, nil
}

func cacheKey(q QueryRequest, strategy string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(q.DocID) + "|" + strings.TrimSpace(strings.ToLower(q.Question)) + "|" + strategy + "|" + strconv.Itoa(q.TopK)))
	return fmt.Sprintf("%x", sum)
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		respondError(w, 405, "POST required")
		return
	}
	s.Requests.Add(1)
	r.Body = http.MaxBytesReader(w, r.Body, 32<<10)
	var q QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
		respondError(w, 400, "invalid JSON query")
		return
	}
	q.Question = strings.TrimSpace(q.Question)
	if q.Question == "" || len(q.Question) > 10000 {
		respondError(w, 400, "question must contain 1–10000 characters")
		return
	}
	strat, err := strategyFor(q)
	if err != nil {
		respondError(w, 400, err.Error())
		return
	}
	q.Strategy = strat
	key := cacheKey(q, strat)
	wantsStream := strings.Contains(r.Header.Get("Accept"), "text/event-stream")
	if prev, ok := s.Cache.Get(key); ok {
		s.Hits.Add(1)
		prev.CacheHit = true
		prev.DurationMS = 0
		if wantsStream {
			stream := newEventWriter(w)
			stream.Emit("final", prev)
		} else {
			respond(w, 200, prev)
		}
		return
	}
	var emit EventFn = func(string, any) {}
	if wantsStream {
		stream := newEventWriter(w)
		emit = stream.Emit
		emit("start", map[string]string{"strategy": strat})
	}
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()
	sources, err := s.Retrieve(ctx, q)
	if err != nil {
		s.Failures.Add(1)
		if wantsStream {
			emit("error", map[string]string{"error": "document retrieval failed: " + err.Error()})
		} else {
			respondError(w, 502, "document retrieval failed: "+err.Error())
		}
		return
	}
	if wantsStream {
		emit("retrieved", map[string]int{"chunks": len(sources)})
	}
	result, err := s.Council.Run(ctx, q, sources, emit)
	if err != nil {
		s.Failures.Add(1)
		if wantsStream {
			emit("error", map[string]string{"error": err.Error()})
		} else {
			respondError(w, 502, err.Error())
		}
		return
	}
	s.Cache.Set(key, result)
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

func multipartWriter(buf *bytes.Buffer, filename string, source io.Reader) string {
	writer := multipart.NewWriter(buf)
	part, err := writer.CreateFormFile("file", filename)
	if err == nil {
		_, _ = io.Copy(part, source)
	}
	_ = writer.Close()
	return writer.FormDataContentType()
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		respondError(w, 405, "POST required")
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
	// Preserve original multipart payload so the Python RAG service can inspect the filename.
	var buf bytes.Buffer
	writer := multipartWriter(&buf, header.Filename, file)
	req, err := http.NewRequestWithContext(r.Context(), "POST", s.RAGURL+"/ingest", &buf)
	if err != nil {
		respondError(w, 500, err.Error())
		return
	}
	req.Header.Set("Content-Type", writer)
	res, err := s.HTTP.Do(req)
	if err != nil {
		respondError(w, 502, "RAG service unavailable: "+err.Error())
		return
	}
	defer res.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(res.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(res.Body, 1<<20))
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

func main() {
	models := strings.Split(env("COUNCIL_MODELS", "model-a,model-b,model-c"), ",")
	for i := range models {
		models[i] = strings.TrimSpace(models[i])
	}
	c := &Council{Provider: newConfiguredModel(), Members: models, Chair: env("CHAIR_MODEL", "model-a"), Timeout: 35 * time.Second}
	srv := NewServer(c, env("RAG_URL", "http://localhost:8000"))
	port := env("PORT", "8080")
	log.Printf("CouncilAI starter listening on :%s (MOCK_LLM=%s)", port, env("MOCK_LLM", "false"))
	if err := http.ListenAndServe(":"+port, srv.Routes()); err != nil {
		log.Fatal(err)
	}
}
