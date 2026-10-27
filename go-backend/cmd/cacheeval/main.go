// Command cacheeval measures semantic-cache decisions on labelled question
// pairs with the production code path: real BGE embeddings from the RAG
// service's /embed endpoint, and the production RedisSemanticCache
// (RediSearch KNN, cosine distance, threshold rule and number guard) on a
// real Redis Stack. It uses its own index and key prefix and drops them
// when done, so the application's semantic cache is never touched.
//
// For each pair it stores q1, then looks up q2 at every threshold with the
// real Get, and records the cosine similarity of the two embeddings.
//
//	go run ./cmd/cacheeval -pairs pairs.jsonl -out decisions.jsonl \
//	  -rag http://python-rag:8000 -redis redis:6379
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"example.com/councilai-starter/internal/cache"
)

type pair struct {
	ID         string `json:"id"`
	Q1         string `json:"q1"`
	Q2         string `json:"q2"`
	Equivalent bool   `json:"equivalent"`
	Category   string `json:"category"`
	Source     string `json:"source"`
	Split      string `json:"split"`
}

type decision struct {
	ID              string          `json:"id"`
	Equivalent      bool            `json:"equivalent"`
	Category        string          `json:"category"`
	Source          string          `json:"source"`
	Split           string          `json:"split"`
	Similarity      float64         `json:"similarity"`       // cosine of the two BGE vectors (float64)
	RedisSimilarity *float64        `json:"redis_similarity"` // 1 - RediSearch distance, when the guard allowed a hit
	NumbersMatch    bool            `json:"numbers_match"`
	Hits            map[string]bool `json:"production_hit"` // threshold -> real Get result
	EmbeddingModel  string          `json:"embedding_model"`
}

type embedder struct {
	url    string
	client *http.Client
	memo   map[string][]float32
	model  string
}

func (e *embedder) embed(ctx context.Context, text string) ([]float32, error) {
	if v, ok := e.memo[text]; ok {
		return v, nil
	}
	body, _ := json.Marshal(map[string]string{"text": text})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.url+"/embed", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	res, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("/embed returned %d", res.StatusCode)
	}
	var out struct {
		Embedding []float32 `json:"embedding"`
		Model     string    `json:"model"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return nil, err
	}
	e.memo[text], e.model = out.Embedding, out.Model
	return out.Embedding, nil
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func main() {
	pairsPath := flag.String("pairs", "", "labelled pairs JSONL")
	outPath := flag.String("out", "", "output decisions JSONL")
	ragURL := flag.String("rag", "http://python-rag:8000", "RAG service base URL (for /embed)")
	redisAddr := flag.String("redis", "redis:6379", "Redis Stack address")
	thresholds := flag.String("thresholds", "0.70,0.75,0.80,0.85,0.88,0.90,0.92,0.95,0.97", "similarity thresholds to evaluate with the real Get")
	flag.Parse()
	if *pairsPath == "" || *outPath == "" {
		log.Fatal("-pairs and -out are required")
	}
	var ts []float64
	for _, s := range strings.Split(*thresholds, ",") {
		t, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil || t <= 0 || t > 1 {
			log.Fatalf("bad threshold %q", s)
		}
		ts = append(ts, t)
	}
	in, err := os.Open(*pairsPath)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		log.Fatal(err)
	}
	out, err := os.Create(*outPath)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	ctx := context.Background()
	client := cache.NewClient(*redisAddr, "", 0)
	defer client.Close()
	run := strconv.FormatInt(time.Now().UnixNano(), 36)
	index, prefix := "idx:councileval:"+run, "councileval:"+run+":"
	sc := cache.NewRedisSemanticCacheWithBreaker(client, cache.NewCircuitBreaker("cacheeval", cache.DefaultConfig()), time.Hour).WithNames(index, prefix)
	if err := sc.EnsureIndex(ctx); err != nil {
		log.Fatalf("ensure index: %v", err)
	}
	defer func() {
		if err := client.Do(context.Background(), "FT.DROPINDEX", index, "DD").Err(); err != nil {
			log.Printf("drop index: %v", err)
		}
	}()
	emb := &embedder{url: strings.TrimRight(*ragURL, "/"), client: &http.Client{Timeout: 60 * time.Second}, memo: map[string][]float32{}}

	enc := json.NewEncoder(out)
	sc2 := bufio.NewScanner(in)
	sc2.Buffer(make([]byte, 1<<20), 1<<20)
	n := 0
	start := time.Now()
	for sc2.Scan() {
		var p pair
		if err := json.Unmarshal(sc2.Bytes(), &p); err != nil {
			log.Fatalf("pair %d: %v", n+1, err)
		}
		v1, err := emb.embed(ctx, p.Q1)
		if err != nil {
			log.Fatalf("%s: embed: %v", p.ID, err)
		}
		v2, err := emb.embed(ctx, p.Q2)
		if err != nil {
			log.Fatalf("%s: embed: %v", p.ID, err)
		}
		h := sha256.Sum256([]byte(p.ID))
		scope := cache.Scope{DocID: "councileval", Key: hex.EncodeToString(h[:12])}
		if err := sc.Put(ctx, scope, p.Q1, v1, map[string]string{"pair": p.ID}); err != nil {
			log.Fatalf("%s: put: %v", p.ID, err)
		}
		d := decision{ID: p.ID, Equivalent: p.Equivalent, Category: p.Category, Source: p.Source, Split: p.Split,
			Similarity: cosine(v1, v2), NumbersMatch: cache.NumbersMatch(p.Q1, p.Q2), Hits: map[string]bool{}, EmbeddingModel: emb.model}
		var dest map[string]string
		if hit, ok, err := sc.Get(ctx, scope, p.Q2, v2, 1e-9, &dest); err != nil {
			log.Fatalf("%s: get: %v", p.ID, err)
		} else if ok {
			s := hit.Similarity
			d.RedisSimilarity = &s
		}
		for _, t := range ts {
			_, ok, err := sc.Get(ctx, scope, p.Q2, v2, t, &dest)
			if err != nil {
				log.Fatalf("%s: get: %v", p.ID, err)
			}
			d.Hits[strconv.FormatFloat(t, 'f', 3, 64)] = ok
		}
		if err := enc.Encode(d); err != nil {
			log.Fatal(err)
		}
		n++
	}
	if err := sc2.Err(); err != nil {
		log.Fatal(err)
	}
	log.Printf("cacheeval: %d pairs, %d thresholds, embedding model %s, %s", n, len(ts), emb.model, time.Since(start).Round(time.Millisecond))
}
