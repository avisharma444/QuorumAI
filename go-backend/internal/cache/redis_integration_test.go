package cache

// Integration tests against a REAL Redis Stack (RediSearch). They run when
// REDIS_TEST_ADDR is set (for example redis:6379 on the compose network)
// and use unique key prefixes and index names, so they never touch the
// service's own data. RAG_TEST_URL enables tests with real BGE vectors from
// the Python /embed endpoint; REDIS_PLAIN_TEST_ADDR points at a Redis
// WITHOUT RediSearch.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

func redisAddr(t *testing.T) string {
	t.Helper()
	addr := os.Getenv("REDIS_TEST_ADDR")
	if addr == "" {
		t.Skip("REDIS_TEST_ADDR not set")
	}
	return addr
}

func uniq(t *testing.T) string {
	return fmt.Sprintf("it%d_%s", time.Now().UnixNano(), strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_")))
}

func newSemantic(t *testing.T, client *redis.Client) *RedisSemanticCache {
	t.Helper()
	id := uniq(t)
	c := NewRedisSemanticCacheWithBreaker(client, NewCircuitBreaker("it", DefaultConfig()), 0).WithNames("idx:"+id, id+":")
	t.Cleanup(func() {
		// Own client: the test's client is closed by its defer before cleanups run.
		cl := NewClient(redisAddr(t), "", 0)
		defer cl.Close()
		ctx := context.Background()
		cl.Do(ctx, "FT.DROPINDEX", "idx:"+id, "DD")
		if keys, _ := cl.Keys(ctx, id+":*").Result(); len(keys) > 0 {
			cl.Del(ctx, keys...)
		}
	})
	return c
}

// rotated returns a unit vector at cosine similarity cos to e0.
func rotated(cos float64) []float32 {
	v := make([]float32, VectorDim)
	v[0], v[1] = float32(cos), float32(math.Sqrt(1-cos*cos))
	return v
}

func TestIntegrationExactCache(t *testing.T) {
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := NewRedisCacheWithBreaker(client, nil, 0)
	ctx := context.Background()
	key := uniq(t) + ":l1"
	defer client.Del(ctx, key, key+":short")

	var out map[string]string
	if hit, err := c.Get(ctx, key, &out); hit || err != nil {
		t.Fatalf("miss expected: %v %v", hit, err)
	}
	if err := c.Set(ctx, key, map[string]string{"answer": "Mirevale — \"quoted\" ✓"}); err != nil {
		t.Fatal(err)
	}
	if ttl := client.TTL(ctx, key).Val(); ttl < 59*time.Minute || ttl > time.Hour {
		t.Fatalf("L1 TTL %s, want 1h", ttl)
	}
	if hit, err := c.Get(ctx, key, &out); !hit || err != nil || out["answer"] != "Mirevale — \"quoted\" ✓" {
		t.Fatalf("%v %v %v", hit, err, out)
	}
	// A new client (as after a Go restart) sees the same entry.
	client2 := NewClient(redisAddr(t), "", 0)
	defer client2.Close()
	if hit, _ := NewRedisCacheWithBreaker(client2, nil, 0).Get(ctx, key, &out); !hit {
		t.Fatal("entry must survive a client (backend) restart")
	}
	// Expiry.
	short := key + ":short"
	if err := c.SetWithTTL(ctx, short, "x", time.Second); err != nil {
		t.Fatal(err)
	}
	time.Sleep(1200 * time.Millisecond)
	var s string
	if hit, _ := c.Get(ctx, short, &s); hit {
		t.Fatal("expired entry returned")
	}
	// Corrupt JSON: reported, never returned, deleted, and not a breaker failure.
	client.Set(ctx, key, "{not json", time.Minute)
	if hit, err := c.Get(ctx, key, &out); hit || !errors.Is(err, ErrCorruptEntry) {
		t.Fatalf("corrupt entry: %v %v", hit, err)
	}
	if client.Exists(ctx, key).Val() != 0 {
		t.Fatal("corrupt entry must be deleted")
	}
	if _, f, _, _ := c.CircuitBreaker().Counts(); f != 0 {
		t.Fatal("misses and corrupt entries are not Redis failures")
	}
}

// REDIS_AUTH_TEST_ADDR points at a Redis started with --requirepass secret.
// (A server without a password accepts any AUTH from go-redis, so the
// wrong-password case needs a protected server.)
func TestIntegrationWrongPasswordFails(t *testing.T) {
	addr := os.Getenv("REDIS_AUTH_TEST_ADDR")
	if addr == "" {
		t.Skip("REDIS_AUTH_TEST_ADDR not set")
	}
	bad := NewClient(addr, "definitely-wrong", 0)
	defer bad.Close()
	cb := NewCircuitBreaker("auth", DefaultConfig())
	l1 := NewRedisCacheWithBreaker(bad, cb, 0)
	var v string
	for i := 0; i < 3; i++ {
		if _, err := l1.Get(context.Background(), "k", &v); err == nil || !strings.Contains(strings.ToUpper(err.Error()), "WRONGPASS") {
			t.Fatalf("attempt %d: %v", i, err)
		}
	}
	if cb.State() != StateOpen {
		t.Fatal("authentication failures are Redis failures")
	}
	good := NewClient(addr, "secret", 0)
	defer good.Close()
	if err := NewRedisCacheWithBreaker(good, nil, 0).Ping(context.Background()); err != nil {
		t.Fatalf("correct password: %v", err)
	}
}

func TestIntegrationIndexCreationIsIdempotentAndConcurrent(t *testing.T) {
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := newSemantic(t, client)
	var wg sync.WaitGroup
	errs := make(chan error, 10)
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- c.EnsureIndex(context.Background()) }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	info, err := client.Do(context.Background(), "FT.INFO", c.index).Result()
	if err != nil {
		t.Fatal(err)
	}
	flat := fmt.Sprint(info)
	for _, want := range []string{"FLOAT32", "COSINE", "FLAT", "384", "scope", "doc_id"} {
		if !strings.Contains(flat, want) {
			t.Fatalf("FT.INFO lacks %s: %s", want, flat)
		}
	}
}

func TestIntegrationIncompatibleIndexIsReported(t *testing.T) {
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := newSemantic(t, client)
	// The reference schema: no scope field.
	if err := client.Do(context.Background(), "FT.CREATE", c.index, "ON", "HASH", "PREFIX", "1", c.prefix, "SCHEMA", "doc_id", "TAG").Err(); err != nil {
		t.Fatal(err)
	}
	if err := c.EnsureIndex(context.Background()); !errors.Is(err, ErrIndexSchemaMismatch) {
		t.Fatalf("got %v", err)
	}
}

func TestIntegrationSemanticKNNThresholdAndScope(t *testing.T) {
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := newSemantic(t, client)
	ctx := context.Background()
	if err := c.EnsureIndex(ctx); err != nil {
		t.Fatal(err)
	}
	scope := Scope{DocID: "report_v1.pdf_abc", Key: "scopeA"}
	if err := c.Put(ctx, scope, "what was revenue in q2", rotated(1), map[string]string{"answer": "135"}); err != nil {
		t.Fatal(err)
	}
	key := c.key(scope, "what was revenue in q2")
	if ttl := client.TTL(ctx, key).Val(); ttl < 23*time.Hour || ttl > 24*time.Hour {
		t.Fatalf("L2 TTL %s, want 24h", ttl)
	}
	raw, _ := client.HGet(ctx, key, "vector").Bytes()
	if len(raw) != 1536 {
		t.Fatalf("stored vector bytes %d, want 1536", len(raw))
	}

	var out map[string]string
	cases := []struct {
		cos  float64
		want bool
	}{{1, true}, {0.95, true}, {0.86, true}, {0.84, false}, {0.5, false}}
	for _, tc := range cases {
		hit, found, err := c.Get(ctx, scope, "what was revenue in q2", rotated(tc.cos), 0.85, &out)
		if err != nil || found != tc.want {
			t.Fatalf("cos %.2f: found=%v err=%v hit=%+v", tc.cos, found, err, hit)
		}
		if found && math.Abs(hit.Distance-(1-tc.cos)) > 1e-4 {
			t.Fatalf("cos %.2f: RediSearch distance %v, want %v", tc.cos, hit.Distance, 1-tc.cos)
		}
	}
	// Isolation: same vector, other document / other scope key → miss.
	for _, other := range []Scope{{DocID: "report_v2.pdf_def", Key: "scopeA"}, {DocID: scope.DocID, Key: "scopeB"}} {
		if _, found, err := c.Get(ctx, other, "what was revenue in q2", rotated(1), 0.85, &out); found || err != nil {
			t.Fatalf("scope %+v leaked: %v", other, err)
		}
	}
	// Number guard: identical vector but a different year is rejected.
	if _, found, _ := c.Get(ctx, scope, "what was revenue in q3", rotated(1), 0.85, &out); found {
		t.Fatal("number guard failed")
	}
	// A RESP3 client parses the map-shaped reply too.
	r3 := redis.NewClient(&redis.Options{Addr: redisAddr(t), Protocol: 3})
	defer r3.Close()
	c3 := &RedisSemanticCache{client: r3, cb: NewCircuitBreaker("r3", DefaultConfig()), ttl: DefaultTTL, index: c.index, prefix: c.prefix, opTimeout: time.Second, ready: true}
	if _, found, err := c3.Get(ctx, scope, "what was revenue in q2", rotated(1), 0.85, &out); !found || err != nil {
		t.Fatalf("RESP3 lookup: %v %v", found, err)
	}
}

func TestIntegrationSemanticCorruptAndExpiry(t *testing.T) {
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := newSemantic(t, client)
	ctx := context.Background()
	c.EnsureIndex(ctx)
	scope := Scope{DocID: "d", Key: "s"}
	key := c.key(scope, "q")
	client.HSet(ctx, key, map[string]any{"doc_id": "d", "scope": "s", "question": "q", "response": "{broken", "vector": Float32ToBytes(rotated(1))})
	client.Expire(ctx, key, time.Minute)
	var out any
	if _, found, err := c.Get(ctx, scope, "q", rotated(1), 0.85, &out); found || !errors.Is(err, ErrCorruptEntry) {
		t.Fatalf("corrupt: %v %v", found, err)
	}
	short := NewRedisSemanticCacheWithBreaker(client, nil, time.Second).WithNames(c.index, c.prefix)
	short.ready = true
	if err := short.Put(ctx, Scope{DocID: "d", Key: "t"}, "q", rotated(1), "x"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(1300 * time.Millisecond)
	if _, found, _ := c.Get(ctx, Scope{DocID: "d", Key: "t"}, "q", rotated(1), 0.85, &out); found {
		t.Fatal("expired L2 entry returned")
	}
}

func TestIntegrationLostIndexIsRecreated(t *testing.T) {
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := newSemantic(t, client)
	ctx := context.Background()
	c.EnsureIndex(ctx)
	client.Do(ctx, "FT.DROPINDEX", c.index)
	var out any
	_, found, err := c.Get(ctx, Scope{DocID: "d", Key: "s"}, "q", rotated(1), 0.85, &out)
	if found || err != nil {
		t.Fatalf("missing index must be a plain miss: %v", err)
	}
	if _, _, _, failures := c.cb.Counts(); failures != 0 || c.cb.State() != StateClosed {
		t.Fatalf("a missing index is not a Redis failure (failures=%d)", failures)
	}
	if err := c.Put(ctx, Scope{DocID: "d", Key: "s"}, "q", rotated(1), "v"); err != nil {
		t.Fatalf("index not recreated: %v", err)
	}
	if _, found, _ := c.Get(ctx, Scope{DocID: "d", Key: "s"}, "q", rotated(1), 0.85, &out); !found {
		t.Fatal("lookup after recreation failed")
	}
}

func TestIntegrationPlainRedisWithoutSearch(t *testing.T) {
	addr := os.Getenv("REDIS_PLAIN_TEST_ADDR")
	if addr == "" {
		t.Skip("REDIS_PLAIN_TEST_ADDR not set")
	}
	client := NewClient(addr, "", 0)
	defer client.Close()
	cb := NewCircuitBreaker("plain", DefaultConfig())
	c := NewRedisSemanticCacheWithBreaker(client, cb, 0)
	if err := c.EnsureIndex(context.Background()); !errors.Is(err, ErrSearchUnavailable) {
		t.Fatalf("got %v", err)
	}
	if cb.State() != StateClosed {
		t.Fatal("missing module must not trip the shared breaker (L1 still works)")
	}
	if err := NewRedisCacheWithBreaker(client, cb, 0).Set(context.Background(), uniq(t), "v"); err != nil {
		t.Fatalf("L1 on plain Redis: %v", err)
	}
}

func TestIntegrationRedisOOMRejection(t *testing.T) {
	addr := os.Getenv("REDIS_PLAIN_TEST_ADDR") // a throwaway server we may reconfigure
	if addr == "" {
		t.Skip("REDIS_PLAIN_TEST_ADDR not set")
	}
	client := NewClient(addr, "", 0)
	defer client.Close()
	ctx := context.Background()
	client.ConfigSet(ctx, "maxmemory", "1")
	client.ConfigSet(ctx, "maxmemory-policy", "noeviction")
	defer client.ConfigSet(ctx, "maxmemory", "0")
	err := NewRedisCacheWithBreaker(client, nil, 0).Set(ctx, uniq(t), strings.Repeat("x", 1000))
	if err == nil || !strings.Contains(strings.ToUpper(err.Error()), "OOM") {
		t.Fatalf("expected OOM rejection, got %v", err)
	}
}

// ── Real BGE vectors ────────────────────────────────────────────────

func embed(t *testing.T, text string) []float32 {
	t.Helper()
	base := os.Getenv("RAG_TEST_URL")
	body, _ := json.Marshal(map[string]string{"text": text})
	resp, err := http.Post(base+"/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Embedding []float32 `json:"embedding"`
		Model     string    `json:"model"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || out.Model != "BAAI/bge-small-en-v1.5" {
		t.Fatalf("embed: %v %+v", err, out.Model)
	}
	return out.Embedding
}

func cosine(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / math.Sqrt(na*nb)
}

func TestIntegrationRealBGEVectors(t *testing.T) {
	if os.Getenv("RAG_TEST_URL") == "" {
		t.Skip("RAG_TEST_URL not set")
	}
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	c := newSemantic(t, client)
	ctx := context.Background()
	c.EnsureIndex(ctx)
	scope := Scope{DocID: "zog", Key: "s"}
	original := "Who sets household water prices in Zogland?"
	if err := c.Put(ctx, scope, strings.ToLower(original), embed(t, original), "answer"); err != nil {
		t.Fatal(err)
	}
	var out string
	// Identical text: distance ~0 through real RediSearch.
	hit, found, err := c.Get(ctx, scope, strings.ToLower(original), embed(t, original), 0.85, &out)
	if !found || err != nil || hit.Distance > 1e-3 {
		t.Fatalf("identical: %+v %v %v", hit, found, err)
	}
	// RediSearch's distance agrees with an independently computed cosine.
	para := "Which organisation decides what households pay for water in Zogland?"
	pv := embed(t, para)
	want := 1 - cosine(embed(t, original), pv)
	hit, found, _ = c.Get(ctx, scope, strings.ToLower(para), pv, 0.01, &out)
	if !found || math.Abs(hit.Distance-want) > 1e-3 {
		t.Fatalf("paraphrase distance %v, independent %v", hit.Distance, want)
	}
	t.Logf("paraphrase similarity %.4f (threshold 0.85 → hit=%v)", 1-hit.Distance, Accept(hit.Distance, 0.85))
	// Near-miss with different numbers is rejected by the guard even at a permissive threshold.
	if err := c.Put(ctx, scope, "what was revenue in 2024?", embed(t, "What was revenue in 2024?"), "2024 answer"); err != nil {
		t.Fatal(err)
	}
	v25 := embed(t, "What was revenue in 2025?")
	t.Logf("2024 vs 2025 similarity %.4f", cosine(embed(t, "What was revenue in 2024?"), v25))
	if _, found, _ := c.Get(ctx, scope, "what was revenue in 2025?", v25, 0.5, &out); found && out == "2024 answer" {
		t.Fatal("2025 question reused the 2024 answer")
	}
}
