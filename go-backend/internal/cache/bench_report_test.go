package cache

// Redis-only latency report against a real Redis Stack. Runs only with
// CACHE_BENCH=1 and REDIS_TEST_ADDR set; prints one JSON line.
//
//   CACHE_BENCH=1 REDIS_TEST_ADDR=redis:6379 go test -run TestCacheLatencyReport -v ./internal/cache/

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"testing"
	"time"
)

func pct(samples []time.Duration) map[string]float64 {
	s := append([]time.Duration(nil), samples...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	at := func(p float64) float64 {
		return float64(s[int(math.Min(float64(len(s)-1), p*float64(len(s))))].Microseconds()) / 1000
	}
	var sum time.Duration
	for _, d := range s {
		sum += d
	}
	return map[string]float64{"n": float64(len(s)), "mean_ms": float64(sum.Microseconds()) / 1000 / float64(len(s)), "p50_ms": at(0.50), "p95_ms": at(0.95), "p99_ms": at(0.99)}
}

func randUnit(r *rand.Rand) []float32 {
	v := make([]float32, VectorDim)
	var n float64
	for i := range v {
		x := r.NormFloat64()
		v[i] = float32(x)
		n += x * x
	}
	for i := range v {
		v[i] /= float32(math.Sqrt(n))
	}
	return v
}

func TestCacheLatencyReport(t *testing.T) {
	if os.Getenv("CACHE_BENCH") != "1" {
		t.Skip("CACHE_BENCH not set")
	}
	client := NewClient(redisAddr(t), "", 0)
	defer client.Close()
	ctx := context.Background()
	r := rand.New(rand.NewSource(42))
	const iters, entriesInScope, entriesOtherScopes = 1000, 2000, 2000

	l1 := NewRedisCacheWithBreaker(client, nil, time.Minute)
	payload := map[string]string{"answer": string(make([]byte, 2048))} // ~2 KB, like a council answer
	prefix := uniq(t) + ":l1:"
	var set, getHit, getMiss []time.Duration
	for i := 0; i < iters; i++ {
		key := fmt.Sprintf("%s%d", prefix, i)
		t0 := time.Now()
		if err := l1.Set(ctx, key, payload); err != nil {
			t.Fatal(err)
		}
		set = append(set, time.Since(t0))
		var out map[string]string
		t0 = time.Now()
		if hit, err := l1.Get(ctx, key, &out); !hit || err != nil {
			t.Fatal(err)
		}
		getHit = append(getHit, time.Since(t0))
		t0 = time.Now()
		if hit, _ := l1.Get(ctx, key+":absent", &out); hit {
			t.Fatal("unexpected hit")
		}
		getMiss = append(getMiss, time.Since(t0))
	}
	if keys, _ := client.Keys(ctx, prefix+"*").Result(); len(keys) > 0 {
		client.Del(ctx, keys...)
	}

	l2 := newSemantic(t, client)
	if err := l2.EnsureIndex(ctx); err != nil {
		t.Fatal(err)
	}
	scope := Scope{DocID: "bench-doc", Key: "bench-scope"}
	var put []time.Duration
	for i := 0; i < entriesInScope; i++ {
		t0 := time.Now()
		if err := l2.Put(ctx, scope, fmt.Sprintf("question %d", i), randUnit(r), "answer"); err != nil {
			t.Fatal(err)
		}
		put = append(put, time.Since(t0))
	}
	for i := 0; i < entriesOtherScopes; i++ {
		if err := l2.Put(ctx, Scope{DocID: fmt.Sprintf("other-%d", i%20), Key: "other"}, fmt.Sprintf("q %d", i), randUnit(r), "x"); err != nil {
			t.Fatal(err)
		}
	}
	var knn []time.Duration
	var out string
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		if _, _, err := l2.Get(ctx, scope, "question", randUnit(r), 0.85, &out); err != nil {
			t.Fatal(err)
		}
		knn = append(knn, time.Since(t0))
	}
	info, _ := client.Do(ctx, "FT.INFO", l2.index).Result()
	report := map[string]any{
		"redis_addr": redisAddr(t), "go": runtime.Version(), "goarch": runtime.GOARCH, "cpus": runtime.NumCPU(),
		"iterations": iters, "l1_payload_bytes": 2048 + 13,
		"l1_set": pct(set), "l1_get_hit": pct(getHit), "l1_get_miss": pct(getMiss),
		"l2_put": pct(put), "l2_knn_search": pct(knn),
		"l2_index": map[string]any{"entries_in_scope": entriesInScope, "entries_other_scopes": entriesOtherScopes, "type": "FLAT/COSINE/FLOAT32/384", "ft_info_excerpt": excerpt(info)},
	}
	b, _ := json.Marshal(report)
	fmt.Println("CACHE_LATENCY_REPORT " + string(b))
}

func excerpt(info any) map[string]string {
	out := map[string]string{}
	if r, ok := info.([]any); ok {
		for i := 0; i+1 < len(r); i += 2 {
			switch k := toString(r[i]); k {
			case "num_docs", "vector_index_sz_mb", "inverted_sz_mb", "total_indexing_time":
				out[k] = toString(r[i+1])
			}
		}
	}
	return out
}
