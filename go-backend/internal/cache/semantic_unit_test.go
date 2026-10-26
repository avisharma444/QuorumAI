package cache

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"net"
	"testing"
	"time"
)

func unitVec(i int) []float32 {
	v := make([]float32, VectorDim)
	v[i] = 1
	return v
}

func TestFloat32ToBytesIsLittleEndianIEEE754(t *testing.T) {
	v := []float32{1, -0.5, 0, 3.14159, float32(math.SmallestNonzeroFloat32)}
	b := Float32ToBytes(v)
	if len(b) != 20 {
		t.Fatalf("len %d", len(b))
	}
	for i, want := range v {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])); got != want {
			t.Fatalf("index %d: %v != %v", i, got, want)
		}
	}
	if binary.LittleEndian.Uint32(b[0:]) != 0x3f800000 {
		t.Fatal("1.0 must encode as 00 00 80 3f")
	}
	// The bytes are a copy: mutating the input later does not change them.
	v[0] = 7
	if math.Float32frombits(binary.LittleEndian.Uint32(b[0:])) != 1 {
		t.Fatal("encoding must not alias the input slice")
	}
	if len(Float32ToBytes(nil)) != 0 || len(Float32ToBytes(make([]float32, VectorDim))) != 1536 {
		t.Fatal("sizes")
	}
}

func TestValidateVector(t *testing.T) {
	if err := ValidateVector(unitVec(3)); err != nil {
		t.Fatal(err)
	}
	nan := unitVec(0)
	nan[5] = float32(math.NaN())
	inf := unitVec(0)
	inf[5] = float32(math.Inf(1))
	for name, v := range map[string][]float32{"short": make([]float32, 383), "long": make([]float32, 385), "nan": nan, "inf": inf, "zero": make([]float32, VectorDim)} {
		if err := ValidateVector(v); !errors.Is(err, ErrInvalidVector) {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestThresholdBoundary(t *testing.T) {
	if got := MaxDistance(0.85); math.Abs(got-0.15) > 1e-12 {
		t.Fatalf("max distance %v", got)
	}
	cases := []struct {
		dist float64
		want bool
	}{{0, true}, {0.1499999, true}, {MaxDistance(0.85), true}, {MaxDistance(0.85) + 1e-9, false}, {0.2, false}, {math.NaN(), false}, {-0.01, false}}
	for _, c := range cases {
		if Accept(c.dist, 0.85) != c.want {
			t.Fatalf("Accept(%v)=%v", c.dist, !c.want)
		}
	}
}

func TestNumbersMatch(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"What was revenue in 2024?", "what was the revenue for 2024", true},
		{"What was revenue in 2024?", "What was revenue in 2025?", false},
		{"Q2 revenue in 2024", "2024 revenue for Q2", true},
		{"price 1,250 dollars", "price 1250 dollars", false},
		{"what is the refund policy", "explain the refund policy", true},
	}
	for _, c := range cases {
		if NumbersMatch(c.a, c.b) != c.want {
			t.Fatalf("%q vs %q", c.a, c.b)
		}
	}
}

func TestSanitizeTag(t *testing.T) {
	if got := SanitizeTag("report_v1.pdf-ab cd,{x}"); got != `report_v1\.pdf\-ab\ cd\,\{x\}` {
		t.Fatalf("got %q", got)
	}
	if SanitizeTag("abc123") != "abc123" {
		t.Fatal("plain tags unchanged")
	}
}

func TestParseSearchReplyRESP2AndRESP3(t *testing.T) {
	resp2 := []any{int64(2), "semcache:a", []any{"response", `{"a":1}`, "score", "0.01", "question", "q1"}, "semcache:b", []any{"score", "0.3"}}
	docs, err := ParseSearchReply(resp2)
	if err != nil || len(docs) != 2 || docs[0]["response"] != `{"a":1}` || docs[1]["score"] != "0.3" {
		t.Fatalf("%v %v", docs, err)
	}
	resp3 := map[any]any{"total_results": int64(1), "results": []any{map[any]any{"id": "semcache:a", "extra_attributes": map[any]any{"response": "{}", "score": "0.02"}}}}
	docs, err = ParseSearchReply(resp3)
	if err != nil || len(docs) != 1 || docs[0]["score"] != "0.02" {
		t.Fatalf("%v %v", docs, err)
	}
	if docs, err := ParseSearchReply([]any{int64(0)}); err != nil || len(docs) != 0 {
		t.Fatal("empty result")
	}
	for name, bad := range map[string]any{"string": "OK", "nil": nil, "bad total": []any{"x"}, "bad fields": []any{int64(1), "k", "notalist"}, "map without results": map[any]any{}} {
		if _, err := ParseSearchReply(bad); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

// Unreachable Redis: operations fail with real network errors, count as
// failures and open the shared breaker; later calls fail fast.
func TestUnreachableRedisOpensBreaker(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close() // nothing listens here now
	client := NewClient(addr, "", 0)
	defer client.Close()
	cb := NewCircuitBreaker("redis", DefaultConfig())
	l1 := NewRedisCacheWithBreaker(client, cb, 0)
	l2 := NewRedisSemanticCacheWithBreaker(client, cb, 0)
	ctx := context.Background()
	var v string
	for i := 0; i < 3; i++ {
		if _, err := l1.Get(ctx, "k", &v); err == nil || errors.Is(err, ErrCircuitOpen) {
			t.Fatalf("attempt %d: %v", i, err)
		}
	}
	if cb.State() != StateOpen {
		t.Fatalf("breaker %s", cb.State())
	}
	start := time.Now()
	if _, _, err := l2.Get(ctx, Scope{DocID: "d", Key: "k"}, "q", unitVec(1), 0.85, &v); !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("L2 must share the open breaker: %v", err)
	}
	if err := l1.Set(ctx, "k", "v"); !errors.Is(err, ErrCircuitOpen) || time.Since(start) > 50*time.Millisecond {
		t.Fatal("open breaker must fail fast")
	}
}

func TestSemanticInputValidationNeedsNoRedis(t *testing.T) {
	c := NewRedisSemanticCacheWithBreaker(NewClient("127.0.0.1:1", "", 0), nil, 0)
	var v any
	if _, _, err := c.Get(context.Background(), Scope{}, "q", unitVec(1), 0.85, &v); err == nil {
		t.Fatal("empty scope accepted")
	}
	if _, _, err := c.Get(context.Background(), Scope{DocID: "d", Key: "k"}, "q", make([]float32, 3), 0.85, &v); !errors.Is(err, ErrInvalidVector) {
		t.Fatal("bad vector accepted")
	}
	if _, _, err := c.Get(context.Background(), Scope{DocID: "d", Key: "k"}, "q", unitVec(1), 1.5, &v); err == nil {
		t.Fatal("bad threshold accepted")
	}
	if err := c.Put(context.Background(), Scope{DocID: "d", Key: "k"}, "q", unitVec(1), make(chan int)); err == nil {
		t.Fatal("unmarshalable response accepted")
	}
}

func TestSelectHitRules(t *testing.T) {
	var out map[string]string
	doc := func(score, q, resp string) map[string]string {
		return map[string]string{"score": score, "question": q, "response": resp}
	}
	// Malformed or missing scores are errors, never perfect matches.
	for _, bad := range []string{"", "abc", "NaN", "inf"} {
		if _, found, err := SelectHit([]map[string]string{doc(bad, "q", `{"a":"x"}`)}, "q", 0.85, &out); found || err == nil {
			t.Fatalf("score %q accepted", bad)
		}
	}
	// Nearest fails the number guard; the next candidate within threshold is used.
	docs := []map[string]string{doc("0.01", "revenue 2024", `{"a":"2024"}`), doc("0.05", "revenue 2025", `{"a":"2025"}`), doc("0.30", "revenue 2025", `{"a":"far"}`)}
	hit, found, err := SelectHit(docs, "revenue 2025", 0.85, &out)
	if !found || err != nil || out["a"] != "2025" || hit.Distance != 0.05 {
		t.Fatalf("%+v %v %v %v", hit, found, err, out)
	}
	// Results are distance-sorted: stop at the first one beyond the threshold.
	if _, found, _ := SelectHit([]map[string]string{doc("0.2", "q", `{}`), doc("0.01", "q", `{}`)}, "q", 0.85, &out); found {
		t.Fatal("must stop at the first candidate beyond the threshold")
	}
	if _, found, err := SelectHit([]map[string]string{doc("0.01", "q", "")}, "q", 0.85, &out); found || err == nil {
		t.Fatal("empty response accepted")
	}
	if _, found, err := SelectHit(nil, "q", 0.85, &out); found || err != nil {
		t.Fatal("no results is a plain miss")
	}
}
