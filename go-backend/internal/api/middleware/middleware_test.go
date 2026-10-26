package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"example.com/councilai-starter/internal/auth"
	"example.com/councilai-starter/internal/cache"
)

const (
	secret = "mw-test-secret-ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789"
	uid    = "5d1e9a2b-7c3f-4a8e-9b6d-0e1f2a3b4c5d"
)

func TestBearerToken(t *testing.T) {
	ok := map[string]string{"Bearer a.b.c": "a.b.c", "bearer a.b.c": "a.b.c", "BEARER   a.b.c  ": "a.b.c"}
	for h, want := range ok {
		if got, valid := BearerToken(h); !valid || got != want {
			t.Fatalf("%q -> %q %v", h, got, valid)
		}
	}
	for _, h := range []string{"", "Bearer", "Bearer ", "Basic a.b.c", "Token a.b.c", "Bearer a.b", "Bearer a.b.c d", "a.b.c"} {
		if _, valid := BearerToken(h); valid {
			t.Fatalf("%q accepted", h)
		}
	}
}

func TestAuthMiddleware(t *testing.T) {
	jm, _ := auth.NewJWTManager(secret, time.Hour)
	var reached atomic.Int32
	var seen string
	h := Auth(jm)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached.Add(1)
		seen, _ = UserID(r.Context())
	}))
	good, _ := jm.GenerateToken(uid)
	other, _ := auth.NewJWTManager("another-secret-with-enough-entropy-0123456789abcd", time.Hour)
	foreign, _ := other.GenerateToken(uid)
	expiredMgr, _ := auth.NewJWTManager(secret, time.Hour)
	expiredMgr.SetClock(func() time.Time { return time.Now().Add(-2 * time.Hour) })
	expired, _ := expiredMgr.GenerateToken(uid)
	for name, header := range map[string]string{"missing": "", "basic": "Basic dXNlcjpwYXNz", "malformed": "Bearer x", "wrong secret": "Bearer " + foreign, "expired": "Bearer " + expired} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/api/v1/query", nil)
		if header != "" {
			req.Header.Set("Authorization", header)
		}
		req.Header.Set("X-User-ID", uid) // client-supplied identity is ignored
		h.ServeHTTP(rec, req)
		if rec.Code != 401 || rec.Header().Get("WWW-Authenticate") == "" {
			t.Fatalf("%s: %d", name, rec.Code)
		}
	}
	if reached.Load() != 0 {
		t.Fatal("handler reached without a valid token")
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/query", nil)
	req.Header.Set("Authorization", "Bearer "+good)
	h.ServeHTTP(rec, req)
	if rec.Code != 200 || seen != uid {
		t.Fatalf("%d %q", rec.Code, seen)
	}
	if _, ok := UserID(context.Background()); ok {
		t.Fatal("no identity without authentication (the reference returned \"anonymous\")")
	}
}

func TestByIPIgnoresForwardedHeaders(t *testing.T) {
	req := httptest.NewRequest("POST", "/api/v1/login", nil)
	req.RemoteAddr = "10.1.2.3:5555"
	req.Header.Set("X-Forwarded-For", "1.1.1.1")
	if id, _ := ByIP(req); id != "ip:10.1.2.3" {
		t.Fatal(id)
	}
}

// ── Real Redis ──────────────────────────────────────────────────────

func limiter(t *testing.T, rps float64, burst int, policy FailurePolicy) (*RateLimiter, string) {
	t.Helper()
	addr := os.Getenv("REDIS_TEST_ADDR")
	if addr == "" {
		t.Skip("REDIS_TEST_ADDR not set")
	}
	client := cache.NewClient(addr, "", 0)
	t.Cleanup(func() { client.Close() })
	prefix := fmt.Sprintf("it%d", time.Now().UnixNano())
	rl, err := NewRateLimiter(client, nil, rps, burst, prefix, policy)
	if err != nil {
		t.Fatal(err)
	}
	return rl, prefix
}

func TestIntegrationGCRABurstThenRefill(t *testing.T) {
	rl, _ := limiter(t, 20, 3, FailClosed) // 1 request per 50 ms, burst 3
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		d, err := rl.Allow(ctx, "u1")
		if err != nil || !d.Allowed || d.Remaining != 2-i {
			t.Fatalf("burst %d: %+v %v", i, d, err)
		}
	}
	d, _ := rl.Allow(ctx, "u1")
	if d.Allowed || d.RetryAfter <= 0 || d.RetryAfter > 60*time.Millisecond {
		t.Fatalf("4th request: %+v", d)
	}
	time.Sleep(d.RetryAfter + 5*time.Millisecond)
	if d, _ := rl.Allow(ctx, "u1"); !d.Allowed {
		t.Fatal("one token must refill after the emission interval")
	}
	if d, _ := rl.Allow(ctx, "u1"); d.Allowed {
		t.Fatal("only one token refilled")
	}
	if d, _ := rl.Allow(ctx, "u2"); !d.Allowed || d.Remaining != 2 {
		t.Fatal("users have independent quotas")
	}
}

func TestIntegrationConcurrentRequestsCannotExceedBurst(t *testing.T) {
	rl, _ := limiter(t, 0.5, 5, FailClosed)
	other, _ := limiter(t, 0.5, 5, FailClosed)
	other.prefix = rl.prefix // a second Go instance sharing the same Redis keys
	var allowed atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 60; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			l := rl
			if i%2 == 1 {
				l = other
			}
			if d, err := l.Allow(context.Background(), "shared-user"); err == nil && d.Allowed {
				allowed.Add(1)
			}
		}(i)
	}
	wg.Wait()
	if allowed.Load() != 5 {
		t.Fatalf("allowed %d of 60 concurrent requests across two instances, want exactly 5", allowed.Load())
	}
}

func TestIntegrationLimiterMemoryIsBounded(t *testing.T) {
	rl, _ := limiter(t, 10, 4, FailClosed)
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		_, _ = rl.Allow(ctx, "bounded")
	}
	ttl := rl.client.PTTL(ctx, rl.prefix+"bounded").Val()
	if ttl <= 0 || ttl > 500*time.Millisecond {
		t.Fatalf("limiter key TTL %s must be at most burst*interval", ttl)
	}
	if n := len(rl.client.Keys(ctx, rl.prefix+"*").Val()); n != 1 {
		t.Fatalf("one key per identity, got %d", n)
	}
}

func TestIntegrationMiddlewareReturns429WithRetryAfter(t *testing.T) {
	rl, _ := limiter(t, 1, 2, FailClosed)
	h := rl.Middleware(func(r *http.Request) (string, bool) { return r.Header.Get("X-Test-User"), true })(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	codes := []int{}
	var last *httptest.ResponseRecorder
	for i := 0; i < 3; i++ {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/", nil)
		req.Header.Set("X-Test-User", "u")
		h.ServeHTTP(rec, req)
		codes = append(codes, rec.Code)
		last = rec
	}
	if fmt.Sprint(codes) != "[200 200 429]" {
		t.Fatalf("codes %v", codes)
	}
	ra, _ := strconv.Atoi(last.Header().Get("Retry-After"))
	var body map[string]string
	_ = json.Unmarshal(last.Body.Bytes(), &body)
	if ra != 1 || body["error"] != "rate limit exceeded" || last.Header().Get("X-RateLimit-Remaining") != "0" {
		t.Fatalf("Retry-After %q body %v", last.Header().Get("Retry-After"), body)
	}
}

func TestRedisOutagePolicies(t *testing.T) {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	dead := l.Addr().String()
	l.Close()
	for _, tc := range []struct {
		policy FailurePolicy
		code   int
	}{{FailClosed, 503}, {FailOpen, 200}} {
		client := cache.NewClient(dead, "", 0)
		rl, _ := NewRateLimiter(client, nil, 10, 10, "x", tc.policy)
		reached := false
		h := rl.Middleware(func(*http.Request) (string, bool) { return "u", true })(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached = true }))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/", nil))
		if rec.Code != tc.code || reached != (tc.policy == FailOpen) {
			t.Fatalf("%s: %d reached=%v", tc.policy, rec.Code, reached)
		}
		client.Close()
	}
	if _, err := NewRateLimiter(nil, nil, 10, 10, "x", "sometimes"); err == nil {
		t.Fatal("unknown policy accepted")
	}
	if _, err := NewRateLimiter(nil, nil, 0, 10, "x", FailOpen); err == nil {
		t.Fatal("rps 0 accepted")
	}
}
