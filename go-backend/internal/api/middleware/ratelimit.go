package middleware

import (
	"go.opentelemetry.io/otel/attribute"

	"example.com/councilai-starter/internal/metrics"
	"example.com/councilai-starter/internal/telemetry"

	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"example.com/councilai-starter/internal/cache"
)

// gcraScript is an atomic Generic Cell Rate Algorithm (token-bucket
// equivalent) on Redis server time, so every Go instance shares one clock
// and one limit. KEYS[1]: limiter key. ARGV[1]: emission interval (µs),
// ARGV[2]: burst. Returns {allowed, remaining, retry_after_µs}.
//
// The reference incremented a counter and refreshed a 60 s EXPIRE on every
// request: burst was unused, and a client that kept sending requests kept
// extending the window, so after rps*60 requests it was blocked until it
// stopped for a full minute.
const gcraScript = `
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000000 + tonumber(t[2])
local interval = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local tat = tonumber(redis.call('GET', KEYS[1]) or now)
if tat < now then tat = now end
local new_tat = tat + interval
local allow_at = new_tat - burst * interval
if now < allow_at then
  return {0, 0, allow_at - now}
end
redis.call('SET', KEYS[1], new_tat, 'PX', math.ceil((new_tat - now) / 1000) + 1)
return {1, math.floor((now - allow_at) / interval), 0}
`

// FailurePolicy decides what happens when Redis cannot answer.
type FailurePolicy string

const (
	// FailClosed rejects requests (503) while the limiter is unavailable:
	// no unmetered model inference. Production default.
	FailClosed FailurePolicy = "closed"
	// FailOpen allows requests (reference behaviour): available, but an
	// outage removes the cost ceiling on model calls.
	FailOpen FailurePolicy = "open"
)

// Decision is one limiter answer.
type Decision struct {
	Allowed    bool
	Remaining  int
	RetryAfter time.Duration
}

// RateLimiter enforces `rps` sustained requests per second with bursts of
// up to `burst` per identity, across all Go instances sharing Redis.
type RateLimiter struct {
	client    *redis.Client
	cb        cache.CircuitBreaker
	script    *redis.Script
	interval  time.Duration
	burst     int
	prefix    string
	policy    FailurePolicy
	opTimeout time.Duration
	name      string // metric label: the prefix ("user", "auth")
}

// NewRateLimiter builds a limiter. prefix separates independent limits
// (for example per-user API calls and per-IP login attempts).
func NewRateLimiter(client *redis.Client, cb cache.CircuitBreaker, rps float64, burst int, prefix string, policy FailurePolicy) (*RateLimiter, error) {
	if rps <= 0 || burst < 1 {
		return nil, errors.New("rate limit requires rps > 0 and burst >= 1")
	}
	if policy != FailOpen && policy != FailClosed {
		return nil, fmt.Errorf("unknown rate-limit failure policy %q", policy)
	}
	if cb == nil {
		cb = cache.NewCircuitBreaker("redis", cache.DefaultConfig())
	}
	return &RateLimiter{
		client: client, cb: cb, script: redis.NewScript(gcraScript),
		interval: time.Duration(float64(time.Second) / rps), burst: burst,
		prefix: "councilai:v1:ratelimit:" + prefix + ":", policy: policy, opTimeout: 300 * time.Millisecond, name: limiterLabel(prefix),
	}, nil
}

// Allow consumes one request for identity.
func (rl *RateLimiter) Allow(ctx context.Context, identity string) (Decision, error) {
	var res []any
	err := rl.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, rl.opTimeout)
		defer cancel()
		v, err := rl.script.Run(opCtx, rl.client, []string{rl.prefix + identity}, rl.interval.Microseconds(), rl.burst).Slice()
		res = v
		return err
	})
	if err != nil {
		return Decision{}, err
	}
	if len(res) != 3 {
		return Decision{}, fmt.Errorf("unexpected limiter reply %v", res)
	}
	allowed, _ := res[0].(int64)
	remaining, _ := res[1].(int64)
	retry, _ := res[2].(int64)
	return Decision{Allowed: allowed == 1, Remaining: int(remaining), RetryAfter: time.Duration(retry) * time.Microsecond}, nil
}

// Middleware limits requests by identity(r); requests without an identity
// are rejected (this middleware runs after authentication).
func (rl *RateLimiter) Middleware(identity func(*http.Request) (string, bool)) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id, ok := identity(r)
			if !ok {
				writeJSONError(w, http.StatusUnauthorized, "unauthenticated")
				return
			}
			ctx, span := telemetry.Start(r.Context(), "ratelimit.check", attribute.String("ratelimit.limiter", rl.name))
			d, err := rl.Allow(ctx, id)
			span.SetAttributes(attribute.Bool("ratelimit.allowed", err == nil && d.Allowed))
			if err != nil {
				telemetry.Fail(span, "limiter_unavailable")
			}
			span.End()
			if err != nil {
				if r.Context().Err() != nil {
					return // client gone
				}
				metrics.DependencyErrors.WithLabelValues("redis", "rate_limit").Inc()
				if rl.policy == FailOpen {
					slog.Warn("rate limiter unavailable; failing open", "error", err.Error())
					next.ServeHTTP(w, r)
					return
				}
				slog.Error("rate limiter unavailable; failing closed", "error", err.Error())
				metrics.RateLimitRejections.WithLabelValues(rl.name, "unavailable").Inc()
				w.Header().Set("Retry-After", "5")
				writeJSONError(w, http.StatusServiceUnavailable, "rate limiter unavailable; try again shortly")
				return
			}
			w.Header().Set("X-RateLimit-Limit", strconv.Itoa(rl.burst))
			w.Header().Set("X-RateLimit-Remaining", strconv.Itoa(d.Remaining))
			if !d.Allowed {
				secs := int(math.Ceil(d.RetryAfter.Seconds()))
				if secs < 1 {
					secs = 1
				}
				w.Header().Set("Retry-After", strconv.Itoa(secs))
				metrics.RateLimitRejections.WithLabelValues(rl.name, "limited").Inc()
				writeJSONError(w, http.StatusTooManyRequests, "rate limit exceeded")
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// ByUser identifies requests by authenticated user UUID.
func ByUser(r *http.Request) (string, bool) { return UserID(r.Context()) }

// ByIP identifies requests by the TCP peer address. X-Forwarded-For is not
// trusted (clients could forge it to get unlimited attempts).
func ByIP(r *http.Request) (string, bool) {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	return "ip:" + host, host != ""
}

// limiterLabel keeps the metric label bounded even for unexpected prefixes.
func limiterLabel(prefix string) string {
	switch prefix {
	case "user", "auth":
		return prefix
	}
	return "other"
}
