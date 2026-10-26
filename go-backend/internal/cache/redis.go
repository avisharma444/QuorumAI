package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// DefaultExactTTL is the reference L1 TTL.
const DefaultExactTTL = time.Hour

// DefaultOpTimeout bounds one Redis round trip so a slow Redis cannot stall
// requests (the reference relied only on the request context).
const DefaultOpTimeout = 500 * time.Millisecond

// ErrCorruptEntry means a cached value could not be decoded. It is a miss,
// not a Redis failure, and the caller must not use the value.
var ErrCorruptEntry = errors.New("corrupt cache entry")

// NewClient builds the shared go-redis client. RESP2 is selected so
// FT.SEARCH replies have one documented shape (the reference relied on
// go-redis v9's default RESP3 but parsed RESP2 arrays, so it never matched).
func NewClient(addr, password string, db int) *redis.Client {
	return redis.NewClient(&redis.Options{
		Addr:         addr,
		Password:     password,
		DB:           db,
		Protocol:     2,
		DialTimeout:  time.Second,
		ReadTimeout:  time.Second,
		WriteTimeout: time.Second,
		MaxRetries:   1,
	})
}

// RedisCache is the L1 exact-match cache: JSON strings under exact keys,
// with TTL, protected by the shared circuit breaker.
type RedisCache struct {
	client    redis.Cmdable
	closer    func() error
	ttl       time.Duration
	cb        CircuitBreaker
	opTimeout time.Duration
}

// NewRedisCacheWithBreaker builds an L1 cache on an existing client and breaker.
func NewRedisCacheWithBreaker(client *redis.Client, cb CircuitBreaker, ttl time.Duration) *RedisCache {
	if ttl <= 0 {
		ttl = DefaultExactTTL
	}
	if cb == nil {
		cb = NewCircuitBreaker("redis", DefaultConfig())
	}
	return &RedisCache{client: client, closer: client.Close, ttl: ttl, cb: cb, opTimeout: DefaultOpTimeout}
}

// SetOpTimeout changes the per-operation timeout.
func (c *RedisCache) SetOpTimeout(d time.Duration) {
	if d > 0 {
		c.opTimeout = d
	}
}

// TTL returns the default TTL.
func (c *RedisCache) TTL() time.Duration { return c.ttl }

// CircuitBreaker returns the breaker guarding this cache.
func (c *RedisCache) CircuitBreaker() CircuitBreaker { return c.cb }

// Ping checks connectivity through the breaker.
func (c *RedisCache) Ping(ctx context.Context) error {
	return c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, c.opTimeout)
		defer cancel()
		return c.client.Ping(opCtx).Err()
	})
}

// Get loads key into dest. A missing key is (false, nil). A value that does
// not decode returns ErrCorruptEntry and is deleted best-effort.
func (c *RedisCache) Get(ctx context.Context, key string, dest any) (bool, error) {
	var raw string
	var miss bool
	err := c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, c.opTimeout)
		defer cancel()
		v, err := c.client.Get(opCtx, key).Result()
		if errors.Is(err, redis.Nil) {
			miss = true
			return nil // a miss is a healthy Redis answer
		}
		raw = v
		return err
	})
	if err != nil {
		if errors.Is(err, ErrCircuitOpen) {
			return false, ErrCircuitOpen
		}
		return false, fmt.Errorf("redis get failed: %w", err)
	}
	if miss {
		return false, nil
	}
	if err := json.Unmarshal([]byte(raw), dest); err != nil {
		_ = c.Delete(context.WithoutCancel(ctx), key)
		return false, fmt.Errorf("%w: %v", ErrCorruptEntry, err)
	}
	return true, nil
}

// Set stores value as JSON with the default TTL.
func (c *RedisCache) Set(ctx context.Context, key string, value any) error {
	return c.SetWithTTL(ctx, key, value, c.ttl)
}

// SetWithTTL stores value as JSON with an explicit TTL.
func (c *RedisCache) SetWithTTL(ctx context.Context, key string, value any, ttl time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("cache marshal failed: %w", err)
	}
	err = c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, c.opTimeout)
		defer cancel()
		return c.client.Set(opCtx, key, data, ttl).Err()
	})
	if err != nil {
		if errors.Is(err, ErrCircuitOpen) {
			return ErrCircuitOpen
		}
		return fmt.Errorf("redis set failed: %w", err)
	}
	return nil
}

// Delete removes key.
func (c *RedisCache) Delete(ctx context.Context, key string) error {
	return c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, c.opTimeout)
		defer cancel()
		return c.client.Del(opCtx, key).Err()
	})
}

// Close closes the client.
func (c *RedisCache) Close() error {
	if c.closer != nil {
		return c.closer()
	}
	return nil
}
