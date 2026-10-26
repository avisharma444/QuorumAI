package api

import (
	"sync"
	"time"
)

// MemoryCache is the in-process TTL cache used until the Redis phase. It
// holds query results and, as a second instance, document summaries (the
// reference keeps both in Redis, with the same 1-hour TTL). Contents are lost
// when the backend restarts and are not shared between replicas.
type MemoryCache[V any] struct {
	mu     sync.RWMutex
	values map[string]cacheEntry[V]
	ttl    time.Duration
}

type cacheEntry[V any] struct {
	Data    V
	Expires time.Time
}

func NewMemoryCache[V any](ttl time.Duration) *MemoryCache[V] {
	return &MemoryCache[V]{values: make(map[string]cacheEntry[V]), ttl: ttl}
}

func (c *MemoryCache[V]) Get(key string) (V, bool) {
	c.mu.RLock()
	entry, ok := c.values[key]
	c.mu.RUnlock()
	if ok && time.Now().After(entry.Expires) {
		c.mu.Lock()
		if e, still := c.values[key]; still && time.Now().After(e.Expires) {
			delete(c.values, key)
		}
		c.mu.Unlock()
		ok = false
	}
	if !ok {
		var zero V
		return zero, false
	}
	return entry.Data, true
}

func (c *MemoryCache[V]) Set(key string, value V) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values[key] = cacheEntry[V]{Data: value, Expires: time.Now().Add(c.ttl)}
}
