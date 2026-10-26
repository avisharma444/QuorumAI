// Package cachetest provides in-memory cache doubles for API unit tests.
// They are never wired into the server binary; production uses Redis and
// RediSearch (see internal/cache), and Redis behaviour itself is covered by
// integration tests against a real Redis Stack.
package cachetest

import (
	"context"
	"encoding/json"
	"math"
	"sort"
	"sync"
	"time"

	"example.com/councilai-starter/internal/cache"
)

// Exact is an in-memory stand-in for cache.RedisCache.
type Exact struct {
	mu     sync.Mutex
	data   map[string][]byte
	ttls   map[string]time.Duration
	GetErr error
	SetErr error
	Gets   int
	Sets   int
}

func NewExact() *Exact { return &Exact{data: map[string][]byte{}, ttls: map[string]time.Duration{}} }

func (e *Exact) Get(ctx context.Context, key string, dest any) (bool, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.Gets++
	if e.GetErr != nil {
		return false, e.GetErr
	}
	raw, ok := e.data[key]
	if !ok {
		return false, nil
	}
	if err := json.Unmarshal(raw, dest); err != nil {
		return false, cache.ErrCorruptEntry
	}
	return true, nil
}

func (e *Exact) Set(ctx context.Context, key string, value any) error {
	return e.SetWithTTL(ctx, key, value, cache.DefaultExactTTL)
}

func (e *Exact) SetWithTTL(ctx context.Context, key string, value any, ttl time.Duration) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.Sets++
	if e.SetErr != nil {
		return e.SetErr
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return err
	}
	e.data[key], e.ttls[key] = raw, ttl
	return nil
}

// Keys lists stored keys, sorted.
func (e *Exact) Keys() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	keys := make([]string, 0, len(e.data))
	for k := range e.data {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Put stores raw bytes (to simulate corrupt entries).
func (e *Exact) PutRaw(key string, raw []byte) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.data[key] = raw
}

// TTL returns the TTL a key was stored with.
func (e *Exact) TTL(key string) time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ttls[key]
}

type entry struct {
	scope    cache.Scope
	question string
	vector   []float32
	data     []byte
}

// Semantic is an in-memory stand-in for cache.RedisSemanticCache with the
// same scope, threshold and number-guard rules. Its cosine computation only
// emulates RediSearch for API unit tests.
type Semantic struct {
	mu      sync.Mutex
	entries []entry
	GetErr  error
	PutErr  error
	Gets    int
	Puts    int
}

func NewSemantic() *Semantic { return &Semantic{} }

func (s *Semantic) EnsureIndex(context.Context) error { return nil }
func (s *Semantic) Close() error                      { return nil }

func (s *Semantic) Put(ctx context.Context, scope cache.Scope, question string, vector []float32, response any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Puts++
	if s.PutErr != nil {
		return s.PutErr
	}
	if err := cache.ValidateVector(vector); err != nil {
		return err
	}
	raw, err := json.Marshal(response)
	if err != nil {
		return err
	}
	s.entries = append(s.entries, entry{scope, question, append([]float32(nil), vector...), raw})
	return nil
}

func (s *Semantic) Get(ctx context.Context, scope cache.Scope, question string, vector []float32, threshold float64, dest any) (cache.SemanticHit, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Gets++
	if s.GetErr != nil {
		return cache.SemanticHit{}, false, s.GetErr
	}
	if err := cache.ValidateVector(vector); err != nil {
		return cache.SemanticHit{}, false, err
	}
	type cand struct {
		e    entry
		dist float64
	}
	var cands []cand
	for _, e := range s.entries {
		if e.scope == scope {
			cands = append(cands, cand{e, cosineDistance(vector, e.vector)})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	for _, c := range cands {
		if !cache.Accept(c.dist, threshold) {
			break
		}
		if !cache.NumbersMatch(question, c.e.question) {
			continue
		}
		if err := json.Unmarshal(c.e.data, dest); err != nil {
			return cache.SemanticHit{}, false, cache.ErrCorruptEntry
		}
		return cache.SemanticHit{Distance: c.dist, Similarity: 1 - c.dist, Question: c.e.question}, true, nil
	}
	return cache.SemanticHit{}, false, nil
}

// Len returns the number of stored entries.
func (s *Semantic) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}

func cosineDistance(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
}
