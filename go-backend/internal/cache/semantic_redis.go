package cache

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// Reference constants.
const (
	SemanticIndexName = "idx:semantic_cache"
	SemanticKeyPrefix = "semcache:"
	VectorDim         = 384
	DefaultTTL        = 24 * time.Hour
	// DefaultThreshold is the reference cosine-similarity threshold.
	DefaultThreshold = 0.85
	knnCandidates    = 3
)

var (
	// ErrSearchUnavailable means Redis answered but RediSearch is not loaded.
	ErrSearchUnavailable = errors.New("RediSearch is not available on this Redis server; semantic cache disabled")
	// ErrIndexSchemaMismatch means an index with our name exists with an
	// incompatible schema (for example the reference's index without the
	// scope field). It is not dropped automatically.
	ErrIndexSchemaMismatch = errors.New("semantic cache index exists with an incompatible schema")
	// ErrInvalidVector rejects vectors that cannot be stored or compared.
	ErrInvalidVector = errors.New("invalid embedding vector")
)

// Scope isolates semantic-cache entries. DocID is the reference's tag
// filter. Key is a hex digest of every other input that changes an answer
// (namespace, resolved strategy, model and prompt fingerprint, retrieval
// settings, embedding model); only entries with an identical Key can match.
type Scope struct {
	DocID string
	Key   string
}

// SemanticHit describes an accepted match.
type SemanticHit struct {
	Distance   float64 // cosine distance reported by RediSearch
	Similarity float64 // 1 - distance
	Question   string  // the cached question that matched
}

// SemanticCache is the reference interface, extended with scope and the
// question text used by the hit guard.
type SemanticCache interface {
	EnsureIndex(ctx context.Context) error
	Put(ctx context.Context, scope Scope, question string, vector []float32, response any) error
	Get(ctx context.Context, scope Scope, question string, vector []float32, threshold float64, dest any) (SemanticHit, bool, error)
	Close() error
}

// RedisSemanticCache stores answers in Redis HASHes indexed by a RediSearch
// FLAT/COSINE/FLOAT32 vector index and looks them up with KNN FT.SEARCH.
type RedisSemanticCache struct {
	client    *redis.Client
	cb        CircuitBreaker
	ttl       time.Duration
	index     string
	prefix    string
	opTimeout time.Duration

	mu    sync.Mutex
	ready bool
}

// NewRedisSemanticCacheWithBreaker builds the L2 cache on a shared client and breaker.
func NewRedisSemanticCacheWithBreaker(client *redis.Client, cb CircuitBreaker, ttl time.Duration) *RedisSemanticCache {
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	if cb == nil {
		cb = NewCircuitBreaker("redis", DefaultConfig())
	}
	return &RedisSemanticCache{client: client, cb: cb, ttl: ttl, index: SemanticIndexName, prefix: SemanticKeyPrefix, opTimeout: DefaultOpTimeout}
}

// WithNames overrides the index name and key prefix (integration tests use
// unique names so they never touch the service's index).
func (c *RedisSemanticCache) WithNames(index, prefix string) *RedisSemanticCache {
	c.index, c.prefix = index, prefix
	return c
}

// SetOpTimeout changes the per-operation timeout.
func (c *RedisSemanticCache) SetOpTimeout(d time.Duration) {
	if d > 0 {
		c.opTimeout = d
	}
}

// TTL returns the entry TTL.
func (c *RedisSemanticCache) TTL() time.Duration { return c.ttl }

// ── Vector encoding ─────────────────────────────────────────────────

// Float32ToBytes encodes v as little-endian IEEE 754 float32s, the layout
// RediSearch expects for TYPE FLOAT32. The reference reinterpreted the
// slice's memory with unsafe, which is only correct on little-endian hosts
// and aliases the caller's slice; this copy is portable and owns its bytes.
func Float32ToBytes(v []float32) []byte {
	out := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(f))
	}
	return out
}

// ValidateVector checks dimension, finiteness and a non-zero norm (cosine
// distance is undefined for the zero vector).
func ValidateVector(v []float32) error {
	if len(v) != VectorDim {
		return fmt.Errorf("%w: dimension %d, want %d", ErrInvalidVector, len(v), VectorDim)
	}
	var norm float64
	for _, f := range v {
		x := float64(f)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return fmt.Errorf("%w: non-finite value", ErrInvalidVector)
		}
		norm += x * x
	}
	if norm == 0 {
		return fmt.Errorf("%w: zero vector", ErrInvalidVector)
	}
	return nil
}

// ── Query helpers ───────────────────────────────────────────────────

// SanitizeTag escapes every character that is not a letter, digit or
// underscore, as RediSearch TAG syntax requires. The reference escaped
// only a fixed list (missing spaces, commas, quotes, ...).
func SanitizeTag(tag string) string {
	var b strings.Builder
	for _, r := range tag {
		if r == '_' || (r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		} else {
			b.WriteByte('\\')
			b.WriteRune(r)
		}
	}
	return b.String()
}

var numberPattern = regexp.MustCompile(`\d+(?:[.,]\d+)*`)

// NumbersMatch reports whether two questions mention the same numbers.
// Embeddings place "revenue in 2024" and "revenue in 2025" very close, so a
// semantic hit is rejected when the numbers differ.
func NumbersMatch(a, b string) bool {
	na, nb := numberPattern.FindAllString(a, -1), numberPattern.FindAllString(b, -1)
	sort.Strings(na)
	sort.Strings(nb)
	return strings.Join(na, "|") == strings.Join(nb, "|")
}

// MaxDistance converts a similarity threshold into the largest accepted
// cosine distance: similarity = 1 - distance, so distance <= 1 - threshold.
func MaxDistance(threshold float64) float64 { return 1 - threshold }

// distanceEpsilon absorbs float32 rounding in RediSearch's cosine distance:
// identical vectors can come back as a tiny negative (or positive) distance.
// The cache-safety evaluation found identical questions rejected because of
// a distance like -1.2e-7.
const distanceEpsilon = 1e-6

// Accept applies the threshold rule. A slightly negative distance is
// rounding noise for (near-)identical vectors and counts as 0; the upper
// bound stays exact.
func Accept(distance, threshold float64) bool {
	return !math.IsNaN(distance) && distance >= -distanceEpsilon && distance <= MaxDistance(threshold)
}

// clampDistance maps a cosine distance into its valid range [0, 2].
func clampDistance(d float64) float64 { return math.Min(2, math.Max(0, d)) }

func (c *RedisSemanticCache) key(scope Scope, question string) string {
	q := sha256.Sum256([]byte(question))
	s := scope.Key
	if len(s) > 16 {
		s = s[:16]
	}
	return c.prefix + s + ":" + hex.EncodeToString(q[:8])
}

func isScopeValid(scope Scope) error {
	if scope.DocID == "" || scope.Key == "" {
		return errors.New("semantic cache scope requires a document and a scope key")
	}
	return nil
}

// ── Index ───────────────────────────────────────────────────────────

func (c *RedisSemanticCache) isReady() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ready
}

func (c *RedisSemanticCache) setReady(v bool) {
	c.mu.Lock()
	c.ready = v
	c.mu.Unlock()
}

// EnsureIndex creates the index if needed. It is idempotent and safe to
// call concurrently (FT.CREATE is atomic in Redis; "Index already exists"
// is success after a schema probe). Other errors are returned as errors;
// the reference also treated "BUSY" replies as success.
func (c *RedisSemanticCache) EnsureIndex(ctx context.Context) error {
	err := c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, 5*c.opTimeout)
		defer cancel()
		err := c.client.Do(opCtx,
			"FT.CREATE", c.index, "ON", "HASH", "PREFIX", "1", c.prefix,
			"SCHEMA",
			"doc_id", "TAG",
			"scope", "TAG",
			"vector", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", strconv.Itoa(VectorDim), "DISTANCE_METRIC", "COSINE",
		).Err()
		if err == nil {
			return nil
		}
		msg := strings.ToLower(err.Error())
		switch {
		case strings.Contains(msg, "index already exists"):
			info, ierr := c.client.Do(opCtx, "FT.INFO", c.index).Result()
			if ierr != nil {
				return ierr
			}
			if missing := missingAttributes(info, "doc_id", "scope", "vector"); len(missing) > 0 {
				return Neutral(fmt.Errorf("%w: missing %s", ErrIndexSchemaMismatch, strings.Join(missing, ", ")))
			}
			return nil
		case strings.Contains(msg, "unknown command"):
			return Neutral(fmt.Errorf("%w: %v", ErrSearchUnavailable, err))
		}
		return err
	})
	c.setReady(err == nil)
	if err != nil && errors.Is(err, ErrCircuitOpen) {
		return ErrCircuitOpen
	}
	return err
}

// missingAttributes lists required fields absent from an FT.INFO reply
// (RESP2: flat key/value array whose "attributes" entry lists one array per
// field with an "identifier"; RESP3: the equivalent maps).
func missingAttributes(info any, want ...string) []string {
	have := map[string]bool{}
	var attrs []any
	switch r := info.(type) {
	case []any:
		for i := 0; i+1 < len(r); i += 2 {
			if toString(r[i]) == "attributes" {
				attrs, _ = r[i+1].([]any)
			}
		}
	case map[any]any:
		attrs, _ = r["attributes"].([]any)
	}
	for _, a := range attrs {
		switch f := a.(type) {
		case []any:
			for i := 0; i+1 < len(f); i += 2 {
				if toString(f[i]) == "identifier" {
					have[toString(f[i+1])] = true
				}
			}
		case map[any]any:
			have[toString(f["identifier"])] = true
		}
	}
	var missing []string
	for _, w := range want {
		if !have[w] {
			missing = append(missing, w)
		}
	}
	return missing
}

func isMissingIndex(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "no such index") || strings.Contains(msg, "unknown index name")
}

// ── Put / Get ───────────────────────────────────────────────────────

// Put stores (question, vector, response) for scope. HSET and EXPIRE run in
// one MULTI transaction so an entry never exists without its TTL.
func (c *RedisSemanticCache) Put(ctx context.Context, scope Scope, question string, vector []float32, response any) error {
	if err := isScopeValid(scope); err != nil {
		return err
	}
	if err := ValidateVector(vector); err != nil {
		return err
	}
	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to marshal cache response: %w", err)
	}
	if !c.isReady() {
		if err := c.EnsureIndex(ctx); err != nil {
			return err
		}
	}
	key := c.key(scope, question)
	err = c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, c.opTimeout)
		defer cancel()
		_, err := c.client.TxPipelined(opCtx, func(p redis.Pipeliner) error {
			p.HSet(opCtx, key, map[string]any{
				"doc_id":   scope.DocID,
				"scope":    scope.Key,
				"question": question,
				"response": string(data),
				"vector":   Float32ToBytes(vector),
			})
			p.Expire(opCtx, key, c.ttl)
			return nil
		})
		return err
	})
	if err != nil {
		if errors.Is(err, ErrCircuitOpen) {
			return ErrCircuitOpen
		}
		return fmt.Errorf("failed to store semantic cache entry: %w", err)
	}
	return nil
}

// Get returns the nearest cached response in scope whose cosine distance is
// within the threshold and whose question mentions the same numbers.
func (c *RedisSemanticCache) Get(ctx context.Context, scope Scope, question string, vector []float32, threshold float64, dest any) (SemanticHit, bool, error) {
	if err := isScopeValid(scope); err != nil {
		return SemanticHit{}, false, err
	}
	if err := ValidateVector(vector); err != nil {
		return SemanticHit{}, false, err
	}
	if threshold <= 0 || threshold > 1 {
		return SemanticHit{}, false, fmt.Errorf("semantic threshold must be in (0, 1], got %v", threshold)
	}
	query := fmt.Sprintf("(@doc_id:{%s} @scope:{%s})=>[KNN %d @vector $vec AS score]", SanitizeTag(scope.DocID), SanitizeTag(scope.Key), knnCandidates)
	var reply any
	err := c.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, c.opTimeout)
		defer cancel()
		res, err := c.client.Do(opCtx,
			"FT.SEARCH", c.index, query,
			"PARAMS", "2", "vec", Float32ToBytes(vector),
			"SORTBY", "score", "ASC",
			"RETURN", "3", "response", "score", "question",
			"LIMIT", "0", strconv.Itoa(knnCandidates),
			"DIALECT", "2",
		).Result()
		if isMissingIndex(err) {
			// Redis lost the index (restart without persistence, FLUSHALL).
			// Recreate it later; this lookup is a miss, not a Redis failure.
			c.setReady(false)
			return Neutral(err)
		}
		reply = res
		return err
	})
	if err != nil {
		if isMissingIndex(err) {
			_ = c.EnsureIndex(context.WithoutCancel(ctx))
			return SemanticHit{}, false, nil
		}
		if errors.Is(err, ErrCircuitOpen) {
			return SemanticHit{}, false, ErrCircuitOpen
		}
		return SemanticHit{}, false, fmt.Errorf("ft.search failed: %w", err)
	}
	docs, err := ParseSearchReply(reply)
	if err != nil {
		return SemanticHit{}, false, err
	}
	return SelectHit(docs, question, threshold, dest)
}

// SelectHit applies the hit rules to KNN results sorted by distance: the
// score must parse (a missing or malformed score is an error, never a
// match; the reference parsed it to 0, a perfect match), lie within the
// threshold, and the cached question must mention the same numbers.
func SelectHit(docs []map[string]string, question string, threshold float64, dest any) (SemanticHit, bool, error) {
	for _, d := range docs {
		score, perr := strconv.ParseFloat(d["score"], 64)
		if perr != nil || math.IsNaN(score) || math.IsInf(score, 0) {
			return SemanticHit{}, false, fmt.Errorf("malformed FT.SEARCH score %q", d["score"])
		}
		if !Accept(score, threshold) {
			break // results are sorted by distance; nothing further can pass
		}
		if !NumbersMatch(question, d["question"]) {
			continue
		}
		if d["response"] == "" {
			return SemanticHit{}, false, errors.New("semantic cache entry has no response")
		}
		if err := json.Unmarshal([]byte(d["response"]), dest); err != nil {
			return SemanticHit{}, false, fmt.Errorf("%w: %v", ErrCorruptEntry, err)
		}
		score = clampDistance(score)
		return SemanticHit{Distance: score, Similarity: 1 - score, Question: d["question"]}, true, nil
	}
	return SemanticHit{}, false, nil
}

// ParseSearchReply converts an FT.SEARCH reply into one field map per
// document, in reply order. It accepts the RESP2 array form
// [total, key, [field, value, ...], ...] and the RESP3 map form
// {total_results, results: [{id, extra_attributes: {...}}]}.
func ParseSearchReply(reply any) ([]map[string]string, error) {
	switch r := reply.(type) {
	case []any:
		if len(r) == 0 {
			return nil, errors.New("empty FT.SEARCH reply")
		}
		if _, ok := r[0].(int64); !ok {
			return nil, fmt.Errorf("unexpected FT.SEARCH total %T", r[0])
		}
		var docs []map[string]string
		for i := 1; i+1 < len(r); i += 2 {
			fields, ok := r[i+1].([]any)
			if !ok {
				return nil, fmt.Errorf("unexpected FT.SEARCH fields %T", r[i+1])
			}
			doc := map[string]string{}
			for j := 0; j+1 < len(fields); j += 2 {
				doc[toString(fields[j])] = toString(fields[j+1])
			}
			docs = append(docs, doc)
		}
		return docs, nil
	case map[any]any:
		results, ok := r["results"].([]any)
		if !ok {
			return nil, errors.New("FT.SEARCH RESP3 reply without results")
		}
		var docs []map[string]string
		for _, item := range results {
			m, ok := item.(map[any]any)
			if !ok {
				return nil, fmt.Errorf("unexpected FT.SEARCH result %T", item)
			}
			attrs, _ := m["extra_attributes"].(map[any]any)
			doc := map[string]string{}
			for k, v := range attrs {
				doc[toString(k)] = toString(v)
			}
			docs = append(docs, doc)
		}
		return docs, nil
	}
	return nil, fmt.Errorf("unexpected FT.SEARCH reply type %T", reply)
}

func toString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []byte:
		return string(x)
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	}
	return fmt.Sprint(v)
}

// Close closes the client.
func (c *RedisSemanticCache) Close() error { return c.client.Close() }
