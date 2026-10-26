// Package memory stores recent multi-turn conversation history in Redis
// lists (reference internal/memory/conversation.go).
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/redis/go-redis/v9"

	"example.com/councilai-starter/internal/cache"
)

// Reference defaults: 10 exchanges, 24 h TTL.
const (
	DefaultMaxTurns = 10
	DefaultTTL      = 24 * time.Hour
	keyPrefix       = "councilai:v1:conv:"
)

// Turn is one message (reference).
type Turn struct {
	Role      string    `json:"role"` // "user" | "assistant"
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// ErrInvalidSession rejects malformed session or user identifiers.
var ErrInvalidSession = errors.New("invalid session identifier")

var sessionPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// ValidSessionID reports whether s may be used as a session identifier.
func ValidSessionID(s string) bool { return sessionPattern.MatchString(s) }

// ConversationStore keeps the last maxTurns exchanges per (user, session).
type ConversationStore struct {
	client    *redis.Client
	cb        cache.CircuitBreaker
	maxTurns  int
	ttl       time.Duration
	opTimeout time.Duration
}

// NewConversationStore builds a store on a shared client and breaker.
func NewConversationStore(client *redis.Client, cb cache.CircuitBreaker, maxTurns int, ttl time.Duration) *ConversationStore {
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	if cb == nil {
		cb = cache.NewCircuitBreaker("redis", cache.DefaultConfig())
	}
	return &ConversationStore{client: client, cb: cb, maxTurns: maxTurns, ttl: ttl, opTimeout: 500 * time.Millisecond}
}

// Key is the per-user, per-session list key. The user UUID is part of the
// key, so the same session ID under two users never shares history.
func Key(userID, sessionID string) string { return keyPrefix + userID + ":" + sessionID }

func validate(userID, sessionID string) error {
	if userID == "" || !ValidSessionID(sessionID) {
		return ErrInvalidSession
	}
	return nil
}

// Append adds one turn (reference API), atomically with the trim and TTL.
func (s *ConversationStore) Append(ctx context.Context, userID, sessionID string, turn Turn) error {
	return s.appendTurns(ctx, userID, sessionID, turn)
}

// AppendExchange adds a user turn and the assistant's answer in one
// transaction, so history never contains a question without its answer.
// The reference used two separate appends, each RPUSH/LTRIM/EXPIRE as
// separate commands.
func (s *ConversationStore) AppendExchange(ctx context.Context, userID, sessionID string, user, assistant Turn) error {
	return s.appendTurns(ctx, userID, sessionID, user, assistant)
}

func (s *ConversationStore) appendTurns(ctx context.Context, userID, sessionID string, turns ...Turn) error {
	if err := validate(userID, sessionID); err != nil {
		return err
	}
	values := make([]any, 0, len(turns))
	for _, t := range turns {
		if t.Timestamp.IsZero() {
			t.Timestamp = time.Now().UTC()
		}
		b, err := json.Marshal(t)
		if err != nil {
			return fmt.Errorf("failed to marshal turn: %w", err)
		}
		values = append(values, string(b))
	}
	k := Key(userID, sessionID)
	return s.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, s.opTimeout)
		defer cancel()
		_, err := s.client.TxPipelined(opCtx, func(p redis.Pipeliner) error {
			p.RPush(opCtx, k, values...)
			p.LTrim(opCtx, k, int64(-2*s.maxTurns), -1) // keep the last maxTurns exchanges
			p.Expire(opCtx, k, s.ttl)
			return nil
		})
		return err
	})
}

// GetHistory returns up to the last `limit` exchanges (0 = all kept), oldest
// first. One LRANGE with a negative start replaces the reference's LLEN +
// LRANGE pair, which could race with a concurrent append.
func (s *ConversationStore) GetHistory(ctx context.Context, userID, sessionID string, limit int) ([]Turn, error) {
	if err := validate(userID, sessionID); err != nil {
		return nil, err
	}
	start := int64(0)
	if limit > 0 {
		start = int64(-2 * limit)
	}
	var raw []string
	err := s.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, s.opTimeout)
		defer cancel()
		var err error
		raw, err = s.client.LRange(opCtx, Key(userID, sessionID), start, -1).Result()
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("failed to read conversation history: %w", err)
	}
	turns := make([]Turn, 0, len(raw))
	for _, r := range raw {
		var t Turn
		if err := json.Unmarshal([]byte(r), &t); err != nil || (t.Role != "user" && t.Role != "assistant") {
			continue // never feed a corrupt entry to a model
		}
		turns = append(turns, t)
	}
	return turns, nil
}

// Clear deletes one session of one user.
func (s *ConversationStore) Clear(ctx context.Context, userID, sessionID string) error {
	if err := validate(userID, sessionID); err != nil {
		return err
	}
	return s.cb.Execute(ctx, func() error {
		opCtx, cancel := context.WithTimeout(ctx, s.opTimeout)
		defer cancel()
		return s.client.Del(opCtx, Key(userID, sessionID)).Err()
	})
}

// Close closes the client.
func (s *ConversationStore) Close() error { return s.client.Close() }
