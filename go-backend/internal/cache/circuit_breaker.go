// Package cache provides the Redis exact-match cache, the RediSearch
// semantic cache and the circuit breaker that protects both (reference
// internal/cache).
package cache

import (
	"example.com/councilai-starter/internal/metrics"
	"strings"

	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
)

// State is the breaker state.
type State int

const (
	StateClosed State = iota
	StateHalfOpen
	StateOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "CLOSED"
	case StateHalfOpen:
		return "HALF_OPEN"
	case StateOpen:
		return "OPEN"
	}
	return "UNKNOWN"
}

var (
	// ErrCircuitOpen is returned without contacting Redis while the breaker
	// is open, or when half-open probe slots are taken.
	ErrCircuitOpen = errors.New("circuit breaker is open: cache operation rejected")
	// ErrTooManyRequests is kept for reference compatibility.
	ErrTooManyRequests = ErrCircuitOpen
)

// CircuitBreaker is the reference interface.
type CircuitBreaker interface {
	Execute(ctx context.Context, op func() error) error
	State() State
	Reset()
	Trip()
	Counts() (consecutiveSuccesses, consecutiveFailures int, totalSuccesses, totalFailures uint64)
	Allow() bool
}

// Config holds breaker settings (reference defaults: 3 / 2 / 10s / 1).
type Config struct {
	FailureThreshold int
	SuccessThreshold int
	Timeout          time.Duration
	HalfOpenMaxCalls int
	NowFunc          func() time.Time // injectable clock for tests
}

// DefaultConfig returns the reference production defaults.
func DefaultConfig() Config {
	return Config{FailureThreshold: 3, SuccessThreshold: 2, Timeout: 10 * time.Second, HalfOpenMaxCalls: 1}
}

// neutralError marks an outcome that says nothing about Redis health (for
// example a missing RediSearch index that is being recreated).
type neutralError struct{ err error }

func (e neutralError) Error() string { return e.err.Error() }
func (e neutralError) Unwrap() error { return e.err }

// Neutral wraps err so Execute returns it without counting it as a failure.
func Neutral(err error) error {
	if err == nil {
		return nil
	}
	return neutralError{err}
}

// Breaker is a thread-safe circuit breaker. The lock is never held while
// the protected operation runs.
//
// Differences from the reference implementation:
//   - Each state change starts a new generation. An operation that began in
//     an earlier generation (for example a slow call started while CLOSED
//     that finishes after the breaker opened) does not change the counters
//     of the current state; the reference applied such stale results.
//   - Outcomes are classified: an error caused by the caller's own
//     context ending (request cancelled or timed out) and errors wrapped
//     with Neutral are not Redis failures. The reference counted every
//     error, so client disconnects could open the breaker.
//   - A half-open probe slot is always released, whatever the outcome
//     (success, failure, neutral, panic).
type Breaker struct {
	mu                   sync.Mutex
	name                 string
	cfg                  Config
	state                State
	generation           uint64
	consecutiveFailures  int
	consecutiveSuccesses int
	totalSuccesses       uint64
	totalFailures        uint64
	openedAt             time.Time
	halfOpenCalls        int
}

// NewCircuitBreaker builds a breaker; zero config fields take the defaults.
func NewCircuitBreaker(name string, cfg Config) *Breaker {
	d := DefaultConfig()
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = d.FailureThreshold
	}
	if cfg.SuccessThreshold <= 0 {
		cfg.SuccessThreshold = d.SuccessThreshold
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = d.Timeout
	}
	if cfg.HalfOpenMaxCalls <= 0 {
		cfg.HalfOpenMaxCalls = d.HalfOpenMaxCalls
	}
	return &Breaker{name: name, cfg: cfg}
}

func (cb *Breaker) now() time.Time {
	if cb.cfg.NowFunc != nil {
		return cb.cfg.NowFunc()
	}
	return time.Now()
}

// Name returns the breaker's name.
func (cb *Breaker) Name() string { return cb.name }

// State reports the state, showing HALF_OPEN once the cooldown has passed.
func (cb *Breaker) State() State {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	if cb.state == StateOpen && cb.now().Sub(cb.openedAt) >= cb.cfg.Timeout {
		return StateHalfOpen
	}
	return cb.state
}

// Counts returns consecutive and total outcomes.
func (cb *Breaker) Counts() (int, int, uint64, uint64) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.consecutiveSuccesses, cb.consecutiveFailures, cb.totalSuccesses, cb.totalFailures
}

// setStateLocked changes state and starts a new generation.
func (cb *Breaker) setStateLocked(s State) {
	if cb.state != s {
		log.Printf("[CircuitBreaker %s] %s -> %s", cb.name, cb.state, s)
		metrics.BreakerTransitions.WithLabelValues(cb.name, strings.ToLower(s.String())).Inc()
	}
	cb.state = s
	cb.generation++
	cb.consecutiveFailures, cb.consecutiveSuccesses, cb.halfOpenCalls = 0, 0, 0
	if s == StateOpen {
		cb.openedAt = cb.now()
	}
}

// Reset closes the breaker.
func (cb *Breaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.setStateLocked(StateClosed)
}

// Trip opens the breaker.
func (cb *Breaker) Trip() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.setStateLocked(StateOpen)
}

// ticket identifies an admitted call.
type ticket struct {
	generation uint64
	probe      bool
}

// admit decides whether a call may run.
func (cb *Breaker) admit() (ticket, error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	switch cb.state {
	case StateOpen:
		if cb.now().Sub(cb.openedAt) < cb.cfg.Timeout {
			return ticket{}, ErrCircuitOpen
		}
		cb.setStateLocked(StateHalfOpen)
		fallthrough
	case StateHalfOpen:
		if cb.halfOpenCalls >= cb.cfg.HalfOpenMaxCalls {
			return ticket{}, ErrCircuitOpen
		}
		cb.halfOpenCalls++
		return ticket{generation: cb.generation, probe: true}, nil
	}
	return ticket{generation: cb.generation}, nil
}

type outcome int

const (
	outcomeSuccess outcome = iota
	outcomeFailure
	outcomeNeutral
)

// finish records an admitted call's outcome.
func (cb *Breaker) finish(t ticket, o outcome) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	switch o {
	case outcomeSuccess:
		cb.totalSuccesses++
	case outcomeFailure:
		cb.totalFailures++
	}
	if t.generation != cb.generation {
		return // the state changed while the call ran; its slot no longer exists
	}
	if t.probe && cb.halfOpenCalls > 0 {
		cb.halfOpenCalls--
	}
	switch o {
	case outcomeSuccess:
		cb.consecutiveFailures = 0
		cb.consecutiveSuccesses++
		if cb.state == StateHalfOpen && cb.consecutiveSuccesses >= cb.cfg.SuccessThreshold {
			cb.setStateLocked(StateClosed)
		}
	case outcomeFailure:
		cb.consecutiveSuccesses = 0
		cb.consecutiveFailures++
		if cb.state == StateHalfOpen || cb.consecutiveFailures >= cb.cfg.FailureThreshold {
			cb.setStateLocked(StateOpen)
		}
	}
}

// Allow admits one call for callers that report the outcome themselves
// with RecordSuccess/RecordFailure (reference API). In half-open state it
// takes a probe slot, which the following Record call releases.
func (cb *Breaker) Allow() bool {
	_, err := cb.admit()
	return err == nil
}

// RecordSuccess reports a success for a call admitted with Allow.
func (cb *Breaker) RecordSuccess() { cb.finishCurrent(outcomeSuccess) }

// RecordFailure reports a failure for a call admitted with Allow.
func (cb *Breaker) RecordFailure(err error) {
	if err != nil {
		cb.finishCurrent(outcomeFailure)
	}
}

func (cb *Breaker) finishCurrent(o outcome) {
	cb.mu.Lock()
	t := ticket{generation: cb.generation, probe: cb.state == StateHalfOpen}
	cb.mu.Unlock()
	cb.finish(t, o)
}

// Execute runs op if the breaker admits it and records the outcome.
func (cb *Breaker) Execute(ctx context.Context, op func() error) (err error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	t, err := cb.admit()
	if err != nil {
		return err
	}
	defer func() {
		if r := recover(); r != nil {
			cb.finish(t, outcomeFailure)
			panic(r)
		}
	}()
	err = op()
	cb.finish(t, classify(ctx, err))
	return err
}

func classify(ctx context.Context, err error) outcome {
	switch {
	case err == nil:
		return outcomeSuccess
	case errors.As(err, new(neutralError)):
		return outcomeNeutral
	case ctx != nil && ctx.Err() != nil:
		// The caller gave up; this says nothing about Redis.
		return outcomeNeutral
	}
	return outcomeFailure
}

// String describes the breaker for logs and diagnostics.
func (cb *Breaker) String() string {
	s, f, ts, tf := cb.Counts()
	return fmt.Sprintf("%s state=%s consecutive(success=%d failure=%d) total(success=%d failure=%d)", cb.name, cb.State(), s, f, ts, tf)
}
