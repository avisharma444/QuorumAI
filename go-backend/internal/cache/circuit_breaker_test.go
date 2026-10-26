package cache

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

func newTestBreaker() (*Breaker, *clock) {
	clk := &clock{t: time.Unix(1_700_000_000, 0)}
	cfg := DefaultConfig()
	cfg.NowFunc = clk.now
	return NewCircuitBreaker("test", cfg), clk
}

var errRedis = errors.New("dial tcp: connection refused")

func fail(cb *Breaker) error {
	return cb.Execute(context.Background(), func() error { return errRedis })
}
func ok(cb *Breaker) error { return cb.Execute(context.Background(), func() error { return nil }) }

func TestDefaultsMatchReference(t *testing.T) {
	d := DefaultConfig()
	if d.FailureThreshold != 3 || d.SuccessThreshold != 2 || d.Timeout != 10*time.Second || d.HalfOpenMaxCalls != 1 {
		t.Fatalf("%+v", d)
	}
	cb := NewCircuitBreaker("x", Config{})
	if cb.cfg.FailureThreshold != 3 || cb.State() != StateClosed {
		t.Fatal("zero config must take defaults")
	}
}

func TestClosedToOpenAfterConsecutiveFailures(t *testing.T) {
	cb, _ := newTestBreaker()
	fail(cb)
	fail(cb)
	ok(cb) // a success resets the consecutive count
	fail(cb)
	fail(cb)
	if cb.State() != StateClosed {
		t.Fatal("2 consecutive failures must not open")
	}
	fail(cb)
	if cb.State() != StateOpen {
		t.Fatalf("3 consecutive failures must open, got %s", cb.State())
	}
	called := false
	if err := cb.Execute(context.Background(), func() error { called = true; return nil }); !errors.Is(err, ErrCircuitOpen) || called {
		t.Fatal("open breaker must fail fast without running the operation")
	}
	_, _, ts, tf := cb.Counts()
	if ts != 1 || tf != 5 {
		t.Fatalf("totals %d/%d", ts, tf)
	}
}

func TestOpenToHalfOpenToClosed(t *testing.T) {
	cb, clk := newTestBreaker()
	cb.Trip()
	clk.advance(9 * time.Second)
	if !errors.Is(ok(cb), ErrCircuitOpen) {
		t.Fatal("still cooling down")
	}
	clk.advance(time.Second)
	if cb.State() != StateHalfOpen {
		t.Fatalf("after cooldown State reports %s", cb.State())
	}
	if err := ok(cb); err != nil {
		t.Fatal(err)
	}
	if cb.State() != StateHalfOpen {
		t.Fatal("one probe success must not close (threshold 2)")
	}
	if err := ok(cb); err != nil || cb.State() != StateClosed {
		t.Fatalf("second probe success must close: %v %s", err, cb.State())
	}
}

func TestHalfOpenFailureReopens(t *testing.T) {
	cb, clk := newTestBreaker()
	cb.Trip()
	clk.advance(10 * time.Second)
	fail(cb)
	if cb.State() != StateOpen {
		t.Fatal("probe failure must reopen")
	}
	clk.advance(5 * time.Second)
	if !errors.Is(ok(cb), ErrCircuitOpen) {
		t.Fatal("cooldown restarts after reopening")
	}
}

func TestHalfOpenAdmitsOnlyOneConcurrentProbe(t *testing.T) {
	cb, clk := newTestBreaker()
	cb.Trip()
	clk.advance(10 * time.Second)
	release := make(chan struct{})
	started := make(chan struct{})
	var ran atomic.Int32
	done := make(chan error, 1)
	go func() {
		done <- cb.Execute(context.Background(), func() error {
			ran.Add(1)
			close(started)
			<-release
			return nil
		})
	}()
	<-started
	var rejected atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if errors.Is(cb.Execute(context.Background(), func() error { ran.Add(1); return nil }), ErrCircuitOpen) {
				rejected.Add(1)
			}
		}()
	}
	wg.Wait()
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if ran.Load() != 1 || rejected.Load() != 20 {
		t.Fatalf("ran=%d rejected=%d", ran.Load(), rejected.Load())
	}
	if err := ok(cb); err != nil || cb.State() != StateClosed {
		t.Fatalf("slot must be released for the next probe: %v %s", err, cb.State())
	}
}

func TestCancellationDoesNotCountOrStrandProbe(t *testing.T) {
	cb, clk := newTestBreaker()
	for i := 0; i < 5; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		_ = cb.Execute(ctx, func() error { cancel(); return ctx.Err() })
	}
	if cb.State() != StateClosed {
		t.Fatal("client cancellations must not open the breaker")
	}
	cb.Trip()
	clk.advance(10 * time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	_ = cb.Execute(ctx, func() error { cancel(); return ctx.Err() })
	if cb.State() != StateHalfOpen {
		t.Fatalf("cancelled probe must leave the breaker half-open, got %s", cb.State())
	}
	if err := ok(cb); err != nil {
		t.Fatalf("probe slot stranded by cancellation: %v", err)
	}
	cancelled, cancel2 := context.WithCancel(context.Background())
	cancel2()
	if err := cb.Execute(cancelled, func() error { t.Fatal("must not run"); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestNeutralErrorsDoNotCount(t *testing.T) {
	cb, _ := newTestBreaker()
	for i := 0; i < 10; i++ {
		_ = cb.Execute(context.Background(), func() error { return Neutral(errors.New("no such index")) })
	}
	if cb.State() != StateClosed {
		t.Fatal("neutral outcomes must not open the breaker")
	}
	if _, _, ts, tf := cb.Counts(); ts != 0 || tf != 0 {
		t.Fatalf("neutral outcomes counted: %d/%d", ts, tf)
	}
}

func TestPanicCountsAsFailureAndReleasesSlot(t *testing.T) {
	cb, clk := newTestBreaker()
	cb.Trip()
	clk.advance(10 * time.Second)
	func() {
		defer func() { _ = recover() }()
		_ = cb.Execute(context.Background(), func() error { panic("boom") })
	}()
	if cb.State() != StateOpen {
		t.Fatal("panicking probe must reopen")
	}
	clk.advance(10 * time.Second)
	if err := ok(cb); err != nil {
		t.Fatalf("slot not released after panic: %v", err)
	}
}

// A call that started before a state change must not affect the new state.
func TestStaleResultsAreIgnored(t *testing.T) {
	cb, clk := newTestBreaker()
	release := make(chan struct{})
	started := make(chan struct{})
	done := make(chan struct{})
	go func() {
		_ = cb.Execute(context.Background(), func() error { close(started); <-release; return errRedis })
		close(done)
	}()
	<-started
	cb.Trip()
	clk.advance(10 * time.Second)
	if err := ok(cb); err != nil { // first half-open probe succeeds
		t.Fatal(err)
	}
	close(release) // the old CLOSED-era call now fails
	<-done
	if cb.State() != StateHalfOpen {
		t.Fatalf("stale failure reopened the breaker: %s", cb.State())
	}
	if err := ok(cb); err != nil || cb.State() != StateClosed {
		t.Fatalf("recovery blocked by stale result: %v %s", err, cb.State())
	}
}

func TestResetAndAllowAPI(t *testing.T) {
	cb, clk := newTestBreaker()
	cb.Trip()
	if cb.Allow() {
		t.Fatal("Allow while open")
	}
	cb.Reset()
	if !cb.Allow() || cb.State() != StateClosed {
		t.Fatal("Reset must close")
	}
	cb.Trip()
	clk.advance(10 * time.Second)
	if !cb.Allow() || cb.Allow() {
		t.Fatal("half-open Allow admits exactly one probe")
	}
	cb.RecordSuccess()
	if !cb.Allow() {
		t.Fatal("RecordSuccess must release the probe slot")
	}
	cb.RecordFailure(errRedis)
	if cb.State() != StateOpen {
		t.Fatal("RecordFailure in half-open must reopen")
	}
}

func TestConcurrentMixedTrafficIsRaceFree(t *testing.T) {
	cb, clk := newTestBreaker()
	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			switch i % 4 {
			case 0:
				_ = fail(cb)
			case 1:
				_ = ok(cb)
			case 2:
				clk.advance(time.Second)
				_ = cb.State()
			default:
				cb.Counts()
			}
		}(i)
	}
	wg.Wait()
	_, _, ts, tf := cb.Counts()
	if ts+tf > 100 {
		t.Fatalf("more outcomes than executed operations: %d", ts+tf)
	}
}
