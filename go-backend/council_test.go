package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── Test doubles ────────────────────────────────────────────────────

// stageOf maps a system prompt to the pipeline stage that sent it.
func stageOf(system string) string {
	switch {
	case strings.HasPrefix(system, "REFLECTION_AGENT"):
		return "reflect"
	case strings.HasPrefix(system, "CHAIRMAN_AGENT: revise"):
		return "revise"
	case strings.HasPrefix(system, "CHAIRMAN_AGENT"):
		return "chair"
	case strings.HasPrefix(system, "REVIEW_AGENT"):
		return "review"
	case strings.HasPrefix(system, "COUNCIL_MEMBER"):
		return "member"
	case strings.HasPrefix(system, "DIRECT_AGENT"):
		return "direct"
	}
	return "unknown"
}

type reply struct {
	text string
	err  error
}

// fakeModel returns scripted replies keyed by "stage/model" and records every
// call. Unscripted calls fall back to MockModel. hook, if set, runs before
// the reply and can block or fail the call.
type fakeModel struct {
	mu      sync.Mutex
	replies map[string]reply
	calls   map[string]int
	prompts map[string]string
	hook    func(ctx context.Context, stage, model string) error
}

func newFake(replies map[string]reply) *fakeModel {
	if replies == nil {
		replies = map[string]reply{}
	}
	return &fakeModel{replies: replies, calls: map[string]int{}, prompts: map[string]string{}}
}

func (f *fakeModel) Generate(ctx context.Context, model, system, prompt string) (string, error) {
	stage := stageOf(system)
	key := stage + "/" + model
	f.mu.Lock()
	f.calls[stage]++
	f.prompts[key] = prompt
	r, scripted := f.replies[key]
	f.mu.Unlock()
	if f.hook != nil {
		if err := f.hook(ctx, stage, model); err != nil {
			return "", err
		}
	}
	if !scripted {
		return (&MockModel{}).Generate(ctx, model, system, prompt)
	}
	return r.text, r.err
}

func (f *fakeModel) count(stage string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[stage]
}

func (f *fakeModel) prompt(stage, model string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.prompts[stage+"/"+model]
}

// eventLog collects emitted events safely from many goroutines.
type eventLog struct {
	mu     sync.Mutex
	names  []string
	values []any
}

func (e *eventLog) emit(name string, v any) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.names = append(e.names, name)
	e.values = append(e.values, v)
}

func (e *eventLog) has(name string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, n := range e.names {
		if n == name {
			return true
		}
	}
	return false
}

func testCouncil(m Model) *Council {
	return &Council{Provider: m, Members: []string{"a", "b", "c"}, Chair: "chair", Timeout: 2 * time.Second}
}

func runCouncil(t *testing.T, m Model, strategy string) (QueryResult, *eventLog, error) {
	t.Helper()
	log := &eventLog{}
	res, err := testCouncil(m).Run(context.Background(), QueryRequest{Question: "Compare X and Y", Strategy: strategy}, nil, log.emit)
	return res, log, err
}

func mustRun(t *testing.T, m Model, strategy string) (QueryResult, *eventLog) {
	t.Helper()
	res, log, err := runCouncil(t, m, strategy)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	return res, log
}

func ranking(r string) reply { return reply{text: "RANKING: " + r + "\nREASONING: because."} }

var errProvider = errors.New("provider down")

// ── Ranking parser ──────────────────────────────────────────────────

func TestParseRankingAccepts(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		n    int
		want []int
	}{
		{"plain", "RANKING: B, A, C\nREASONING: B is best.", 3, []int{1, 0, 2}},
		{"brackets", "RANKING: [B, A, C]", 3, []int{1, 0, 2}},
		{"extra whitespace", "   RANKING :   B ,A,    C   ", 3, []int{1, 0, 2}},
		{"lowercase", "ranking: c, b, a", 3, []int{2, 1, 0}},
		{"quoted like the prompt example", `RANKING: "B, A, C"`, 3, []int{1, 0, 2}},
		{"trailing period", "RANKING: B, A, C.", 3, []int{1, 0, 2}},
		{"preamble before ranking", "Here is my review.\n\nRANKING: A, C, B\nREASONING: x", 3, []int{0, 2, 1}},
		{"two candidates", "RANKING: B, A", 2, []int{1, 0}},
		{"single candidate", "RANKING: A", 1, []int{0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseRanking(tc.raw, tc.n)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestParseRankingRejects(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		n    int
	}{
		{"empty response", "", 3},
		{"missing ranking section", "REASONING: A is best.", 3},
		{"old best_index JSON vote", `{"best_index": 0, "reason": "good"}`, 3},
		{"empty JSON object", "{}", 3},
		{"irrelevant text", "Sure! I think B is the best answer overall.", 3},
		{"ranking keyword mid-sentence", "My ranking: B, A, C", 3},
		{"blank ranking", "RANKING:\nREASONING: none", 3},
		{"empty brackets", "RANKING: []", 3},
		{"duplicate labels", "RANKING: B, B, A", 3},
		{"all the same label", "RANKING: A, A, A", 3},
		{"letter beyond candidates", "RANKING: A, B, D", 3},
		{"too many labels", "RANKING: A, B, C, D", 3},
		{"incomplete ranking", "RANKING: B, A", 3},
		{"only a winner", "RANKING: B", 3},
		{"numeric indices", "RANKING: 1, 0, 2", 3},
		{"space separated", "RANKING: B A C", 3},
		{"arrow separated", "RANKING: B > A > C", 3},
		{"verbose labels", "RANKING: Response B, Response A, Response C", 3},
		{"empty slot", "RANKING: B, , A", 3},
		{"single candidate out of range", "RANKING: B", 1},
		{"no candidates", "RANKING: A", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got, err := parseRanking(tc.raw, tc.n); err == nil {
				t.Fatalf("expected rejection, got %v", got)
			}
		})
	}
}

func TestExtractReasoning(t *testing.T) {
	if got := extractReasoning("RANKING: A, B\nreasoning:  A cites the evidence.\nB does not."); got != "A cites the evidence.\nB does not." {
		t.Fatalf("got %q", got)
	}
	if got := extractReasoning("RANKING: A, B"); got != "" {
		t.Fatalf("expected empty reasoning, got %q", got)
	}
}

// ── Scoring and selection ───────────────────────────────────────────

func validReview(r ...string) Review { return Review{Ranking: r} }

func TestScoreRanking(t *testing.T) {
	got := scoreRanking([]int{1, 0, 2})
	want := map[string]int{"B": 3, "A": 2, "C": 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// The worked example from the checkpoint spec: B wins with 8 points.
func TestAggregateWorkedExample(t *testing.T) {
	reviews := []Review{validReview("B", "A", "C"), validReview("B", "C", "A"), validReview("A", "B", "C")}
	totals, valid := aggregateScores(reviews, 3)
	if !reflect.DeepEqual(totals, []int{6, 8, 4}) || valid != 3 {
		t.Fatalf("totals %v valid %d", totals, valid)
	}
	if best := pickBestIndex(totals); best != 1 {
		t.Fatalf("expected B (1) to win, got %d", best)
	}
}

func TestAggregateIgnoresInvalidReviews(t *testing.T) {
	reviews := []Review{
		validReview("C", "B", "A"),
		{Error: "review request failed: timeout"},
		{Error: "invalid ranking: duplicate label A", Raw: "RANKING: A, A, A"},
		{Ranking: []string{"A", "B"}}, // wrong length can never count
		// A review marked as failed never counts, even if it carries a ranking.
		{Error: "review request failed: late", Ranking: []string{"A", "B", "C"}},
	}
	totals, valid := aggregateScores(reviews, 3)
	if !reflect.DeepEqual(totals, []int{1, 2, 3}) || valid != 1 {
		t.Fatalf("totals %v valid %d", totals, valid)
	}
}

func TestTieBreakIsDeterministic(t *testing.T) {
	for i := 0; i < 1000; i++ {
		if got := pickBestIndex([]int{5, 5, 2}); got != 0 {
			t.Fatalf("tie must go to earliest candidate, got %d", got)
		}
		if got := pickBestIndex([]int{2, 5, 5}); got != 1 {
			t.Fatalf("tie must go to earliest candidate, got %d", got)
		}
	}
	if got := pickLongest([]Candidate{{Answer: "xx"}, {Answer: "yyy"}, {Answer: "zzz"}}); got != 1 {
		t.Fatalf("longest tie must go to earliest candidate, got %d", got)
	}
}

// ── Full pipeline ───────────────────────────────────────────────────

func TestNonFirstCandidateWinsByWeightedRanking(t *testing.T) {
	f := newFake(map[string]reply{
		"review/a":    ranking("B, A, C"),
		"review/b":    ranking("B, C, A"),
		"review/c":    ranking("A, B, C"),
		"chair/chair": {text: "chairman synthesis"},
	})
	res, _ := mustRun(t, f, "council")
	if res.Selected != "B" || res.Answer != "chairman synthesis" || res.Source != "chairman:chair" || !res.PeerReviewed {
		t.Fatalf("unexpected result: %+v", res)
	}
	scores := []int{res.Candidates[0].Score, res.Candidates[1].Score, res.Candidates[2].Score}
	if !reflect.DeepEqual(scores, []int{6, 8, 4}) {
		t.Fatalf("scores %v", scores)
	}
	r := res.Reviews[1]
	if r.Reviewer != "b" || r.Index != 1 || !reflect.DeepEqual(r.Ranking, []string{"B", "C", "A"}) ||
		r.Reasoning != "because." || r.Raw == "" || r.Scores["B"] != 3 || r.Scores["A"] != 1 {
		t.Fatalf("review not fully recorded: %+v", r)
	}
	chair := f.prompt("chair", "chair")
	for _, want := range []string{"=== Response A ===", "=== Response C ===", "--- Reviewer 1 ---", "--- Reviewer 3 ---", "RANKING: B, C, A"} {
		if !strings.Contains(chair, want) {
			t.Fatalf("chairman prompt missing %q:\n%s", want, chair)
		}
	}
}

func TestEqualScoresResolveToEarliestCandidate(t *testing.T) {
	f := newFake(map[string]reply{
		"review/a":    ranking("B, A, C"),
		"review/b":    ranking("A, B, C"),
		"review/c":    {err: errProvider},
		"chair/chair": {err: errProvider},
		"member/a":    {text: "answer one"},
		"member/b":    {text: "answer two"},
	})
	res, _ := mustRun(t, f, "council")
	if res.Selected != "A" || res.Answer != "answer one" {
		t.Fatalf("tie should select A: %+v", res)
	}
}

func TestReviewersSeeAnonymizedAnswers(t *testing.T) {
	c := &Council{Provider: nil, Members: []string{"secret-gpt", "secret-claude"}, Chair: "chair", Timeout: time.Second}
	f := newFake(map[string]reply{
		"member/secret-gpt":    {text: "Answer about X."},
		"member/secret-claude": {text: "Answer about Y."},
	})
	c.Provider = f
	if _, err := c.Run(context.Background(), QueryRequest{Question: "Q?", Strategy: "council"}, nil, nil); err != nil {
		t.Fatal(err)
	}
	p := f.prompt("review", "secret-gpt")
	if strings.Contains(p, "secret-") {
		t.Fatalf("review prompt leaks model identity:\n%s", p)
	}
	for _, want := range []string{"=== Response A ===\nAnswer about X.", "=== Response B ===\nAnswer about Y.", "RANKING:", "REASONING:"} {
		if !strings.Contains(p, want) {
			t.Fatalf("review prompt missing %q:\n%s", want, p)
		}
	}
}

// When member b fails, labels compress to A=a, B=c, and rankings must map back
// to the right models.
func TestRankingMapsCorrectlyWhenACandidateFails(t *testing.T) {
	f := newFake(map[string]reply{
		"member/a":    {text: "answer from a"},
		"member/b":    {err: errProvider},
		"member/c":    {text: "answer from c"},
		"review/a":    ranking("B, A"),
		"review/b":    ranking("B, A"),
		"review/c":    ranking("A, B"),
		"chair/chair": {err: errProvider},
	})
	res, log := mustRun(t, f, "council")
	if len(res.Candidates) != 2 || res.Candidates[0].Model != "a" || res.Candidates[0].Label != "A" ||
		res.Candidates[1].Model != "c" || res.Candidates[1].Label != "B" {
		t.Fatalf("candidates %+v", res.Candidates)
	}
	if strings.Contains(f.prompt("review", "a"), "Response C") {
		t.Fatal("failed candidate must not appear in the review prompt")
	}
	// Reviewer b's own answer failed, but it still reviews (reference behaviour).
	if len(res.Reviews) != 3 || f.count("review") != 3 {
		t.Fatalf("expected 3 reviewer slots, got %d", len(res.Reviews))
	}
	if res.Selected != "B" || res.Answer != "answer from c" || res.Candidates[1].Score != 5 {
		t.Fatalf("wrong mapping: %+v", res)
	}
	if !log.has("candidate") || !log.has("fallback") {
		t.Fatalf("events: %v", log.names)
	}
}

// ── Concurrency and cancellation ────────────────────────────────────

// barrier releases callers only once n of them are waiting at the same time.
// If calls ran one after another, the first would time out instead.
type barrier struct {
	mu      sync.Mutex
	n       int
	arrived int
	open    chan struct{}
}

func newBarrier(n int) *barrier { return &barrier{n: n, open: make(chan struct{})} }

func (b *barrier) wait() error {
	b.mu.Lock()
	b.arrived++
	if b.arrived == b.n {
		close(b.open)
	}
	b.mu.Unlock()
	select {
	case <-b.open:
		return nil
	case <-time.After(time.Second):
		return errors.New("barrier timeout: calls were not concurrent")
	}
}

func TestCandidatesAndReviewsRunConcurrently(t *testing.T) {
	barriers := map[string]*barrier{"member": newBarrier(3), "review": newBarrier(3)}
	f := newFake(nil)
	f.hook = func(ctx context.Context, stage, model string) error {
		if b, ok := barriers[stage]; ok {
			return b.wait()
		}
		return nil
	}
	res, _ := mustRun(t, f, "council")
	for _, r := range res.Reviews {
		if r.Error != "" {
			t.Fatalf("review failed (not concurrent?): %+v", r)
		}
	}
	if len(res.Candidates) != 3 || res.Source != "chairman:chair" {
		t.Fatalf("unexpected result: %+v", res)
	}
}

func TestCancellationStopsTheRun(t *testing.T) {
	started := make(chan struct{}, 3)
	f := newFake(nil)
	f.hook = func(ctx context.Context, stage, model string) error {
		started <- struct{}{}
		<-ctx.Done()
		return ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for i := 0; i < 3; i++ {
			<-started
		}
		cancel()
	}()
	log := &eventLog{}
	c := testCouncil(f)
	c.Timeout = time.Minute
	done := make(chan error, 1)
	go func() {
		_, err := c.Run(ctx, QueryRequest{Question: "Q", Strategy: "council"}, nil, log.emit)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not stop after cancellation")
	}
	if f.count("review") != 0 || log.has("final") {
		t.Fatalf("cancelled run continued: events %v", log.names)
	}
}

// Cancelling during peer review must return the context error, not a
// degraded "longest candidate" answer that would then be cached.
func TestCancellationDuringReviewIsNotAFallback(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	f := newFake(nil)
	f.hook = func(c context.Context, stage, model string) error {
		if stage == "review" {
			cancel()
			<-c.Done()
			return c.Err()
		}
		return nil
	}
	_, err := testCouncil(f).Run(ctx, QueryRequest{Question: "Q", Strategy: "council"}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestStageTimeoutFailsOnlyTheSlowModel(t *testing.T) {
	f := newFake(nil)
	f.hook = func(ctx context.Context, stage, model string) error {
		if stage == "member" && model == "c" {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}
	c := testCouncil(f)
	c.Timeout = 50 * time.Millisecond
	res, err := c.Run(context.Background(), QueryRequest{Question: "Q", Strategy: "council"}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Candidates) != 2 || !res.PeerReviewed {
		t.Fatalf("slow model should be dropped, rest continue: %+v", res)
	}
}

// ── Failure handling ────────────────────────────────────────────────

func TestAllMembersFail(t *testing.T) {
	f := newFake(map[string]reply{"member/a": {err: errProvider}, "member/b": {err: errProvider}, "member/c": {text: "   "}})
	_, log, err := runCouncil(t, f, "council")
	if err == nil || !strings.Contains(err.Error(), "all council members failed") {
		t.Fatalf("expected failure, got %v", err)
	}
	if f.count("review") != 0 || log.has("final") {
		t.Fatal("no review or final answer after total failure")
	}
}

func TestSingleSurvivorSkipsReviewAndChairman(t *testing.T) {
	f := newFake(map[string]reply{"member/a": {err: errProvider}, "member/b": {text: "only answer"}, "member/c": {err: errProvider}})
	res, log := mustRun(t, f, "council_deep")
	if res.Answer != "only answer" || res.Source != "b (single-response)" || res.PeerReviewed {
		t.Fatalf("unexpected: %+v", res)
	}
	if f.count("review")+f.count("chair")+f.count("reflect") != 0 {
		t.Fatal("review, chairman and reflection must be skipped")
	}
	if !log.has("fallback") || !log.has("final") {
		t.Fatalf("events %v", log.names)
	}
}

func TestMixedReviewerOutcomes(t *testing.T) {
	f := newFake(map[string]reply{
		"review/a": {err: errProvider},
		"review/b": {text: "RANKING: A, A, B\nREASONING: A twice."},
		"review/c": ranking("C, B, A"),
	})
	res, log := mustRun(t, f, "council")
	if res.Selected != "C" || !res.PeerReviewed || res.Source != "chairman:chair" {
		t.Fatalf("unexpected: %+v", res)
	}
	a, b, c := res.Reviews[0], res.Reviews[1], res.Reviews[2]
	if a.Reviewer != "a" || !strings.Contains(a.Error, "review request failed") {
		t.Fatalf("failed reviewer slot lost: %+v", a)
	}
	if b.Reviewer != "b" || !strings.Contains(b.Error, "duplicate") || b.Raw == "" || b.Reasoning != "A twice." || b.Ranking != nil || b.Scores != nil {
		t.Fatalf("invalid review not recorded correctly: %+v", b)
	}
	if c.Error != "" || c.Scores["C"] != 3 {
		t.Fatalf("valid review: %+v", c)
	}
	chair := f.prompt("chair", "chair")
	if strings.Contains(chair, "A twice") || strings.Contains(chair, "Reviewer 1") || !strings.Contains(chair, "--- Reviewer 3 ---") {
		t.Fatalf("chairman must only see valid reviews:\n%s", chair)
	}
	if !log.has("review") || !log.has("review_error") {
		t.Fatalf("events %v", log.names)
	}
}

// The starter's old bug: "{}" counted as a vote for candidate 0. Now every
// review is invalid, so the longest candidate (b, not a) wins and the
// chairman is skipped.
func TestInvalidReviewsNeverVoteForA(t *testing.T) {
	f := newFake(map[string]reply{
		"member/a": {text: "short"},
		"member/b": {text: "the longest answer of all"},
		"member/c": {text: "medium one"},
		"review/a": {text: "{}"},
		"review/b": {text: `{"best_index": 0}`},
		"review/c": {text: "RANKING: A"},
	})
	res, log := mustRun(t, f, "council_deep")
	if res.Answer != "the longest answer of all" || res.Source != "b (peer-review-failed-fallback)" || res.PeerReviewed || res.Selected != "" {
		t.Fatalf("unexpected: %+v", res)
	}
	for _, cand := range res.Candidates {
		if cand.Score != 0 {
			t.Fatalf("no score may be invented: %+v", cand)
		}
	}
	if f.count("chair") != 0 || f.count("reflect") != 0 {
		t.Fatal("chairman and reflection must be skipped")
	}
	if !log.has("fallback") {
		t.Fatalf("events %v", log.names)
	}
}

func TestAllReviewRequestsFail(t *testing.T) {
	f := newFake(map[string]reply{
		"member/a": {text: "short"},
		"member/b": {text: "also short"},
		"member/c": {text: "the longest candidate answer here"},
		"review/a": {err: errProvider}, "review/b": {err: errProvider}, "review/c": {err: errProvider},
	})
	res, _ := mustRun(t, f, "council")
	if res.Source != "c (peer-review-failed-fallback)" || f.count("chair") != 0 || len(res.Reviews) != 3 {
		t.Fatalf("unexpected: %+v", res)
	}
}

func TestChairmanFailureReturnsTopRankedNotFirst(t *testing.T) {
	for name, chair := range map[string]reply{"error": {err: errProvider}, "empty": {text: "  "}} {
		t.Run(name, func(t *testing.T) {
			f := newFake(map[string]reply{
				"member/c":    {text: "c's answer"},
				"review/a":    ranking("C, A, B"),
				"review/b":    ranking("C, B, A"),
				"review/c":    ranking("A, C, B"),
				"chair/chair": chair,
			})
			res, log := mustRun(t, f, "council_deep")
			if res.Answer != "c's answer" || res.Source != "c (chairman-failed-fallback)" || res.Selected != "C" || !res.PeerReviewed {
				t.Fatalf("unexpected: %+v", res)
			}
			if f.count("reflect") != 0 || res.Reflection != "" {
				t.Fatal("reflection must not run on a fallback answer")
			}
			if !log.has("fallback") {
				t.Fatalf("events %v", log.names)
			}
		})
	}
}

func TestDeepReflectionRevisesAfterChairman(t *testing.T) {
	f := newFake(map[string]reply{
		"chair/chair":   {text: "draft"},
		"reflect/chair": {text: `{"quality":"needs_revision","reason":"missing Y"}`},
		"revise/chair":  {text: "revised answer"},
	})
	res, log := mustRun(t, f, "council_deep")
	if res.Answer != "revised answer" || res.Reflection != "needs_revision: missing Y" || res.Source != "chairman-revised:chair" {
		t.Fatalf("unexpected: %+v", res)
	}
	if !log.has("reflection") {
		t.Fatalf("events %v", log.names)
	}
	// Plain council mode never reflects.
	f2 := newFake(nil)
	mustRun(t, f2, "council")
	if f2.count("reflect") != 0 {
		t.Fatal("council mode must not reflect")
	}
}

// ── HTTP integration ────────────────────────────────────────────────

func postQuery(t *testing.T, srv *Server, body string, stream bool) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(body))
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)
	return rec
}

type sseEvent struct {
	name string
	data string
}

func parseSSE(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	var name string
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			name = line[7:]
		case strings.HasPrefix(line, "data: "):
			if !json.Valid([]byte(line[6:])) {
				t.Fatalf("invalid JSON in %s event: %s", name, line)
			}
			events = append(events, sseEvent{name, line[6:]})
		}
	}
	return events
}

func TestHTTPJSONResponse(t *testing.T) {
	srv := NewServer(testCouncil(&MockModel{}), "http://unused.invalid")
	rec := postQuery(t, srv, `{"question":"Compare A and B","strategy":"council"}`, false)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var out QueryResult
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Reviews) != 3 || !reflect.DeepEqual(out.Reviews[0].Ranking, []string{"A", "B", "C"}) || !out.PeerReviewed || out.Selected != "A" {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "best_index") {
		t.Fatal("old best_index format still present")
	}
}

func TestSSEEventOrder(t *testing.T) {
	srv := NewServer(testCouncil(&MockModel{}), "http://unused.invalid")
	events := parseSSE(t, postQuery(t, srv, `{"question":"compare A vs B","strategy":"council"}`, true).Body.String())
	var names []string
	for _, e := range events {
		names = append(names, e.name)
	}
	want := []string{"start", "retrieved", "candidate", "candidate", "candidate", "review", "review", "review", "final"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("got events %v, want %v", names, want)
	}
	var final QueryResult
	_ = json.Unmarshal([]byte(events[len(events)-1].data), &final)
	if !final.PeerReviewed || final.Source != "chairman:chair" {
		t.Fatalf("final payload: %s", events[len(events)-1].data)
	}
}

func TestSSEReviewErrorAndFallbackEvents(t *testing.T) {
	f := newFake(map[string]reply{"review/a": {err: errProvider}, "chair/chair": {err: errProvider}})
	srv := NewServer(testCouncil(f), "http://unused.invalid")
	events := parseSSE(t, postQuery(t, srv, `{"question":"compare","strategy":"council"}`, true).Body.String())
	seen := map[string]int{}
	for _, e := range events {
		seen[e.name]++
	}
	if seen["review_error"] != 1 || seen["review"] != 2 || seen["fallback"] != 1 || seen["final"] != 1 {
		t.Fatalf("events %v", seen)
	}
	if events[len(events)-1].name != "final" {
		t.Fatal("final must be last")
	}
}

func TestHTTPCaching(t *testing.T) {
	f := newFake(nil)
	srv := NewServer(testCouncil(f), "http://unused.invalid")
	body := `{"question":"Compare concurrency and parallelism","strategy":"council"}`
	var first, second QueryResult
	_ = json.Unmarshal(postQuery(t, srv, body, false).Body.Bytes(), &first)
	_ = json.Unmarshal(postQuery(t, srv, body, false).Body.Bytes(), &second)
	if first.CacheHit || !second.CacheHit || srv.Hits.Load() != 1 {
		t.Fatal("second identical request must be an exact cache hit")
	}
	if f.count("member") != 3 || f.count("review") != 3 {
		t.Fatalf("cache hit must not call models again: %v", f.calls)
	}
	if !reflect.DeepEqual(first.Reviews, second.Reviews) {
		t.Fatal("cached reviews differ")
	}
}

func TestDirectModeBypassesPeerReview(t *testing.T) {
	f := newFake(map[string]reply{"direct/chair": {text: "direct answer"}})
	srv := NewServer(testCouncil(f), "http://unused.invalid")
	var out QueryResult
	_ = json.Unmarshal(postQuery(t, srv, `{"question":"hi","strategy":"direct"}`, false).Body.Bytes(), &out)
	if out.Answer != "direct answer" || out.Source != "direct:chair" || len(out.Reviews) != 0 {
		t.Fatalf("unexpected: %+v", out)
	}
	if f.count("member")+f.count("review")+f.count("chair") != 0 {
		t.Fatalf("direct mode called council stages: %v", f.calls)
	}
}

func TestDocumentRetrievalFeedsTheCouncil(t *testing.T) {
	var gotDoc string
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotDoc, _ = req["doc_id"].(string)
		fmt.Fprint(w, `{"chunks":[{"text":"The sky is green on Zog.","page":2,"score":0.9}]}`)
	}))
	defer rag.Close()
	f := newFake(nil)
	srv := NewServer(testCouncil(f), rag.URL)
	var out QueryResult
	rec := postQuery(t, srv, `{"question":"What colour is the sky?","doc_id":"doc-1","strategy":"council"}`, false)
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if gotDoc != "doc-1" || len(out.Sources) != 1 || !strings.Contains(out.Sources[0], "page 2") {
		t.Fatalf("retrieval not wired: doc=%q body=%s", gotDoc, rec.Body.String())
	}
	if !strings.Contains(f.prompt("member", "a"), "The sky is green on Zog.") || !strings.Contains(f.prompt("chair", "chair"), "The sky is green on Zog.") {
		t.Fatal("evidence missing from member or chairman prompt")
	}
}

func TestStrategyValidation(t *testing.T) {
	if _, err := strategyFor(QueryRequest{Question: "Q", Strategy: "bogus"}); err == nil {
		t.Fatal("unknown strategy accepted")
	}
	if value, _ := strategyFor(QueryRequest{Question: "hello"}); value != "direct" {
		t.Fatalf("got %q", value)
	}
	if value, _ := strategyFor(QueryRequest{Question: "hello", DocID: "doc1"}); value != "council" {
		t.Fatalf("got %q", value)
	}
}
