package council

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"example.com/councilai-starter/internal/llm/llmtest"
)

// ── Test doubles ────────────────────────────────────────────────────

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

func testCouncil(f *llmtest.Fake) *Council {
	return New(f.Clients("a", "b", "c"), f.Client("chair"), 2*time.Second)
}

func runCouncil(t *testing.T, m *llmtest.Fake, strategy string) (QueryResult, *eventLog, error) {
	t.Helper()
	log := &eventLog{}
	res, err := testCouncil(m).Run(context.Background(), QueryRequest{Question: "Compare X and Y", Strategy: strategy}, Evidence{}, log.emit)
	return res, log, err
}

func mustRun(t *testing.T, m *llmtest.Fake, strategy string) (QueryResult, *eventLog) {
	t.Helper()
	res, log, err := runCouncil(t, m, strategy)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	return res, log
}

// chairJSON is a valid structured chairman reply.
func chairJSON(answer string) llmtest.Reply {
	return llmtest.Reply{Text: fmt.Sprintf(`{"answer":%q,"reasoning":"synthesized","source":"chairman-synthesis"}`, answer)}
}

func ranking(r string) llmtest.Reply {
	return llmtest.Reply{Text: "RANKING: " + r + "\nREASONING: because."}
}

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
	f := llmtest.New(map[string]llmtest.Reply{
		"review/a":    ranking("B, A, C"),
		"review/b":    ranking("B, C, A"),
		"review/c":    ranking("A, B, C"),
		"chair/chair": chairJSON("chairman synthesis"),
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
	chair := f.Prompt("chair", "chair")
	for _, want := range []string{"=== Response A ===", "=== Response C ===", "--- Reviewer 1 ---", "--- Reviewer 3 ---", "RANKING: B, C, A"} {
		if !strings.Contains(chair, want) {
			t.Fatalf("chairman prompt missing %q:\n%s", want, chair)
		}
	}
}

func TestEqualScoresResolveToEarliestCandidate(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"review/a":    ranking("B, A, C"),
		"review/b":    ranking("A, B, C"),
		"review/c":    {Err: errProvider},
		"chair/chair": {Err: errProvider},
		"member/a":    {Text: "answer one"},
		"member/b":    {Text: "answer two"},
	})
	res, _ := mustRun(t, f, "council")
	if res.Selected != "A" || res.Answer != "answer one" {
		t.Fatalf("tie should select A: %+v", res)
	}
}

func TestReviewersSeeAnonymizedAnswers(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"member/secret-gpt":    {Text: "Answer about X."},
		"member/secret-claude": {Text: "Answer about Y."},
	})
	c := New(f.Clients("secret-gpt", "secret-claude"), f.Client("chair"), time.Second)
	if _, err := c.Run(context.Background(), QueryRequest{Question: "Q?", Strategy: "council"}, Evidence{}, nil); err != nil {
		t.Fatal(err)
	}
	p := f.Prompt("review", "secret-gpt")
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
	f := llmtest.New(map[string]llmtest.Reply{
		"member/a":    {Text: "answer from a"},
		"member/b":    {Err: errProvider},
		"member/c":    {Text: "answer from c"},
		"review/a":    ranking("B, A"),
		"review/b":    ranking("A, B"), // must never be used: b's candidate failed
		"review/c":    ranking("B, A"),
		"chair/chair": {Err: errProvider},
	})
	res, log := mustRun(t, f, "council")
	if len(res.Candidates) != 2 || res.Candidates[0].Model != "a" || res.Candidates[0].Label != "A" ||
		res.Candidates[1].Model != "c" || res.Candidates[1].Label != "B" {
		t.Fatalf("candidates %+v", res.Candidates)
	}
	if strings.Contains(f.Prompt("review", "a"), "Response C") {
		t.Fatal("failed candidate must not appear in the review prompt")
	}
	// Member b's own answer failed, so it is not asked to review.
	if len(res.Reviews) != 2 || f.Count("review") != 2 || f.Prompt("review", "b") != "" {
		t.Fatalf("expected reviews from a and c only, got %d", len(res.Reviews))
	}
	if res.Selected != "B" || res.Answer != "answer from c" || res.Candidates[1].Score != 4 {
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
	f := llmtest.New(nil)
	f.Hook = func(ctx context.Context, stage, model string) error {
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
	f := llmtest.New(nil)
	f.Hook = func(ctx context.Context, stage, model string) error {
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
	c.StageTimeout = time.Minute
	done := make(chan error, 1)
	go func() {
		_, err := c.Run(ctx, QueryRequest{Question: "Q", Strategy: "council"}, Evidence{}, log.emit)
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
	if f.Count("review") != 0 || log.has("final") {
		t.Fatalf("cancelled run continued: events %v", log.names)
	}
}

// Cancelling during peer review must return the context error, not a
// degraded "longest candidate" answer that would then be cached.
func TestCancellationDuringReviewIsNotAFallback(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	f := llmtest.New(nil)
	f.Hook = func(c context.Context, stage, model string) error {
		if stage == "review" {
			cancel()
			<-c.Done()
			return c.Err()
		}
		return nil
	}
	_, err := testCouncil(f).Run(ctx, QueryRequest{Question: "Q", Strategy: "council"}, Evidence{}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestStageTimeoutFailsOnlyTheSlowModel(t *testing.T) {
	f := llmtest.New(nil)
	f.Hook = func(ctx context.Context, stage, model string) error {
		if stage == "member" && model == "c" {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}
	c := testCouncil(f)
	c.StageTimeout = 50 * time.Millisecond
	res, err := c.Run(context.Background(), QueryRequest{Question: "Q", Strategy: "council"}, Evidence{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Candidates) != 2 || !res.PeerReviewed {
		t.Fatalf("slow model should be dropped, rest continue: %+v", res)
	}
}

// ── Failure handling ────────────────────────────────────────────────

func TestAllMembersFail(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{"member/a": {Err: errProvider}, "member/b": {Err: errProvider}, "member/c": {Text: "   "}})
	_, log, err := runCouncil(t, f, "council")
	if err == nil || !strings.Contains(err.Error(), "all council members failed") {
		t.Fatalf("expected failure, got %v", err)
	}
	if f.Count("review") != 0 || log.has("final") {
		t.Fatal("no review or final answer after total failure")
	}
}

func TestSingleSurvivorSkipsReviewAndChairman(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{"member/a": {Err: errProvider}, "member/b": {Text: "only answer"}, "member/c": {Err: errProvider}})
	res, log := mustRun(t, f, "council_deep")
	if res.Answer != "only answer" || res.Source != "b (single-response)" || res.PeerReviewed {
		t.Fatalf("unexpected: %+v", res)
	}
	if f.Count("review")+f.Count("chair")+f.Count("reflect") != 0 {
		t.Fatal("review, chairman and reflection must be skipped")
	}
	if !log.has("fallback") || !log.has("final") {
		t.Fatalf("events %v", log.names)
	}
}

func TestMixedReviewerOutcomes(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"review/a": {Err: errProvider},
		"review/b": {Text: "RANKING: A, A, B\nREASONING: A twice."},
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
	chair := f.Prompt("chair", "chair")
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
	f := llmtest.New(map[string]llmtest.Reply{
		"member/a": {Text: "short"},
		"member/b": {Text: "the longest answer of all"},
		"member/c": {Text: "medium one"},
		"review/a": {Text: "{}"},
		"review/b": {Text: `{"best_index": 0}`},
		"review/c": {Text: "RANKING: A"},
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
	if f.Count("chair") != 0 || f.Count("reflect") != 0 {
		t.Fatal("chairman and reflection must be skipped")
	}
	if !log.has("fallback") {
		t.Fatalf("events %v", log.names)
	}
}

func TestAllReviewRequestsFail(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"member/a": {Text: "short"},
		"member/b": {Text: "also short"},
		"member/c": {Text: "the longest candidate answer here"},
		"review/a": {Err: errProvider}, "review/b": {Err: errProvider}, "review/c": {Err: errProvider},
	})
	res, _ := mustRun(t, f, "council")
	if res.Source != "c (peer-review-failed-fallback)" || f.Count("chair") != 0 || len(res.Reviews) != 3 {
		t.Fatalf("unexpected: %+v", res)
	}
}

func TestChairmanFailureReturnsTopRankedNotFirst(t *testing.T) {
	for name, chair := range map[string]llmtest.Reply{"error": {Err: errProvider}, "empty": {Text: "  "}} {
		t.Run(name, func(t *testing.T) {
			f := llmtest.New(map[string]llmtest.Reply{
				"member/c":    {Text: "c's answer"},
				"review/a":    ranking("C, A, B"),
				"review/b":    ranking("C, B, A"),
				"review/c":    ranking("A, C, B"),
				"chair/chair": chair,
			})
			res, log := mustRun(t, f, "council_deep")
			if res.Answer != "c's answer" || res.Source != "c (chairman-failed-fallback)" || res.Selected != "C" || !res.PeerReviewed {
				t.Fatalf("unexpected: %+v", res)
			}
			if f.Count("reflect") != 0 || res.Reflection != nil {
				t.Fatal("reflection must not run on a fallback answer")
			}
			if !log.has("fallback") {
				t.Fatalf("events %v", log.names)
			}
		})
	}
}

func TestDeepReflectionRevisesAfterChairman(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"chair/chair":   chairJSON("draft"),
		"reflect/chair": {Text: `{"quality":"needs_revision","issues":"missing Y","suggestion":"add Y","faithful":true}`},
		"revise/chair":  {Text: `{"answer":"revised answer","reasoning":"added Y","source":"chairman-synthesis"}`},
	})
	res, log := mustRun(t, f, "council_deep")
	if res.Answer != "revised answer" || res.Reasoning != "added Y" || res.Source != "chairman-revised:chair" {
		t.Fatalf("unexpected: %+v", res)
	}
	if r := res.Reflection; r.Status != "completed" || r.Quality != "needs_revision" || r.Issues != "missing Y" || r.Revision.Status != "succeeded" || r.Revision.PreviousAnswer != "draft" {
		t.Fatalf("reflection %+v", r)
	}
	if !log.has("reflection") || !log.has("revision") {
		t.Fatalf("events %v", log.names)
	}
	// Plain council mode never reflects.
	f2 := llmtest.New(nil)
	mustRun(t, f2, "council")
	if f2.Count("reflect") != 0 {
		t.Fatal("council mode must not reflect")
	}
}

// Strategy selection is the LLM router's job; Run only executes a resolved
// strategy and refuses "auto" or anything unknown.
func TestStrategyValidation(t *testing.T) {
	for _, bad := range []string{"", "auto", "bogus"} {
		f := llmtest.New(nil)
		if _, err := testCouncil(f).Run(context.Background(), QueryRequest{Question: "Q", Strategy: bad}, Evidence{}, nil); err == nil {
			t.Fatalf("strategy %q accepted", bad)
		}
		if len(f.Calls()) != 0 {
			t.Fatalf("strategy %q called models", bad)
		}
	}
	for _, ok := range []string{"direct", "council", "council_deep"} {
		if !ValidStrategy(ok) {
			t.Fatalf("%q rejected", ok)
		}
	}
}

func TestEvidenceIsLabelledByKind(t *testing.T) {
	cases := []struct {
		ev        Evidence
		grounding string
		label     string
	}{
		{Evidence{}, "none", "No document context was supplied."},
		{Evidence{Kind: EvidenceDocument, Passages: []string{"[Document d, page 1] text"}}, "document", "Document excerpts from the attached document"},
		{Evidence{Kind: EvidenceWeb, Passages: []string{"Source [1]: T\nURL: u\nContent: c\n"}}, "web", "Web search results (unverified third-party pages"},
		{Evidence{Kind: EvidenceWeb}, "none", "No document context was supplied."}, // empty web results ground nothing
	}
	for _, tc := range cases {
		f := llmtest.New(nil)
		res, err := testCouncil(f).Run(context.Background(), QueryRequest{Question: "Q", Strategy: "council"}, tc.ev, nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.Grounding != tc.grounding {
			t.Fatalf("grounding %q, want %q", res.Grounding, tc.grounding)
		}
		if p := f.Prompt("member", "a"); !strings.Contains(p, tc.label) {
			t.Fatalf("member prompt missing %q:\n%s", tc.label, p)
		}
		chairLabel := map[string]string{"none": "", "document": "Document Excerpts:", "web": "Web Search Results (unverified third-party pages"}[tc.grounding]
		if p := f.Prompt("chair", "chair"); !strings.Contains(p, chairLabel) || (tc.grounding != "document" && strings.Contains(p, "Document Excerpts:")) {
			t.Fatalf("chairman prompt for %s:\n%s", tc.grounding, p)
		}
		if tc.grounding == "web" && strings.Contains(f.Prompt("member", "a"), "Document excerpts") {
			t.Fatal("web results must not be presented as document excerpts")
		}
	}
}

// A member whose candidate failed is not asked to review. The reviews that
// do run keep configuration order and configured slots, the failure is
// reported in the result and the stream, and the chairman labels reviewers by
// their configured slot.
func TestFailedMemberIsNotAskedToReview(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"member/a": {Err: errProvider},
		"member/b": {Text: "answer from b"},
		"member/c": {Text: "answer from c"},
		"review/a": ranking("A, B"),
		"review/b": ranking("B, A"),
		"review/c": ranking("B, A"),
	})
	// Finish c's review before b's so slice order cannot come from timing.
	f.Hook = func(ctx context.Context, stage, model string) error {
		if stage == "review" && model == "b" {
			time.Sleep(50 * time.Millisecond)
		}
		return nil
	}
	res, log := mustRun(t, f, "council")
	if f.Prompt("review", "a") != "" || f.Count("review") != 2 {
		t.Fatalf("failed member a was asked to review (%d review calls)", f.Count("review"))
	}
	if len(res.Reviews) != 2 || res.Reviews[0].Reviewer != "b" || res.Reviews[0].Index != 1 ||
		res.Reviews[1].Reviewer != "c" || res.Reviews[1].Index != 2 {
		t.Fatalf("review order/slots: %+v", res.Reviews)
	}
	if res.Selected != "B" || res.Source != "chairman:chair" || !res.PeerReviewed || len(res.Degraded) != 0 {
		t.Fatalf("unexpected: %+v", res)
	}
	if len(res.FailedCandidates) != 1 || res.FailedCandidates[0].Model != "a" || res.FailedCandidates[0].Error == "" || res.FailedCandidates[0].Label != "" {
		t.Fatalf("failure not reported: %+v", res.FailedCandidates)
	}
	chair := f.Prompt("chair", "chair")
	if strings.Contains(chair, "--- Reviewer 1 ---") || !strings.Contains(chair, "--- Reviewer 2 ---") || !strings.Contains(chair, "--- Reviewer 3 ---") {
		t.Fatalf("chairman reviewer labels:\n%s", chair)
	}
	var skipped []any
	log.mu.Lock()
	for i, n := range log.names {
		if n == "review_skipped" {
			skipped = append(skipped, log.values[i])
		}
	}
	log.mu.Unlock()
	if len(skipped) != 1 {
		t.Fatalf("want one review_skipped event, got %v", skipped)
	}
	if m := skipped[0].(map[string]any); m["reviewer"] != "a" || m["index"] != 0 {
		t.Fatalf("skip event %v", m)
	}
}

// Skipping a failed member's review does not change the existing fallbacks:
// if the reviews that do run are all unusable, the longest candidate wins.
func TestSkippedReviewerStillFallsBackWhenRemainingReviewsFail(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"member/a": {Text: "short"},
		"member/b": {Err: errProvider},
		"member/c": {Text: "the longest candidate answer"},
		"review/a": {Err: errProvider},
		"review/c": {Text: "no ranking here"},
	})
	res, _ := mustRun(t, f, "council")
	if res.Source != "c (peer-review-failed-fallback)" || f.Count("chair") != 0 || len(res.Reviews) != 2 || f.Prompt("review", "b") != "" {
		t.Fatalf("unexpected: %+v", res)
	}
	if len(res.FailedCandidates) != 1 || res.FailedCandidates[0].Model != "b" {
		t.Fatalf("failure not reported: %+v", res.FailedCandidates)
	}
}

// With every member healthy nothing is skipped and every member reviews.
func TestNoReviewSkippedWhenAllCandidatesSucceed(t *testing.T) {
	f := llmtest.New(nil)
	res, log := mustRun(t, f, "council")
	if f.Count("review") != 3 || len(res.Reviews) != 3 || res.FailedCandidates != nil || log.has("review_skipped") {
		t.Fatalf("unexpected: %+v", res)
	}
	for i, r := range res.Reviews {
		if r.Index != i {
			t.Fatalf("review %d has slot %d", i, r.Index)
		}
	}
}
