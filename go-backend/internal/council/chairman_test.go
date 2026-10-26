package council

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/llm/llmtest"
)

// ── Chairman output parsing ─────────────────────────────────────────

func TestParseChairmanAccepts(t *testing.T) {
	type want struct {
		answer, reasoning, source string
		conf                      float64
		confReported, repaired    bool
	}
	cases := map[string]struct {
		raw  string
		want want
	}{
		"complete":            {`{"answer":"A","reasoning":"R","source":"chairman-synthesis","confidence":0.85}`, want{"A", "R", "chairman-synthesis", 0.85, true, false}},
		"fenced":              {"```json\n{\"answer\":\"A\",\"reasoning\":\"R\",\"source\":\"s\",\"confidence\":0.5}\n```", want{"A", "R", "s", 0.5, true, false}},
		"escaped quotes/nl":   {`{"answer":"He said \"hi\"\nline 2","reasoning":"R"}`, want{"He said \"hi\"\nline 2", "R", "", 0, false, false}},
		"unicode":             {`{"answer":"Zürich — 東京 ✓ \u00e9","reasoning":"R"}`, want{"Zürich — 東京 ✓ é", "R", "", 0, false, false}},
		"escaped math":        {`{"answer":"$\\frac{a}{b} + \\alpha$","reasoning":"R"}`, want{`$\frac{a}{b} + \alpha$`, "R", "", 0, false, false}},
		"confidence zero":     {`{"answer":"A","reasoning":"R","confidence":0}`, want{"A", "R", "", 0, true, false}},
		"confidence one":      {`{"answer":"A","reasoning":"R","confidence":1}`, want{"A", "R", "", 1, true, false}},
		"confidence absent":   {`{"answer":"A","reasoning":"R","source":"s"}`, want{"A", "R", "s", 0, false, false}},
		"confidence null":     {`{"answer":"A","reasoning":"R","confidence":null}`, want{"A", "R", "", 0, false, false}},
		"empty reasoning":     {`{"answer":"A","reasoning":""}`, want{"A", "", "", 0, false, false}},
		"extra fields":        {`{"answer":"A","reasoning":"R","citations":[1,2]}`, want{"A", "R", "", 0, false, false}},
		"repair: LaTeX \\a":   {`{"answer":"$\alpha + \Lambda$","reasoning":"R"}`, want{`$\alpha + \Lambda$`, "R", "", 0, false, true}},
		"repair: raw newline": {"{\"answer\":\"line 1\nline 2\",\"reasoning\":\"R\"}", want{"line 1\nline 2", "R", "", 0, false, true}},
	}
	for name, tc := range cases {
		got, err := ParseChairmanResult(tc.raw)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		g := want{got.Answer, got.Reasoning, got.Source, got.Confidence, got.ConfidenceReported, got.Repaired}
		if g != tc.want {
			t.Fatalf("%s:\n got %+v\nwant %+v", name, g, tc.want)
		}
	}
}

func TestParseChairmanRejects(t *testing.T) {
	cases := map[string]string{
		"empty":               ``,
		"prose":               `The answer is 42.`,
		"prose around JSON":   `Here you go: {"answer":"A","reasoning":"R"} Hope that helps!`,
		"multiple objects":    `{"answer":"A","reasoning":"R"}{"answer":"B","reasoning":"R"}`,
		"array":               `[{"answer":"A","reasoning":"R"}]`,
		"truncated":           `{"answer":"A","reasoning":"R`,
		"truncated + raw nl":  "{\"answer\":\"line\nmore",
		"malformed":           `{"answer":"A",,"reasoning":"R"}`,
		"missing answer":      `{"reasoning":"R","confidence":0.9}`,
		"empty answer":        `{"answer":"   ","reasoning":"R"}`,
		"numeric answer":      `{"answer":42,"reasoning":"R"}`,
		"missing reasoning":   `{"answer":"A"}`,
		"object reasoning":    `{"answer":"A","reasoning":{"x":1}}`,
		"numeric source":      `{"answer":"A","reasoning":"R","source":7}`,
		"string confidence":   `{"answer":"A","reasoning":"R","confidence":"0.9"}`,
		"negative confidence": `{"answer":"A","reasoning":"R","confidence":-0.1}`,
		"confidence over 1":   `{"answer":"A","reasoning":"R","confidence":1.5}`,
		"percent confidence":  `{"answer":"A","reasoning":"R","confidence":85}`,
		"overflow confidence": `{"answer":"A","reasoning":"R","confidence":1e999}`,
		"NaN literal":         `{"answer":"A","reasoning":"R","confidence":NaN}`,
		"Infinity literal":    `{"answer":"A","reasoning":"R","confidence":Infinity}`,
		"bool confidence":     `{"answer":"A","reasoning":"R","confidence":true}`,
		"fence with prose":    "```json\nSure! {\"answer\":\"A\",\"reasoning\":\"R\"}\n```",
	}
	for name, raw := range cases {
		if got, err := ParseChairmanResult(raw); err == nil {
			t.Fatalf("%s: accepted %+v", name, got)
		}
	}
}

func TestRepairLeavesValidEscapesAlone(t *testing.T) {
	valid := `{"a":"tab\tnl\nquote\"slash\\uni\u00e9\/x\b\f\r","b":"plain"}`
	if got := repairJSONStrings(valid); got != valid {
		t.Fatalf("valid escapes changed:\n%s\n%s", valid, got)
	}
	// Outside strings nothing changes, even backslashes the decoder will reject.
	if got := repairJSONStrings("{\n  \"a\": 1\n}"); got != "{\n  \"a\": 1\n}" {
		t.Fatalf("whitespace between tokens changed: %q", got)
	}
	// Short or non-hex \u is invalid and gets doubled.
	if got := repairJSONStrings(`{"a":"\u12 and \uzzzz"}`); got != `{"a":"\\u12 and \\uzzzz"}` {
		t.Fatalf("got %s", got)
	}
}

func TestParseReflection(t *testing.T) {
	good, err := ParseReflectionResult(`{"quality":"good","issues":"","suggestion":"","faithful":true,"confidence":0.9}`)
	if err != nil || good.Quality != "good" || !good.Faithful || !good.ConfidenceReported || good.Confidence != 0.9 {
		t.Fatalf("%+v %v", good, err)
	}
	needs, err := ParseReflectionResult("```json\n{\"quality\":\"needs_revision\",\"issues\":\"misses Y\",\"suggestion\":\"add Y\",\"faithful\":false}\n```")
	if err != nil || needs.Quality != "needs_revision" || needs.Faithful || needs.ConfidenceReported || needs.Issues != "misses Y" {
		t.Fatalf("%+v %v", needs, err)
	}
	bad := map[string]string{
		"malformed":          `{"quality":"good"`,
		"prose":              `Looks good to me.`,
		"missing quality":    `{"issues":"","suggestion":"","faithful":true}`,
		"invalid quality":    `{"quality":"excellent","issues":"","suggestion":"","faithful":true}`,
		"quality case":       `{"quality":"Good","issues":"","suggestion":"","faithful":true}`,
		"missing faithful":   `{"quality":"good","issues":"","suggestion":""}`,
		"string faithful":    `{"quality":"good","issues":"","suggestion":"","faithful":"true"}`,
		"missing issues":     `{"quality":"good","suggestion":"","faithful":true}`,
		"missing suggestion": `{"quality":"good","issues":"","faithful":true}`,
		"bad confidence":     `{"quality":"good","issues":"","suggestion":"","faithful":true,"confidence":2}`,
		"reference-shaped":   `{"quality":"good","reason":"fine"}`,
	}
	for name, raw := range bad {
		if got, err := ParseReflectionResult(raw); err == nil {
			t.Fatalf("%s: accepted %+v", name, got)
		}
	}
}

// ── Request construction ────────────────────────────────────────────

func TestChairmanRequestMatchesReference(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"review/a": ranking("B, A, C"),
		"review/b": {Text: "RANKING: A, A, B\nREASONING: broken"},
		"review/c": ranking("C, B, A"),
		"member/a": {Text: "answer from a"},
	})
	ev := Evidence{Kind: EvidenceDocument, Passages: []string{"[Document d, page 1] Fact one.", "[Document d, page 2] Fact two."}}
	if _, err := testCouncil(f).Run(context.Background(), QueryRequest{Question: "Compare X and Y", Strategy: "council"}, ev, nil); err != nil {
		t.Fatal(err)
	}
	opts := f.Options("chair", "chair")
	if opts.Messages[0].Role != "system" || opts.Messages[0].Content != chairmanSystemPrompt || !opts.ResponseJSON || opts.Temperature != nil || opts.EnableSearch {
		t.Fatalf("chairman call settings %+v", opts)
	}
	user := opts.Messages[1].Content
	wantOrder := []string{
		"Question: Compare X and Y\n",
		"\nDocument Excerpts:\n[Document d, page 1] Fact one.\n\n---\n\n[Document d, page 2] Fact two.\n",
		"Individual Responses:\n=== Response A ===\nanswer from a",
		"=== Response B ===", "=== Response C ===",
		"Peer Reviews:\n--- Reviewer 1 ---\nRANKING: B, A, C",
		"--- Reviewer 3 ---\nRANKING: C, B, A",
	}
	pos := 0
	for _, w := range wantOrder {
		i := strings.Index(user[pos:], w)
		if i < 0 {
			t.Fatalf("missing or out of order %q in:\n%s", w, user)
		}
		pos += i + len(w)
	}
	for _, leak := range []string{"Reviewer 2", "A, A, B", "Web Search Results"} {
		if strings.Contains(user, leak) {
			t.Fatalf("chairman input must not contain %q", leak)
		}
	}

	f = llmtest.New(nil)
	web := Evidence{Kind: EvidenceWeb, Passages: []string{"Source [1]: T\nURL: u\nContent: c\n"}}
	if _, err := testCouncil(f).Run(context.Background(), QueryRequest{Question: "Q", Strategy: "council"}, web, nil); err != nil {
		t.Fatal(err)
	}
	if u := f.Prompt("chair", "chair"); !strings.Contains(u, "Web Search Results (unverified third-party pages, not the user's document):\nSource [1]") || strings.Contains(u, "Document Excerpts") {
		t.Fatalf("web evidence provenance lost:\n%s", u)
	}
}

func TestReflectionAndRevisionRequestsMatchReference(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"chair/chair":   {Text: `{"answer":"draft answer","reasoning":"r","source":"chairman-synthesis","confidence":0.6}`},
		"reflect/chair": {Text: `{"quality":"needs_revision","issues":"misses Y","suggestion":"add Y","faithful":false}`},
		"revise/chair":  {Text: `{"answer":"better answer","reasoning":"added Y","source":"chairman-synthesis","confidence":0.7}`},
	})
	ev := Evidence{Kind: EvidenceDocument, Passages: []string{"[Document d, page 1] Fact."}}
	if _, err := testCouncil(f).Run(context.Background(), QueryRequest{Question: "Explain Z", Strategy: "council_deep"}, ev, nil); err != nil {
		t.Fatal(err)
	}
	ro := f.Options("reflect", "chair")
	if ro.Messages[0].Content != reflectionSystemPrompt || ro.Temperature == nil || *ro.Temperature != 0.2 || !ro.ResponseJSON || ro.MaxTokens != 500 {
		t.Fatalf("reflection settings %+v", ro)
	}
	if got, want := ro.Messages[1].Content, "Question: Explain Z\n\nSource Material:\n[Document d, page 1] Fact.\n\nAnswer to evaluate:\ndraft answer"; got != want {
		t.Fatalf("reflection message\n got %q\nwant %q", got, want)
	}
	vo := f.Options("revise", "chair")
	if vo.Messages[0].Content != chairmanSystemPrompt || !vo.ResponseJSON || vo.Temperature != nil {
		t.Fatalf("revision settings %+v", vo)
	}
	for _, w := range []string{"Question: Explain Z\n", "Document Excerpts:\n[Document d, page 1] Fact.", "Your previous answer:\ndraft answer", "- Issues: misses Y", "- Suggestion: add Y", "Maintain accuracy and faithfulness to sources."} {
		if !strings.Contains(vo.Messages[1].Content, w) {
			t.Fatalf("revision message missing %q:\n%s", w, vo.Messages[1].Content)
		}
	}
}

// ── Pipeline behaviour ──────────────────────────────────────────────

func confidenceOf(r QueryResult) string {
	b, _ := json.Marshal(r)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	if v, ok := m["confidence"]; ok {
		return strings.TrimSpace(strings.ReplaceAll(string(mustJSON(v)), " ", ""))
	}
	return "absent"
}

func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

func TestCouncilModeStructuredSynthesis(t *testing.T) {
	r := llmtest.Reply{Text: `{"answer":"Final","reasoning":"B was most complete","source":"chairman-synthesis","confidence":0}`, Usage: llm.Usage{TotalTokens: 30, PromptTokens: 25, CompletionTokens: 5}}
	f := llmtest.New(map[string]llmtest.Reply{"chair/chair": r})
	res, log := mustRun(t, f, "council")
	if res.Answer != "Final" || res.Reasoning != "B was most complete" || res.Source != "chairman:chair" || res.ConfidenceSource != "chairman" {
		t.Fatalf("%+v", res)
	}
	if confidenceOf(res) != "0" {
		t.Fatalf("a reported 0 must stay 0, got %s", confidenceOf(res))
	}
	if s := res.Synthesis; s.Status != "synthesized" || s.Model != "chair" || s.ReportedSource != "chairman-synthesis" || s.Error != "" {
		t.Fatalf("synthesis %+v", s)
	}
	if res.Reflection != nil || f.Count("reflect") != 0 || len(res.Degraded) != 0 || !log.has("synthesis") {
		t.Fatal("council mode must not reflect")
	}
	if res.TokenUsage == nil || res.TokenUsage.TotalTokens != 30 {
		t.Fatalf("chairman usage lost: %+v", res.TokenUsage)
	}
	// Without a reported confidence the field is absent, never 0.
	res, _ = mustRun(t, llmtest.New(nil), "council")
	if confidenceOf(res) != "absent" || res.ConfidenceSource != "" {
		t.Fatalf("confidence invented: %s", confidenceOf(res))
	}
}

func TestMalformedChairmanFallsBack(t *testing.T) {
	for name, reply := range map[string]llmtest.Reply{
		"truncated JSON":   {Text: `{"answer":"half`, Usage: llm.Usage{TotalTokens: 12}},
		"plain text":       {Text: "Here is a lovely answer without JSON."},
		"missing answer":   {Text: `{"reasoning":"r","confidence":0.9}`},
		"bad confidence":   {Text: `{"answer":"A","reasoning":"r","confidence":7}`},
		"provider failure": {Err: errProvider},
	} {
		f := llmtest.New(map[string]llmtest.Reply{
			"chair/chair": reply,
			"member/c":    {Text: "c's answer"},
			"review/a":    ranking("C, A, B"),
			"review/b":    ranking("C, B, A"),
			"review/c":    ranking("A, C, B"),
		})
		res, log := mustRun(t, f, "council_deep")
		if res.Answer != "c's answer" || res.Source != "c (chairman-failed-fallback)" || res.Confidence != nil || res.Reasoning != "" {
			t.Fatalf("%s: %+v", name, res)
		}
		if res.Synthesis == nil || res.Synthesis.Status != "failed" || res.Synthesis.Error == "" || res.Degraded[0] != "chairman-failed-fallback" {
			t.Fatalf("%s: synthesis %+v degraded %v", name, res.Synthesis, res.Degraded)
		}
		if f.Count("reflect")+f.Count("revise") != 0 || res.Reflection != nil || !log.has("fallback") {
			t.Fatalf("%s: reflection after fallback", name)
		}
		if strings.Contains(res.Answer, "{") {
			t.Fatalf("%s: raw output presented as answer", name)
		}
	}
	// Tokens spent on an invalid response are still counted.
	f := llmtest.New(map[string]llmtest.Reply{"chair/chair": {Text: `{"answer":"half`, Usage: llm.Usage{TotalTokens: 12}}})
	res, _ := mustRun(t, f, "council")
	if res.TokenUsage == nil || res.TokenUsage.TotalTokens != 12 {
		t.Fatalf("usage %+v", res.TokenUsage)
	}
}

func TestReflectionGoodNoRevision(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"chair/chair":   {Text: `{"answer":"A1","reasoning":"r","confidence":0.6}`},
		"reflect/chair": {Text: `{"quality":"good","issues":"","suggestion":"","faithful":true,"confidence":0.9}`},
	})
	res, _ := mustRun(t, f, "council_deep")
	r := res.Reflection
	if r.Status != "completed" || r.Quality != "good" || !*r.Faithful || *r.Confidence != 0.9 || r.Revision != nil || f.Count("revise") != 0 {
		t.Fatalf("%+v", r)
	}
	// Reference behaviour: a good audit's confidence replaces the chairman's.
	if res.Answer != "A1" || *res.Confidence != 0.9 || res.ConfidenceSource != "reflection" || len(res.Degraded) != 0 {
		t.Fatalf("%+v", res)
	}
	// Without an audit confidence the chairman's value stays.
	f = llmtest.New(map[string]llmtest.Reply{
		"chair/chair":   {Text: `{"answer":"A1","reasoning":"r","confidence":0.6}`},
		"reflect/chair": {Text: `{"quality":"good","issues":"","suggestion":"","faithful":true}`},
	})
	res, _ = mustRun(t, f, "council_deep")
	if *res.Confidence != 0.6 || res.ConfidenceSource != "chairman" || res.Reflection.Confidence != nil {
		t.Fatalf("%+v", res)
	}
}

// The reference triggers revision only on quality; a "good" audit that also
// says faithful=false is reported as-is and does not revise.
func TestGoodButUnfaithfulIsReportedNotRevised(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"reflect/chair": {Text: `{"quality":"good","issues":"claim 2 unsupported","suggestion":"","faithful":false}`},
	})
	res, _ := mustRun(t, f, "council_deep")
	if f.Count("revise") != 0 || *res.Reflection.Faithful || res.Reflection.Issues != "claim 2 unsupported" {
		t.Fatalf("%+v", res.Reflection)
	}
}

func TestNeedsRevisionRunsExactlyOneRevision(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{
		"chair/chair":   {Text: `{"answer":"draft","reasoning":"r1","confidence":0.6}`},
		"reflect/chair": {Text: `{"quality":"needs_revision","issues":"i","suggestion":"s","faithful":false,"confidence":0.4}`},
		"revise/chair":  {Text: `{"answer":"revised","reasoning":"r2","source":"chairman-synthesis"}`},
	})
	res, log := mustRun(t, f, "council_deep")
	if f.Count("chair") != 1 || f.Count("reflect") != 1 || f.Count("revise") != 1 {
		t.Fatalf("calls %v", f.Calls())
	}
	if res.Answer != "revised" || res.Reasoning != "r2" || res.Source != "chairman-revised:chair" {
		t.Fatalf("%+v", res)
	}
	// The revision reported no confidence, so the draft's 0.6 must not carry over.
	if res.Confidence != nil || res.ConfidenceSource != "" {
		t.Fatalf("stale confidence: %v", *res.Confidence)
	}
	rev := res.Reflection.Revision
	if rev.Status != "succeeded" || rev.PreviousAnswer != "draft" || rev.PreviousSource != "chairman:chair" || len(res.Degraded) != 0 {
		t.Fatalf("%+v", rev)
	}
	// Events: synthesis → reflection → revision → final, with one final.
	var order []string
	for _, n := range log.names {
		if n == "synthesis" || n == "reflection" || n == "revision" || n == "final" {
			order = append(order, n)
		}
	}
	if strings.Join(order, ",") != "synthesis,reflection,revision,final" {
		t.Fatalf("events %v", log.names)
	}
	if final := log.values[len(log.values)-1].(QueryResult); final.Answer != "revised" {
		t.Fatal("final event must carry the revised answer")
	}
	// A revised answer with confidence reports it as the revision's.
	f = llmtest.New(map[string]llmtest.Reply{
		"reflect/chair": {Text: `{"quality":"needs_revision","issues":"i","suggestion":"s","faithful":true}`},
		"revise/chair":  {Text: `{"answer":"revised","reasoning":"r2","confidence":0.75}`},
	})
	res, _ = mustRun(t, f, "council_deep")
	if *res.Confidence != 0.75 || res.ConfidenceSource != "revision" {
		t.Fatalf("%+v", res)
	}
}

func TestReflectionFailuresKeepAnswerWithoutFakeSuccess(t *testing.T) {
	for name, reply := range map[string]llmtest.Reply{
		"provider error":     {Err: errProvider},
		"stage timeout":      {Err: context.DeadlineExceeded},
		"malformed":          {Text: `{"quality":"good"`},
		"invalid quality":    {Text: `{"quality":"excellent","issues":"","suggestion":"","faithful":true}`},
		"missing faithful":   {Text: `{"quality":"good","issues":"","suggestion":""}`},
		"invalid confidence": {Text: `{"quality":"good","issues":"","suggestion":"","faithful":true,"confidence":-1}`},
		"reference schema":   {Text: `{"quality":"good","reason":"ok"}`},
	} {
		f := llmtest.New(map[string]llmtest.Reply{
			"chair/chair":   {Text: `{"answer":"chair answer","reasoning":"r","confidence":0.8}`},
			"reflect/chair": reply,
		})
		res, log := mustRun(t, f, "council_deep")
		r := res.Reflection
		if r.Status != "failed" || r.Error == "" || r.Faithful != nil || r.Confidence != nil || r.Quality != "" {
			t.Fatalf("%s: failed audit reported as success: %+v", name, r)
		}
		if res.Answer != "chair answer" || *res.Confidence != 0.8 || res.ConfidenceSource != "chairman" || f.Count("revise") != 0 {
			t.Fatalf("%s: %+v", name, res)
		}
		if strings.Join(res.Degraded, ",") != "reflection-failed" || !log.has("reflection") {
			t.Fatalf("%s: degraded %v", name, res.Degraded)
		}
	}
}

func TestRevisionFailuresKeepPreviousAnswer(t *testing.T) {
	for name, reply := range map[string]llmtest.Reply{
		"provider error": {Err: errProvider},
		"malformed":      {Text: "I revised it: much better now."},
		"empty answer":   {Text: `{"answer":"","reasoning":"r"}`},
		"bad confidence": {Text: `{"answer":"x","reasoning":"r","confidence":3}`},
	} {
		f := llmtest.New(map[string]llmtest.Reply{
			"chair/chair":   {Text: `{"answer":"original","reasoning":"r1","confidence":0.6}`},
			"reflect/chair": {Text: `{"quality":"needs_revision","issues":"i","suggestion":"s","faithful":false}`},
			"revise/chair":  reply,
		})
		res, log := mustRun(t, f, "council_deep")
		rev := res.Reflection.Revision
		if res.Answer != "original" || res.Reasoning != "r1" || res.Source != "chairman:chair" || *res.Confidence != 0.6 {
			t.Fatalf("%s: previous answer not preserved: %+v", name, res)
		}
		if res.Reflection.Quality != "needs_revision" || rev.Status != "failed" || rev.Error == "" || rev.PreviousAnswer != "original" {
			t.Fatalf("%s: %+v", name, rev)
		}
		if strings.Join(res.Degraded, ",") != "revision-failed" || f.Count("revise") != 1 || !log.has("revision") {
			t.Fatalf("%s: %v %v", name, res.Degraded, f.Calls())
		}
	}
}

func TestCancellationDuringChairmanReflectionAndRevision(t *testing.T) {
	for _, stage := range []string{"chair", "reflect", "revise"} {
		ctx, cancel := context.WithCancel(context.Background())
		f := llmtest.New(map[string]llmtest.Reply{"reflect/chair": {Text: `{"quality":"needs_revision","issues":"i","suggestion":"s","faithful":false}`}})
		f.Hook = func(c context.Context, s, model string) error {
			if s == stage {
				cancel()
				<-c.Done()
				return c.Err()
			}
			return nil
		}
		log := &eventLog{}
		_, err := testCouncil(f).Run(ctx, QueryRequest{Question: "Q", Strategy: "council_deep"}, Evidence{}, log.emit)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("%s: expected context.Canceled, got %v", stage, err)
		}
		if log.has("final") || log.has("fallback") {
			t.Fatalf("%s: cancelled run reported a result: %v", stage, log.names)
		}
	}
}

func TestReflectionHonoursStageTimeout(t *testing.T) {
	f := llmtest.New(nil)
	f.Hook = func(c context.Context, s, model string) error {
		if s == "reflect" {
			<-c.Done()
			return c.Err()
		}
		return nil
	}
	c := testCouncil(f)
	c.StageTimeout = 50 * time.Millisecond
	res, err := c.Run(context.Background(), QueryRequest{Question: "Q", Strategy: "council_deep"}, Evidence{}, nil)
	if err != nil || res.Reflection.Status != "failed" || !strings.Contains(res.Reflection.Error, "deadline") {
		t.Fatalf("%v %+v", err, res.Reflection)
	}
}

func TestNoReflectionAfterEarlyFallbacks(t *testing.T) {
	scripts := map[string]map[string]llmtest.Reply{
		"single survivor":  {"member/a": {Err: errProvider}, "member/b": {Err: errProvider}},
		"no valid reviews": {"review/a": {Text: "{}"}, "review/b": {Text: "{}"}, "review/c": {Text: "{}"}},
		"chairman invalid": {"chair/chair": {Text: "no json"}},
	}
	for name, s := range scripts {
		f := llmtest.New(s)
		res, _ := mustRun(t, f, "council_deep")
		if f.Count("reflect") != 0 || res.Reflection != nil || len(res.Degraded) != 1 {
			t.Fatalf("%s: %+v", name, res)
		}
	}
}

// Mock mode (MOCK_LLM=true) drives every path through the real parsers via
// explicit question markers; the mock never reports a confidence.
func TestMockDirectivesExerciseRealParsers(t *testing.T) {
	members := []llm.LLMClient{&llm.MockClient{Name: "m1"}, &llm.MockClient{Name: "m2"}, &llm.MockClient{Name: "m3"}}
	c := New(members, &llm.MockClient{Name: "chairman"}, time.Second)
	run := func(q string) QueryResult {
		res, err := c.Run(context.Background(), QueryRequest{Question: q, Strategy: "council_deep"}, Evidence{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	if r := run("plain"); r.Synthesis.Status != "synthesized" || r.Reflection.Quality != "good" || r.Confidence != nil || r.Reflection.Revision != nil {
		t.Fatalf("default: %+v", r)
	}
	if r := run("q #reflect=needs_revision"); r.Reflection.Revision.Status != "succeeded" || !strings.HasPrefix(r.Answer, "[MOCK revised]") {
		t.Fatalf("revision: %+v", r)
	}
	if r := run("q #reflect=needs_revision #revise=invalid"); r.Reflection.Revision.Status != "failed" || !strings.HasPrefix(r.Answer, "[MOCK consensus]") {
		t.Fatalf("revision failure: %+v", r)
	}
	if r := run("q #reflect=invalid"); r.Reflection.Status != "failed" {
		t.Fatalf("reflection failure: %+v", r.Reflection)
	}
	if r := run("q #chair=invalid"); r.Synthesis.Status != "failed" || r.Reflection != nil || !strings.Contains(r.Source, "chairman-failed-fallback") {
		t.Fatalf("chairman failure: %+v", r)
	}
}
