package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/llm/llmtest"
)

func routerWith(reply llmtest.Reply) (*Router, *llmtest.Fake) {
	f := llmtest.New(map[string]llmtest.Reply{"route/router": reply})
	return NewRouter(f.Client("router")), f
}

func plan(t *testing.T, text string, doc DocumentContext) (*QueryPlan, *llmtest.Fake) {
	t.Helper()
	r, f := routerWith(llmtest.Reply{Text: text})
	p, err := r.PlanWithDocument(context.Background(), "What is X?", doc)
	if err != nil {
		t.Fatal(err)
	}
	return p, f
}

// ── Strict parsing ──────────────────────────────────────────────────

func TestParsePlanAccepts(t *testing.T) {
	cases := map[string]QueryPlan{
		`{"strategy":"direct","reasoning":"greeting","needs_doc":false}`:                 {Strategy: "direct", Reasoning: "greeting"},
		`{"strategy":"council","reasoning":"needs views","needs_doc":true}`:              {Strategy: "council", Reasoning: "needs views", NeedsDoc: true},
		`{"strategy":"council_deep","reasoning":"high stakes","needs_doc":false}`:        {Strategy: "council_deep", Reasoning: "high stakes"},
		"```json\n{\"strategy\":\"direct\",\"reasoning\":\"r\",\"needs_doc\":true}\n```": {Strategy: "direct", Reasoning: "r", NeedsDoc: true},
		"```\n{\"strategy\":\"council\",\"needs_doc\":false}\n```":                       {Strategy: "council"},
		"  \n{\"strategy\":\"council\",\"needs_doc\":false,\"confidence\":0.9}\n ":       {Strategy: "council"},
		`{"strategy":"direct","reasoning":"  padded  ","needs_doc":false}`:               {Strategy: "direct", Reasoning: "padded"},
	}
	for raw, want := range cases {
		got, err := ParsePlan(raw)
		if err != nil {
			t.Fatalf("%q: %v", raw, err)
		}
		if *got != want {
			t.Fatalf("%q: got %+v want %+v", raw, *got, want)
		}
	}
}

func TestParsePlanRejects(t *testing.T) {
	cases := map[string]string{
		"empty":                   "",
		"whitespace":              "  \n ",
		"prose":                   "I think the council should handle this.",
		"prose around JSON":       `Here is the JSON: {"strategy":"direct","needs_doc":false}`,
		"trailing text":           `{"strategy":"direct","needs_doc":false} because it is simple`,
		"two objects":             `{"strategy":"direct","needs_doc":false}{"strategy":"council","needs_doc":true}`,
		"array":                   `[{"strategy":"direct","needs_doc":false}]`,
		"malformed":               `{"strategy":"direct","needs_doc":false`,
		"missing strategy":        `{"reasoning":"r","needs_doc":false}`,
		"null strategy":           `{"strategy":null,"needs_doc":false}`,
		"unknown strategy":        `{"strategy":"deep","needs_doc":false}`,
		"wrong case strategy":     `{"strategy":"Council","needs_doc":false}`,
		"numeric strategy":        `{"strategy":1,"needs_doc":false}`,
		"missing needs_doc":       `{"strategy":"direct","reasoning":"r"}`,
		"string needs_doc":        `{"strategy":"direct","needs_doc":"false"}`,
		"numeric needs_doc":       `{"strategy":"direct","needs_doc":0}`,
		"numeric reasoning":       `{"strategy":"direct","reasoning":5,"needs_doc":false}`,
		"fence with prose inside": "```json\nSure: {\"strategy\":\"direct\",\"needs_doc\":false}\n```",
		"unterminated fence":      "```json\n{\"strategy\":\"direct\",\"needs_doc\":false}",
	}
	for name, raw := range cases {
		if p, err := ParsePlan(raw); err == nil {
			t.Fatalf("%s: accepted %+v", name, p)
		}
	}
}

// ── Plan: request construction and decisions ────────────────────────

func TestPlanRequestMatchesReference(t *testing.T) {
	_, f := plan(t, `{"strategy":"direct","reasoning":"r","needs_doc":false}`, DocumentContext{})
	opts := f.Options("route", "router")
	if len(opts.Messages) != 2 || opts.Messages[0].Role != "system" || opts.Messages[0].Content != routerSystemPrompt {
		t.Fatalf("system prompt not the reference prompt: %+v", opts.Messages)
	}
	if opts.Temperature == nil || *opts.Temperature != 0.1 || !opts.ResponseJSON || opts.MaxTokens != 200 || opts.EnableSearch {
		t.Fatalf("generation settings %+v", opts)
	}
	if got := f.Prompt("route", "router"); got != "Question: What is X?\nNo document attached." {
		t.Fatalf("user message %q", got)
	}
	_, f = plan(t, `{"strategy":"direct","needs_doc":true}`, DocumentContext{Attached: true, Summary: "About Zogland."})
	if got := f.Prompt("route", "router"); got != "Question: What is X?\nAttached Document Summary:\nAbout Zogland." {
		t.Fatalf("user message %q", got)
	}
	_, f = plan(t, `{"strategy":"direct","needs_doc":false}`, DocumentContext{Attached: true})
	if got := f.Prompt("route", "router"); got != "Question: What is X?\n"+missingSummaryNote {
		t.Fatalf("user message %q", got)
	}
}

func TestPlanReturnsModelDecisionForEachStrategy(t *testing.T) {
	for _, s := range []string{StrategyDirect, StrategyCouncil, StrategyCouncilDeep} {
		p, _ := plan(t, `{"strategy":"`+s+`","reasoning":"because","needs_doc":true}`, DocumentContext{Attached: true, Summary: "S"})
		if p.Strategy != s || !p.NeedsDoc || p.Reasoning != "because" || p.Fallback {
			t.Fatalf("%s: %+v", s, p)
		}
	}
}

func TestDocumentDecisions(t *testing.T) {
	// No document: needs_doc is forced false even if the model says true.
	p, _ := plan(t, `{"strategy":"council","needs_doc":true}`, DocumentContext{})
	if p.NeedsDoc {
		t.Fatal("needs_doc must be false without a document")
	}
	// Document with summary: the model decides, including "irrelevant".
	p, _ = plan(t, `{"strategy":"direct","reasoning":"unrelated greeting","needs_doc":false}`, DocumentContext{Attached: true, Summary: "Tax law."})
	if p.NeedsDoc || p.Fallback {
		t.Fatalf("model judged the document irrelevant: %+v", p)
	}
	// Document without summary: relevance cannot be judged, so the document is kept.
	p, _ = plan(t, `{"strategy":"direct","reasoning":"simple","needs_doc":false}`, DocumentContext{Attached: true})
	if !p.NeedsDoc || p.Strategy != "direct" || !strings.Contains(p.Reasoning, "summary unavailable") {
		t.Fatalf("missing summary must keep the document: %+v", p)
	}
	// Reference signature: a non-empty summary means a document is attached.
	r, _ := routerWith(llmtest.Reply{Text: `{"strategy":"council","needs_doc":true}`})
	if p, _ := r.Plan(context.Background(), "q", ""); p.NeedsDoc {
		t.Fatal("Plan with empty summary means no document")
	}
}

func TestRoutingFailuresFallBackToCouncil(t *testing.T) {
	failures := map[string]llmtest.Reply{
		"model error":   {Err: errors.New("provider HTTP 500")},
		"stage timeout": {Err: context.DeadlineExceeded}, // the router's own timeout, not the caller's
		"invalid JSON":  {Text: "council please"},
		"unknown":       {Text: `{"strategy":"quick","needs_doc":false}`, Usage: llm.Usage{TotalTokens: 9}},
		"wrong type":    {Text: `{"strategy":"direct","needs_doc":"no"}`},
	}
	for name, reply := range failures {
		for _, doc := range []DocumentContext{{}, {Attached: true, Summary: "S"}, {Attached: true}} {
			r, _ := routerWith(reply)
			p, err := r.PlanWithDocument(context.Background(), "q", doc)
			if err != nil {
				t.Fatalf("%s: routing failure must not be an error: %v", name, err)
			}
			if p.Strategy != StrategyCouncil || !p.Fallback || p.NeedsDoc != doc.Attached || !strings.Contains(p.Reasoning, "Router unavailable") {
				t.Fatalf("%s %+v: %+v", name, doc, p)
			}
		}
	}
	// Usage of an answered-but-invalid call is still counted.
	r, _ := routerWith(failures["unknown"])
	if p, _ := r.Plan(context.Background(), "q", ""); p.Usage == nil || p.Usage.TotalTokens != 9 {
		t.Fatalf("usage lost: %+v", p.Usage)
	}
}

func TestCancellationIsNotAFallback(t *testing.T) {
	f := llmtest.New(nil)
	f.Hook = func(ctx context.Context, stage, model string) error {
		<-ctx.Done()
		return ctx.Err()
	}
	r := NewRouter(f.Client("router"))
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)
	p, err := r.PlanWithDocument(ctx, "q", DocumentContext{Attached: true})
	if !errors.Is(err, context.Canceled) || p != nil {
		t.Fatalf("got plan=%+v err=%v", p, err)
	}
	ctx2, cancel2 := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel2()
	if _, err := r.Plan(ctx2, "q", ""); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline: %v", err)
	}
}

func TestPlanUsageReported(t *testing.T) {
	r, _ := routerWith(llmtest.Reply{Text: `{"strategy":"direct","needs_doc":false}`, Usage: llm.Usage{PromptTokens: 80, CompletionTokens: 20, TotalTokens: 100}})
	p, _ := r.Plan(context.Background(), "q", "")
	if p.Usage == nil || p.Usage.TotalTokens != 100 {
		t.Fatalf("usage %+v", p.Usage)
	}
	r, _ = routerWith(llmtest.Reply{Text: `{"strategy":"direct","needs_doc":false}`})
	if p, _ := r.Plan(context.Background(), "q", ""); p.Usage != nil {
		t.Fatal("usage invented")
	}
}

// Mock mode routes through the same parser, controlled by explicit
// directives in the question.
func TestMockRouterDirectives(t *testing.T) {
	r := NewRouter(&llm.MockClient{Name: "router"})
	cases := []struct {
		q        string
		doc      DocumentContext
		strategy string
		needsDoc bool
		fallback bool
	}{
		{"hi #route=direct", DocumentContext{}, "direct", false, false},
		{"hard #route=council_deep #needs_doc=true", DocumentContext{Attached: true, Summary: "S"}, "council_deep", true, false},
		{"off-topic #route=direct #needs_doc=false", DocumentContext{Attached: true, Summary: "S"}, "direct", false, false},
		{"no directive", DocumentContext{Attached: true, Summary: "S"}, "council", true, false},
		{"broken #route=invalid", DocumentContext{}, "council", false, true},
		{"unknown #route=bogus", DocumentContext{}, "council", false, true},
	}
	for _, tc := range cases {
		p, err := r.PlanWithDocument(context.Background(), tc.q, tc.doc)
		if err != nil || p.Strategy != tc.strategy || p.NeedsDoc != tc.needsDoc || p.Fallback != tc.fallback {
			t.Fatalf("%q: %+v %v", tc.q, p, err)
		}
	}
}
