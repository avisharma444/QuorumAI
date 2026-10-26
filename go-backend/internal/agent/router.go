// Package agent provides the LLM query router and the document ingestion
// (summarization) agent.
package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"

	"example.com/councilai-starter/internal/llm"
)

// Strategies the router may choose.
const (
	StrategyDirect      = "direct"
	StrategyCouncil     = "council"
	StrategyCouncilDeep = "council_deep"
)

// IsStrategy reports whether s is one of the three execution strategies.
func IsStrategy(s string) bool {
	return s == StrategyDirect || s == StrategyCouncil || s == StrategyCouncilDeep
}

// QueryPlan describes the execution strategy chosen by the router agent.
type QueryPlan struct {
	Strategy  string `json:"strategy"`  // "direct" | "council" | "council_deep"
	Reasoning string `json:"reasoning"` // why this strategy was chosen
	NeedsDoc  bool   `json:"needs_doc"` // does this query need document context?

	// Fallback is true when the plan is the safe default because routing
	// failed (model error or invalid output), not a model decision.
	Fallback bool `json:"fallback,omitempty"`
	// Usage is the router call's provider-reported token usage, if any.
	Usage *llm.Usage `json:"-"`
}

// DocumentContext is what the router knows about an attached document.
// Attached with an empty Summary means a document was attached but its
// summary could not be found (summarization failed or the cache expired).
type DocumentContext struct {
	Attached bool
	Summary  string
}

// Router classifies a question and decides the execution strategy.
type Router struct {
	client llm.LLMClient
}

// NewRouter creates a router backed by the given client (typically a fast,
// cheap model).
func NewRouter(client llm.LLMClient) *Router {
	return &Router{client: client}
}

// ModelName identifies the router model.
func (r *Router) ModelName() string { return r.client.ModelName() }

// routerSystemPrompt is the reference prompt, verbatim.
const routerSystemPrompt = `You are a query classifier for a multi-LLM council system.
Given a user question, and optionally a summary of an attached document, choose the best strategy:

- "direct": Simple factual question, greeting, meta-question, or something a single model handles well. One model is enough.
- "council": Needs multiple perspectives, nuanced reasoning, or a thorough answer. Use the full multi-model council.
- "council_deep": Complex, ambiguous, high-stakes, or requires careful deliberation with quality verification.

Also decide if the query genuinely needs document context (needs_doc).
If no document summary is provided, set needs_doc to false regardless.
If a document summary IS provided, evaluate if the user's question is related to the topics in the summary. If the question is completely unrelated (e.g., general knowledge, greetings), set needs_doc to false.

Respond ONLY with a raw JSON object and NO OTHER TEXT whatsoever. Do not include markdown code blocks, do not say 'Here is the JSON'. Just output the raw JSON string: {"strategy":"...", "reasoning":"...", "needs_doc": true/false}`

// missingSummaryNote replaces the summary when a document is attached but
// its summary is unavailable. The model still chooses the strategy; the
// document decision is made conservatively afterwards.
const missingSummaryNote = "A document is attached, but its summary is unavailable."

// Plan analyzes a question with the reference signature: a non-empty
// docSummary means a document is attached.
func (r *Router) Plan(ctx context.Context, question string, docSummary string) (*QueryPlan, error) {
	return r.PlanWithDocument(ctx, question, DocumentContext{Attached: docSummary != "", Summary: docSummary})
}

// PlanWithDocument returns an execution plan. Routing failures (model error,
// invalid output) yield the default council plan with Fallback set and a nil
// error. The only error returned is the caller's context error, so a
// cancelled request is never mistaken for a routing fallback.
//
// Document rules:
//   - no document: NeedsDoc is always false (reference rule);
//   - document with summary: the model decides NeedsDoc;
//   - document without summary: NeedsDoc is forced true, because relevance
//     cannot be judged and dropping the user's document would be worse than
//     an unnecessary retrieval (fix for the reference, which treated a
//     missing summary as "no document");
//   - routing failure with a document attached: NeedsDoc is true.
func (r *Router) PlanWithDocument(ctx context.Context, question string, doc DocumentContext) (*QueryPlan, error) {
	var userMsg string
	switch {
	case doc.Attached && doc.Summary != "":
		userMsg = fmt.Sprintf("Question: %s\nAttached Document Summary:\n%s", question, doc.Summary)
	case doc.Attached:
		userMsg = fmt.Sprintf("Question: %s\n%s", question, missingSummaryNote)
	default:
		userMsg = fmt.Sprintf("Question: %s\nNo document attached.", question)
	}

	resp, err := r.client.GenerateChat(ctx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: routerSystemPrompt},
			{Role: "user", Content: userMsg},
		},
		Temperature:  llm.Float64Ptr(0.1), // low temperature for stable classification
		ResponseJSON: true,
		MaxTokens:    200,
	})
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		log.Printf("[Router] Classification failed, defaulting to council: %v", err)
		return defaultPlan(doc.Attached, "router call failed: "+err.Error()), nil
	}

	plan, perr := ParsePlan(resp.Answer)
	if perr != nil {
		log.Printf("[Router] Invalid plan, defaulting to council: %v", perr)
		p := defaultPlan(doc.Attached, "router returned an invalid plan: "+perr.Error())
		p.Usage = usageOf(resp)
		return p, nil
	}
	plan.Usage = usageOf(resp)
	switch {
	case !doc.Attached:
		plan.NeedsDoc = false
	case doc.Summary == "":
		plan.NeedsDoc = true
		plan.Reasoning = strings.TrimSpace(plan.Reasoning + " [Document summary unavailable; document context kept.]")
	}
	log.Printf("[Router] Strategy=%s NeedsDoc=%v Reason=%s", plan.Strategy, plan.NeedsDoc, plan.Reasoning)
	return plan, nil
}

// defaultPlan is the safe default when routing fails: the full council, and
// the document kept if one is attached.
func defaultPlan(docAttached bool, why string) *QueryPlan {
	return &QueryPlan{
		Strategy:  StrategyCouncil,
		Reasoning: "Router unavailable, using default council strategy (" + why + ")",
		NeedsDoc:  docAttached,
		Fallback:  true,
	}
}

func usageOf(resp *llm.Response) *llm.Usage {
	if resp == nil || !resp.Usage.Reported() {
		return nil
	}
	u := resp.Usage
	return &u
}

// rawPlan uses pointers so missing fields can be told apart from zero values.
type rawPlan struct {
	Strategy  *string `json:"strategy"`
	Reasoning *string `json:"reasoning"`
	NeedsDoc  *bool   `json:"needs_doc"`
}

// ParsePlan strictly parses router output. It accepts exactly one JSON
// object, optionally wrapped in a single markdown code fence, with a known
// "strategy" string and a boolean "needs_doc". "reasoning", if present, must
// be a string. Extra fields are ignored. Text outside the object, wrong
// types, unknown strategies and missing fields are errors.
func ParsePlan(raw string) (*QueryPlan, error) {
	body := unfence(strings.TrimSpace(raw))
	if body == "" {
		return nil, errors.New("empty response")
	}
	if !strings.HasPrefix(body, "{") {
		return nil, errors.New("response is not a JSON object")
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(body)))
	var p rawPlan
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("malformed JSON: %w", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("unexpected text after the JSON object")
	}
	if p.Strategy == nil {
		return nil, errors.New(`missing "strategy"`)
	}
	if !IsStrategy(*p.Strategy) {
		return nil, fmt.Errorf("unknown strategy %q", *p.Strategy)
	}
	if p.NeedsDoc == nil {
		return nil, errors.New(`missing boolean "needs_doc"`)
	}
	plan := &QueryPlan{Strategy: *p.Strategy, NeedsDoc: *p.NeedsDoc}
	if p.Reasoning != nil {
		plan.Reasoning = strings.TrimSpace(*p.Reasoning)
	}
	return plan, nil
}

// unfence removes one surrounding ```json ... ``` (or ``` ... ```) fence.
// Anything else is returned unchanged and later rejected if it is not JSON.
func unfence(s string) string {
	if !strings.HasPrefix(s, "```") || !strings.HasSuffix(s, "```") || len(s) < 6 {
		return s
	}
	inner := strings.TrimSuffix(strings.TrimPrefix(s, "```"), "```")
	inner = strings.TrimPrefix(inner, "json")
	return strings.TrimSpace(inner)
}
