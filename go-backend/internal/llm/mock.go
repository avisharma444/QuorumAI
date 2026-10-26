package llm

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// MockClient is an offline, deterministic client. It is used only when
// MOCK_LLM=true is set explicitly, and by tests. It never touches the network
// and is never used as a fallback for a failing real provider.
//
// It picks a reply from the system message (our stage tags COUNCIL_MEMBER,
// REVIEW_AGENT, DIRECT_AGENT, or the reference router, summarizer, chairman
// and reflection prompts). It reports no token usage and no confidence, so
// none is invented.
type MockClient struct {
	Name  string
	Delay time.Duration
}

var mockResponseLabel = regexp.MustCompile(`(?m)^=== Response ([A-Z]) ===$`)

// Generate sends a single user prompt.
func (m *MockClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return m.GenerateChat(ctx, GenerateOptions{Messages: []Message{{Role: "user", Content: prompt}}})
}

// GenerateChat returns a canned, clearly labelled reply.
func (m *MockClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	if m.Delay > 0 {
		select {
		case <-time.After(m.Delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var system, user []string
	for _, msg := range opts.Messages {
		switch msg.Role {
		case "system":
			system = append(system, msg.Content)
		case "user":
			user = append(user, msg.Content)
		}
	}
	sys := strings.Join(system, "\n")
	prompt := strings.Join(user, "\n")
	// "#fail=<name>" in the question makes that mock model's requests fail
	// (failure injection for observability checks; mock mode only).
	if strings.Contains(prompt, "#fail="+m.Name) {
		return nil, fmt.Errorf("mock failure injected for %s", m.ModelName())
	}
	return &Response{Answer: mockAnswer(m.ModelName(), sys, prompt), Model: m.ModelName()}, nil
}

// Mock routing directives, written in the question text (mock mode only):
//
//	#route=direct|council|council_deep   strategy the mock router returns
//	#route=invalid                       malformed output, to exercise the router fallback
//	#needs_doc=true|false                document decision the mock router returns
//
// Without directives the mock router returns "council", with needs_doc true
// only when a document summary is supplied. Real routing always uses the
// configured router model.
var (
	mockRouteDirective    = regexp.MustCompile(`#route=([a-z_]+)`)
	mockNeedsDocDirective = regexp.MustCompile(`#needs_doc=(true|false)`)
)

func mockRoute(prompt string) string {
	strategy, reasoning := "council", "Mock router default; not a real classification."
	if m := mockRouteDirective.FindStringSubmatch(prompt); m != nil {
		if m[1] == "invalid" {
			return "I think this needs the council."
		}
		strategy, reasoning = m[1], "Mock router: strategy taken from #route directive."
	}
	needsDoc := strings.Contains(prompt, "Attached Document Summary:")
	if m := mockNeedsDocDirective.FindStringSubmatch(prompt); m != nil {
		needsDoc = m[1] == "true"
	}
	return fmt.Sprintf(`{"strategy":%q,"reasoning":%q,"needs_doc":%t}`, strategy, reasoning, needsDoc)
}

// Mock reflection directives (mock mode only), written in the question:
//
//	#reflect=needs_revision   audit asks for one revision
//	#reflect=invalid          malformed audit (exercises reflection failure)
//	#revise=invalid           malformed revision (exercises revision failure)
//	#chair=invalid            malformed chairman output (exercises the fallback)
//	#fail=member-2            that mock model's requests fail (any mock name)
//
// Without directives the mock audit returns "good". It never reports a
// confidence value.
func mockReflection(prompt string) string {
	switch {
	case strings.Contains(prompt, "#reflect=invalid"):
		return `{"quality":"excellent"}`
	case strings.Contains(prompt, "#reflect=needs_revision"):
		return `{"quality":"needs_revision","issues":"Mock audit: placeholder issue.","suggestion":"Mock audit: placeholder suggestion.","faithful":false}`
	}
	return `{"quality":"good","issues":"","suggestion":"","faithful":true}`
}

func mockAnswer(model, system, prompt string) string {
	switch {
	case strings.Contains(system, "query classifier for a multi-LLM council"):
		return mockRoute(prompt)
	case strings.Contains(system, "document summarization agent"):
		sample := strings.Join(strings.Fields(strings.TrimPrefix(prompt, "Document Sample:")), " ")
		if len(sample) > 120 {
			sample = sample[:120]
		}
		return "[MOCK summary] Placeholder summary from the offline mock model, not a real summary. Sample begins: " + sample
	case strings.Contains(system, "REVIEW_AGENT"):
		// Rank responses in the order shown. This proves the pipeline works
		// offline; it is not a quality judgment.
		labels := mockResponseLabel.FindAllStringSubmatch(prompt, -1)
		ranking := make([]string, len(labels))
		for i, m := range labels {
			ranking[i] = m[1]
		}
		return "RANKING: " + strings.Join(ranking, ", ") + "\nREASONING: Mock reviewer; deterministic order, not a quality judgment."
	case strings.Contains(system, "quality assurance agent"):
		return mockReflection(prompt)
	case strings.Contains(system, "Chairman of an AI council") && strings.Contains(prompt, "Quality review feedback:"):
		if strings.Contains(prompt, "#revise=invalid") {
			return "Here is my improved answer, without JSON."
		}
		return `{"answer":"[MOCK revised] Representative revised synthesis addressing the review feedback.","reasoning":"Mock revision; not a real model judgment.","source":"chairman-synthesis"}`
	case strings.Contains(system, "Chairman of an AI council"):
		if strings.Contains(prompt, "#chair=invalid") {
			return `{"answer": "unterminated`
		}
		return `{"answer":"[MOCK consensus] Representative council synthesis. Configure real providers to generate meaningful answers.","reasoning":"Mock synthesis; not a real model judgment.","source":"chairman-synthesis"}`
	default:
		short := prompt
		if len(short) > 180 {
			short = short[:180]
		}
		return "[MOCK " + model + "] Candidate response for: " + short
	}
}

// ModelName returns the name prefixed with "mock:".
func (m *MockClient) ModelName() string {
	return "mock:" + m.Name
}
