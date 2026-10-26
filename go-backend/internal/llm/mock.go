package llm

import (
	"context"
	"regexp"
	"strings"
	"time"
)

// MockClient is an offline, deterministic client. It is used only when
// MOCK_LLM=true is set explicitly, and by tests. It never touches the network
// and is never used as a fallback for a failing real provider.
//
// It picks a reply from the stage tag at the start of the system message
// (COUNCIL_MEMBER, REVIEW_AGENT, CHAIRMAN_AGENT, REFLECTION_AGENT). It
// reports no token usage, so none is invented.
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
	return &Response{Answer: mockAnswer(m.ModelName(), sys, prompt), Model: m.ModelName()}, nil
}

func mockAnswer(model, system, prompt string) string {
	switch {
	case strings.Contains(system, "REVIEW_AGENT"):
		// Rank responses in the order shown. This proves the pipeline works
		// offline; it is not a quality judgment.
		labels := mockResponseLabel.FindAllStringSubmatch(prompt, -1)
		ranking := make([]string, len(labels))
		for i, m := range labels {
			ranking[i] = m[1]
		}
		return "RANKING: " + strings.Join(ranking, ", ") + "\nREASONING: Mock reviewer; deterministic order, not a quality judgment."
	case strings.Contains(system, "REFLECTION_AGENT"):
		return `{"quality":"good","reason":"mock validation only"}`
	case strings.Contains(system, "CHAIRMAN_AGENT"):
		return "[MOCK consensus] Representative council synthesis. Configure real providers to generate meaningful answers."
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
