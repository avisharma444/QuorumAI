// Package llmtest provides a scriptable, recording llm.LLMClient for tests.
package llmtest

import (
	"context"
	"strings"
	"sync"

	"example.com/councilai-starter/internal/llm"
)

// StageOf maps a system prompt to the council stage that sent it. Revision
// reuses the chairman prompt (as in the reference), so use StageOfMessages
// when the user message is available.
func StageOf(system string) string { return StageOfMessages(system, "") }

// StageOfMessages identifies the stage from the system and user messages.
func StageOfMessages(system, user string) string {
	switch {
	case strings.HasPrefix(system, "You are a quality assurance agent"):
		return "reflect"
	case strings.HasPrefix(system, "You are the Chairman of an AI council") && strings.Contains(user, "Quality review feedback:"):
		return "revise"
	case strings.HasPrefix(system, "You are the Chairman of an AI council"):
		return "chair"
	case strings.HasPrefix(system, "REVIEW_AGENT"):
		return "review"
	case strings.HasPrefix(system, "COUNCIL_MEMBER"):
		return "member"
	case strings.HasPrefix(system, "DIRECT_AGENT"):
		return "direct"
	case strings.HasPrefix(system, "You are a query classifier"):
		return "route"
	case strings.HasPrefix(system, "You are a document summarization agent"):
		return "summarize"
	}
	return "unknown"
}

// Reply is a scripted result for one stage and model.
type Reply struct {
	Text  string
	Err   error
	Usage llm.Usage
	Model string // served model to report; defaults to the client name
}

// Fake hands out clients that share one script and call log. Replies are
// keyed by "stage/model"; unscripted calls fall back to llm.MockClient.
// Hook, if set before use, runs before each reply and can block or fail it.
type Fake struct {
	mu      sync.Mutex
	replies map[string]Reply
	calls   map[string]int
	prompts map[string]string
	options map[string]llm.GenerateOptions
	Hook    func(ctx context.Context, stage, model string) error
}

// New creates a Fake with the given script (nil is fine).
func New(replies map[string]Reply) *Fake {
	if replies == nil {
		replies = map[string]Reply{}
	}
	return &Fake{replies: replies, calls: map[string]int{}, prompts: map[string]string{}, options: map[string]llm.GenerateOptions{}}
}

// Client returns a client whose ModelName is name.
func (f *Fake) Client(name string) llm.LLMClient { return &client{f: f, name: name} }

// Clients returns one client per name.
func (f *Fake) Clients(names ...string) []llm.LLMClient {
	out := make([]llm.LLMClient, len(names))
	for i, n := range names {
		out[i] = f.Client(n)
	}
	return out
}

// Count returns how many calls a stage received.
func (f *Fake) Count(stage string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[stage]
}

// Calls returns a copy of the per-stage call counts.
func (f *Fake) Calls() map[string]int {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make(map[string]int, len(f.calls))
	for k, v := range f.calls {
		out[k] = v
	}
	return out
}

// Prompt returns the last user prompt a stage sent to model.
func (f *Fake) Prompt(stage, model string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.prompts[stage+"/"+model]
}

// Options returns the last GenerateOptions a stage sent to model.
func (f *Fake) Options(stage, model string) llm.GenerateOptions {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.options[stage+"/"+model]
}

type client struct {
	f    *Fake
	name string
}

func (c *client) ModelName() string { return c.name }

func (c *client) Generate(ctx context.Context, prompt string) (*llm.Response, error) {
	return c.GenerateChat(ctx, llm.GenerateOptions{Messages: []llm.Message{{Role: "user", Content: prompt}}})
}

func (c *client) GenerateChat(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
	var system, user []string
	for _, m := range opts.Messages {
		switch m.Role {
		case "system":
			system = append(system, m.Content)
		case "user":
			user = append(user, m.Content)
		}
	}
	stage := StageOfMessages(strings.Join(system, "\n"), strings.Join(user, "\n"))
	key := stage + "/" + c.name
	c.f.mu.Lock()
	c.f.calls[stage]++
	c.f.prompts[key] = strings.Join(user, "\n")
	c.f.options[key] = opts
	r, scripted := c.f.replies[key]
	hook := c.f.Hook
	c.f.mu.Unlock()
	if hook != nil {
		if err := hook(ctx, stage, c.name); err != nil {
			return nil, err
		}
	}
	if !scripted {
		return (&llm.MockClient{Name: c.name}).GenerateChat(ctx, opts)
	}
	if r.Err != nil {
		return nil, r.Err
	}
	model := r.Model
	if model == "" {
		model = c.name
	}
	return &llm.Response{Answer: r.Text, Model: model, Usage: r.Usage}, nil
}
