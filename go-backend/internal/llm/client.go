// Package llm defines the client interface and provider adapters for Gemini,
// OpenRouter, NVIDIA NIM, and local vLLM servers.
package llm

import "context"

// Message is a single message in a conversation.
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// DefaultMaxOutputTokens is the output cap for council calls that set none
// (LLM_MAX_OUTPUT_TOKENS overrides it). An uncapped call can bill thousands
// of output tokens on a paid model.
const DefaultMaxOutputTokens = 1024

// GenerateOptions provides full control over one generation call.
type GenerateOptions struct {
	Messages     []Message // system + user + assistant turns
	Temperature  *float64  // nil = provider default
	MaxTokens    int       // 0 = provider default
	ResponseJSON bool      // request JSON output if the provider supports it
	EnableSearch bool      // live web-search grounding if the provider supports it (Gemini)
}

// Response is the normalized result of a generation call.
type Response struct {
	Answer string `json:"answer"`
	// Model is the model the provider says served the request, or the
	// configured model when the provider does not say.
	Model string `json:"model"`
	// Confidence is part of the reference interface; no provider reports it,
	// so adapters leave it at zero rather than inventing a value.
	Confidence float64 `json:"confidence,omitempty"`
	// Usage holds provider-reported token counts. All zero means the
	// provider did not report usage; see Usage.Reported.
	Usage Usage `json:"usage"`
}

// Usage tracks token consumption for a single call, exactly as reported.
// CompletionTokens counts all billed output, including reasoning/thinking
// tokens. CostCredits is the charge the provider itself reported (OpenRouter
// reports usage.cost in its credits); 0 means not reported.
type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	CostCredits      float64 `json:"cost_credits,omitempty"`
}

// Reported is false when the provider sent no usage numbers at all.
func (u Usage) Reported() bool {
	return u.PromptTokens != 0 || u.CompletionTokens != 0 || u.TotalTokens != 0
}

// LLMClient is implemented by every provider adapter.
type LLMClient interface {
	// Generate sends a single user prompt. It wraps GenerateChat.
	Generate(ctx context.Context, prompt string) (*Response, error)

	// GenerateChat sends a multi-message conversation with full control over
	// system prompts, temperature, and output format.
	GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error)

	// ModelName returns the model identifier, provider-prefixed for Gemini
	// ("gemini:"), NIM ("nim:") and vLLM ("local:"), as in the reference.
	ModelName() string
}

// Float64Ptr creates a *float64 for GenerateOptions.Temperature.
func Float64Ptr(v float64) *float64 {
	return &v
}
