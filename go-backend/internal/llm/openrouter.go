package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// OpenRouterClient talks to OpenRouter's OpenAI-compatible chat completions
// endpoint.
type OpenRouterClient struct {
	apiKey string
	apiURL string // full chat completions URL
	model  string
	client *http.Client
}

// NewOpenRouterClient builds a client for one OpenRouter model.
func NewOpenRouterClient(apiKey, apiURL, model string, timeout time.Duration) *OpenRouterClient {
	return &OpenRouterClient{apiKey: apiKey, apiURL: apiURL, model: model, client: &http.Client{Timeout: timeout}}
}

// ── OpenAI-compatible request/response types (shared with NIM and vLLM) ──

type openAIRequest struct {
	Model          string            `json:"model"`
	Messages       []openAIMessage   `json:"messages"`
	Temperature    *float64          `json:"temperature,omitempty"`
	MaxTokens      int               `json:"max_tokens,omitempty"`
	ResponseFormat *openAIRespFormat `json:"response_format,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRespFormat struct {
	Type string `json:"type"` // "json_object"
}

type openAIResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	// Some gateways (including OpenRouter) can report an error inside a
	// 200 response.
	Error *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// newOpenAIRequest maps GenerateOptions onto the OpenAI chat format. Roles
// map directly; EnableSearch has no OpenAI-compatible equivalent and is
// ignored, as in the reference.
func newOpenAIRequest(model string, opts GenerateOptions) (openAIRequest, error) {
	if len(opts.Messages) == 0 {
		return openAIRequest{}, errNoMessages
	}
	req := openAIRequest{Model: model, Temperature: opts.Temperature}
	for _, m := range opts.Messages {
		req.Messages = append(req.Messages, openAIMessage{Role: m.Role, Content: m.Content})
	}
	if opts.MaxTokens > 0 {
		req.MaxTokens = opts.MaxTokens
	}
	if opts.ResponseJSON {
		req.ResponseFormat = &openAIRespFormat{Type: "json_object"}
	}
	return req, nil
}

func parseOpenAIResponse(provider string, body []byte, configuredModel string) (*Response, error) {
	var r openAIResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("%s: failed to parse response: %w", provider, err)
	}
	if r.Error != nil {
		return nil, fmt.Errorf("%s: provider error: %s", provider, truncate(r.Error.Message, maxErrorBodyBytes))
	}
	if len(r.Choices) == 0 {
		return nil, fmt.Errorf("no choices in %s response", provider)
	}
	content := r.Choices[0].Message.Content
	if strings.TrimSpace(content) == "" {
		return nil, fmt.Errorf("empty content in %s response (finish_reason %q)", provider, r.Choices[0].FinishReason)
	}
	out := &Response{Answer: content, Model: configuredModel}
	if r.Model != "" {
		out.Model = r.Model
	}
	if r.Usage != nil {
		out.Usage = Usage{PromptTokens: r.Usage.PromptTokens, CompletionTokens: r.Usage.CompletionTokens, TotalTokens: r.Usage.TotalTokens}
	}
	return out, nil
}

// Generate sends a single user prompt.
func (c *OpenRouterClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{Messages: []Message{{Role: "user", Content: prompt}}})
}

// GenerateChat sends a full conversation to OpenRouter.
func (c *OpenRouterClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	reqBody, err := newOpenAIRequest(c.model, opts)
	if err != nil {
		return nil, fmt.Errorf("OpenRouter: %w", err)
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("OpenRouter: failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("OpenRouter: failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	// Optional OpenRouter app-attribution headers.
	req.Header.Set("HTTP-Referer", "https://localhost")
	req.Header.Set("X-Title", "CouncilAI Reproduction")
	body, err := postJSON(c.client, req, "OpenRouter")
	if err != nil {
		return nil, err
	}
	return parseOpenAIResponse("OpenRouter", body, c.model)
}

// ModelName returns the configured model. OpenRouter IDs are not prefixed,
// matching the reference.
func (c *OpenRouterClient) ModelName() string {
	return c.model
}
