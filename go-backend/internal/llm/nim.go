package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// NIMClient talks to NVIDIA NIM microservices (OpenAI-compatible format,
// Bearer authentication).
type NIMClient struct {
	apiKey string
	apiURL string // full chat completions URL
	model  string
	client *http.Client
}

// NewNIMClient builds a client for one NIM model.
func NewNIMClient(apiKey, apiURL, model string, timeout time.Duration) *NIMClient {
	return &NIMClient{apiKey: apiKey, apiURL: apiURL, model: model, client: &http.Client{Timeout: timeout}}
}

// Generate sends a single user prompt.
func (c *NIMClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{Messages: []Message{{Role: "user", Content: prompt}}})
}

// GenerateChat sends a full conversation to NIM.
func (c *NIMClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	reqBody, err := newOpenAIRequest(c.model, opts)
	if err != nil {
		return nil, fmt.Errorf("NIM: %w", err)
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("NIM: failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("NIM: failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "application/json")
	body, err := postJSON(c.client, req, "NIM")
	if err != nil {
		return nil, err
	}
	return parseOpenAIResponse("NIM", body, c.model)
}

// ModelName returns the model prefixed with "nim:".
func (c *NIMClient) ModelName() string {
	return "nim:" + c.model
}
