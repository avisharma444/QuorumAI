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

// VLLMClient talks to a local vLLM server through its OpenAI-compatible API.
// vLLM runs as a local service, so no API key is sent.
type VLLMClient struct {
	apiURL string // full chat completions URL, e.g. http://vllm-inference:8001/v1/chat/completions
	model  string
	client *http.Client
}

// NewVLLMClient builds a client for a local vLLM server.
func NewVLLMClient(apiURL, model string, timeout time.Duration) *VLLMClient {
	return &VLLMClient{apiURL: apiURL, model: model, client: &http.Client{Timeout: timeout}}
}

// Generate sends a single user prompt.
func (c *VLLMClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{Messages: []Message{{Role: "user", Content: prompt}}})
}

// GenerateChat sends a full conversation to the vLLM server.
func (c *VLLMClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	reqBody, err := newOpenAIRequest(c.model, opts)
	if err != nil {
		return nil, fmt.Errorf("vLLM: %w", err)
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("vLLM: failed to marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("vLLM: failed to create request: %w", err)
	}
	body, err := postJSON(c.client, req, "vLLM")
	if err != nil {
		return nil, err
	}
	return parseOpenAIResponse("vLLM", body, c.model)
}

// ModelName returns the model prefixed with "local:".
func (c *VLLMClient) ModelName() string {
	return "local:" + c.model
}

// URL returns the chat completions endpoint this client uses.
func (c *VLLMClient) URL() string {
	return c.apiURL
}

// IsReady reports whether the vLLM server answers its /health endpoint.
func (c *VLLMClient) IsReady(ctx context.Context) bool {
	healthURL := strings.TrimSuffix(c.apiURL, "/v1/chat/completions") + "/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return false
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
