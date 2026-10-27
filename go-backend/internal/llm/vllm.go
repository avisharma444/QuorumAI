package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// Endpoint kinds served by VLLMClient.
const (
	VLLMEndpointLocal  = "vllm"   // a vLLM server's own OpenAI API (has /health)
	VLLMEndpointRunPod = "runpod" // RunPod Serverless vLLM worker, OpenAI route
)

// runPodOpenAIPath matches RunPod's OpenAI-compatible chat route:
// /v2/<ENDPOINT_ID>/openai/v1/chat/completions.
var runPodOpenAIPath = regexp.MustCompile(`^/v2/([A-Za-z0-9_-]+)/openai/v1/chat/completions$`)

// VLLMClient talks to a vLLM server through its OpenAI-compatible API: a
// self-hosted server (no credentials) or a RunPod Serverless endpoint
// (bearer API key). The key is sent only in the Authorization header and is
// never logged or included in errors.
type VLLMClient struct {
	apiURL     string // full chat completions URL
	model      string
	apiKey     string
	kind       string
	endpointID string // RunPod endpoint ID (not a secret)
	healthURL  string
	client     *http.Client
}

// NewVLLMClient builds a client for a self-hosted vLLM server without
// credentials (the original behaviour).
func NewVLLMClient(apiURL, model string, timeout time.Duration) *VLLMClient {
	c, err := NewVLLMClientWithKey(apiURL, "", model, timeout)
	if err != nil { // only possible with a key or a malformed RunPod URL
		return &VLLMClient{apiURL: apiURL, model: model, kind: VLLMEndpointLocal,
			healthURL: strings.TrimSuffix(apiURL, "/v1/chat/completions") + "/health", client: &http.Client{Timeout: timeout}}
	}
	return c
}

// NewVLLMClientWithKey builds a client that sends apiKey as a bearer token
// when it is non-empty. A key is refused over plain HTTP except to loopback
// hosts, so it can never cross a network unencrypted. A URL on
// api.runpod.ai must be the OpenAI chat route.
func NewVLLMClientWithKey(apiURL, apiKey, model string, timeout time.Duration) (*VLLMClient, error) {
	u, err := url.Parse(apiURL)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, fmt.Errorf("vLLM: invalid endpoint URL for model %s", model)
	}
	if u.User != nil || u.RawQuery != "" {
		return nil, fmt.Errorf("vLLM: endpoint URL for model %s must not contain credentials or a query string", model)
	}
	if apiKey != "" && u.Scheme != "https" && !isLoopback(u.Hostname()) {
		return nil, fmt.Errorf("vLLM: refusing to send an API key over plain HTTP to %s (model %s); use https", u.Hostname(), model)
	}
	c := &VLLMClient{apiURL: apiURL, model: model, apiKey: apiKey, kind: VLLMEndpointLocal, client: &http.Client{Timeout: timeout}}
	if m := runPodOpenAIPath.FindStringSubmatch(u.Path); m != nil {
		c.kind, c.endpointID = VLLMEndpointRunPod, m[1]
		c.healthURL = (&url.URL{Scheme: u.Scheme, Host: u.Host, Path: "/v2/" + m[1] + "/health"}).String()
		if apiKey == "" {
			return nil, fmt.Errorf("vLLM: RunPod endpoint %s needs an API key (set VLLM_API_KEY or %s)", m[1], VLLMKeyEnvKey(model))
		}
	} else {
		if strings.EqualFold(u.Hostname(), "api.runpod.ai") {
			return nil, fmt.Errorf("vLLM: RunPod URL for model %s must be https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions", model)
		}
		c.healthURL = strings.TrimSuffix(apiURL, "/v1/chat/completions") + "/health"
	}
	return c, nil
}

func isLoopback(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
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
	c.authorize(req)
	body, err := postJSON(c.client, req, "vLLM")
	if err != nil {
		return nil, err
	}
	return parseOpenAIResponse("vLLM", body, c.model)
}

func (c *VLLMClient) authorize(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

// ModelName returns the model prefixed with "local:" (the vLLM provider).
func (c *VLLMClient) ModelName() string {
	return "local:" + c.model
}

// URL returns the chat completions endpoint this client uses.
func (c *VLLMClient) URL() string {
	return c.apiURL
}

// Describe summarises the endpoint for logs without credentials.
func (c *VLLMClient) Describe() string {
	auth := "no credentials"
	if c.apiKey != "" {
		auth = "bearer auth"
	}
	if c.kind == VLLMEndpointRunPod {
		return fmt.Sprintf("vLLM on RunPod Serverless endpoint %s, %s", c.endpointID, auth)
	}
	host := c.apiURL
	if u, err := url.Parse(c.apiURL); err == nil {
		host = u.Host
	}
	return fmt.Sprintf("vLLM server %s, %s", host, auth)
}

// Readiness is the result of one readiness probe.
type Readiness struct {
	Kind  string `json:"kind"`
	Ready bool   `json:"ready"`
	// Detail explains the state, e.g. "scaled to zero" or "unauthorized".
	Detail string `json:"detail"`
	// RunPod only (nil when not reported): workers and queued jobs.
	WorkersIdle    *int `json:"workers_idle,omitempty"`
	WorkersRunning *int `json:"workers_running,omitempty"`
	JobsInQueue    *int `json:"jobs_in_queue,omitempty"`
	JobsInProgress *int `json:"jobs_in_progress,omitempty"`
}

// Readiness probes the endpoint. A self-hosted server must answer vLLM's
// /health with 200. RunPod does not expose that route: its endpoint-level
// /v2/<id>/health reports workers and jobs (it does not start a worker).
// "Ready" there means the endpoint exists and accepts the key; an endpoint
// scaled to zero is ready, and its first request cold-starts a worker.
func (c *VLLMClient) Readiness(ctx context.Context) Readiness {
	r := Readiness{Kind: c.kind}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.healthURL, nil)
	if err != nil {
		r.Detail = "invalid health URL"
		return r
	}
	c.authorize(req)
	resp, err := c.client.Do(req)
	if err != nil {
		r.Detail = "unreachable: " + transportReason(err)
		return r
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	switch {
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		r.Detail = fmt.Sprintf("unauthorized (HTTP %d): check the API key's access to this endpoint", resp.StatusCode)
		return r
	case resp.StatusCode == http.StatusNotFound:
		r.Detail = "not found (HTTP 404)"
		return r
	case resp.StatusCode != http.StatusOK:
		r.Detail = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return r
	}
	r.Ready = true
	if c.kind != VLLMEndpointRunPod {
		r.Detail = "vLLM /health OK"
		return r
	}
	var h struct {
		Jobs struct {
			InQueue    *int `json:"inQueue"`
			InProgress *int `json:"inProgress"`
		} `json:"jobs"`
		Workers struct {
			Idle    *int `json:"idle"`
			Running *int `json:"running"`
		} `json:"workers"`
	}
	if err := json.Unmarshal(body, &h); err != nil {
		r.Detail = "endpoint reachable; health body not understood"
		return r
	}
	r.WorkersIdle, r.WorkersRunning = h.Workers.Idle, h.Workers.Running
	r.JobsInQueue, r.JobsInProgress = h.Jobs.InQueue, h.Jobs.InProgress
	if deref(h.Workers.Idle)+deref(h.Workers.Running) == 0 {
		r.Detail = "endpoint reachable, scaled to zero: the first request cold-starts a worker"
	} else {
		r.Detail = "endpoint reachable, worker available"
	}
	return r
}

// IsReady reports Readiness(ctx).Ready.
func (c *VLLMClient) IsReady(ctx context.Context) bool {
	return c.Readiness(ctx).Ready
}

func deref(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func transportReason(err error) string {
	var ne net.Error
	switch {
	case errors.Is(err, context.DeadlineExceeded), errors.As(err, &ne) && ne.Timeout():
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	}
	return "connection failed"
}
