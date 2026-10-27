package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// A RunPod-shaped route on a loopback test server (plain HTTP is allowed to
// loopback only).
const rpPath = "/v2/ep-test_123/openai/v1/chat/completions"

func TestRemoteVLLMSendsBearerTokenAndParsesUsage(t *testing.T) {
	srv := newFakeProvider(t, 200, openAIOK)
	c, err := NewVLLMClientWithKey(srv.URL+rpPath, testKey, "microsoft/Phi-4-mini-instruct", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.GenerateChat(context.Background(), fullOpts)
	if err != nil {
		t.Fatal(err)
	}
	req := srv.last(t)
	if req.Path != rpPath || req.Header.Get("Authorization") != "Bearer "+testKey {
		t.Fatalf("path %s auth %q", req.Path, req.Header.Get("Authorization"))
	}
	if req.Body["model"] != "microsoft/Phi-4-mini-instruct" || req.Body["max_tokens"] != 256.0 {
		t.Fatalf("body %v", req.Body)
	}
	if resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 || resp.Model != "served/model-v2" {
		t.Fatalf("usage/model %+v", resp)
	}
	if c.ModelName() != "local:microsoft/Phi-4-mini-instruct" {
		t.Fatal(c.ModelName())
	}
	if d := c.Describe(); !strings.Contains(d, "RunPod Serverless endpoint ep-test_123") || strings.Contains(d, testKey) {
		t.Fatalf("describe %q", d)
	}
}

func TestRemoteVLLMURLRules(t *testing.T) {
	cases := []struct {
		url, key, wantErr string
	}{
		{"http://gpu.example.com/v1/chat/completions", testKey, "plain HTTP"},
		{"http://api.runpod.ai" + rpPath, testKey, "plain HTTP"},
		{"https://api.runpod.ai/v2/abc/runsync", testKey, "must be https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions"},
		{"https://api.runpod.ai" + rpPath, "", "needs an API key"},
		{"https://user:pw@gpu.example.com/v1/chat/completions", "", "must not contain credentials"},
		{"https://gpu.example.com/v1/chat/completions?api_key=x", "", "must not contain credentials"},
		{"not a url", "", "invalid endpoint URL"},
	}
	for _, tc := range cases {
		_, err := NewVLLMClientWithKey(tc.url, tc.key, "m", time.Second)
		if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
			t.Errorf("%s: want %q, got %v", tc.url, tc.wantErr, err)
		}
		if err != nil && strings.Contains(err.Error(), testKey) {
			t.Errorf("key leaked: %v", err)
		}
	}
	for _, ok := range []struct{ url, key string }{
		{"https://api.runpod.ai" + rpPath, testKey},
		{"https://gpu.example.com/v1/chat/completions", testKey},
		{"http://localhost:8001/v1/chat/completions", testKey},
		{"http://127.0.0.1:8001/v1/chat/completions", testKey},
		{"http://vllm-inference:8001/v1/chat/completions", ""}, // compose service, no key: unchanged behaviour
	} {
		if _, err := NewVLLMClientWithKey(ok.url, ok.key, "m", time.Second); err != nil {
			t.Errorf("%s: %v", ok.url, err)
		}
	}
}

func TestRunPodReadinessUsesEndpointHealthNotVLLMHealth(t *testing.T) {
	var paths []string
	var auth atomic.Value
	status, body := 200, `{"jobs":{"completed":4,"failed":0,"inProgress":0,"inQueue":0,"retried":0},"workers":{"idle":0,"running":0}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		auth.Store(r.Header.Get("Authorization"))
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()
	c, err := NewVLLMClientWithKey(srv.URL+rpPath, testKey, "m", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	r := c.Readiness(context.Background())
	if !r.Ready || r.Kind != VLLMEndpointRunPod || !strings.Contains(r.Detail, "scaled to zero") || r.WorkersRunning == nil || *r.WorkersRunning != 0 {
		t.Fatalf("%+v", r)
	}
	if paths[0] != "GET /v2/ep-test_123/health" || auth.Load() != "Bearer "+testKey {
		t.Fatalf("probe %v auth %v", paths, auth.Load())
	}
	body = `{"jobs":{"inQueue":2,"inProgress":1},"workers":{"idle":0,"running":1}}`
	r = c.Readiness(context.Background())
	if !r.Ready || *r.JobsInQueue != 2 || *r.JobsInProgress != 1 || !strings.Contains(r.Detail, "worker available") {
		t.Fatalf("%+v", r)
	}
	for code, want := range map[int]string{401: "unauthorized", 403: "unauthorized", 404: "not found", 500: "HTTP 500"} {
		status = code
		if r := c.Readiness(context.Background()); r.Ready || !strings.Contains(r.Detail, want) || strings.Contains(r.Detail, testKey) {
			t.Fatalf("%d: %+v", code, r)
		}
	}
	if c.IsReady(context.Background()) {
		t.Fatal("IsReady must follow Readiness")
	}
	dead, _ := NewVLLMClientWithKey("http://127.0.0.1:1"+rpPath, testKey, "m", time.Second)
	if r := dead.Readiness(context.Background()); r.Ready || !strings.HasPrefix(r.Detail, "unreachable") {
		t.Fatalf("%+v", r)
	}
}

func TestVLLMKeyOverridesAndNoLeaks(t *testing.T) {
	model := "microsoft/Phi-4-mini-instruct"
	if VLLMKeyEnvKey(model) != "VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_API_KEY" {
		t.Fatal(VLLMKeyEnvKey(model))
	}
	srv := newFakeProvider(t, 200, openAIOK)
	t.Setenv(VLLMURLEnvKey(model), srv.URL+rpPath)
	c, err := NewClientFromProvider("local", model, ProviderKeys{VLLM: "global-key-AAAA"}, allURLs, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Generate(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if got := srv.last(t).Header.Get("Authorization"); got != "Bearer global-key-AAAA" {
		t.Fatalf("global key: %q", got)
	}
	t.Setenv(VLLMKeyEnvKey(model), "per-model-key-BBBB")
	c, _ = NewClientFromProvider("local", model, ProviderKeys{VLLM: "global-key-AAAA"}, allURLs, time.Second)
	_, _ = c.Generate(context.Background(), "hi")
	if got := srv.last(t).Header.Get("Authorization"); got != "Bearer per-model-key-BBBB" {
		t.Fatalf("per-model key: %q", got)
	}
	// A RunPod URL without any key is a configuration error that names the variables, not values.
	t.Setenv(VLLMKeyEnvKey(model), "")
	_, err = NewClientFromProvider("local", model, ProviderKeys{}, allURLs, time.Second)
	if err == nil || !strings.Contains(err.Error(), "VLLM_API_KEY") {
		t.Fatalf("missing key: %v", err)
	}
	// Provider errors never contain the key.
	bad := newFakeProvider(t, 401, `{"error":"Unauthorized"}`)
	c, _ = NewVLLMClientWithKey(bad.URL+rpPath, testKey, model, time.Second)
	_, err = c.Generate(context.Background(), "hi")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 || strings.Contains(err.Error(), testKey) {
		t.Fatalf("401: %v", err)
	}
}

func TestRemoteVLLMTimeoutAndCancellationReachTheServer(t *testing.T) {
	gone := make(chan struct{}, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body) // the server notices a disconnect only after the body is read
		<-r.Context().Done()               // a cold-starting or stuck worker
		gone <- struct{}{}
	}))
	defer srv.Close()

	c, _ := NewVLLMClientWithKey(srv.URL+rpPath, testKey, "m", 150*time.Millisecond)
	start := time.Now()
	_, err := c.Generate(context.Background(), "hi")
	var ne interface{ Timeout() bool }
	if err == nil || !errors.As(err, &ne) || !ne.Timeout() || time.Since(start) > 2*time.Second {
		t.Fatalf("client timeout: %v after %s", err, time.Since(start))
	}

	c, _ = NewVLLMClientWithKey(srv.URL+rpPath, testKey, "m", time.Minute)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(100 * time.Millisecond); cancel() }()
	if _, err := c.Generate(ctx, "hi"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	for i := 0; i < 2; i++ {
		select {
		case <-gone: // the connection was closed, so the server saw both requests end
		case <-time.After(2 * time.Second):
			t.Fatal("server never observed the client giving up")
		}
	}
}
