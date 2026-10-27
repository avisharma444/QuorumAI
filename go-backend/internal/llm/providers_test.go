package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// testKey is a fake credential. Tests assert it never appears in errors.
const testKey = "sk-test-SECRET-0123456789"

// captured is one request a fake provider received.
type captured struct {
	Method string
	Path   string
	Query  string
	Header http.Header
	Body   map[string]any
}

// fakeProvider is an httptest server that records requests and answers with
// a fixed status and body.
type fakeProvider struct {
	*httptest.Server
	mu       sync.Mutex
	requests []captured
	status   int
	body     string
}

func newFakeProvider(t *testing.T, status int, body string) *fakeProvider {
	t.Helper()
	f := &fakeProvider{status: status, body: body}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var decoded map[string]any
		_ = json.Unmarshal(raw, &decoded)
		f.mu.Lock()
		f.requests = append(f.requests, captured{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Clone(), decoded})
		f.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(f.status)
		_, _ = io.WriteString(w, f.body)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeProvider) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

func (f *fakeProvider) last(t *testing.T) captured {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		t.Fatal("provider received no request")
	}
	return f.requests[len(f.requests)-1]
}

// providerCase builds each adapter against a test server URL.
type providerCase struct {
	name     string
	gemini   bool
	newAt    func(url string, timeout time.Duration) LLMClient
	wantName string
}

var allProviders = []providerCase{
	{"openrouter", false, func(u string, to time.Duration) LLMClient {
		return NewOpenRouterClient(testKey, u+"/api/v1/chat/completions", "meta/llama-test", to)
	}, "meta/llama-test"},
	{"gemini", true, func(u string, to time.Duration) LLMClient {
		return NewGeminiClientWithURL(testKey, u+"/v1beta", "gemini-test", to)
	}, "gemini:gemini-test"},
	{"nvidia-nim", false, func(u string, to time.Duration) LLMClient {
		return NewNIMClient(testKey, u+"/v1/chat/completions", "nvidia/nim-test", to)
	}, "nim:nvidia/nim-test"},
	{"vllm", false, func(u string, to time.Duration) LLMClient {
		return NewVLLMClient(u+"/v1/chat/completions", "phi-test", to)
	}, "local:phi-test"},
}

const openAIOK = `{"id":"x","model":"served/model-v2","choices":[{"message":{"role":"assistant","content":"hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`
const geminiOK = `{"candidates":[{"content":{"parts":[{"text":"hello there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7,"totalTokenCount":18},"modelVersion":"gemini-test-001"}`

func okBody(p providerCase) string {
	if p.gemini {
		return geminiOK
	}
	return openAIOK
}

var fullOpts = GenerateOptions{
	Messages: []Message{
		{Role: "system", Content: "be terse"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
		{Role: "user", Content: "again"},
	},
	Temperature:  Float64Ptr(0.3),
	MaxTokens:    256,
	ResponseJSON: true,
}

// ── Request mapping ─────────────────────────────────────────────────

func TestOpenAICompatibleRequestSerialization(t *testing.T) {
	for _, p := range allProviders {
		if p.gemini {
			continue
		}
		t.Run(p.name, func(t *testing.T) {
			srv := newFakeProvider(t, 200, openAIOK)
			if _, err := p.newAt(srv.URL, time.Second).GenerateChat(context.Background(), fullOpts); err != nil {
				t.Fatal(err)
			}
			req := srv.last(t)
			if req.Method != http.MethodPost || !strings.HasSuffix(req.Path, "/chat/completions") {
				t.Fatalf("wrong endpoint %s %s", req.Method, req.Path)
			}
			wantModel := strings.TrimPrefix(strings.TrimPrefix(p.wantName, "nim:"), "local:")
			if req.Body["model"] != wantModel || req.Body["temperature"] != 0.3 || req.Body["max_tokens"] != 256.0 {
				t.Fatalf("body %v", req.Body)
			}
			if rf, _ := req.Body["response_format"].(map[string]any); rf["type"] != "json_object" {
				t.Fatalf("JSON mode not requested: %v", req.Body)
			}
			msgs, _ := json.Marshal(req.Body["messages"])
			want := `[{"content":"be terse","role":"system"},{"content":"hi","role":"user"},{"content":"hello","role":"assistant"},{"content":"again","role":"user"}]`
			if string(msgs) != want {
				t.Fatalf("messages %s", msgs)
			}
			auth := req.Header.Get("Authorization")
			if p.name == "vllm" {
				if auth != "" {
					t.Fatalf("vLLM must not send credentials, got %q", auth)
				}
			} else if auth != "Bearer "+testKey {
				t.Fatalf("auth header %q", auth)
			}
			if p.name == "openrouter" && req.Header.Get("X-Title") == "" {
				t.Fatal("missing OpenRouter attribution header")
			}
		})
	}
}

func TestOpenAICompatibleOmitsUnsetOptions(t *testing.T) {
	srv := newFakeProvider(t, 200, openAIOK)
	c := NewOpenRouterClient(testKey, srv.URL, "m", time.Second)
	if _, err := c.Generate(context.Background(), "just this"); err != nil {
		t.Fatal(err)
	}
	body := srv.last(t).Body
	for _, k := range []string{"temperature", "max_tokens", "response_format"} {
		if _, ok := body[k]; ok {
			t.Fatalf("%s must be omitted so the provider default applies: %v", k, body)
		}
	}
	msgs, _ := json.Marshal(body["messages"])
	if string(msgs) != `[{"content":"just this","role":"user"}]` {
		t.Fatalf("Generate must send one user message, got %s", msgs)
	}
}

func TestGeminiRequestMapping(t *testing.T) {
	srv := newFakeProvider(t, 200, geminiOK)
	opts := fullOpts
	opts.Messages = append([]Message{{Role: "system", Content: "second rule"}}, opts.Messages...)
	opts.EnableSearch = true
	if _, err := NewGeminiClientWithURL(testKey, srv.URL+"/v1beta", "gemini-test", time.Second).GenerateChat(context.Background(), opts); err != nil {
		t.Fatal(err)
	}
	req := srv.last(t)
	if req.Path != "/v1beta/models/gemini-test:generateContent" {
		t.Fatalf("path %s", req.Path)
	}
	if req.Header.Get("x-goog-api-key") != testKey || strings.Contains(req.Query, testKey) || req.Header.Get("Authorization") != "" {
		t.Fatalf("key must travel only in x-goog-api-key: query=%q", req.Query)
	}
	got, _ := json.Marshal(req.Body)
	want := `{"contents":[{"parts":[{"text":"hi"}],"role":"user"},{"parts":[{"text":"hello"}],"role":"model"},{"parts":[{"text":"again"}],"role":"user"}],` +
		`"generationConfig":{"maxOutputTokens":256,"responseMimeType":"application/json","temperature":0.3},` +
		`"systemInstruction":{"parts":[{"text":"second rule"},{"text":"be terse"}]},"tools":[{"googleSearch":{}}]}`
	if string(got) != want {
		t.Fatalf("gemini body\n got %s\nwant %s", got, want)
	}
}

func TestGeminiOmitsUnsetOptions(t *testing.T) {
	srv := newFakeProvider(t, 200, geminiOK)
	if _, err := NewGeminiClientWithURL(testKey, srv.URL, "g", time.Second).Generate(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	body := srv.last(t).Body
	for _, k := range []string{"generationConfig", "systemInstruction", "tools"} {
		if _, ok := body[k]; ok {
			t.Fatalf("%s must be omitted: %v", k, body)
		}
	}
}

func TestGeminiRejectsInvalidConversations(t *testing.T) {
	c := NewGeminiClientWithURL(testKey, "http://127.0.0.1:1", "g", time.Second)
	for name, msgs := range map[string][]Message{
		"system only":  {{Role: "system", Content: "x"}},
		"no messages":  nil,
		"unknown role": {{Role: "tool", Content: "x"}},
	} {
		if _, err := c.GenerateChat(context.Background(), GenerateOptions{Messages: msgs}); err == nil {
			t.Fatalf("%s: expected error", name)
		}
	}
}

func TestEmptyConversationFailsBeforeNetwork(t *testing.T) {
	for _, p := range allProviders {
		srv := newFakeProvider(t, 200, okBody(p))
		if _, err := p.newAt(srv.URL, time.Second).GenerateChat(context.Background(), GenerateOptions{}); err == nil {
			t.Fatalf("%s: expected error", p.name)
		}
		if srv.count() != 0 {
			t.Fatalf("%s: request sent for an empty conversation", p.name)
		}
	}
}

// ── Response parsing ────────────────────────────────────────────────

func TestSuccessfulResponseParsing(t *testing.T) {
	for _, p := range allProviders {
		t.Run(p.name, func(t *testing.T) {
			srv := newFakeProvider(t, 200, okBody(p))
			c := p.newAt(srv.URL, time.Second)
			resp, err := c.Generate(context.Background(), "hi")
			if err != nil {
				t.Fatal(err)
			}
			wantServed := "served/model-v2"
			if p.gemini {
				wantServed = "gemini-test-001"
			}
			if resp.Answer != "hello there" || resp.Model != wantServed || resp.Confidence != 0 {
				t.Fatalf("resp %+v", resp)
			}
			if resp.Usage != (Usage{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18}) || !resp.Usage.Reported() {
				t.Fatalf("usage %+v", resp.Usage)
			}
			if c.ModelName() != p.wantName {
				t.Fatalf("ModelName %q, want %q", c.ModelName(), p.wantName)
			}
		})
	}
}

func TestMissingUsageIsNotInvented(t *testing.T) {
	bodies := map[bool]string{
		false: `{"choices":[{"message":{"content":"ok"}}]}`,
		true:  `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`,
	}
	for _, p := range allProviders {
		srv := newFakeProvider(t, 200, bodies[p.gemini])
		resp, err := p.newAt(srv.URL, time.Second).Generate(context.Background(), "hi")
		if err != nil {
			t.Fatalf("%s: %v", p.name, err)
		}
		if resp.Usage.Reported() {
			t.Fatalf("%s: invented usage %+v", p.name, resp.Usage)
		}
		// Without a served-model field the configured model is reported.
		if resp.Model == "" {
			t.Fatalf("%s: empty model", p.name)
		}
	}
}

func TestGeminiJoinsTextPartsAndSkipsThoughts(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"internal","thought":true},{"text":"Part one. "},{"text":"Part two."}]}}]}`
	srv := newFakeProvider(t, 200, body)
	resp, err := NewGeminiClientWithURL(testKey, srv.URL, "g", time.Second).Generate(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Answer != "Part one. Part two." {
		t.Fatalf("answer %q", resp.Answer)
	}
}

func TestMalformedAndEmptyResponses(t *testing.T) {
	openAIBad := map[string]string{
		"not JSON":             `<html>oops</html>`,
		"no choices":           `{"choices":[]}`,
		"empty content":        `{"choices":[{"message":{"content":"   "},"finish_reason":"length"}]}`,
		"error inside 200":     `{"error":{"message":"upstream overloaded","code":502}}`,
		"error beside choices": `{"choices":[{"message":{"content":"partial"}}],"error":{"message":"upstream overloaded"}}`,
		"empty object":         `{}`,
		"truncated JSON body":  `{"choices":[{"message":`,
	}
	geminiBad := map[string]string{
		"not JSON":         `<html>oops</html>`,
		"no candidates":    `{"candidates":[]}`,
		"blocked prompt":   `{"promptFeedback":{"blockReason":"SAFETY"}}`,
		"no text parts":    `{"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}]}`,
		"only thoughts":    `{"candidates":[{"content":{"parts":[{"text":"x","thought":true}]}}]}`,
		"empty object":     `{}`,
		"whitespace parts": `{"candidates":[{"content":{"parts":[{"text":"  "}]}}]}`,
	}
	for _, p := range allProviders {
		cases := openAIBad
		if p.gemini {
			cases = geminiBad
		}
		for name, body := range cases {
			t.Run(p.name+"/"+name, func(t *testing.T) {
				srv := newFakeProvider(t, 200, body)
				resp, err := p.newAt(srv.URL, time.Second).Generate(context.Background(), "hi")
				if err == nil {
					t.Fatalf("expected error, got %+v", resp)
				}
			})
		}
	}
	srv := newFakeProvider(t, 200, `{"promptFeedback":{"blockReason":"SAFETY"}}`)
	_, err := NewGeminiClientWithURL(testKey, srv.URL, "g", time.Second).Generate(context.Background(), "hi")
	if !strings.Contains(err.Error(), "SAFETY") {
		t.Fatalf("block reason should be reported: %v", err)
	}
}

// ── Errors, timeouts, cancellation, secrets ─────────────────────────

func TestHTTPErrorStatuses(t *testing.T) {
	for _, p := range allProviders {
		for _, status := range []int{400, 401, 429, 500} {
			srv := newFakeProvider(t, status, `{"error":{"message":"status body"}}`)
			_, err := p.newAt(srv.URL, time.Second).Generate(context.Background(), "hi")
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.StatusCode != status || !strings.Contains(apiErr.Body, "status body") {
				t.Fatalf("%s %d: got %v", p.name, status, err)
			}
			if strings.Contains(err.Error(), testKey) {
				t.Fatalf("%s %d: key leaked: %v", p.name, status, err)
			}
		}
	}
}

func TestErrorBodyIsTruncated(t *testing.T) {
	srv := newFakeProvider(t, 500, strings.Repeat("x", 10_000))
	_, err := NewOpenRouterClient(testKey, srv.URL, "m", time.Second).Generate(context.Background(), "hi")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || len(apiErr.Body) > maxErrorBodyBytes+len("…") {
		t.Fatalf("body not truncated: %d bytes", len(apiErr.Body))
	}
}

// blockingServer holds every request until the client gives up.
func blockingServer(t *testing.T) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(func() { close(release); srv.Close() })
	return srv
}

func TestClientTimeout(t *testing.T) {
	for _, p := range allProviders {
		srv := blockingServer(t)
		start := time.Now()
		_, err := p.newAt(srv.URL, 50*time.Millisecond).Generate(context.Background(), "hi")
		var netErr net.Error
		if !errors.As(err, &netErr) || !netErr.Timeout() {
			t.Fatalf("%s: expected a timeout error, got %v", p.name, err)
		}
		if time.Since(start) > 2*time.Second {
			t.Fatalf("%s: timeout not enforced", p.name)
		}
	}
}

func TestContextCancellation(t *testing.T) {
	for _, p := range allProviders {
		srv := blockingServer(t)
		ctx, cancel := context.WithCancel(context.Background())
		time.AfterFunc(30*time.Millisecond, cancel)
		_, err := p.newAt(srv.URL, time.Minute).Generate(ctx, "hi")
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("%s: expected context.Canceled, got %v", p.name, err)
		}
		deadline, cancel2 := context.WithTimeout(context.Background(), 30*time.Millisecond)
		_, err = p.newAt(srv.URL, time.Minute).Generate(deadline, "hi")
		cancel2()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("%s: expected context.DeadlineExceeded, got %v", p.name, err)
		}
	}
}

// Transport errors include the request URL. The reference put the Gemini
// key in the URL, which would leak here.
func TestNetworkErrorsDoNotLeakCredentials(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	deadURL := srv.URL
	srv.Close()
	for _, p := range allProviders {
		_, err := p.newAt(deadURL, time.Second).Generate(context.Background(), "hi")
		if err == nil {
			t.Fatalf("%s: expected connection error", p.name)
		}
		if strings.Contains(err.Error(), testKey) {
			t.Fatalf("%s: key leaked in error: %v", p.name, err)
		}
	}
}

func TestVLLMIsReady(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()
	if !NewVLLMClient(srv.URL+"/v1/chat/completions", "m", time.Second).IsReady(context.Background()) {
		t.Fatal("expected ready")
	}
	if NewVLLMClient("http://127.0.0.1:1/v1/chat/completions", "m", time.Second).IsReady(context.Background()) {
		t.Fatal("expected not ready")
	}
}

func TestAdaptersAreSafeForConcurrentUse(t *testing.T) {
	for _, p := range allProviders {
		srv := newFakeProvider(t, 200, okBody(p))
		c := p.newAt(srv.URL, time.Second)
		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, err := c.Generate(context.Background(), "hi"); err != nil {
					t.Error(err)
				}
			}()
		}
		wg.Wait()
		if got := srv.count(); got != 8 {
			t.Fatalf("%s: %d requests", p.name, got)
		}
	}
}

func TestUsageReported(t *testing.T) {
	if (Usage{}).Reported() {
		t.Fatal("zero usage is unreported")
	}
	if !reflect.DeepEqual(Usage{TotalTokens: 1}.Reported(), true) {
		t.Fatal("any non-zero field counts as reported")
	}
}

func TestOpenRouterReportedCostAndGeminiThinkingTokens(t *testing.T) {
	or := `{"model":"openai/gpt-4.1-nano","choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.0000123}}`
	r, err := parseOpenAIResponse("openrouter", []byte(or), "openai/gpt-4.1-nano")
	if err != nil || r.Usage.CostCredits != 0.0000123 || r.Usage.CompletionTokens != 5 {
		t.Fatalf("%+v %v", r, err)
	}
	r, _ = parseOpenAIResponse("openrouter", []byte(openAIOK), "x")
	if r.Usage.CostCredits != 0 {
		t.Fatal("cost must not be invented when the provider sent none")
	}
	g := `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"thoughtsTokenCount":300,"totalTokenCount":420}}`
	gr, err := parseGeminiResponse([]byte(g), "gemini-2.5-flash")
	if err != nil || gr.Usage.CompletionTokens != 320 || gr.Usage.PromptTokens+gr.Usage.CompletionTokens != gr.Usage.TotalTokens {
		t.Fatalf("thinking tokens must count as billed output: %+v %v", gr.Usage, err)
	}
}
