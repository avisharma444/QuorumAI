package llm

import (
	"context"
	"strings"
	"testing"
	"time"
)

var allKeys = ProviderKeys{Gemini: "g-" + testKey, OpenRouter: "or-" + testKey, NVIDIANim: "nim-" + testKey}
var allURLs = ProviderURLs{OpenRouter: "https://or.example/v1/chat/completions", NVIDIANim: "https://nim.example/v1/chat/completions", VLLM: "http://vllm.example/v1/chat/completions"}

func TestFactoryBuildsEachProvider(t *testing.T) {
	cases := []struct {
		provider, model, wantName string
		check                     func(LLMClient) bool
	}{
		{"openrouter", "openrouter/free", "openrouter/free", func(c LLMClient) bool { _, ok := c.(*OpenRouterClient); return ok }},
		{"gemini", "gemini-2.5-flash", "gemini:gemini-2.5-flash", func(c LLMClient) bool { _, ok := c.(*GeminiClient); return ok }},
		{"nvidia-nim", "meta/llama-3.1-8b-instruct", "nim:meta/llama-3.1-8b-instruct", func(c LLMClient) bool { _, ok := c.(*NIMClient); return ok }},
		{"local", "microsoft/Phi-4-mini-instruct", "local:microsoft/Phi-4-mini-instruct", func(c LLMClient) bool { _, ok := c.(*VLLMClient); return ok }},
	}
	for _, tc := range cases {
		c, err := NewClientFromProvider(tc.provider, tc.model, allKeys, allURLs, time.Second)
		if err != nil {
			t.Fatalf("%s: %v", tc.provider, err)
		}
		if !tc.check(c) || c.ModelName() != tc.wantName {
			t.Fatalf("%s: got %T %q", tc.provider, c, c.ModelName())
		}
	}
}

func TestFactoryUnknownProvider(t *testing.T) {
	_, err := NewClientFromProvider("openai", "gpt-x", allKeys, allURLs, time.Second)
	if err == nil || !strings.Contains(err.Error(), `unknown provider "openai"`) || !strings.Contains(err.Error(), "nvidia-nim") {
		t.Fatalf("got %v", err)
	}
	if IsSupportedProvider("openai") || !IsSupportedProvider("local") {
		t.Fatal("IsSupportedProvider mismatch")
	}
}

func TestFactoryMissingCredentials(t *testing.T) {
	for provider, envVar := range map[string]string{"gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "nvidia-nim": "NVIDIA_NIM_API_KEY"} {
		// Other providers' keys are set; only this one is missing.
		keys := allKeys
		switch provider {
		case "gemini":
			keys.Gemini = ""
		case "openrouter":
			keys.OpenRouter = ""
		case "nvidia-nim":
			keys.NVIDIANim = ""
		}
		c, err := NewClientFromProvider(provider, "m", keys, allURLs, time.Second)
		if err == nil || c != nil || !strings.Contains(err.Error(), envVar) {
			t.Fatalf("%s: got client=%v err=%v", provider, c, err)
		}
		if strings.Contains(err.Error(), testKey) {
			t.Fatalf("%s: error leaks a key: %v", provider, err)
		}
	}
	// vLLM needs no key.
	if _, err := NewClientFromProvider("local", "m", ProviderKeys{}, allURLs, time.Second); err != nil {
		t.Fatal(err)
	}
}

func TestFactoryRequiresModel(t *testing.T) {
	if _, err := NewClientFromProvider("openrouter", "  ", allKeys, allURLs, time.Second); err == nil {
		t.Fatal("empty model accepted")
	}
}

func TestFactoryVLLMEndpointOverride(t *testing.T) {
	model := "microsoft/Phi-4-mini-instruct"
	if key := VLLMURLEnvKey(model); key != "VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_URL" {
		t.Fatalf("env key %q", key)
	}
	c, _ := NewClientFromProvider("local", model, allKeys, allURLs, time.Second)
	if got := c.(*VLLMClient).URL(); got != allURLs.VLLM {
		t.Fatalf("shared URL expected, got %s", got)
	}
	t.Setenv("VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_URL", "http://phi-server:9000/v1/chat/completions")
	c, _ = NewClientFromProvider("local", model, allKeys, allURLs, time.Second)
	if got := c.(*VLLMClient).URL(); got != "http://phi-server:9000/v1/chat/completions" {
		t.Fatalf("override ignored, got %s", got)
	}
	other, _ := NewClientFromProvider("local", "qwen/Qwen2.5-7B", allKeys, allURLs, time.Second)
	if got := other.(*VLLMClient).URL(); got != allURLs.VLLM {
		t.Fatalf("override must apply only to its model, got %s", got)
	}
	none, _ := NewClientFromProvider("local", "x", allKeys, ProviderURLs{}, time.Second)
	if got := none.(*VLLMClient).URL(); got != DefaultVLLMURL {
		t.Fatalf("default expected, got %s", got)
	}
}

func TestFactoryClientsAreIndependent(t *testing.T) {
	a, _ := NewClientFromProvider("openrouter", "model-a", allKeys, allURLs, time.Second)
	b, _ := NewClientFromProvider("openrouter", "model-b", allKeys, allURLs, time.Second)
	if a == b || a.ModelName() == b.ModelName() {
		t.Fatal("each slot must get its own client")
	}
}

func TestMockClient(t *testing.T) {
	m := &MockClient{Name: "x"}
	resp, err := m.GenerateChat(context.Background(), GenerateOptions{Messages: []Message{
		{Role: "system", Content: "REVIEW_AGENT: rank"},
		{Role: "user", Content: "=== Response A ===\nfoo\n\n=== Response B ===\nbar"},
	}})
	if err != nil || !strings.HasPrefix(resp.Answer, "RANKING: A, B\n") || resp.Usage.Reported() || m.ModelName() != "mock:x" {
		t.Fatalf("resp %+v err %v", resp, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (&MockClient{Name: "x", Delay: time.Hour}).Generate(ctx, "hi"); err == nil {
		t.Fatal("mock must honour cancellation")
	}
}
