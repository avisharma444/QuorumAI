package main

import (
	"strings"
	"testing"
	"time"

	"example.com/councilai-starter/internal/config"
	"example.com/councilai-starter/internal/llm"
)

const secret = "sk-test-SECRET-0123456789"

func load(t *testing.T, vars map[string]string) *config.Config {
	t.Helper()
	cfg, err := config.LoadFrom("../../../config.yaml", func(k string) string { return vars[k] })
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestMockModeNeedsNoCredentials(t *testing.T) {
	clients, err := buildClients(load(t, map[string]string{"MOCK_LLM": "true"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(clients.Members) != 3 || clients.Chairman.ModelName() != "mock:chairman" || clients.Members[2].ModelName() != "mock:member-3" {
		t.Fatalf("%+v", clients)
	}
	for _, c := range append(clients.Members, clients.Chairman, clients.Router, clients.Ingestion) {
		if _, ok := c.(*llm.MockClient); !ok {
			t.Fatalf("expected mock client, got %T", c)
		}
	}
}

// With real providers and no keys, startup fails and names every role and
// missing variable. It never falls back to mock clients.
func TestMissingCredentialsFailStartupExplicitly(t *testing.T) {
	clients, err := buildClients(load(t, map[string]string{"MOCK_LLM": "false"}))
	if err == nil {
		t.Fatalf("expected configuration error, got %+v", clients)
	}
	for _, want := range []string{
		"council member 1: openrouter: OPENROUTER_API_KEY is required",
		"council member 3: openrouter: OPENROUTER_API_KEY is required",
		"chairman: gemini: GEMINI_API_KEY is required",
		"router: gemini: GEMINI_API_KEY is required",
		"ingestion: gemini: GEMINI_API_KEY is required",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing %q in:\n%v", want, err)
		}
	}
	if clients.Chairman != nil || len(clients.Members) != 0 {
		t.Fatal("no clients may be returned with a configuration error")
	}
}

func TestEachRoleGetsItsOwnProviderModelAndTimeout(t *testing.T) {
	type call struct {
		provider, model string
		timeout         time.Duration
	}
	var calls []call
	orig := newClient
	t.Cleanup(func() { newClient = orig })
	newClient = func(p, m string, k llm.ProviderKeys, u llm.ProviderURLs, to time.Duration) (llm.LLMClient, error) {
		calls = append(calls, call{p, m, to})
		return orig(p, m, k, u, to)
	}
	cfg := load(t, map[string]string{
		"COUNCIL_1_PROVIDER": "openrouter", "COUNCIL_1_MODEL": "meta/llama-x",
		"COUNCIL_2_PROVIDER": "gemini", "COUNCIL_2_MODEL": "gemini-2.5-pro",
		"COUNCIL_3_PROVIDER": "nvidia-nim", "COUNCIL_3_MODEL": "nvidia/nemotron",
		"CHAIRMAN_PROVIDER": "local", "CHAIRMAN_MODEL": "phi-4",
		"OPENROUTER_API_KEY": secret, "GEMINI_API_KEY": secret, "NVIDIA_NIM_API_KEY": secret,
		"LLM_TIMEOUT": "100s", "STAGE_TIMEOUT": "20s",
	})
	clients, err := buildClients(cfg)
	if err != nil {
		t.Fatal(err)
	}
	names := []string{clients.Members[0].ModelName(), clients.Members[1].ModelName(), clients.Members[2].ModelName(), clients.Chairman.ModelName()}
	want := []string{"meta/llama-x", "gemini:gemini-2.5-pro", "nim:nvidia/nemotron", "local:phi-4"}
	for i := range want {
		if names[i] != want[i] {
			t.Fatalf("got %v, want %v", names, want)
		}
	}
	wantCalls := []call{
		{"openrouter", "meta/llama-x", 100 * time.Second},
		{"gemini", "gemini-2.5-pro", 100 * time.Second},
		{"nvidia-nim", "nvidia/nemotron", 100 * time.Second},
		{"local", "phi-4", 20 * time.Second},
		{"gemini", "gemini-2.5-flash", routerTimeout},
		{"gemini", "gemini-2.5-flash", ingestionTimeout},
	}
	if len(calls) != len(wantCalls) {
		t.Fatalf("calls %+v", calls)
	}
	for i := range wantCalls {
		if calls[i] != wantCalls[i] {
			t.Fatalf("call %d: got %+v want %+v", i, calls[i], wantCalls[i])
		}
	}
	for _, line := range startupSummary(cfg, clients) {
		if strings.Contains(line, secret) {
			t.Fatalf("startup summary leaks a key: %s", line)
		}
	}
}

func TestEmptySlotsAreSkipped(t *testing.T) {
	clients, err := buildClients(load(t, map[string]string{"MOCK_LLM": "true", "COUNCIL_SIZE": "5"}))
	if err != nil {
		t.Fatal(err)
	}
	// Slots 4 and 5 have no model, so the reference skips them.
	if len(clients.Members) != 3 {
		t.Fatalf("got %d members", len(clients.Members))
	}
}
