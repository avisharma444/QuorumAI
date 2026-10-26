package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// repoConfig is the real config.yaml at the repository root.
const repoConfig = "../../../config.yaml"

func env(vars map[string]string) func(string) string {
	return func(k string) string { return vars[k] }
}

func writeYAML(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadsRepositoryConfig(t *testing.T) {
	cfg, err := LoadFrom(repoConfig, env(nil))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.CouncilSize != 3 || len(cfg.CouncilSlots) != 3 {
		t.Fatalf("slots %+v", cfg.CouncilSlots)
	}
	for _, s := range cfg.CouncilSlots {
		if s != (ModelSlot{"openrouter", "openrouter/free"}) {
			t.Fatalf("slot %+v", s)
		}
	}
	want := ModelSlot{"gemini", "gemini-2.5-flash"}
	if cfg.ChairmanSlot != want || cfg.RouterSlot != want || cfg.IngestionSlot != want {
		t.Fatalf("roles %+v %+v %+v", cfg.ChairmanSlot, cfg.RouterSlot, cfg.IngestionSlot)
	}
	if cfg.StageTimeout != 30*time.Second || cfg.LLMTimeout != 120*time.Second || cfg.RequestTimeout != 120*time.Second {
		t.Fatalf("timeouts %s %s %s", cfg.StageTimeout, cfg.LLMTimeout, cfg.RequestTimeout)
	}
	if cfg.OpenRouterURL != "https://openrouter.ai/api/v1/chat/completions" || cfg.MockLLM {
		t.Fatalf("providers %+v", cfg)
	}
}

func TestEnvironmentOverridesEachSlotIndependently(t *testing.T) {
	cfg, err := LoadFrom(repoConfig, env(map[string]string{
		"COUNCIL_1_PROVIDER": "gemini", "COUNCIL_1_MODEL": "gemini-2.5-pro",
		"COUNCIL_2_PROVIDER": "nvidia-nim", "COUNCIL_2_MODEL": "meta/llama-3.1-70b-instruct",
		"COUNCIL_3_PROVIDER": "local", "COUNCIL_3_MODEL": "microsoft/Phi-4-mini-instruct",
		"CHAIRMAN_PROVIDER": "openrouter", "CHAIRMAN_MODEL": "anthropic/claude-x",
		"ROUTER_MODEL":  "gemini-2.5-flash-lite",
		"STAGE_TIMEOUT": "12s", "REQUEST_TIMEOUT": "90s",
		"OPENROUTER_API_KEY": "or-key", "GEMINI_API_KEY": "g-key",
		"NVIDIA_NIM_URL": "https://nim.internal/v1/chat/completions",
		"MOCK_LLM":       "false",
	}))
	if err != nil {
		t.Fatal(err)
	}
	want := []ModelSlot{{"gemini", "gemini-2.5-pro"}, {"nvidia-nim", "meta/llama-3.1-70b-instruct"}, {"local", "microsoft/Phi-4-mini-instruct"}}
	for i, s := range want {
		if cfg.CouncilSlots[i] != s {
			t.Fatalf("slot %d: %+v", i+1, cfg.CouncilSlots[i])
		}
	}
	if cfg.ChairmanSlot != (ModelSlot{"openrouter", "anthropic/claude-x"}) || cfg.RouterSlot != (ModelSlot{"gemini", "gemini-2.5-flash-lite"}) || cfg.IngestionSlot.Model != "gemini-2.5-flash" {
		t.Fatalf("roles %+v %+v %+v", cfg.ChairmanSlot, cfg.RouterSlot, cfg.IngestionSlot)
	}
	if cfg.StageTimeout != 12*time.Second || cfg.RequestTimeout != 90*time.Second || cfg.LLMTimeout != 120*time.Second {
		t.Fatal("timeout overrides")
	}
	keys, urls := cfg.Keys(), cfg.URLs()
	if keys.OpenRouter != "or-key" || keys.Gemini != "g-key" || keys.NVIDIANim != "" || urls.NVIDIANim != "https://nim.internal/v1/chat/completions" {
		t.Fatalf("keys/urls %+v %+v", keys, urls)
	}
}

func TestDefaultsWithoutConfigFile(t *testing.T) {
	cfg, err := LoadFrom("", env(map[string]string{"COUNCIL_1_MODEL": "openrouter/free"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ServerPort != "8080" || cfg.CouncilSize != 3 || cfg.CouncilSlots[0] != (ModelSlot{"openrouter", "openrouter/free"}) || cfg.CouncilSlots[1].Model != "" {
		t.Fatalf("%+v", cfg)
	}
	if cfg.ChairmanSlot != (ModelSlot{"gemini", "gemini-3-flash-preview"}) || cfg.StageTimeout != 30*time.Second {
		t.Fatalf("reference defaults %+v", cfg.ChairmanSlot)
	}
}

func TestNoCouncilMembersIsAnError(t *testing.T) {
	_, err := LoadFrom("", env(nil))
	if err == nil || !strings.Contains(err.Error(), "no council members configured") {
		t.Fatalf("got %v", err)
	}
}

func TestEmptyFileIsAllowed(t *testing.T) {
	if _, err := LoadFrom(writeYAML(t, ""), env(map[string]string{"COUNCIL_1_MODEL": "m"})); err != nil {
		t.Fatal(err)
	}
}

func TestInvalidConfigurationIsReportedNotDefaulted(t *testing.T) {
	cases := map[string]struct {
		yaml string
		env  map[string]string
		want string
	}{
		"bad duration":         {"", map[string]string{"COUNCIL_1_MODEL": "m", "STAGE_TIMEOUT": "thirty"}, "STAGE_TIMEOUT"},
		"negative duration":    {"council:\n  timeouts:\n    llm: \"-5s\"\n", map[string]string{"COUNCIL_1_MODEL": "m"}, "LLM_TIMEOUT"},
		"bad council size":     {"", map[string]string{"COUNCIL_1_MODEL": "m", "COUNCIL_SIZE": "lots"}, "COUNCIL_SIZE"},
		"council size too big": {"", map[string]string{"COUNCIL_1_MODEL": "m", "COUNCIL_SIZE": "27"}, "between 1 and 26"},
		"bad MOCK_LLM":         {"", map[string]string{"COUNCIL_1_MODEL": "m", "MOCK_LLM": "yes please"}, "MOCK_LLM"},
		"unsupported member":   {"", map[string]string{"COUNCIL_2_PROVIDER": "openai", "COUNCIL_2_MODEL": "gpt"}, `council member 2: unsupported provider "openai"`},
		"unsupported chairman": {"", map[string]string{"COUNCIL_1_MODEL": "m", "CHAIRMAN_PROVIDER": "anthropic"}, `chairman: unsupported provider "anthropic"`},
		"legacy env var":       {"", map[string]string{"COUNCIL_1_MODEL": "m", "MODEL_API_KEY": "old"}, "MODEL_API_KEY is no longer used"},
		"legacy RAG_URL":       {"", map[string]string{"COUNCIL_1_MODEL": "m", "RAG_URL": "http://x"}, "RAG_SERVICE_URL"},
		"typo in yaml":         {"council:\n  chairmen:\n    provider: gemini\n", map[string]string{"COUNCIL_1_MODEL": "m"}, "chairmen"},
		"keys in yaml":         {"keys:\n  gemini: \"abc\"\n", map[string]string{"COUNCIL_1_MODEL": "m"}, "keys"},
		"malformed yaml":       {"council: [unclosed\n", map[string]string{"COUNCIL_1_MODEL": "m"}, "parsing"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			path := ""
			if tc.yaml != "" {
				path = writeYAML(t, tc.yaml)
			}
			cfg, err := LoadFrom(path, env(tc.env))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want error containing %q, got cfg=%v err=%v", tc.want, cfg, err)
			}
		})
	}
}

func TestAllProblemsReportedTogether(t *testing.T) {
	_, err := LoadFrom("", env(map[string]string{"STAGE_TIMEOUT": "x", "MOCK_LLM": "maybe", "CHAIR_MODEL": "old"}))
	if err == nil {
		t.Fatal("expected errors")
	}
	for _, want := range []string{"STAGE_TIMEOUT", "MOCK_LLM", "CHAIR_MODEL", "no council members"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing %q in:\n%v", want, err)
		}
	}
}

func TestMissingFileIsAnError(t *testing.T) {
	if _, err := LoadFrom(filepath.Join(t.TempDir(), "absent.yaml"), env(nil)); err == nil {
		t.Fatal("explicit missing file accepted")
	}
}

func TestWebSearchSwitch(t *testing.T) {
	cfg, err := LoadFrom(repoConfig, env(nil))
	if err != nil || !cfg.WebSearch {
		t.Fatalf("web search should default on (reference behaviour): %v", err)
	}
	cfg, _ = LoadFrom(repoConfig, env(map[string]string{"WEB_SEARCH_ENABLED": "false"}))
	if cfg.WebSearch {
		t.Fatal("WEB_SEARCH_ENABLED=false ignored")
	}
	cfg, _ = LoadFrom(writeYAML(t, "rag:\n  web_search: false\n"), env(map[string]string{"COUNCIL_1_MODEL": "m"}))
	if cfg.WebSearch {
		t.Fatal("rag.web_search: false ignored")
	}
	if _, err := LoadFrom(repoConfig, env(map[string]string{"WEB_SEARCH_ENABLED": "sometimes"})); err == nil {
		t.Fatal("invalid boolean accepted")
	}
}

func TestRedisConfig(t *testing.T) {
	cfg, err := LoadFrom(repoConfig, env(nil))
	if err != nil {
		t.Fatal(err)
	}
	r := cfg.Redis
	if !r.Enabled || r.Addr != "redis:6379" || r.DB != 0 || r.ExactTTL != time.Hour || r.SemanticTTL != 24*time.Hour ||
		r.SemanticThreshold != 0.85 || r.FailureThreshold != 3 || r.SuccessThreshold != 2 || r.BreakerTimeout != 10*time.Second || r.OpTimeout != 500*time.Millisecond {
		t.Fatalf("reference defaults: %+v", r)
	}
	cfg, err = LoadFrom(repoConfig, env(map[string]string{"REDIS_ADDR": "cache:6380", "REDIS_PASSWORD": "pw", "SEMANTIC_CACHE_THRESHOLD": "0.92", "REDIS_ENABLED": "false", "CIRCUIT_BREAKER_TIMEOUT": "3s"}))
	if err != nil || cfg.Redis.Addr != "cache:6380" || cfg.Redis.Password != "pw" || cfg.Redis.SemanticThreshold != 0.92 || cfg.Redis.Enabled || cfg.Redis.BreakerTimeout != 3*time.Second {
		t.Fatalf("overrides: %+v %v", cfg.Redis, err)
	}
	for name, vars := range map[string]map[string]string{
		"non-zero db":    {"REDIS_DB": "2"},
		"threshold > 1":  {"SEMANTIC_CACHE_THRESHOLD": "1.5"},
		"threshold text": {"SEMANTIC_CACHE_THRESHOLD": "high"},
		"bad ttl":        {"REDIS_SEMANTIC_TTL": "forever"},
		"zero failures":  {"CIRCUIT_BREAKER_FAILURE_THRESHOLD": "0"},
	} {
		if _, err := LoadFrom(repoConfig, env(vars)); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}
