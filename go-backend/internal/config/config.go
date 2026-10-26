// Package config loads settings from config.yaml and overlays environment
// variables, using the reference project's names. API keys are read only
// from the environment.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"example.com/councilai-starter/internal/llm"
)

// ModelSlot is a provider and model pair.
type ModelSlot struct {
	Provider string
	Model    string
}

func (s ModelSlot) String() string { return s.Provider + "/" + s.Model }

// Config is the backend configuration in scope for this phase.
type Config struct {
	ConfigFile    string // file that was loaded, "" if none was found
	ServerPort    string
	RAGServiceURL string
	MockLLM       bool // explicit offline mode; no provider is contacted

	OpenRouterAPIKey string
	OpenRouterURL    string
	GeminiAPIKey     string
	GeminiURL        string
	NVIDIANimAPIKey  string
	NVIDIANimURL     string
	VLLMURL          string

	CouncilSize   int
	CouncilSlots  []ModelSlot
	ChairmanSlot  ModelSlot
	RouterSlot    ModelSlot
	IngestionSlot ModelSlot

	StageTimeout   time.Duration // per model call inside a council stage
	LLMTimeout     time.Duration // HTTP client timeout for council members
	RequestTimeout time.Duration // whole query
}

// Keys returns the provider credentials.
func (c *Config) Keys() llm.ProviderKeys {
	return llm.ProviderKeys{Gemini: c.GeminiAPIKey, OpenRouter: c.OpenRouterAPIKey, NVIDIANim: c.NVIDIANimAPIKey}
}

// URLs returns the provider endpoints.
func (c *Config) URLs() llm.ProviderURLs {
	return llm.ProviderURLs{OpenRouter: c.OpenRouterURL, Gemini: c.GeminiURL, NVIDIANim: c.NVIDIANimURL, VLLM: c.VLLMURL}
}

type yamlSlot struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

// yamlSchema is the subset of the reference config.yaml used so far. There
// is deliberately no "keys" section: credentials belong in the environment.
type yamlSchema struct {
	Server struct {
		Port string `yaml:"port"`
	} `yaml:"server"`
	RAG struct {
		ServiceURL string `yaml:"service_url"`
	} `yaml:"rag"`
	Providers struct {
		OpenRouterURL string `yaml:"openrouter_url"`
		GeminiURL     string `yaml:"gemini_url"`
		NVIDIANimURL  string `yaml:"nvidia_nim_url"`
		VLLMURL       string `yaml:"vllm_url"`
	} `yaml:"providers"`
	Council struct {
		Size      int        `yaml:"size"`
		Slots     []yamlSlot `yaml:"slots"`
		Chairman  yamlSlot   `yaml:"chairman"`
		Router    yamlSlot   `yaml:"router"`
		Ingestion yamlSlot   `yaml:"ingestion"`
		Timeouts  struct {
			Stage   string `yaml:"stage"`
			LLM     string `yaml:"llm"`
			Request string `yaml:"request"`
		} `yaml:"timeouts"`
	} `yaml:"council"`
}

// SearchPaths are tried in order when CONFIG_FILE is not set, as in the
// reference.
var SearchPaths = []string{"/app/config.yaml", "config.yaml", "../config.yaml", "../../config.yaml"}

// legacyEnv are the starter's old variables. Setting one now fails startup
// instead of being silently ignored.
var legacyEnv = map[string]string{
	"MODEL_API_KEY":  "OPENROUTER_API_KEY / GEMINI_API_KEY / NVIDIA_NIM_API_KEY",
	"MODEL_BASE_URL": "OPENROUTER_URL / NVIDIA_NIM_URL / VLLM_URL",
	"COUNCIL_MODELS": "COUNCIL_<n>_PROVIDER and COUNCIL_<n>_MODEL",
	"CHAIR_MODEL":    "CHAIRMAN_PROVIDER and CHAIRMAN_MODEL",
	"RAG_URL":        "RAG_SERVICE_URL",
}

// Load reads the config file (CONFIG_FILE, or the first of SearchPaths that
// exists) and the process environment.
func Load() (*Config, error) {
	path := os.Getenv("CONFIG_FILE")
	if path == "" {
		for _, p := range SearchPaths {
			if _, err := os.Stat(p); err == nil {
				path = p
				break
			}
		}
	}
	return LoadFrom(path, os.Getenv)
}

// LoadFrom loads from an explicit file ("" for none) and environment lookup
// function, then validates. All problems are reported together.
func LoadFrom(path string, getenv func(string) string) (*Config, error) {
	var y yamlSchema
	if path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("config: reading %s: %w", path, err)
		}
		dec := yaml.NewDecoder(bytes.NewReader(raw))
		dec.KnownFields(true) // a typo such as "chairmen:" is an error, not a silent default
		if err := dec.Decode(&y); err != nil && !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("config: parsing %s: %w", path, err)
		}
	}

	l := loader{getenv: getenv}
	cfg := &Config{ConfigFile: path}
	cfg.ServerPort = l.str("SERVER_PORT", y.Server.Port, "8080")
	cfg.RAGServiceURL = l.str("RAG_SERVICE_URL", y.RAG.ServiceURL, "http://python-rag:8000")
	cfg.MockLLM = l.boolean("MOCK_LLM", false)

	cfg.OpenRouterAPIKey = getenv("OPENROUTER_API_KEY")
	cfg.GeminiAPIKey = getenv("GEMINI_API_KEY")
	cfg.NVIDIANimAPIKey = getenv("NVIDIA_NIM_API_KEY")
	cfg.OpenRouterURL = l.str("OPENROUTER_URL", y.Providers.OpenRouterURL, "https://openrouter.ai/api/v1/chat/completions")
	cfg.GeminiURL = l.str("GEMINI_URL", y.Providers.GeminiURL, llm.DefaultGeminiURL)
	cfg.NVIDIANimURL = l.str("NVIDIA_NIM_URL", y.Providers.NVIDIANimURL, "https://integrate.api.nvidia.com/v1/chat/completions")
	cfg.VLLMURL = l.str("VLLM_URL", y.Providers.VLLMURL, llm.DefaultVLLMURL)

	cfg.CouncilSize = l.integer("COUNCIL_SIZE", y.Council.Size, 3)
	if cfg.CouncilSize < 1 || cfg.CouncilSize > 26 {
		l.fail("COUNCIL_SIZE must be between 1 and 26, got %d", cfg.CouncilSize)
		cfg.CouncilSize = 0
	}
	for i := 0; i < cfg.CouncilSize; i++ {
		var def yamlSlot
		if i < len(y.Council.Slots) {
			def = y.Council.Slots[i]
		}
		if def.Provider == "" {
			def.Provider = llm.ProviderOpenRouter // reference default for unlisted slots
		}
		n := i + 1
		cfg.CouncilSlots = append(cfg.CouncilSlots, ModelSlot{
			Provider: l.str(fmt.Sprintf("COUNCIL_%d_PROVIDER", n), def.Provider, ""),
			Model:    l.str(fmt.Sprintf("COUNCIL_%d_MODEL", n), def.Model, ""),
		})
	}
	cfg.ChairmanSlot = l.slot("CHAIRMAN", y.Council.Chairman)
	cfg.RouterSlot = l.slot("ROUTER", y.Council.Router)
	cfg.IngestionSlot = l.slot("INGESTION", y.Council.Ingestion)

	cfg.StageTimeout = l.duration("STAGE_TIMEOUT", y.Council.Timeouts.Stage, 30*time.Second)
	cfg.LLMTimeout = l.duration("LLM_TIMEOUT", y.Council.Timeouts.LLM, 120*time.Second)
	cfg.RequestTimeout = l.duration("REQUEST_TIMEOUT", y.Council.Timeouts.Request, 120*time.Second)

	for old, replacement := range legacyEnv {
		if getenv(old) != "" {
			l.fail("%s is no longer used; set %s instead (see .env.example)", old, replacement)
		}
	}
	l.errs = append(l.errs, cfg.validateSlots()...)
	if len(l.errs) > 0 {
		return nil, errors.Join(l.errs...)
	}
	return cfg, nil
}

// validateSlots checks provider names and that at least one member has a
// model. Credentials are checked by llm.NewClientFromProvider when clients
// are built, so there is one source of truth for them.
func (c *Config) validateSlots() []error {
	var errs []error
	active := 0
	for i, s := range c.CouncilSlots {
		if s.Model == "" {
			continue // the reference skips slots without a model
		}
		active++
		if !llm.IsSupportedProvider(s.Provider) {
			errs = append(errs, fmt.Errorf("council member %d: unsupported provider %q (valid: gemini, openrouter, nvidia-nim, local)", i+1, s.Provider))
		}
	}
	if active == 0 && c.CouncilSize > 0 {
		errs = append(errs, errors.New("no council members configured; set council.slots in config.yaml or COUNCIL_1_PROVIDER and COUNCIL_1_MODEL"))
	}
	for name, s := range map[string]ModelSlot{"chairman": c.ChairmanSlot, "router": c.RouterSlot, "ingestion": c.IngestionSlot} {
		if !llm.IsSupportedProvider(s.Provider) {
			errs = append(errs, fmt.Errorf("%s: unsupported provider %q (valid: gemini, openrouter, nvidia-nim, local)", name, s.Provider))
		}
	}
	return errs
}

// loader applies "environment, then file, then default" and collects errors
// instead of silently falling back on invalid values.
type loader struct {
	getenv func(string) string
	errs   []error
}

func (l *loader) fail(format string, args ...any) {
	l.errs = append(l.errs, fmt.Errorf(format, args...))
}

func (l *loader) str(key, fromFile, def string) string {
	if v := strings.TrimSpace(l.getenv(key)); v != "" {
		return v
	}
	if fromFile != "" {
		return fromFile
	}
	return def
}

func (l *loader) slot(prefix string, fromFile yamlSlot) ModelSlot {
	// Reference defaults for chairman, router and ingestion.
	return ModelSlot{
		Provider: l.str(prefix+"_PROVIDER", fromFile.Provider, llm.ProviderGemini),
		Model:    l.str(prefix+"_MODEL", fromFile.Model, "gemini-3-flash-preview"),
	}
}

func (l *loader) boolean(key string, def bool) bool {
	v := strings.TrimSpace(l.getenv(key))
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		l.fail("%s must be true or false, got %q", key, v)
		return def
	}
	return b
}

func (l *loader) integer(key string, fromFile, def int) int {
	if v := strings.TrimSpace(l.getenv(key)); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			l.fail("%s must be an integer, got %q", key, v)
			return def
		}
		return n
	}
	if fromFile != 0 {
		return fromFile
	}
	return def
}

func (l *loader) duration(key, fromFile string, def time.Duration) time.Duration {
	raw := l.str(key, fromFile, "")
	if raw == "" {
		return def
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		l.fail("%s / timeouts must be a positive duration like \"30s\", got %q", key, raw)
		return def
	}
	return d
}
