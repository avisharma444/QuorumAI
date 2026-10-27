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
	WebSearch     bool // web grounding for questions without a document (reference default: on)

	Redis RedisConfig

	Auth         AuthConfig
	Database     DatabaseConfig
	RateLimit    RateLimitConfig
	Conversation ConversationConfig
	CORSOrigins  []string // explicit browser origins; empty = CORS off

	Observability ObservabilityConfig

	OpenRouterAPIKey string
	OpenRouterURL    string
	GeminiAPIKey     string
	GeminiURL        string
	NVIDIANimAPIKey  string
	VLLMAPIKey       string // optional bearer token for a remote vLLM endpoint
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

	// MaxOutputTokens caps every council model call (candidates, reviews,
	// chairman, direct, revision) that does not set a smaller cap itself.
	MaxOutputTokens int
}

// ObservabilityConfig: Prometheus endpoint and OpenTelemetry tracing.
type ObservabilityConfig struct {
	MetricsAddr   string  // internal listener for /metrics ("" disables)
	TraceExporter string  // noop (default), stdout, otlp
	OTLPEndpoint  string  // host:port of an OTLP/HTTP receiver (e.g. jaeger:4318)
	SampleRatio   float64 // of new root traces (incoming sampled parents are always followed)
}

// AuthConfig holds JWT and account settings. Secrets come only from the
// environment; validation happens at startup (auth.NewJWTManager).
type AuthConfig struct {
	AppEnv        string // "production" (default) or "development"
	JWTSecret     string
	JWTExpiration time.Duration // reference: 24h
	BcryptCost    int
	DemoUser      bool   // only when APP_ENV=development and AUTH_DEMO_PASSWORD is set
	DemoPassword  string // never logged
}

// DatabaseConfig is the PostgreSQL connection (reference database section).
type DatabaseConfig struct {
	URL             string
	Host            string
	Port            string
	User            string
	Password        string
	Name            string
	SSLMode         string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

// RateLimitConfig: per-user API limit (reference rps/burst) and a per-IP
// limit for register/login.
type RateLimitConfig struct {
	RPS           float64
	Burst         int
	AuthRPS       float64
	AuthBurst     int
	FailurePolicy string // "closed" (default) or "open" (reference)
}

// ConversationConfig: Redis conversation memory (reference: 10 turns, 24h).
type ConversationConfig struct {
	MaxTurns         int
	TTL              time.Duration
	HistoryExchanges int // earlier exchanges sent to the models (reference: 5)
}

// RedisConfig configures the Redis Stack caches (reference redis section).
type RedisConfig struct {
	Enabled           bool // false: no caching at all (there is no in-process fallback)
	Addr              string
	Password          string
	DB                int
	OpTimeout         time.Duration // per Redis round trip
	ExactTTL          time.Duration // L1 and document summaries (reference: 1h)
	SemanticTTL       time.Duration // L2 (reference: 24h)
	SemanticThreshold float64       // cosine similarity (reference: 0.85)
	SemanticReuse     bool          // L2 answer reuse; off by default (measured false positives)
	FailureThreshold  int           // breaker: consecutive failures to open (reference: 3)
	SuccessThreshold  int           // breaker: half-open successes to close (reference: 2)
	BreakerTimeout    time.Duration // breaker cooldown (reference: 10s)
}

// Keys returns the provider credentials.
func (c *Config) Keys() llm.ProviderKeys {
	return llm.ProviderKeys{Gemini: c.GeminiAPIKey, OpenRouter: c.OpenRouterAPIKey, NVIDIANim: c.NVIDIANimAPIKey, VLLM: c.VLLMAPIKey}
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
		Port                   string   `yaml:"port"`
		SemanticCacheThreshold *float64 `yaml:"semantic_cache_threshold"`
		JWTExpiration          string   `yaml:"jwt_expiration"`
		CORSAllowedOrigins     []string `yaml:"cors_allowed_origins"`
		RateLimit              struct {
			RPS           float64 `yaml:"rps"`
			Burst         int     `yaml:"burst"`
			AuthRPS       float64 `yaml:"auth_rps"`
			AuthBurst     int     `yaml:"auth_burst"`
			FailurePolicy string  `yaml:"failure_policy"`
		} `yaml:"rate_limit"`
	} `yaml:"server"`
	Database struct {
		Host            string `yaml:"host"`
		Port            string `yaml:"port"`
		User            string `yaml:"user"`
		DBName          string `yaml:"dbname"`
		SSLMode         string `yaml:"sslmode"`
		MaxOpenConns    int    `yaml:"max_open_conns"`
		MaxIdleConns    int    `yaml:"max_idle_conns"`
		ConnMaxLifetime string `yaml:"conn_max_lifetime"`
		ConnMaxIdleTime string `yaml:"conn_max_idle_time"`
	} `yaml:"database"`
	Observability struct {
		MetricsAddr string `yaml:"metrics_addr"`
		Tracing     struct {
			Exporter     string  `yaml:"exporter"`
			OTLPEndpoint string  `yaml:"otlp_endpoint"`
			SampleRatio  float64 `yaml:"sample_ratio"`
		} `yaml:"tracing"`
	} `yaml:"observability"`
	Conversation struct {
		MaxTurns         int    `yaml:"max_turns"`
		TTL              string `yaml:"ttl"`
		HistoryExchanges int    `yaml:"history_exchanges"`
	} `yaml:"conversation"`
	Redis struct {
		Enabled          *bool  `yaml:"enabled"`
		SemanticEnabled  *bool  `yaml:"semantic_cache_enabled"`
		Addr             string `yaml:"addr"`
		Password         string `yaml:"password"`
		DB               int    `yaml:"db"`
		OpTimeout        string `yaml:"op_timeout"`
		ExactTTL         string `yaml:"exact_ttl"`
		SemanticTTL      string `yaml:"semantic_ttl"`
		FailureThreshold int    `yaml:"circuit_breaker_failure_threshold"`
		SuccessThreshold int    `yaml:"circuit_breaker_success_threshold"`
		Timeout          string `yaml:"circuit_breaker_timeout"`
	} `yaml:"redis"`
	RAG struct {
		ServiceURL string `yaml:"service_url"`
		WebSearch  *bool  `yaml:"web_search"`
		// Read by the Python RAG service; accepted here so strict parsing passes.
		ChromaDBPath       string `yaml:"chroma_db_path"`
		EmbeddingModel     string `yaml:"embedding_model"`
		EmbeddingDimension int    `yaml:"embedding_dimension"`
		RerankerModel      string `yaml:"reranker_model"`
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
	webDefault := true
	if y.RAG.WebSearch != nil {
		webDefault = *y.RAG.WebSearch
	}
	cfg.WebSearch = l.boolean("WEB_SEARCH_ENABLED", webDefault)

	cfg.OpenRouterAPIKey = getenv("OPENROUTER_API_KEY")
	cfg.GeminiAPIKey = getenv("GEMINI_API_KEY")
	cfg.NVIDIANimAPIKey = getenv("NVIDIA_NIM_API_KEY")
	cfg.VLLMAPIKey = getenv("VLLM_API_KEY")
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
	cfg.MaxOutputTokens = l.integer("LLM_MAX_OUTPUT_TOKENS", 0, llm.DefaultMaxOutputTokens)
	if cfg.MaxOutputTokens <= 0 {
		l.fail("LLM_MAX_OUTPUT_TOKENS must be a positive integer, got %d", cfg.MaxOutputTokens)
	}

	redisDefault := true
	if y.Redis.Enabled != nil {
		redisDefault = *y.Redis.Enabled
	}
	r := &cfg.Redis
	r.Enabled = l.boolean("REDIS_ENABLED", redisDefault)
	r.Addr = l.str("REDIS_ADDR", y.Redis.Addr, "redis:6379")
	r.Password = getenv("REDIS_PASSWORD")
	if r.Password == "" {
		r.Password = y.Redis.Password
	}
	r.DB = l.integer("REDIS_DB", y.Redis.DB, 0)
	if r.DB != 0 {
		l.fail("REDIS_DB must be 0: RediSearch indexes only work in database 0")
	}
	r.OpTimeout = l.duration("REDIS_OP_TIMEOUT", y.Redis.OpTimeout, 500*time.Millisecond)
	r.ExactTTL = l.duration("REDIS_EXACT_TTL", y.Redis.ExactTTL, time.Hour)
	r.SemanticTTL = l.duration("REDIS_SEMANTIC_TTL", y.Redis.SemanticTTL, 24*time.Hour)
	r.FailureThreshold = l.integer("CIRCUIT_BREAKER_FAILURE_THRESHOLD", y.Redis.FailureThreshold, 3)
	r.SuccessThreshold = l.integer("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", y.Redis.SuccessThreshold, 2)
	r.BreakerTimeout = l.duration("CIRCUIT_BREAKER_TIMEOUT", y.Redis.Timeout, 10*time.Second)
	thr := 0.85
	if y.Server.SemanticCacheThreshold != nil {
		thr = *y.Server.SemanticCacheThreshold
	}
	if v := strings.TrimSpace(getenv("SEMANTIC_CACHE_THRESHOLD")); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			l.fail("SEMANTIC_CACHE_THRESHOLD must be a number, got %q", v)
		}
		thr = f
	}
	if thr <= 0 || thr > 1 {
		l.fail("semantic cache threshold must be in (0, 1], got %v", thr)
	}
	r.SemanticThreshold = thr
	if r.FailureThreshold < 1 || r.SuccessThreshold < 1 {
		l.fail("circuit breaker thresholds must be at least 1")
	}
	semDefault := false
	if y.Redis.SemanticEnabled != nil {
		semDefault = *y.Redis.SemanticEnabled
	}
	r.SemanticReuse = l.boolean("SEMANTIC_CACHE_ENABLED", semDefault)

	a := &cfg.Auth
	a.AppEnv = strings.ToLower(l.str("APP_ENV", "", "production"))
	if a.AppEnv != "production" && a.AppEnv != "development" {
		l.fail("APP_ENV must be production or development, got %q", a.AppEnv)
	}
	a.JWTSecret = getenv("JWT_SECRET")
	a.JWTExpiration = l.duration("JWT_EXPIRATION", y.Server.JWTExpiration, 24*time.Hour)
	a.BcryptCost = l.integer("BCRYPT_COST", 0, 12)
	if a.BcryptCost < 10 || a.BcryptCost > 15 {
		l.fail("BCRYPT_COST must be between 10 and 15")
	}
	a.DemoPassword = getenv("AUTH_DEMO_PASSWORD")
	a.DemoUser = a.DemoPassword != ""
	if a.DemoUser && a.AppEnv != "development" {
		l.fail("AUTH_DEMO_PASSWORD is only allowed with APP_ENV=development")
	}

	d := &cfg.Database
	d.URL = getenv("DATABASE_URL")
	d.Host = l.str("DB_HOST", y.Database.Host, "postgres")
	d.Port = l.str("DB_PORT", y.Database.Port, "5432")
	d.User = l.str("DB_USER", y.Database.User, "")
	d.Password = getenv("DB_PASSWORD")
	d.Name = l.str("DB_NAME", y.Database.DBName, "")
	d.SSLMode = l.str("DB_SSLMODE", y.Database.SSLMode, "disable")
	d.MaxOpenConns = l.integer("DB_MAX_OPEN_CONNS", y.Database.MaxOpenConns, 10)
	d.MaxIdleConns = l.integer("DB_MAX_IDLE_CONNS", y.Database.MaxIdleConns, 2)
	d.ConnMaxLifetime = l.duration("DB_CONN_MAX_LIFETIME", y.Database.ConnMaxLifetime, time.Hour)
	d.ConnMaxIdleTime = l.duration("DB_CONN_MAX_IDLE_TIME", y.Database.ConnMaxIdleTime, 30*time.Minute)

	rl := &cfg.RateLimit
	rl.RPS = l.float("RATE_LIMIT_RPS", y.Server.RateLimit.RPS, 10)
	rl.Burst = l.integer("RATE_LIMIT_BURST", y.Server.RateLimit.Burst, 20)
	rl.AuthRPS = l.float("AUTH_RATE_LIMIT_RPS", y.Server.RateLimit.AuthRPS, 0.2)
	rl.AuthBurst = l.integer("AUTH_RATE_LIMIT_BURST", y.Server.RateLimit.AuthBurst, 10)
	rl.FailurePolicy = strings.ToLower(l.str("RATE_LIMIT_FAILURE_POLICY", y.Server.RateLimit.FailurePolicy, "closed"))
	if rl.RPS <= 0 || rl.Burst < 1 || rl.AuthRPS <= 0 || rl.AuthBurst < 1 {
		l.fail("rate limits need rps > 0 and burst >= 1")
	}
	if rl.FailurePolicy != "closed" && rl.FailurePolicy != "open" {
		l.fail("RATE_LIMIT_FAILURE_POLICY must be closed or open, got %q", rl.FailurePolicy)
	}

	cv := &cfg.Conversation
	cv.MaxTurns = l.integer("CONVERSATION_MAX_TURNS", y.Conversation.MaxTurns, 10)
	cv.TTL = l.duration("CONVERSATION_TTL", y.Conversation.TTL, 24*time.Hour)
	cv.HistoryExchanges = l.integer("CONVERSATION_HISTORY_EXCHANGES", y.Conversation.HistoryExchanges, 5)
	if cv.MaxTurns < 1 || cv.HistoryExchanges < 0 || cv.HistoryExchanges > cv.MaxTurns {
		l.fail("conversation: max_turns >= 1 and 0 <= history_exchanges <= max_turns")
	}

	ob := &cfg.Observability
	ob.MetricsAddr = l.str("METRICS_ADDR", y.Observability.MetricsAddr, ":9464")
	ob.TraceExporter = strings.ToLower(l.str("OTEL_TRACES_EXPORTER", y.Observability.Tracing.Exporter, "noop"))
	switch ob.TraceExporter {
	case "none":
		ob.TraceExporter = "noop"
	case "noop", "stdout", "otlp":
	default:
		l.fail("OTEL_TRACES_EXPORTER must be noop, stdout or otlp, got %q", ob.TraceExporter)
	}
	ob.OTLPEndpoint = strings.TrimPrefix(strings.TrimPrefix(l.str("OTEL_EXPORTER_OTLP_ENDPOINT", y.Observability.Tracing.OTLPEndpoint, ""), "http://"), "https://")
	if ob.TraceExporter == "otlp" && ob.OTLPEndpoint == "" {
		l.fail("OTEL_TRACES_EXPORTER=otlp needs OTEL_EXPORTER_OTLP_ENDPOINT (host:port)")
	}
	ob.SampleRatio = l.float("OTEL_TRACES_SAMPLER_ARG", y.Observability.Tracing.SampleRatio, 1.0)
	if ob.SampleRatio <= 0 || ob.SampleRatio > 1 {
		l.fail("OTEL_TRACES_SAMPLER_ARG must be in (0, 1], got %v", ob.SampleRatio)
	}

	cfg.CORSOrigins = y.Server.CORSAllowedOrigins
	if v := strings.TrimSpace(getenv("CORS_ALLOWED_ORIGINS")); v != "" {
		cfg.CORSOrigins = nil
		for _, o := range strings.Split(v, ",") {
			if o = strings.TrimSpace(o); o != "" {
				cfg.CORSOrigins = append(cfg.CORSOrigins, o)
			}
		}
	}
	for _, o := range cfg.CORSOrigins {
		if o == "*" {
			l.fail("CORS: wildcard origin is not allowed; list explicit origins")
		}
	}

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

func (l *loader) float(key string, fromFile, def float64) float64 {
	if v := strings.TrimSpace(l.getenv(key)); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			l.fail("%s must be a number, got %q", key, v)
			return def
		}
		return f
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
