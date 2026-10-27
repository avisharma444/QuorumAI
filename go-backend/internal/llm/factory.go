package llm

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// Supported provider names, as used in config.yaml.
const (
	ProviderGemini     = "gemini"
	ProviderOpenRouter = "openrouter"
	ProviderNVIDIANim  = "nvidia-nim"
	ProviderLocal      = "local" // vLLM
)

// DefaultVLLMURL is used when neither config nor a per-model override names
// a vLLM endpoint.
const DefaultVLLMURL = "http://vllm-inference:8001/v1/chat/completions"

// ProviderKeys holds API credentials. They come from environment variables
// only and must never be logged.
type ProviderKeys struct {
	Gemini     string
	OpenRouter string
	NVIDIANim  string
	VLLM       string // optional bearer token for remote vLLM (e.g. RunPod Serverless)
}

// ProviderURLs holds provider endpoints.
type ProviderURLs struct {
	OpenRouter string
	Gemini     string // base URL; empty means DefaultGeminiURL
	NVIDIANim  string
	VLLM       string
}

// IsSupportedProvider reports whether NewClientFromProvider knows provider.
func IsSupportedProvider(provider string) bool {
	switch provider {
	case ProviderGemini, ProviderOpenRouter, ProviderNVIDIANim, ProviderLocal:
		return true
	}
	return false
}

// NewClientFromProvider builds the adapter for provider and model. Errors
// name the missing environment variable but never include key values.
func NewClientFromProvider(provider, model string, keys ProviderKeys, urls ProviderURLs, timeout time.Duration) (LLMClient, error) {
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("%s: model name is required", provider)
	}
	switch provider {
	case ProviderGemini:
		if keys.Gemini == "" {
			return nil, fmt.Errorf("gemini: GEMINI_API_KEY is required (model %s)", model)
		}
		return NewGeminiClientWithURL(keys.Gemini, urls.Gemini, model, timeout), nil

	case ProviderOpenRouter:
		if keys.OpenRouter == "" {
			return nil, fmt.Errorf("openrouter: OPENROUTER_API_KEY is required (model %s)", model)
		}
		if urls.OpenRouter == "" {
			return nil, fmt.Errorf("openrouter: endpoint URL is required (model %s)", model)
		}
		return NewOpenRouterClient(keys.OpenRouter, urls.OpenRouter, model, timeout), nil

	case ProviderNVIDIANim:
		if keys.NVIDIANim == "" {
			return nil, fmt.Errorf("nvidia-nim: NVIDIA_NIM_API_KEY is required (model %s)", model)
		}
		if urls.NVIDIANim == "" {
			return nil, fmt.Errorf("nvidia-nim: endpoint URL is required (model %s)", model)
		}
		return NewNIMClient(keys.NVIDIANim, urls.NVIDIANim, model, timeout), nil

	case ProviderLocal:
		// Each local model can have its own server:
		// VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_URL overrides urls.VLLM.
		// VLLM_<MODEL>_API_KEY likewise overrides VLLM_API_KEY.
		vllmURL := os.Getenv(VLLMURLEnvKey(model))
		if vllmURL == "" {
			vllmURL = urls.VLLM
		}
		if vllmURL == "" {
			vllmURL = DefaultVLLMURL
		}
		key := os.Getenv(VLLMKeyEnvKey(model))
		if key == "" {
			key = keys.VLLM
		}
		return NewVLLMClientWithKey(vllmURL, key, model, timeout)

	default:
		return nil, fmt.Errorf("unknown provider %q for model %s (valid: gemini, openrouter, nvidia-nim, local)", provider, model)
	}
}

// VLLMURLEnvKey is the per-model endpoint override variable for model,
// e.g. "microsoft/Phi-4-mini-instruct" → VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_URL.
func VLLMURLEnvKey(model string) string {
	return "VLLM_" + normalizeModelName(model) + "_URL"
}

// VLLMKeyEnvKey is the per-model bearer-token variable for model, e.g.
// "microsoft/Phi-4-mini-instruct" → VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_API_KEY.
func VLLMKeyEnvKey(model string) string {
	return "VLLM_" + normalizeModelName(model) + "_API_KEY"
}

// normalizeModelName converts a model identifier to an uppercase, env-safe
// string.
func normalizeModelName(model string) string {
	var r []rune
	for _, ch := range model {
		if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') {
			r = append(r, ch)
		} else {
			r = append(r, '_')
		}
	}
	return strings.ToUpper(strings.Trim(string(r), "_"))
}
