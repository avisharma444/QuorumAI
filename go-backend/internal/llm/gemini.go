package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// DefaultGeminiURL is the Gemini REST API base URL.
const DefaultGeminiURL = "https://generativelanguage.googleapis.com/v1beta"

// GeminiClient talks to the native Google Gemini REST API.
type GeminiClient struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
}

// NewGeminiClient builds a client for one Gemini model on the public API.
func NewGeminiClient(apiKey, model string, timeout time.Duration) *GeminiClient {
	return NewGeminiClientWithURL(apiKey, DefaultGeminiURL, model, timeout)
}

// NewGeminiClientWithURL builds a client against a custom base URL (used by
// configuration overrides and tests).
func NewGeminiClientWithURL(apiKey, baseURL, model string, timeout time.Duration) *GeminiClient {
	if baseURL == "" {
		baseURL = DefaultGeminiURL
	}
	return &GeminiClient{apiKey: apiKey, baseURL: strings.TrimRight(baseURL, "/"), model: model, client: &http.Client{Timeout: timeout}}
}

// ── Gemini API types ────────────────────────────────────────────────

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
}

type geminiTool struct {
	GoogleSearch *struct{} `json:"googleSearch,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerationConfig struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	MaxOutputTokens  int      `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string   `json:"responseMimeType,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text    string `json:"text"`
				Thought bool   `json:"thought"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

func buildGeminiGenerationConfig(opts GenerateOptions) *geminiGenerationConfig {
	cfg := geminiGenerationConfig{Temperature: opts.Temperature}
	if opts.MaxTokens > 0 {
		cfg.MaxOutputTokens = opts.MaxTokens
	}
	if opts.ResponseJSON {
		cfg.ResponseMimeType = "application/json"
	}
	if cfg == (geminiGenerationConfig{}) {
		return nil
	}
	return &cfg
}

// buildGeminiRequest maps chat messages onto Gemini's format: system
// messages become systemInstruction, "assistant" becomes role "model".
func buildGeminiRequest(opts GenerateOptions) (*geminiRequest, error) {
	req := geminiRequest{GenerationConfig: buildGeminiGenerationConfig(opts)}
	if opts.EnableSearch {
		req.Tools = []geminiTool{{GoogleSearch: &struct{}{}}}
	}
	for _, msg := range opts.Messages {
		part := geminiPart{Text: msg.Content}
		switch msg.Role {
		case "system":
			// The reference kept only the last system message; keep them all.
			if req.SystemInstruction == nil {
				req.SystemInstruction = &geminiContent{}
			}
			req.SystemInstruction.Parts = append(req.SystemInstruction.Parts, part)
		case "user":
			req.Contents = append(req.Contents, geminiContent{Role: "user", Parts: []geminiPart{part}})
		case "assistant":
			req.Contents = append(req.Contents, geminiContent{Role: "model", Parts: []geminiPart{part}})
		default:
			return nil, fmt.Errorf("unsupported message role %q", msg.Role)
		}
	}
	if len(req.Contents) == 0 {
		return nil, fmt.Errorf("at least one user or assistant message is required")
	}
	return &req, nil
}

func parseGeminiResponse(body []byte, model string) (*Response, error) {
	var g geminiResponse
	if err := json.Unmarshal(body, &g); err != nil {
		return nil, fmt.Errorf("Gemini: failed to parse response: %w", err)
	}
	if len(g.Candidates) == 0 {
		if g.PromptFeedback != nil && g.PromptFeedback.BlockReason != "" {
			return nil, fmt.Errorf("Gemini blocked the prompt: %s", g.PromptFeedback.BlockReason)
		}
		return nil, fmt.Errorf("no content in Gemini response")
	}
	// Grounded answers can arrive in several text parts; thought parts are
	// internal reasoning and are not part of the answer.
	var text strings.Builder
	for _, p := range g.Candidates[0].Content.Parts {
		if !p.Thought {
			text.WriteString(p.Text)
		}
	}
	if strings.TrimSpace(text.String()) == "" {
		return nil, fmt.Errorf("no content in Gemini response (finishReason %q)", g.Candidates[0].FinishReason)
	}
	out := &Response{Answer: text.String(), Model: model}
	if g.ModelVersion != "" {
		out.Model = g.ModelVersion
	}
	if u := g.UsageMetadata; u != nil {
		out.Usage = Usage{PromptTokens: u.PromptTokenCount, CompletionTokens: u.CandidatesTokenCount, TotalTokens: u.TotalTokenCount}
	}
	return out, nil
}

// Generate sends a single user prompt.
func (c *GeminiClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{Messages: []Message{{Role: "user", Content: prompt}}})
}

// GenerateChat sends a full conversation to Gemini's generateContent.
func (c *GeminiClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	gReq, err := buildGeminiRequest(opts)
	if err != nil {
		return nil, fmt.Errorf("Gemini: %w", err)
	}
	payload, err := json.Marshal(gReq)
	if err != nil {
		return nil, fmt.Errorf("Gemini: failed to marshal request: %w", err)
	}
	// The key goes in a header, not the URL: Go's transport errors include
	// the URL, so the reference's ?key= query leaked it into logs.
	endpoint := c.baseURL + "/models/" + url.PathEscape(c.model) + ":generateContent"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("Gemini: failed to create request: %w", err)
	}
	req.Header.Set("x-goog-api-key", c.apiKey)
	body, err := postJSON(c.client, req, "Gemini")
	if err != nil {
		return nil, err
	}
	return parseGeminiResponse(body, c.model)
}

// ModelName returns the model prefixed with "gemini:".
func (c *GeminiClient) ModelName() string {
	return "gemini:" + c.model
}
