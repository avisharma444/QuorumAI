package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

type Model interface {
	Generate(context.Context, string, string, string) (string, error)
}

type APIModel struct {
	Key     string
	BaseURL string
	Client  *http.Client
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (m *APIModel) Generate(ctx context.Context, model, system, prompt string) (string, error) {
	if m.Key == "" {
		return "", errors.New("no MODEL_API_KEY configured; use MOCK_LLM=true for offline testing")
	}
	body, err := json.Marshal(map[string]any{
		"model":       model,
		"messages":    []chatMessage{{"system", system}, {"user", prompt}},
		"temperature": 0.2,
		"max_tokens":  700,
	})
	if err != nil {
		return "", err
	}
	url := strings.TrimRight(m.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+m.Key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://localhost")
	req.Header.Set("X-Title", "CouncilAI Starter")
	r, err := m.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		return "", err
	}
	if r.StatusCode < 200 || r.StatusCode >= 300 {
		return "", fmt.Errorf("provider HTTP %d: %s", r.StatusCode, string(raw[:min(len(raw), 500)]))
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 || strings.TrimSpace(out.Choices[0].Message.Content) == "" {
		return "", errors.New("empty model response")
	}
	return out.Choices[0].Message.Content, nil
}

type MockModel struct{ Delay time.Duration }

var mockResponseLabel = regexp.MustCompile(`(?m)^=== Response ([A-Z]) ===$`)

func (m *MockModel) Generate(ctx context.Context, model, system, prompt string) (string, error) {
	if m.Delay > 0 {
		select {
		case <-time.After(m.Delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	switch {
	case strings.Contains(system, "REVIEW_AGENT"):
		// Rank the responses in the order they appear. This proves the
		// pipeline works offline; it is not a quality judgment.
		labels := mockResponseLabel.FindAllStringSubmatch(prompt, -1)
		ranking := make([]string, len(labels))
		for i, m := range labels {
			ranking[i] = m[1]
		}
		return "RANKING: " + strings.Join(ranking, ", ") + "\nREASONING: Mock reviewer; deterministic order, not a quality judgment.", nil
	case strings.Contains(system, "CHAIRMAN_AGENT"):
		return "[MOCK consensus] Representative council synthesis. Configure a real API key to generate meaningful answers.", nil
	case strings.Contains(system, "REFLECTION_AGENT"):
		return `{"quality":"good","reason":"mock validation only"}`, nil
	default:
		short := prompt
		if len(short) > 180 {
			short = short[:180]
		}
		return "[MOCK " + model + "] Candidate response for: " + short, nil
	}
}

func newConfiguredModel() Model {
	if strings.EqualFold(os.Getenv("MOCK_LLM"), "true") {
		return &MockModel{}
	}
	base := env("MODEL_BASE_URL", "https://openrouter.ai/api/v1")
	return &APIModel{Key: os.Getenv("MODEL_API_KEY"), BaseURL: base, Client: &http.Client{Timeout: 45 * time.Second}}
}
