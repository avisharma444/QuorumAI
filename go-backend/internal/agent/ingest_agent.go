package agent

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

	"example.com/councilai-starter/internal/llm"
)

// IngestAgent handles ingestion-time tasks, currently document summarization.
type IngestAgent struct {
	client llm.LLMClient
}

// NewIngestAgent creates an agent backed by the given client.
func NewIngestAgent(client llm.LLMClient) *IngestAgent {
	return &IngestAgent{client: client}
}

// ModelName identifies the ingestion model.
func (a *IngestAgent) ModelName() string { return a.client.ModelName() }

// summarizePrompt is the reference prompt, verbatim.
const summarizePrompt = `You are a document summarization agent for a RAG pipeline.
Your job is to read the provided Beginning-Middle-End (BME) sample of a document and generate a concise 3-4 sentence summary of its core topics.
This summary will be used by a routing agent to determine if future user questions actually require retrieving content from this document.
Focus strictly on the factual subjects, entities, and themes present in the text.`

// ErrEmptyPreview is returned when there is no text to summarize.
var ErrEmptyPreview = errors.New("preview text is empty")

// SummarizeDocument summarizes a BME preview with the configured model
// (temperature 0.2, at most 200 tokens).
func (a *IngestAgent) SummarizeDocument(ctx context.Context, previewText string) (string, error) {
	if strings.TrimSpace(previewText) == "" {
		return "", ErrEmptyPreview
	}
	resp, err := a.client.GenerateChat(ctx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: summarizePrompt},
			{Role: "user", Content: fmt.Sprintf("Document Sample:\n%s", previewText)},
		},
		Temperature: llm.Float64Ptr(0.2),
		MaxTokens:   200,
	})
	if err != nil {
		log.Printf("[IngestAgent] Summarization failed: %v", err)
		return "", err
	}
	summary := strings.TrimSpace(resp.Answer)
	if summary == "" {
		return "", errors.New("summarization returned an empty summary")
	}
	return summary, nil
}
