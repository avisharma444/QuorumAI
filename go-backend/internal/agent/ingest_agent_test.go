package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/llm/llmtest"
)

func TestSummarizeRequestMatchesReference(t *testing.T) {
	f := llmtest.New(map[string]llmtest.Reply{"summarize/ingest": {Text: "  The document covers Zogland's geography and capital, Mirevale.  "}})
	a := NewIngestAgent(f.Client("ingest"))
	got, err := a.SummarizeDocument(context.Background(), "Zogland facts.\nThe capital is Mirevale.")
	if err != nil {
		t.Fatal(err)
	}
	if got != "The document covers Zogland's geography and capital, Mirevale." {
		t.Fatalf("summary %q", got)
	}
	opts := f.Options("summarize", "ingest")
	if opts.Messages[0].Content != summarizePrompt || opts.Messages[1].Content != "Document Sample:\nZogland facts.\nThe capital is Mirevale." {
		t.Fatalf("messages %+v", opts.Messages)
	}
	if opts.Temperature == nil || *opts.Temperature != 0.2 || opts.MaxTokens != 200 || opts.ResponseJSON || opts.EnableSearch {
		t.Fatalf("settings %+v", opts)
	}
	if a.ModelName() != "ingest" {
		t.Fatal("agent must report its own model")
	}
}

func TestSummarizeEmptyPreviewSkipsModel(t *testing.T) {
	f := llmtest.New(nil)
	a := NewIngestAgent(f.Client("ingest"))
	for _, preview := range []string{"", "  \n\t "} {
		if _, err := a.SummarizeDocument(context.Background(), preview); !errors.Is(err, ErrEmptyPreview) {
			t.Fatalf("%q: %v", preview, err)
		}
	}
	if f.Count("summarize") != 0 {
		t.Fatal("model called for an empty preview")
	}
}

func TestSummarizeFailures(t *testing.T) {
	for name, reply := range map[string]llmtest.Reply{
		"model error":   {Err: errors.New("HTTP 429")},
		"empty summary": {Text: "   "},
	} {
		f := llmtest.New(map[string]llmtest.Reply{"summarize/ingest": reply})
		if s, err := NewIngestAgent(f.Client("ingest")).SummarizeDocument(context.Background(), "text"); err == nil || s != "" {
			t.Fatalf("%s: got %q, %v", name, s, err)
		}
	}
	f := llmtest.New(nil)
	f.Hook = func(ctx context.Context, _, _ string) error { <-ctx.Done(); return ctx.Err() }
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)
	if _, err := NewIngestAgent(f.Client("ingest")).SummarizeDocument(ctx, "text"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}

// The agent uses whichever provider adapter it is given; here a real
// OpenAI-compatible adapter against a local server.
func TestSummarizeThroughProviderAdapter(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		fmt.Fprint(w, `{"choices":[{"message":{"content":"A summary from the provider."}}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}`)
	}))
	defer srv.Close()
	a := NewIngestAgent(llm.NewVLLMClient(srv.URL+"/v1/chat/completions", "summarizer-model", time.Second))
	got, err := a.SummarizeDocument(context.Background(), "sample")
	if err != nil || got != "A summary from the provider." {
		t.Fatalf("%q %v", got, err)
	}
	if body["model"] != "summarizer-model" || body["temperature"] != 0.2 || body["max_tokens"] != 200.0 {
		t.Fatalf("request %v", body)
	}
	if a.ModelName() != "local:summarizer-model" {
		t.Fatal(a.ModelName())
	}
}
