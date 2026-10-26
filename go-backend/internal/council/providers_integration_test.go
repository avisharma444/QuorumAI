package council

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"example.com/councilai-starter/internal/llm"
	"example.com/councilai-starter/internal/llm/llmtest"
)

// stageServer is a fake provider endpoint that speaks either the OpenAI or
// the Gemini wire format and answers according to the council stage that
// sent the request. Every reply reports 10/5/15 tokens.
type stageServer struct {
	*httptest.Server
	mu     sync.Mutex
	stages []string
	bodies map[string]map[string]any // last request body per stage
}

func newStageServer(t *testing.T, gemini bool, reply func(stage string) (string, int)) *stageServer {
	t.Helper()
	s := &stageServer{bodies: map[string]map[string]any{}}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		stage := llmtest.StageOf(systemText(body, gemini))
		s.mu.Lock()
		s.stages = append(s.stages, stage)
		s.bodies[stage] = body
		s.mu.Unlock()
		text, status := reply(stage)
		if status != http.StatusOK {
			http.Error(w, `{"error":{"message":"injected failure"}}`, status)
			return
		}
		quoted, _ := json.Marshal(text)
		if gemini {
			fmt.Fprintf(w, `{"candidates":[{"content":{"parts":[{"text":%s}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},"modelVersion":"gemini-served-001"}`, quoted)
		} else {
			fmt.Fprintf(w, `{"model":"served:%s","choices":[{"message":{"content":%s}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`, body["model"], quoted)
		}
	}))
	t.Cleanup(s.Close)
	return s
}

func systemText(body map[string]any, gemini bool) string {
	if gemini {
		si, _ := body["systemInstruction"].(map[string]any)
		parts, _ := si["parts"].([]any)
		if len(parts) > 0 {
			p, _ := parts[0].(map[string]any)
			s, _ := p["text"].(string)
			return s
		}
		return ""
	}
	msgs, _ := body["messages"].([]any)
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if mm["role"] == "system" {
			s, _ := mm["content"].(string)
			return s
		}
	}
	return ""
}

func (s *stageServer) seen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.stages...)
}

func (s *stageServer) body(stage string) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bodies[stage]
}

const fakeKey = "sk-test-SECRET-0123456789"

type replies map[string]string

func script(r replies) func(string) (string, int) {
	return func(stage string) (string, int) {
		if text, ok := r[stage]; ok {
			if strings.HasPrefix(text, "HTTP ") {
				var code int
				fmt.Sscanf(text, "HTTP %d", &code)
				return "", code
			}
			return text, http.StatusOK
		}
		return "unscripted stage " + stage, http.StatusOK
	}
}

// mixedCouncil wires OpenRouter, Gemini and NIM members plus a vLLM chairman,
// each pointing at its own fake server.
func mixedCouncil(t *testing.T, or, gm, nim, chair replies) (*Council, map[string]*stageServer) {
	t.Helper()
	s := map[string]*stageServer{
		"openrouter": newStageServer(t, false, script(or)),
		"gemini":     newStageServer(t, true, script(gm)),
		"nim":        newStageServer(t, false, script(nim)),
		"chair":      newStageServer(t, false, script(chair)),
	}
	members := []llm.LLMClient{
		llm.NewOpenRouterClient(fakeKey, s["openrouter"].URL+"/api/v1/chat/completions", "or/model", 5*time.Second),
		llm.NewGeminiClientWithURL(fakeKey, s["gemini"].URL+"/v1beta", "gemini-test", 5*time.Second),
		llm.NewNIMClient(fakeKey, s["nim"].URL+"/v1/chat/completions", "nim/model", 5*time.Second),
	}
	chairman := llm.NewVLLMClient(s["chair"].URL+"/v1/chat/completions", "chair-model", 5*time.Second)
	return New(members, chairman, 5*time.Second), s
}

func TestMixedProviderCouncilEndToEnd(t *testing.T) {
	c, s := mixedCouncil(t,
		replies{"member": "OpenRouter answer", "review": "RANKING: B, C, A\nREASONING: B cites more."},
		replies{"member": "Gemini answer", "review": "RANKING: B, A, C\nREASONING: B is complete."},
		replies{"member": "NIM answer", "review": "RANKING: C, B, A\nREASONING: C is concise."},
		replies{"chair": `{"answer":"Chairman synthesis","reasoning":"B was strongest","source":"chairman-synthesis","confidence":0.8}`},
	)
	log := &eventLog{}
	res, err := c.Run(context.Background(), QueryRequest{Question: "Compare X and Y", Strategy: "council"}, Evidence{}, log.emit)
	if err != nil {
		t.Fatal(err)
	}
	// Totals: A=1+2+1=4, B=3+3+2=8, C=2+1+3=6, so Gemini's answer (B) ranks first.
	if res.Answer != "Chairman synthesis" || res.Source != "chairman:local:chair-model" || res.Selected != "B" {
		t.Fatalf("unexpected result: %+v", res)
	}
	wantModels := []string{"or/model", "gemini:gemini-test", "nim:nim/model"}
	wantServed := []string{"served:or/model", "gemini-served-001", "served:nim/model"}
	wantScores := []int{4, 8, 6}
	for i, cand := range res.Candidates {
		if cand.Model != wantModels[i] || cand.ServedModel != wantServed[i] || cand.Score != wantScores[i] ||
			cand.Usage == nil || cand.Usage.TotalTokens != 15 {
			t.Fatalf("candidate %d: %+v", i, cand)
		}
	}
	for i, r := range res.Reviews {
		if r.Reviewer != wantModels[i] || r.Error != "" || r.Usage == nil || r.ServedModel != wantServed[i] {
			t.Fatalf("review %d: %+v", i, r)
		}
	}
	// 3 candidates + 3 reviews + 1 chairman call, 15 tokens each.
	if res.TokenUsage == nil || *res.TokenUsage != (llm.Usage{PromptTokens: 70, CompletionTokens: 35, TotalTokens: 105}) {
		t.Fatalf("token usage %+v", res.TokenUsage)
	}
	for name, want := range map[string]string{"openrouter": "member,review", "gemini": "member,review", "nim": "member,review", "chair": "chair"} {
		got := s[name].seen()
		if strings.Join(sortedCopy(got), ",") != want {
			t.Fatalf("%s saw stages %v, want %s", name, got, want)
		}
	}
	// Gemini members use search grounding when there is no document, as in the
	// reference; review calls do not.
	if _, ok := s["gemini"].body("member")["tools"]; !ok {
		t.Fatal("Gemini member call should enable googleSearch without document context")
	}
	if _, ok := s["gemini"].body("review")["tools"]; ok {
		t.Fatal("review calls must not enable search")
	}
	// Reviewers see anonymous labels, not model identities.
	reviewBody, _ := json.Marshal(s["nim"].body("review"))
	for _, id := range wantModels {
		if strings.Contains(string(reviewBody), id) && id != "nim/model" { // own model field is expected
			t.Fatalf("review prompt leaks %s", id)
		}
	}
	if !strings.Contains(string(reviewBody), "=== Response B ===\\nGemini answer") {
		t.Fatalf("review prompt missing anonymized answer: %s", reviewBody)
	}
	names := strings.Join(log.names, ",")
	if strings.Count(names, "candidate") != 3 || strings.Count(names, "review") != 3 || !strings.HasSuffix(names, "final") {
		t.Fatalf("events %s", names)
	}
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	for i := range out {
		for j := i + 1; j < len(out); j++ {
			if out[j] < out[i] {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out
}

func TestMixedProviderPartialFailure(t *testing.T) {
	c, _ := mixedCouncil(t,
		replies{"member": "OpenRouter answer", "review": "RANKING: B, A\nREASONING: x"},
		replies{"member": "Gemini answer", "review": "RANKING: B, A\nREASONING: x"},
		replies{"member": "HTTP 500", "review": "RANKING: A, B\nREASONING: x"},
		replies{"chair": "HTTP 401"},
	)
	res, err := c.Run(context.Background(), QueryRequest{Question: "Q", Strategy: "council_deep"}, Evidence{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Candidates) != 2 || res.Candidates[1].Model != "gemini:gemini-test" || res.Candidates[1].Label != "B" {
		t.Fatalf("candidates %+v", res.Candidates)
	}
	// NIM's answer failed but it still reviewed; B = 2+2+1 = 5 beats A = 4.
	if res.Selected != "B" || res.Answer != "Gemini answer" || res.Source != "gemini:gemini-test (chairman-failed-fallback)" || res.Reflection != nil {
		t.Fatalf("result %+v", res)
	}
	if res.TokenUsage.TotalTokens != 15*5 { // 2 candidates + 3 reviews; failed calls report nothing
		t.Fatalf("usage %+v", res.TokenUsage)
	}
}

func TestDirectAndDeepWithGeminiChairman(t *testing.T) {
	chair := newStageServer(t, true, script(replies{
		"direct": "direct answer", "chair": `{"answer":"draft","reasoning":"r","source":"chairman-synthesis"}`, "reflect": `{"quality":"good","issues":"","suggestion":"","faithful":true}`,
	}))
	chairman := llm.NewGeminiClientWithURL(fakeKey, chair.URL, "gemini-chair", 5*time.Second)
	f := llmtest.New(nil)
	c := New(f.Clients("a", "b", "c"), chairman, 5*time.Second)

	res, err := c.Run(context.Background(), QueryRequest{Question: "hi", Strategy: "direct"}, Evidence{}, nil)
	if err != nil || res.Answer != "direct answer" || res.Source != "direct:gemini:gemini-chair" || res.TokenUsage.TotalTokens != 15 {
		t.Fatalf("direct: %+v %v", res, err)
	}
	if _, ok := chair.body("direct")["tools"]; !ok {
		t.Fatal("direct Gemini call without documents should enable search")
	}
	if f.Count("member")+f.Count("review") != 0 {
		t.Fatal("direct mode must bypass the council")
	}
	if _, err := c.Run(context.Background(), QueryRequest{Question: "hi", Strategy: "direct"}, Evidence{Kind: EvidenceDocument, Passages: []string{"evidence"}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, ok := chair.body("direct")["tools"]; ok {
		t.Fatal("search must be off when document evidence is supplied")
	}

	res, err = c.Run(context.Background(), QueryRequest{Question: "Compare", Strategy: "council_deep"}, Evidence{}, nil)
	if err != nil || res.Answer != "draft" || res.Reflection == nil || res.Reflection.Quality != "good" || !*res.Reflection.Faithful {
		t.Fatalf("deep: %+v %v", res, err)
	}
	gen, _ := chair.body("reflect")["generationConfig"].(map[string]any)
	if gen["temperature"] != 0.2 || gen["responseMimeType"] != "application/json" || gen["maxOutputTokens"] != 500.0 {
		t.Fatalf("reflection settings not mapped: %v", gen)
	}
	if res.TokenUsage == nil || res.TokenUsage.TotalTokens != 30 { // chairman + reflection; mocks report none
		t.Fatalf("usage %+v", res.TokenUsage)
	}
}

func TestUsageIsOmittedWhenNoProviderReportsIt(t *testing.T) {
	res, _ := mustRun(t, llmtest.New(nil), "council")
	if res.TokenUsage != nil || res.Candidates[0].Usage != nil || res.Reviews[0].Usage != nil {
		t.Fatalf("usage invented: %+v", res.TokenUsage)
	}
}
