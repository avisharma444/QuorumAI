package council

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"example.com/councilai-starter/internal/llm"
)

// ChairmanResult is the chairman's structured output (reference contract).
// Confidence is the model's own claim; ConfidenceReported says whether the
// model sent one, so a missing value is never read as 0.0.
type ChairmanResult struct {
	Answer             string  `json:"answer"`
	Reasoning          string  `json:"reasoning"`
	Source             string  `json:"source"`
	Confidence         float64 `json:"confidence"`
	ConfidenceReported bool    `json:"-"`
	// Repaired is true when the JSON needed the narrow string repair in
	// repairJSONStrings before it parsed.
	Repaired bool `json:"-"`
}

// ReflectionResult is the quality audit's structured output (reference contract).
type ReflectionResult struct {
	Quality            string  `json:"quality"` // "good" | "needs_revision"
	Issues             string  `json:"issues"`
	Suggestion         string  `json:"suggestion"`
	Faithful           bool    `json:"faithful"`
	Confidence         float64 `json:"confidence"`
	ConfidenceReported bool    `json:"-"`
}

// chairmanSystemPrompt is the reference prompt, verbatim.
const chairmanSystemPrompt = `You are the Chairman of an AI council. Multiple AI models have independently answered a question, and their peers have reviewed and ranked the answers. Your job is to synthesize the best possible final answer.

Your task:
1. Consider all responses and the peer review feedback
2. Identify the strongest and weakest points across all answers
3. Synthesize a single, authoritative final answer
4. Rate your confidence in the final answer (0.0 to 1.0)

Respond ONLY in JSON format:
{
  "answer": "Your synthesized final answer",
  "reasoning": "How you synthesized the answer and which responses contributed most",
  "source": "chairman-synthesis",
  "confidence": 0.85
}`

// reflectionSystemPrompt is the reference prompt, verbatim.
const reflectionSystemPrompt = `You are a quality assurance agent reviewing an AI-generated answer. Your job is to critically evaluate the answer for accuracy, completeness, and faithfulness to source material.

Evaluate the answer on these criteria:
1. Faithfulness — Is every claim grounded in the provided sources? (If no sources provided, is the answer factually reasonable?)
2. Relevance — Does the answer actually address the question asked?
3. Completeness — Are there important aspects the answer missed?
4. Consistency — Are there internal contradictions?

Respond ONLY in JSON format:
{
  "quality": "good" or "needs_revision",
  "issues": "Description of problems found (empty string if none)",
  "suggestion": "How to improve (empty string if good)",
  "faithful": true/false,
  "confidence": 0.0-1.0
}`

// ── Prompt assembly ─────────────────────────────────────────────────

// evidenceSection renders evidence for chairman, reflection and revision
// prompts. Document evidence keeps the reference heading; web results get
// their own heading so they are never presented as the user's document.
func evidenceSection(ev Evidence, documentHeading string) string {
	if len(ev.Passages) == 0 {
		return ""
	}
	heading := documentHeading
	if ev.Kind == EvidenceWeb {
		heading = "Web Search Results (unverified third-party pages, not the user's document):"
	}
	return fmt.Sprintf("\n%s\n%s\n", heading, strings.Join(ev.Passages, "\n\n---\n\n"))
}

// chairmanUserMessage follows the reference layout: question, evidence,
// anonymized responses, then valid peer reviews only (a malformed review
// carries no trustworthy ranking).
func chairmanUserMessage(question string, ev Evidence, cands []Candidate, reviews []Review) string {
	var reviewParts []string
	for _, r := range reviews {
		if r.Error == "" {
			reviewParts = append(reviewParts, fmt.Sprintf("--- Reviewer %d ---\n%s", r.Index+1, r.Raw))
		}
	}
	reviewsText := "No peer reviews available."
	if len(reviewParts) > 0 {
		reviewsText = strings.Join(reviewParts, "\n\n")
	}
	return fmt.Sprintf("Question: %s\n%s\nIndividual Responses:\n%s\n\nPeer Reviews:\n%s",
		question, evidenceSection(ev, "Document Excerpts:"), anonymizedBlock(cands), reviewsText)
}

func reflectionUserMessage(question string, ev Evidence, answer string) string {
	return fmt.Sprintf("Question: %s\n%s\nAnswer to evaluate:\n%s", question, evidenceSection(ev, "Source Material:"), answer)
}

func revisionUserMessage(question string, ev Evidence, previous string, r ReflectionResult) string {
	return fmt.Sprintf(`Question: %s
%s
Your previous answer:
%s

Quality review feedback:
- Issues: %s
- Suggestion: %s

Please revise your answer to address the issues identified. Maintain accuracy and faithfulness to sources.`,
		question, evidenceSection(ev, "Document Excerpts:"), previous, r.Issues, r.Suggestion)
}

// ── Model calls ─────────────────────────────────────────────────────

// chairmanSynthesize asks the chairman for the structured synthesis. A
// response that does not validate is an error; the caller falls back.
func (c *Council) chairmanSynthesize(ctx context.Context, question string, ev Evidence, cands []Candidate, reviews []Review) (*ChairmanResult, *llm.Usage, error) {
	resp, err := c.call(ctx, "chairman", c.Chairman, chairmanSystemPrompt, chairmanUserMessage(question, ev, cands, reviews), func(o *llm.GenerateOptions) {
		o.ResponseJSON = true
	})
	if err != nil {
		return nil, nil, fmt.Errorf("chairman generation failed: %w", err)
	}
	result, err := ParseChairmanResult(resp.Answer)
	if err != nil {
		return nil, reported(resp), fmt.Errorf("chairman returned invalid output: %w", err)
	}
	return result, reported(resp), nil
}

// reflectOn runs the quality audit with the reference settings.
func (c *Council) reflectOn(ctx context.Context, question string, ev Evidence, answer string) (*ReflectionResult, *llm.Usage, error) {
	resp, err := c.call(ctx, "reflection", c.Chairman, reflectionSystemPrompt, reflectionUserMessage(question, ev, answer), func(o *llm.GenerateOptions) {
		o.Temperature, o.ResponseJSON, o.MaxTokens = llm.Float64Ptr(0.2), true, 500
	})
	if err != nil {
		return nil, nil, fmt.Errorf("reflection failed: %w", err)
	}
	result, err := ParseReflectionResult(resp.Answer)
	if err != nil {
		return nil, reported(resp), fmt.Errorf("reflection returned invalid output: %w", err)
	}
	return result, reported(resp), nil
}

// reviseAnswer asks the chairman for one revision, validated like synthesis.
func (c *Council) reviseAnswer(ctx context.Context, question string, ev Evidence, previous string, r ReflectionResult) (*ChairmanResult, *llm.Usage, error) {
	resp, err := c.call(ctx, "revision", c.Chairman, chairmanSystemPrompt, revisionUserMessage(question, ev, previous, r), func(o *llm.GenerateOptions) {
		o.ResponseJSON = true
	})
	if err != nil {
		return nil, nil, fmt.Errorf("revision failed: %w", err)
	}
	result, err := ParseChairmanResult(resp.Answer)
	if err != nil {
		return nil, reported(resp), fmt.Errorf("revision returned invalid output: %w", err)
	}
	return result, reported(resp), nil
}

// ── Structured parsing ──────────────────────────────────────────────

type rawChairman struct {
	Answer     *string  `json:"answer"`
	Reasoning  *string  `json:"reasoning"`
	Source     *string  `json:"source"`
	Confidence *float64 `json:"confidence"`
}

// ParseChairmanResult validates chairman (and revision) output.
//
// Required: "answer" (non-empty string) and "reasoning" (string; the
// synthesis rationale the API reports). Optional: "source" (the reference
// itself fills a missing source) and "confidence" (absent stays absent;
// present must be a number in [0, 1]). Null counts as absent.
func ParseChairmanResult(raw string) (*ChairmanResult, error) {
	var p rawChairman
	repaired, err := decodeObject(raw, &p)
	if err != nil {
		return nil, err
	}
	if p.Answer == nil || strings.TrimSpace(*p.Answer) == "" {
		return nil, errors.New(`missing or empty "answer"`)
	}
	if p.Reasoning == nil {
		return nil, errors.New(`missing "reasoning"`)
	}
	out := &ChairmanResult{Answer: strings.TrimSpace(*p.Answer), Reasoning: strings.TrimSpace(*p.Reasoning), Repaired: repaired}
	if p.Source != nil {
		out.Source = strings.TrimSpace(*p.Source)
	}
	if p.Confidence != nil {
		if err := checkConfidence(*p.Confidence); err != nil {
			return nil, err
		}
		out.Confidence, out.ConfidenceReported = *p.Confidence, true
	}
	return out, nil
}

type rawReflection struct {
	Quality    *string  `json:"quality"`
	Issues     *string  `json:"issues"`
	Suggestion *string  `json:"suggestion"`
	Faithful   *bool    `json:"faithful"`
	Confidence *float64 `json:"confidence"`
}

// ParseReflectionResult validates the audit. Required: "quality" ("good"
// or "needs_revision"), "issues" and "suggestion" (strings, may be empty)
// and boolean "faithful". "confidence" is optional and validated like the
// chairman's.
func ParseReflectionResult(raw string) (*ReflectionResult, error) {
	var p rawReflection
	if _, err := decodeObject(raw, &p); err != nil {
		return nil, err
	}
	if p.Quality == nil || (*p.Quality != "good" && *p.Quality != "needs_revision") {
		return nil, errors.New(`"quality" must be "good" or "needs_revision"`)
	}
	if p.Issues == nil || p.Suggestion == nil {
		return nil, errors.New(`missing "issues" or "suggestion"`)
	}
	if p.Faithful == nil {
		return nil, errors.New(`missing boolean "faithful"`)
	}
	out := &ReflectionResult{Quality: *p.Quality, Issues: strings.TrimSpace(*p.Issues), Suggestion: strings.TrimSpace(*p.Suggestion), Faithful: *p.Faithful}
	if p.Confidence != nil {
		if err := checkConfidence(*p.Confidence); err != nil {
			return nil, err
		}
		out.Confidence, out.ConfidenceReported = *p.Confidence, true
	}
	return out, nil
}

func checkConfidence(v float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
		return fmt.Errorf(`"confidence" must be between 0 and 1, got %v`, v)
	}
	return nil
}

// decodeObject strictly decodes exactly one JSON object (optionally inside
// one markdown code fence) into out. If the strict decode fails, it tries
// once more after repairJSONStrings and reports whether that was needed.
// Text outside the object is rejected rather than cut away (the reference
// took everything between the first "{" and the last "}").
func decodeObject(raw string, out any) (repaired bool, err error) {
	body := unfence(strings.TrimSpace(raw))
	if body == "" {
		return false, errors.New("empty response")
	}
	if !strings.HasPrefix(body, "{") {
		return false, errors.New("response is not a JSON object")
	}
	if firstErr := decodeStrict(body, out); firstErr != nil {
		fixed := repairJSONStrings(body)
		if fixed == body || decodeStrict(fixed, out) != nil {
			return false, fmt.Errorf("malformed JSON: %w", firstErr)
		}
		return true, nil
	}
	return false, nil
}

func decodeStrict(body string, out any) error {
	dec := json.NewDecoder(bytes.NewReader([]byte(body)))
	if err := dec.Decode(out); err != nil {
		return err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return errors.New("unexpected text after the JSON object")
	}
	return nil
}

// unfence removes one surrounding ```json ... ``` (or ``` ... ```) fence.
func unfence(s string) string {
	if !strings.HasPrefix(s, "```") || !strings.HasSuffix(s, "```") || len(s) < 6 {
		return s
	}
	inner := strings.TrimSuffix(strings.TrimPrefix(s, "```"), "```")
	inner = strings.TrimPrefix(inner, "json")
	return strings.TrimSpace(inner)
}

// repairJSONStrings fixes two common, unambiguous model mistakes inside
// string literals only:
//
//  1. raw control characters (a literal newline, tab, ...) are escaped;
//  2. a backslash that does not start a valid JSON escape (LaTeX such as
//     \alpha or \Lambda) is doubled so it survives as a literal backslash.
//
// Valid escapes (\" \\ \/ \b \f \n \r \t \uXXXX) are never changed.
// Limitation shared with the reference: LaTeX that happens to form a valid
// escape (\frac → form feed + "rac", \theta → tab + "heta") cannot be told
// apart from an intended escape and is left as the model wrote it.
func repairJSONStrings(s string) string {
	var b strings.Builder
	b.Grow(len(s) + 16)
	inString := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if !inString {
			if ch == '"' {
				inString = true
			}
			b.WriteByte(ch)
			continue
		}
		switch {
		case ch == '"':
			inString = false
			b.WriteByte(ch)
		case ch == '\\':
			if i+1 < len(s) && validEscape(s, i+1) {
				b.WriteByte(ch)
				b.WriteByte(s[i+1])
				i++
			} else {
				b.WriteString(`\\`)
			}
		case ch < 0x20:
			fmt.Fprintf(&b, `\u%04x`, ch)
		default:
			b.WriteByte(ch)
		}
	}
	return b.String()
}

// validEscape reports whether s[i] (the character after a backslash) starts
// a valid JSON escape.
func validEscape(s string, i int) bool {
	switch s[i] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return true
	case 'u':
		if i+4 >= len(s) {
			return false
		}
		for _, h := range s[i+1 : i+5] {
			if !strings.ContainsRune("0123456789abcdefABCDEF", h) {
				return false
			}
		}
		return true
	}
	return false
}
