package api

import (
	"encoding/json"
	"strings"
	"testing"

	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm/llmtest"
)

const (
	needsRevision = `{"quality":"needs_revision","issues":"misses Y","suggestion":"add Y","faithful":false}`
	revised       = `{"answer":"revised final","reasoning":"added Y","source":"chairman-synthesis","confidence":0.7}`
)

func TestDeepSSEStreamsRevisionBeforeSingleFinal(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{
		"route/router":  route("council_deep", false),
		"chair/chair":   {Text: `{"answer":"draft","reasoning":"r","source":"chairman-synthesis","confidence":0.5}`},
		"reflect/chair": {Text: needsRevision},
		"revise/chair":  {Text: revised},
	})
	events := parseSSE(t, postQuery(t, e.srv, `{"question":"deep sse question"}`, true).Body.String())
	var names []string
	for _, ev := range events {
		names = append(names, ev.name)
	}
	want := "start,routing,retrieved,candidate,candidate,candidate,review,review,review,synthesis,reflection,revision,final"
	if strings.Join(names, ",") != want {
		t.Fatalf("events\n got %s\nwant %s", strings.Join(names, ","), want)
	}
	if strings.Count(strings.Join(names, ","), "final") != 1 {
		t.Fatal("exactly one final event")
	}
	var final council.QueryResult
	_ = json.Unmarshal([]byte(events[len(events)-1].data), &final)
	if final.Answer != "revised final" || final.Reasoning != "added Y" || *final.Confidence != 0.7 || final.ConfidenceSource != "revision" {
		t.Fatalf("final %+v", final)
	}
	r := final.Reflection
	if r.Quality != "needs_revision" || *r.Faithful || r.Issues != "misses Y" || r.Revision.Status != "succeeded" || r.Revision.PreviousAnswer != "draft" {
		t.Fatalf("reflection %+v", r)
	}
	if final.Routing == nil || final.Routing.Strategy != "council_deep" || final.Grounding != "web" {
		t.Fatalf("routing/grounding lost: %+v", final.Routing)
	}
	// The synthesis event must not carry an answer that is later replaced.
	if strings.Contains(events[9].data, "draft") {
		t.Fatalf("synthesis event leaked the pre-revision answer: %s", events[9].data)
	}
}

func TestJSONResponseShape(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{
		"route/router": route("council", false),
		"chair/chair":  {Text: `{"answer":"A","reasoning":"R","source":"chairman-synthesis"}`},
	})
	_, _, body := e.query(t, `{"question":"shape question"}`)
	var m map[string]any
	_ = json.Unmarshal([]byte(body), &m)
	if _, ok := m["confidence"]; ok {
		t.Fatalf("confidence must be absent when the chairman gave none: %s", body)
	}
	for _, k := range []string{"answer", "reasoning", "source", "synthesis", "candidates", "reviews", "routing", "grounding", "confidence_note"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("missing %q in %s", k, body)
		}
	}
	if _, ok := m["reflection"]; ok {
		t.Fatal("council mode has no reflection")
	}
}

func TestOnlyCompleteAnswersAreCached(t *testing.T) {
	cases := map[string]struct {
		replies  map[string]llmtest.Reply
		cached   bool
		degraded string
	}{
		"complete deep with revision": {map[string]llmtest.Reply{"route/router": route("council_deep", false), "reflect/chair": {Text: needsRevision}, "revise/chair": {Text: revised}}, true, ""},
		"chairman malformed":          {map[string]llmtest.Reply{"route/router": route("council", false), "chair/chair": {Text: "not json"}}, false, "chairman-failed-fallback"},
		"reflection failed":           {map[string]llmtest.Reply{"route/router": route("council_deep", false), "reflect/chair": {Text: "{}"}}, false, "reflection-failed"},
		"revision failed":             {map[string]llmtest.Reply{"route/router": route("council_deep", false), "reflect/chair": {Text: needsRevision}, "revise/chair": {Text: "nope"}}, false, "revision-failed"},
		"single survivor":             {map[string]llmtest.Reply{"route/router": route("council", false), "member/a": {Err: errProvider}, "member/b": {Err: errProvider}}, false, "single-response"},
	}
	for name, tc := range cases {
		e := newEnv(t, tc.replies)
		_, first, _ := e.query(t, `{"question":"cache question"}`)
		if strings.Join(first.Degraded, ",") != tc.degraded {
			t.Fatalf("%s: degraded %v", name, first.Degraded)
		}
		_, second, _ := e.query(t, `{"question":"cache question"}`)
		if second.CacheHit != tc.cached {
			t.Fatalf("%s: cache hit %v, want %v", name, second.CacheHit, tc.cached)
		}
	}
}
