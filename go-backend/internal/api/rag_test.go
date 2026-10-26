package api

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm/llmtest"
)

func TestExcerptLabelsUseOnlyReportedProvenance(t *testing.T) {
	cases := map[string]council.DocumentChunk{
		"[Document d]":                   {Content: "x"},
		"[Document d, page 3]":           {PageNumber: 3},
		"[Document d, pages 2-4]":        {PageNumber: 2, PageEnd: 4},
		"[Document d, page 1, table]":    {PageNumber: 1, ChunkType: "table"},
		"[Document d, pages 1-2, table]": {PageNumber: 1, PageEnd: 2, ChunkType: "table"},
	}
	for want, c := range cases {
		if got := excerptLabel("d", c); got != want {
			t.Fatalf("got %q want %q", got, want)
		}
	}
}

func TestRetrieveParsesReferenceChunkContract(t *testing.T) {
	var body string
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body = string(raw)
		fmt.Fprint(w, `{"chunks":[
		  {"content":"| North | 120 | 135 |","chunk_type":"table","page_number":1,"chunk_index":1,"metadata":{"ocr_method":"layout_aware","page_end":2,"vector_distance":5.5,"rerank_score":3.25}},
		  {"content":"Plain text chunk","chunk_type":"paragraph","page_number":0,"chunk_index":4,"metadata":{"ocr_method":"plain_text","vector_distance":6.0}}],
		  "reranked":false,"rerank_error":"RuntimeError: OOM","candidates":6,"distance_metric":"l2"}`)
	}))
	defer rag.Close()
	srv := newTestServer(testCouncil(llmtest.New(nil)), rag.URL)
	excerpts, info, err := srv.Retrieve(context.Background(), council.QueryRequest{Question: "q", DocID: "doc-1", TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, `"rerank":true`) {
		t.Fatalf("request %s", body)
	}
	if excerpts[0] != "[Document doc-1, pages 1-2, table] | North | 120 | 135 |" || excerpts[1] != "[Document doc-1] Plain text chunk" {
		t.Fatalf("excerpts %q", excerpts)
	}
	if info.Reranked || info.RerankError != "RuntimeError: OOM" || info.Candidates != 6 || info.DistanceMetric != "l2" {
		t.Fatalf("info %+v", info)
	}
	a, b := info.Chunks[0], info.Chunks[1]
	if a.ChunkType != "table" || *a.RerankScore != 3.25 || *a.VectorDistance != 5.5 || a.PageEnd != 2 || a.OCRMethod != "layout_aware" {
		t.Fatalf("chunk %+v", a)
	}
	if b.RerankScore != nil || b.OCRConfidence != nil || b.PageNumber != 0 {
		t.Fatalf("absent metadata must stay absent: %+v", b)
	}
}

func TestQueryResponseCarriesRetrievedChunks(t *testing.T) {
	e := newEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	_, res, _ := e.query(t, `{"question":"capital?","doc_id":"doc-1"}`)
	r := res.Retrieval
	if r == nil || !r.Reranked || len(r.Chunks) != 1 || r.Chunks[0].PageNumber != 1 || *r.Chunks[0].RerankScore != 7.5 {
		t.Fatalf("retrieval %+v", r)
	}
	if !strings.Contains(e.llm.Prompt("member", "a"), "[Document doc-1, page 1] The capital of Zogland is Mirevale.") {
		t.Fatal("member prompt lacks the labelled excerpt")
	}
	// Web-grounded answers carry no document retrieval block.
	e = newEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	if _, res, _ := e.query(t, `{"question":"general"}`); res.Retrieval != nil || res.Grounding != "web" {
		t.Fatalf("%+v", res.Retrieval)
	}
}
