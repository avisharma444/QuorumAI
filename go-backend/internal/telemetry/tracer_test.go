package telemetry

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
)

var traceparentRE = regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-0[01]$`)

func TestLifecycleAndShutdownFlush(t *testing.T) {
	tr, exp := NewTestTracer()
	_, span := tr.StartSpan(context.Background(), "lifecycle")
	span.End()
	if n := len(exp.GetSpans()); n != 1 {
		t.Fatalf("spans %d", n)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := tr.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if err := tr.Shutdown(ctx); err != nil {
		t.Fatalf("second shutdown must be harmless: %v", err)
	}
}

func TestParentChildAndConcurrentSiblings(t *testing.T) {
	tr, exp := NewTestTracer()
	ctx, root := Start(context.Background(), "root")
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, s := Start(ctx, "child")
			time.Sleep(time.Millisecond)
			s.End()
		}()
	}
	wg.Wait()
	root.End()
	spans := tr.GetSpans()
	if len(spans) != 6 {
		t.Fatalf("spans %d", len(spans))
	}
	rootID := root.SpanContext().SpanID()
	for _, s := range spans {
		if s.SpanContext.TraceID() != root.SpanContext().TraceID() {
			t.Fatal("span in another trace")
		}
		if s.Name == "child" && s.Parent.SpanID() != rootID {
			t.Fatal("goroutine child lost its parent")
		}
	}
	exp.Reset()
}

func TestInjectExtractRoundTrip(t *testing.T) {
	tr, _ := NewTestTracer()
	ctx, span := tr.StartSpan(context.Background(), "client")
	defer span.End()
	req := httptest.NewRequest("POST", "http://python-rag:8000/retrieve", nil)
	tr.InjectHTTPHeaders(ctx, req)
	m := traceparentRE.FindStringSubmatch(req.Header.Get("traceparent"))
	if m == nil || m[1] != span.SpanContext().TraceID().String() || m[2] != span.SpanContext().SpanID().String() {
		t.Fatalf("traceparent %q", req.Header.Get("traceparent"))
	}
	in := tr.ExtractHTTPHeaders(context.Background(), req)
	sc := trace.SpanContextFromContext(in)
	if !sc.IsRemote() || sc.TraceID() != span.SpanContext().TraceID() {
		t.Fatalf("extracted %+v", sc)
	}
	// A child of the extracted context continues the same trace.
	_, server := tr.StartSpan(in, "server")
	if server.SpanContext().TraceID() != span.SpanContext().TraceID() {
		t.Fatal("server span not in the caller's trace")
	}
	server.End()
}

func TestParentBasedSampling(t *testing.T) {
	// Ratio 0.000001 would drop almost every new root, but a sampled
	// incoming parent must always be continued.
	tr, err := NewTracerProvider(Config{ExporterType: ExporterMemory, SampleRatio: 0.000001})
	if err != nil {
		t.Fatal(err)
	}
	h := http.Header{}
	h.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	ctx := Extract(context.Background(), h)
	_, s := tr.StartSpan(ctx, "continued")
	if !s.SpanContext().IsSampled() {
		t.Fatal("sampled parent was not followed")
	}
	s.End()
	h.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
	_, s = tr.StartSpan(Extract(context.Background(), h), "unsampled")
	if s.SpanContext().IsSampled() {
		t.Fatal("unsampled parent must stay unsampled")
	}
	s.End()
}

func TestNoopModeRecordsNothingButPropagates(t *testing.T) {
	tr, err := NewTracerProvider(DefaultConfig())
	if err != nil || tr.Exporter() != ExporterNoop {
		t.Fatal(err)
	}
	h := http.Header{}
	h.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	ctx, s := Start(Extract(context.Background(), h), "x")
	if s.IsRecording() {
		t.Fatal("noop span recorded")
	}
	req := httptest.NewRequest("GET", "http://rag/", nil)
	Inject(ctx, req)
	if !strings.Contains(req.Header.Get("traceparent"), "4bf92f3577b34da6a3ce929d0e0e4736") {
		t.Fatal("noop mode must still forward the incoming trace id")
	}
	if tr.Shutdown(context.Background()) != nil {
		t.Fatal("noop shutdown")
	}
}

func TestStdoutExporterWritesOnFlush(t *testing.T) {
	var buf bytes.Buffer
	tr, err := NewTracerProvider(Config{ExporterType: ExporterStdout, StdoutWriter: &buf})
	if err != nil {
		t.Fatal(err)
	}
	_, s := Start(context.Background(), "stdout-span")
	s.End()
	_ = tr.Shutdown(context.Background())
	if !strings.Contains(buf.String(), "stdout-span") {
		t.Fatal("span not written")
	}
}

func TestUnreachableOTLPCollectorDoesNotBlock(t *testing.T) {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	dead := l.Addr().String()
	l.Close()
	var errs atomic.Int32
	InstallErrorHandler(func() { errs.Add(1) }, time.Hour)
	defer otel.SetErrorHandler(otel.ErrorHandlerFunc(func(error) {}))
	tr, err := NewTracerProvider(Config{ExporterType: ExporterOTLP, OTLPEndpoint: dead, OTLPInsecure: true})
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for i := 0; i < 200; i++ {
		_, s := Start(context.Background(), "work")
		s.End()
	}
	if d := time.Since(start); d > 200*time.Millisecond {
		t.Fatalf("span creation blocked on the exporter: %s", d)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	if err := tr.ForceFlush(ctx); err == nil && errs.Load() == 0 {
		t.Fatal("export to a closed port reported no error")
	}
	_ = tr.Shutdown(ctx)
	if _, err := NewTracerProvider(Config{ExporterType: ExporterOTLP}); err == nil {
		t.Fatal("otlp without endpoint accepted")
	}
	if _, err := NewTracerProvider(Config{ExporterType: "zipkin"}); err == nil {
		t.Fatal("unknown exporter accepted")
	}
}

func TestIDsFromContext(t *testing.T) {
	if TraceIDFromContext(context.Background()) != "" || SpanIDFromContext(context.Background()) != "" {
		t.Fatal("ids without a span")
	}
	NewTestTracer()
	ctx, s := Start(context.Background(), "x")
	defer s.End()
	if len(TraceIDFromContext(ctx)) != 32 || len(SpanIDFromContext(ctx)) != 16 {
		t.Fatal("ids")
	}
	var n NoopTracerProvider
	if c, _ := n.StartSpan(context.Background(), "x"); c == nil {
		t.Fatal("noop ctx")
	}
}
