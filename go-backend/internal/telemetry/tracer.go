// Package telemetry configures OpenTelemetry tracing and W3C trace-context
// propagation (reference: internal/telemetry/tracer.go).
//
// Exporters: memory (synchronous, for tests), stdout (diagnostics), noop
// (default: no tracing) and otlp (OTLP/HTTP to a collector such as Jaeger,
// an extension: the reference had no persistent exporter and its main.go
// hard-coded noop). Export runs in a background batch processor, so an
// unreachable collector never blocks or fails a request; SDK errors go to
// the ErrorHandler (logged, rate-limited, and counted).
package telemetry

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// InstrumentationName names the tracer used by all CouncilAI spans.
const InstrumentationName = "councilai-go-backend"

// TracerProvider is the reference contract for tracing operations.
type TracerProvider interface {
	StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span)
	InjectHTTPHeaders(ctx context.Context, req *http.Request)
	ExtractHTTPHeaders(ctx context.Context, req *http.Request) context.Context
}

// ShutdownableTracerProvider adds lifecycle termination (flush + stop).
type ShutdownableTracerProvider interface {
	TracerProvider
	Shutdown(ctx context.Context) error
}

// ExporterType selects where spans go.
type ExporterType string

const (
	ExporterMemory ExporterType = "memory"
	ExporterStdout ExporterType = "stdout"
	ExporterNoop   ExporterType = "noop"
	ExporterOTLP   ExporterType = "otlp"
)

// Config holds tracer settings.
type Config struct {
	ServiceName    string
	ServiceVersion string
	Environment    string
	ExporterType   ExporterType
	SampleRatio    float64 // of new root traces; sampled parents are always followed
	StdoutWriter   io.Writer
	OTLPEndpoint   string // host:port, e.g. jaeger:4318 (OTLP/HTTP)
	OTLPInsecure   bool
}

// DefaultConfig returns the production-safe default: tracing off.
func DefaultConfig() Config {
	return Config{ServiceName: InstrumentationName, ServiceVersion: "2.0.0", Environment: "production",
		ExporterType: ExporterNoop, SampleRatio: 1.0, StdoutWriter: os.Stdout}
}

// Tracer implements ShutdownableTracerProvider.
type Tracer struct {
	provider   trace.TracerProvider
	sdk        *sdktrace.TracerProvider // nil in noop mode
	tracer     trace.Tracer
	propagator propagation.TextMapPropagator
	inMemory   *tracetest.InMemoryExporter
	exporter   ExporterType
}

// Propagator is the W3C TraceContext + Baggage composite (reference).
func Propagator() propagation.TextMapPropagator {
	return propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{})
}

// NewTracerProvider builds the provider for cfg and installs it (and the
// propagator) as the OpenTelemetry globals.
func NewTracerProvider(cfg Config) (*Tracer, error) {
	if cfg.ServiceName == "" {
		cfg.ServiceName = InstrumentationName
	}
	if cfg.ServiceVersion == "" {
		cfg.ServiceVersion = "2.0.0"
	}
	if cfg.SampleRatio <= 0 || cfg.SampleRatio > 1 {
		cfg.SampleRatio = 1.0
	}
	if cfg.ExporterType == "" {
		cfg.ExporterType = ExporterNoop
	}
	prop := Propagator()
	otel.SetTextMapPropagator(prop)

	// Noop: no SDK at all, so disabled tracing costs nothing. Incoming
	// trace context is still propagated to the RAG service unchanged.
	if cfg.ExporterType == ExporterNoop {
		np := noop.NewTracerProvider()
		otel.SetTracerProvider(np)
		return &Tracer{provider: np, tracer: np.Tracer(InstrumentationName), propagator: prop, exporter: ExporterNoop}, nil
	}

	res, err := resource.Merge(resource.Default(), resource.NewWithAttributes(semconv.SchemaURL,
		semconv.ServiceNameKey.String(cfg.ServiceName),
		semconv.ServiceVersionKey.String(cfg.ServiceVersion),
		semconv.DeploymentEnvironmentKey.String(cfg.Environment),
	))
	if err != nil {
		return nil, fmt.Errorf("otel resource: %w", err)
	}
	// ParentBased: a sampled incoming traceparent is always continued, so a
	// distributed trace is never cut in half (the reference's plain ratio
	// sampler could drop children of sampled parents).
	var root sdktrace.Sampler = sdktrace.AlwaysSample()
	if cfg.SampleRatio < 1 {
		root = sdktrace.TraceIDRatioBased(cfg.SampleRatio)
	}
	opts := []sdktrace.TracerProviderOption{sdktrace.WithResource(res), sdktrace.WithSampler(sdktrace.ParentBased(root))}

	var mem *tracetest.InMemoryExporter
	switch cfg.ExporterType {
	case ExporterMemory:
		mem = tracetest.NewInMemoryExporter()
		opts = append(opts, sdktrace.WithSyncer(mem)) // synchronous: deterministic tests
	case ExporterStdout:
		w := cfg.StdoutWriter
		if w == nil {
			w = os.Stdout
		}
		exp, err := stdouttrace.New(stdouttrace.WithWriter(w))
		if err != nil {
			return nil, fmt.Errorf("stdout exporter: %w", err)
		}
		opts = append(opts, sdktrace.WithBatcher(exp))
	case ExporterOTLP:
		if cfg.OTLPEndpoint == "" {
			return nil, fmt.Errorf("otlp exporter needs an endpoint (OTEL_EXPORTER_OTLP_ENDPOINT)")
		}
		eopts := []otlptracehttp.Option{otlptracehttp.WithEndpoint(cfg.OTLPEndpoint), otlptracehttp.WithTimeout(5 * time.Second)}
		if cfg.OTLPInsecure {
			eopts = append(eopts, otlptracehttp.WithInsecure())
		}
		// New does not connect; an unreachable collector surfaces later as
		// export errors, never as a startup or request failure.
		exp, err := otlptracehttp.New(context.Background(), eopts...)
		if err != nil {
			return nil, fmt.Errorf("otlp exporter: %w", err)
		}
		opts = append(opts, sdktrace.WithBatcher(exp, sdktrace.WithMaxQueueSize(2048), sdktrace.WithBatchTimeout(2*time.Second)))
	default:
		return nil, fmt.Errorf("unknown trace exporter %q (memory, stdout, noop, otlp)", cfg.ExporterType)
	}
	tp := sdktrace.NewTracerProvider(opts...)
	otel.SetTracerProvider(tp)
	return &Tracer{provider: tp, sdk: tp, tracer: tp.Tracer(InstrumentationName, trace.WithInstrumentationVersion(cfg.ServiceVersion)),
		propagator: prop, inMemory: mem, exporter: cfg.ExporterType}, nil
}

// NewInMemoryTracerProvider returns a tracer with a synchronous in-memory exporter.
func NewInMemoryTracerProvider() (*Tracer, *tracetest.InMemoryExporter, error) {
	t, err := NewTracerProvider(Config{ServiceName: "test-tracer", ExporterType: ExporterMemory, SampleRatio: 1})
	if err != nil {
		return nil, nil, err
	}
	return t, t.inMemory, nil
}

// NewTestTracer is NewInMemoryTracerProvider that panics on error.
func NewTestTracer() (*Tracer, *tracetest.InMemoryExporter) {
	t, exp, err := NewInMemoryTracerProvider()
	if err != nil {
		panic(err)
	}
	return t, exp
}

// Exporter reports the configured exporter type.
func (t *Tracer) Exporter() ExporterType { return t.exporter }

// StartSpan starts a child of the span in ctx.
func (t *Tracer) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if ctx == nil {
		ctx = context.Background()
	}
	if t == nil || t.tracer == nil {
		return ctx, trace.SpanFromContext(ctx)
	}
	return t.tracer.Start(ctx, name, opts...)
}

// InjectHTTPHeaders writes traceparent/tracestate/baggage into req.
func (t *Tracer) InjectHTTPHeaders(ctx context.Context, req *http.Request) {
	if req == nil {
		return
	}
	if ctx == nil {
		ctx = req.Context()
	}
	Inject(ctx, req)
}

// ExtractHTTPHeaders returns ctx carrying the remote span context of req.
func (t *Tracer) ExtractHTTPHeaders(ctx context.Context, req *http.Request) context.Context {
	if ctx == nil {
		ctx = context.Background()
		if req != nil {
			ctx = req.Context()
		}
	}
	if req == nil {
		return ctx
	}
	return Propagator().Extract(ctx, propagation.HeaderCarrier(req.Header))
}

// Shutdown flushes queued spans and stops the provider.
func (t *Tracer) Shutdown(ctx context.Context) error {
	if t == nil || t.sdk == nil {
		return nil
	}
	return t.sdk.Shutdown(ctx)
}

// ForceFlush exports queued spans now.
func (t *Tracer) ForceFlush(ctx context.Context) error {
	if t == nil || t.sdk == nil {
		return nil
	}
	return t.sdk.ForceFlush(ctx)
}

// GetSpans returns spans recorded by the in-memory exporter.
func (t *Tracer) GetSpans() []tracetest.SpanStub {
	if t == nil || t.inMemory == nil {
		return nil
	}
	return t.inMemory.GetSpans()
}

// Reset clears the in-memory exporter.
func (t *Tracer) Reset() {
	if t != nil && t.inMemory != nil {
		t.inMemory.Reset()
	}
}

// ── Helpers used by the instrumentation ─────────────────────────────

// Start starts a span on the global provider (noop until one is installed).
func Start(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	return otel.Tracer(InstrumentationName).Start(ctx, name, trace.WithAttributes(attrs...))
}

// StartServer starts a server-kind span (an incoming request).
func StartServer(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	return otel.Tracer(InstrumentationName).Start(ctx, name, trace.WithSpanKind(trace.SpanKindServer), trace.WithAttributes(attrs...))
}

// StartClient starts a client-kind span (an outgoing call).
func StartClient(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	return otel.Tracer(InstrumentationName).Start(ctx, name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
}

// Inject writes the current trace context into an outgoing request.
func Inject(ctx context.Context, req *http.Request) {
	if req.Header == nil {
		req.Header = make(http.Header)
	}
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))
}

// Extract reads an incoming request's trace context.
func Extract(ctx context.Context, h http.Header) context.Context {
	return otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier(h))
}

// RecordError marks span as failed with a short category. err's text is
// recorded; callers must not pass errors that embed user content.
func RecordError(span trace.Span, err error) {
	if span == nil || err == nil {
		return
	}
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

// Fail marks span as failed with a fixed category (no error text).
func Fail(span trace.Span, category string) {
	span.SetAttributes(attribute.String("error.type", category))
	span.SetStatus(codes.Error, category)
}

// TraceIDFromContext returns the active trace ID or "".
func TraceIDFromContext(ctx context.Context) string {
	sc := trace.SpanFromContext(ctx).SpanContext()
	if !sc.IsValid() {
		return ""
	}
	return sc.TraceID().String()
}

// SpanIDFromContext returns the active span ID or "".
func SpanIDFromContext(ctx context.Context) string {
	sc := trace.SpanFromContext(ctx).SpanContext()
	if !sc.IsValid() {
		return ""
	}
	return sc.SpanID().String()
}

// InstallErrorHandler routes OpenTelemetry SDK errors (failed exports) to
// count(), and logs at most one line per interval so an unreachable
// collector cannot flood the logs.
func InstallErrorHandler(count func(), interval time.Duration) {
	var mu sync.Mutex
	var last time.Time
	otel.SetErrorHandler(otel.ErrorHandlerFunc(func(err error) {
		count()
		mu.Lock()
		defer mu.Unlock()
		if time.Since(last) >= interval {
			last = time.Now()
			log.Printf("[Telemetry] trace export problem (requests are unaffected): %v", err)
		}
	}))
}

// NoopTracerProvider is the reference's zero-cost provider.
type NoopTracerProvider struct{}

// NewNoopTracerProvider returns a no-op TracerProvider.
func NewNoopTracerProvider() TracerProvider { return &NoopTracerProvider{} }

func (n *NoopTracerProvider) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if ctx == nil {
		ctx = context.Background()
	}
	return ctx, trace.SpanFromContext(ctx)
}
func (n *NoopTracerProvider) InjectHTTPHeaders(ctx context.Context, req *http.Request) {}
func (n *NoopTracerProvider) ExtractHTTPHeaders(ctx context.Context, req *http.Request) context.Context {
	if ctx == nil {
		if req != nil {
			return req.Context()
		}
		return context.Background()
	}
	return ctx
}
func (n *NoopTracerProvider) Shutdown(ctx context.Context) error { return nil }
