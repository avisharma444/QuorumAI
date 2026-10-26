package middleware

import (
	"context"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"example.com/councilai-starter/internal/metrics"
	"example.com/councilai-starter/internal/telemetry"
)

type requestInfo struct{ UserID string }

var requestInfoKey = ctxKey{"request_info"}

func requestInfoFrom(ctx context.Context) *requestInfo {
	info, _ := ctx.Value(requestInfoKey).(*requestInfo)
	return info
}

// StatusClientClosedRequest labels requests whose client disconnected before
// the handler finished (nginx convention); it is never sent to a client.
const StatusClientClosedRequest = 499

// Logging is the reference's LoggingMiddleware, extended. For every request
// it:
//   - continues an incoming W3C traceparent (or starts a trace) with a
//     server span named "<METHOD> <route pattern>";
//   - counts the request and observes its latency with the chi route
//     pattern as the label (never the raw path; unmatched paths are
//     "unmatched"), tracks in-flight requests and client disconnects;
//   - writes one structured log line.
//
// Headers (tokens), bodies (questions, passwords), query strings and raw
// unmatched paths never reach logs, labels or span attributes. The wrapper
// keeps http.Flusher, so SSE events are flushed as they are written, and an
// SSE request's latency covers the whole stream.
func Logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		metrics.InFlight.Inc()
		defer metrics.InFlight.Dec()

		ctx := telemetry.Extract(r.Context(), r.Header)
		ctx, span := telemetry.StartServer(ctx, "HTTP "+r.Method, attribute.String("http.request.method", r.Method))
		defer span.End()

		info := &requestInfo{}
		ww := chimw.NewWrapResponseWriter(w, r.ProtoMajor)
		if id := telemetry.TraceIDFromContext(ctx); id != "" {
			ww.Header().Set("X-Trace-ID", id)
		}
		next.ServeHTTP(ww, r.WithContext(context.WithValue(ctx, requestInfoKey, info)))

		route := "unmatched"
		if rc := chi.RouteContext(r.Context()); rc != nil && rc.RoutePattern() != "" {
			route = rc.RoutePattern()
		}
		// A client that went away before the handler finished is labelled
		// 499 whatever the handler then tried to write (a 502 written to a
		// closed connection is not a server error). net/http sends 200 when
		// a handler writes nothing.
		status := ww.Status()
		disconnected := r.Context().Err() != nil
		switch {
		case disconnected:
			status = StatusClientClosedRequest
		case status == 0:
			status = http.StatusOK
		}
		elapsed := time.Since(start)
		metrics.RequestCount.WithLabelValues(r.Method, route, strconv.Itoa(status)).Inc()
		metrics.RequestLatency.WithLabelValues(r.Method, route).Observe(elapsed.Seconds())
		if disconnected {
			metrics.ClientDisconnects.WithLabelValues(route).Inc()
			span.SetAttributes(attribute.Bool("http.client_disconnected", true))
		}

		span.SetName(r.Method + " " + route)
		span.SetAttributes(attribute.String("http.route", route), attribute.Int("http.response.status_code", status))
		if status >= 500 {
			span.SetStatus(codes.Error, "HTTP "+strconv.Itoa(status))
		}
		slog.Info("request",
			"method", r.Method, "route", route, "status", status, "bytes", ww.BytesWritten(),
			"duration_ms", elapsed.Milliseconds(), "request_id", chimw.GetReqID(r.Context()),
			"trace_id", telemetry.TraceIDFromContext(ctx), "user_id", info.UserID)
	})
}
