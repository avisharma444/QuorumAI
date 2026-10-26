package middleware

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	chimw "github.com/go-chi/chi/v5/middleware"
)

type requestInfo struct{ UserID string }

var requestInfoKey = ctxKey{"request_info"}

func requestInfoFrom(ctx context.Context) *requestInfo {
	info, _ := ctx.Value(requestInfoKey).(*requestInfo)
	return info
}

// Logging writes one structured line per request: method, path (no query
// string), status, size, duration, request ID and authenticated user ID.
// Headers (tokens), bodies (questions, passwords) and query strings are
// never logged.
func Logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		info := &requestInfo{}
		ww := chimw.NewWrapResponseWriter(w, r.ProtoMajor) // keeps http.Flusher for SSE
		next.ServeHTTP(ww, r.WithContext(context.WithValue(r.Context(), requestInfoKey, info)))
		slog.Info("request",
			"method", r.Method, "path", r.URL.Path, "status", ww.Status(), "bytes", ww.BytesWritten(),
			"duration_ms", time.Since(start).Milliseconds(), "request_id", chimw.GetReqID(r.Context()), "user_id", info.UserID)
	})
}
