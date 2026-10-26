// Package middleware provides authentication, rate limiting and request
// logging for the HTTP API (reference internal/api/middleware).
package middleware

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"context"
	"encoding/json"
	"net/http"
	"strings"

	"example.com/councilai-starter/internal/auth"
)

type ctxKey struct{ name string }

// userIDKey holds the authenticated user's UUID (typed key: no collisions
// with other packages' string keys).
var userIDKey = ctxKey{"user_id"}

// WithUserID returns ctx carrying an authenticated user ID.
func WithUserID(ctx context.Context, userID string) context.Context {
	if info := requestInfoFrom(ctx); info != nil {
		info.UserID = userID
	}
	return context.WithValue(ctx, userIDKey, userID)
}

// UserID returns the authenticated user ID and whether one is present. The
// reference returned "anonymous" for unauthenticated requests, which a
// handler could mistake for an identity.
func UserID(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(userIDKey).(string)
	return id, ok && id != ""
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// BearerToken extracts the token from "Authorization: Bearer <token>".
func BearerToken(header string) (string, bool) {
	scheme, token, ok := strings.Cut(header, " ")
	if !ok || !strings.EqualFold(scheme, "bearer") {
		return "", false
	}
	token = strings.TrimSpace(token)
	if token == "" || strings.ContainsAny(token, " \t") || strings.Count(token, ".") != 2 {
		return "", false
	}
	return token, true
}

// Auth validates the Bearer JWT and stores the token's user UUID in the
// request context. Any client-supplied user identity is ignored.
func Auth(jwtManager *auth.JWTManager) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			header := r.Header.Get("Authorization")
			if header == "" {
				w.Header().Set("WWW-Authenticate", `Bearer realm="councilai"`)
				authFailed(r, "missing_token")
				writeJSONError(w, http.StatusUnauthorized, "missing authorization header")
				return
			}
			token, ok := BearerToken(header)
			if !ok {
				w.Header().Set("WWW-Authenticate", `Bearer realm="councilai", error="invalid_request"`)
				authFailed(r, "malformed_header")
				writeJSONError(w, http.StatusUnauthorized, "invalid authorization header")
				return
			}
			claims, err := jwtManager.ValidateToken(token)
			if err != nil {
				w.Header().Set("WWW-Authenticate", `Bearer realm="councilai", error="invalid_token"`)
				authFailed(r, "invalid_token")
				writeJSONError(w, http.StatusUnauthorized, "invalid or expired token")
				return
			}
			next.ServeHTTP(w, r.WithContext(WithUserID(r.Context(), claims.UserID)))
		})
	}
}

// authFailed labels the request span with a fixed category; the token and
// the validation error text are never recorded.
func authFailed(r *http.Request, category string) {
	trace.SpanFromContext(r.Context()).SetAttributes(attribute.String("auth.failure", category))
}
