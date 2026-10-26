package api

import (
	"example.com/councilai-starter/internal/metrics"

	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"example.com/councilai-starter/internal/api/middleware"
	"example.com/councilai-starter/internal/auth"
	"example.com/councilai-starter/internal/memory"
)

type credentials struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type authResponse struct {
	Token     string `json:"token"`
	UserID    string `json:"user_id"` // the user's UUID (the reference returned the username)
	Username  string `json:"username"`
	ExpiresIn int64  `json:"expires_in"` // seconds
	Message   string `json:"message,omitempty"`
}

func decodeCredentials(w http.ResponseWriter, r *http.Request) (credentials, bool) {
	r.Body = http.MaxBytesReader(w, r.Body, 4<<10)
	var c credentials
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		respondError(w, http.StatusBadRequest, "invalid request body")
		return c, false
	}
	return c, true
}

func (s *Server) issue(w http.ResponseWriter, status int, u *auth.User, msg string) {
	token, err := s.JWT.GenerateToken(u.ID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "failed to issue token")
		return
	}
	respond(w, status, authResponse{Token: token, UserID: u.ID, Username: u.Username, ExpiresIn: int64(s.JWT.Expiration() / time.Second), Message: msg})
}

// handleRegister: POST /api/v1/register (reference HandleRegister).
func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	c, ok := decodeCredentials(w, r)
	if !ok {
		return
	}
	username, err := auth.NormalizeUsername(c.Username)
	if err != nil {
		respondError(w, http.StatusBadRequest, err.Error())
		return
	}
	hash, err := auth.HashPassword(c.Password, s.BcryptCost)
	if err != nil {
		if errors.Is(err, auth.ErrInvalidInput) {
			respondError(w, http.StatusBadRequest, err.Error())
		} else {
			respondError(w, http.StatusInternalServerError, "registration failed")
		}
		return
	}
	u, err := s.Users.CreateUser(r.Context(), username, hash)
	switch {
	case errors.Is(err, auth.ErrUserAlreadyExists):
		respondError(w, http.StatusConflict, "username is already taken")
		return
	case err != nil:
		slog.Error("registration failed", "error", err.Error())
		metrics.DependencyErrors.WithLabelValues("postgres", "users").Inc()
		respondError(w, http.StatusServiceUnavailable, "user database unavailable")
		return
	}
	s.issue(w, http.StatusCreated, u, "user created")
}

// handleLogin: POST /api/v1/login. Unknown users and wrong passwords get
// the same 401 and the same bcrypt work, so responses do not reveal which
// usernames exist.
func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	c, ok := decodeCredentials(w, r)
	if !ok {
		return
	}
	username, uerr := auth.NormalizeUsername(c.Username)
	if uerr != nil || c.Password == "" {
		auth.CheckPassword("", c.Password)
		respondError(w, http.StatusUnauthorized, "invalid credentials")
		return
	}
	u, err := s.Users.GetUserByUsername(r.Context(), username)
	switch {
	case errors.Is(err, auth.ErrUserNotFound):
		auth.CheckPassword("", c.Password)
		respondError(w, http.StatusUnauthorized, "invalid credentials")
		return
	case err != nil:
		slog.Error("login lookup failed", "error", err.Error())
		metrics.DependencyErrors.WithLabelValues("postgres", "users").Inc()
		respondError(w, http.StatusServiceUnavailable, "user database unavailable")
		return
	}
	if !auth.CheckPassword(u.PasswordHash, c.Password) {
		respondError(w, http.StatusUnauthorized, "invalid credentials")
		return
	}
	s.issue(w, http.StatusOK, u, "")
}

// ── Conversation endpoints ──────────────────────────────────────────

type sessionRequest struct {
	SessionID string `json:"session_id"`
}

func (s *Server) sessionFromRequest(w http.ResponseWriter, r *http.Request) (string, string, bool) {
	userID, ok := middleware.UserID(r.Context())
	if !ok {
		respondError(w, http.StatusUnauthorized, "unauthenticated")
		return "", "", false
	}
	if s.Memory == nil {
		respondError(w, http.StatusServiceUnavailable, "conversation memory not enabled")
		return "", "", false
	}
	r.Body = http.MaxBytesReader(w, r.Body, 4<<10)
	var req sessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || !memory.ValidSessionID(req.SessionID) {
		respondError(w, http.StatusBadRequest, "session_id must be 1-64 characters of A-Z, a-z, 0-9, _ or -")
		return "", "", false
	}
	return userID, req.SessionID, true
}

// handleClearConversation: POST /api/v1/conversation/clear (reference).
// Only the authenticated user's session is affected.
func (s *Server) handleClearConversation(w http.ResponseWriter, r *http.Request) {
	userID, session, ok := s.sessionFromRequest(w, r)
	if !ok {
		return
	}
	if err := s.Memory.Clear(r.Context(), userID, session); err != nil {
		metrics.DependencyErrors.WithLabelValues("redis", "conversation").Inc()
		respondError(w, http.StatusServiceUnavailable, "conversation memory unavailable")
		return
	}
	respond(w, http.StatusOK, map[string]string{"status": "cleared", "session_id": session})
}

// handleConversationHistory: POST /api/v1/conversation/history returns the
// authenticated user's turns for a session (for the UI).
func (s *Server) handleConversationHistory(w http.ResponseWriter, r *http.Request) {
	userID, session, ok := s.sessionFromRequest(w, r)
	if !ok {
		return
	}
	turns, err := s.Memory.GetHistory(r.Context(), userID, session, 0)
	if err != nil {
		metrics.DependencyErrors.WithLabelValues("redis", "conversation").Inc()
		respondError(w, http.StatusServiceUnavailable, "conversation memory unavailable")
		return
	}
	if turns == nil {
		turns = []memory.Turn{}
	}
	respond(w, http.StatusOK, map[string]any{"session_id": session, "turns": turns})
}

// ConversationMemory is the store the server needs (memory.ConversationStore in production).
type ConversationMemory interface {
	GetHistory(ctx context.Context, userID, sessionID string, limit int) ([]memory.Turn, error)
	AppendExchange(ctx context.Context, userID, sessionID string, user, assistant memory.Turn) error
	Clear(ctx context.Context, userID, sessionID string) error
}
