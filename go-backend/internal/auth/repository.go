package auth

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"golang.org/x/crypto/bcrypt"
)

// Domain errors (reference).
var (
	ErrUserNotFound      = errors.New("user not found")
	ErrUserAlreadyExists = errors.New("user already exists")
	ErrInvalidInput      = errors.New("invalid user input")
)

// User is a persistent identity. The password hash never leaves the server
// (json:"-").
type User struct {
	ID           string    `json:"id"`
	Username     string    `json:"username"`
	PasswordHash string    `json:"-"`
	CreatedAt    time.Time `json:"created_at"`
}

// UserRepository is the reference contract.
type UserRepository interface {
	CreateUser(ctx context.Context, username, passwordHash string) (*User, error)
	GetUserByUsername(ctx context.Context, username string) (*User, error)
	Ping(ctx context.Context) error
	Close() error
}

// DocumentGrant records that a user may use a document.
type DocumentGrant struct {
	UserID     string
	DocID      string
	Filename   string
	ChunkCount int
	OCRMethod  string
}

// DocumentAccess stores per-user document authorization. Documents are
// content-addressed, so the same file uploaded by two users is one RAG
// document with two independent grants.
type DocumentAccess interface {
	GrantDocument(ctx context.Context, g DocumentGrant) error
	HasDocumentAccess(ctx context.Context, userID, docID string) (bool, error)
}

var usernamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_.-]{2,31}$`)

// NormalizeUsername trims and lower-cases a username and validates it:
// 3–32 characters of a-z, 0-9, '_', '.', '-', starting with a letter or digit.
func NormalizeUsername(u string) (string, error) {
	u = strings.ToLower(strings.TrimSpace(u))
	if !usernamePattern.MatchString(u) {
		return "", fmt.Errorf("%w: username must be 3-32 characters (a-z, 0-9, _ . -)", ErrInvalidInput)
	}
	return u, nil
}

// ValidatePassword enforces 8–72 bytes (bcrypt reads at most 72 bytes; the
// reference accepted any non-empty password, silently truncating long ones).
func ValidatePassword(p string) error {
	if len(p) < 8 || len(p) > 72 {
		return fmt.Errorf("%w: password must be 8-72 bytes", ErrInvalidInput)
	}
	return nil
}

// HashPassword bcrypt-hashes a validated password.
func HashPassword(p string, cost int) (string, error) {
	if err := ValidatePassword(p); err != nil {
		return "", err
	}
	if cost == 0 {
		cost = bcrypt.DefaultCost
	}
	h, err := bcrypt.GenerateFromPassword([]byte(p), cost)
	if err != nil {
		return "", fmt.Errorf("password hashing failed: %w", err)
	}
	return string(h), nil
}

// dummyHash is compared against when a username does not exist, so login
// takes the same time whether or not the user exists.
var dummyHash, _ = bcrypt.GenerateFromPassword([]byte("timing-equaliser-not-a-password"), bcrypt.DefaultCost)

// CheckPassword compares in constant time; with hash == "" it burns the
// same bcrypt cost and returns false.
func CheckPassword(hash, password string) bool {
	if hash == "" {
		_ = bcrypt.CompareHashAndPassword(dummyHash, []byte(password))
		return false
	}
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}
