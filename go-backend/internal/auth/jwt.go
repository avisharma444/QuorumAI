// Package auth provides JWT issuance and validation, the user and document
// access repositories (PostgreSQL), and credential validation.
package auth

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// Issuer is the reference issuer claim.
const Issuer = "council-ai"

// MinSecretBytes is the minimum HMAC key length accepted (256 bits, the
// HS256 output size).
const MinSecretBytes = 32

// placeholderPhrases mark copied example values (the reference default is
// "council-ai-secret-change-me"); a secret containing one is rejected.
var placeholderPhrases = []string{"change-me", "changeme", "change_me", "your-secret", "replace-me", "example-secret", "council-ai-secret"}

// ErrWeakSecret rejects secrets that are too short or are known placeholders.
var ErrWeakSecret = errors.New("JWT secret is too weak")

// Claims are the reference claims: user_id plus registered claims. UserID
// and Subject both carry the user's immutable UUID (the reference put the
// mutable username there).
type Claims struct {
	UserID string `json:"user_id"`
	jwt.RegisteredClaims
}

// JWTManager signs and validates HMAC-SHA256 tokens.
type JWTManager struct {
	secret     []byte
	expiration time.Duration
	now        func() time.Time
}

// ValidateSecret enforces the secret policy.
func ValidateSecret(secret string) error {
	if len(secret) < MinSecretBytes {
		return fmt.Errorf("%w: need at least %d bytes, got %d", ErrWeakSecret, MinSecretBytes, len(secret))
	}
	lower := strings.ToLower(secret)
	for _, w := range placeholderPhrases {
		if strings.Contains(lower, w) {
			return fmt.Errorf("%w: contains a well-known placeholder", ErrWeakSecret)
		}
	}
	distinct := map[rune]bool{}
	for _, r := range secret {
		distinct[r] = true
	}
	if len(distinct) < 10 {
		return fmt.Errorf("%w: too few distinct characters", ErrWeakSecret)
	}
	return nil
}

// NewJWTManager validates the secret and builds a manager. There is no
// fallback secret: a missing or weak secret is a startup error.
func NewJWTManager(secret string, expiration time.Duration) (*JWTManager, error) {
	if err := ValidateSecret(secret); err != nil {
		return nil, err
	}
	if expiration <= 0 {
		return nil, errors.New("JWT expiration must be positive")
	}
	return &JWTManager{secret: []byte(secret), expiration: expiration, now: time.Now}, nil
}

// SetClock overrides the clock (tests).
func (m *JWTManager) SetClock(now func() time.Time) { m.now = now }

// Expiration returns the configured token lifetime.
func (m *JWTManager) Expiration() time.Duration { return m.expiration }

// GenerateToken issues a token for a user UUID.
func (m *JWTManager) GenerateToken(userID string) (string, error) {
	if !IsUUID(userID) {
		return "", fmt.Errorf("refusing to issue a token for non-UUID subject")
	}
	now := m.now()
	claims := Claims{
		UserID: userID,
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   userID,
			Issuer:    Issuer,
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(m.expiration)),
		},
	}
	return jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString(m.secret)
}

// ValidateToken verifies signature (HS256 only), expiry, issuer, issued-at
// and the user claims. The reference accepted any HMAC algorithm and did
// not check the issuer.
func (m *JWTManager) ValidateToken(tokenString string) (*Claims, error) {
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(tokenString, claims,
		func(t *jwt.Token) (any, error) { return m.secret, nil },
		jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Alg()}),
		jwt.WithIssuer(Issuer),
		jwt.WithExpirationRequired(),
		jwt.WithIssuedAt(),
		jwt.WithLeeway(5*time.Second),
		jwt.WithTimeFunc(m.now),
	)
	if err != nil {
		return nil, fmt.Errorf("invalid token: %w", err)
	}
	if !token.Valid {
		return nil, errors.New("invalid token")
	}
	if claims.IssuedAt == nil {
		return nil, errors.New("invalid token: missing iat")
	}
	if !IsUUID(claims.UserID) || claims.Subject != claims.UserID {
		return nil, errors.New("invalid token: missing or inconsistent user identity")
	}
	return claims, nil
}

var uuidPattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// IsUUID reports whether s is a lower-case canonical UUID (as PostgreSQL returns).
func IsUUID(s string) bool { return uuidPattern.MatchString(s) }
