package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

const (
	goodSecret = "k8Jd2mQx7Lp4Rz9Vb1Nc6Ht3Wy5Fs0Ga-local-test"
	uid        = "3f2b8c1e-6a4d-4e9f-8b7a-1c2d3e4f5a6b"
)

func manager(t *testing.T) *JWTManager {
	t.Helper()
	m, err := NewJWTManager(goodSecret, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func sign(t *testing.T, method jwt.SigningMethod, key any, claims jwt.Claims) string {
	t.Helper()
	s, err := jwt.NewWithClaims(method, claims).SignedString(key)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func claimsAt(now time.Time, mutate func(*Claims)) *Claims {
	c := &Claims{UserID: uid, RegisteredClaims: jwt.RegisteredClaims{
		Subject: uid, Issuer: Issuer, IssuedAt: jwt.NewNumericDate(now), ExpiresAt: jwt.NewNumericDate(now.Add(time.Hour)),
	}}
	if mutate != nil {
		mutate(c)
	}
	return c
}

func TestValidTokenRoundTrip(t *testing.T) {
	m := manager(t)
	tok, err := m.GenerateToken(uid)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(tok, goodSecret) {
		t.Fatal("token contains the secret")
	}
	c, err := m.ValidateToken(tok)
	if err != nil || c.UserID != uid || c.Subject != uid || c.Issuer != Issuer || c.ExpiresAt == nil || c.IssuedAt == nil {
		t.Fatalf("%+v %v", c, err)
	}
	if _, err := m.GenerateToken("alice"); err == nil {
		t.Fatal("tokens are issued for UUIDs only (the reference used usernames)")
	}
}

func TestExpiredAndNotYetValidTokens(t *testing.T) {
	m := manager(t)
	base := time.Now()
	tok, _ := m.GenerateToken(uid)
	m.SetClock(func() time.Time { return base.Add(time.Hour + time.Minute) })
	if _, err := m.ValidateToken(tok); !errors.Is(err, jwt.ErrTokenExpired) {
		t.Fatalf("expired: %v", err)
	}
	m.SetClock(time.Now)
	future := sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(base.Add(time.Hour), nil))
	if _, err := m.ValidateToken(future); err == nil {
		t.Fatal("token issued in the future accepted")
	}
}

func TestRejectedTokens(t *testing.T) {
	m := manager(t)
	now := time.Now()
	rsaKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	valid := sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, nil))
	parts := strings.Split(valid, ".")
	tamperedPayload, _ := base64.RawURLEncoding.DecodeString(parts[1])
	tampered := parts[0] + "." + base64.RawURLEncoding.EncodeToString([]byte(strings.Replace(string(tamperedPayload), uid, "aaaaaaaa-6a4d-4e9f-8b7a-1c2d3e4f5a6b", 2))) + "." + parts[2]
	noneTok, _ := jwt.NewWithClaims(jwt.SigningMethodNone, claimsAt(now, nil)).SignedString(jwt.UnsafeAllowNoneSignatureType)

	cases := map[string]string{
		"wrong secret":        sign(t, jwt.SigningMethodHS256, []byte(strings.Repeat("z", 40)+"Q1w2E3r4T5"), claimsAt(now, nil)),
		"HS512 (other HMAC)":  sign(t, jwt.SigningMethodHS512, []byte(goodSecret), claimsAt(now, nil)),
		"RS256":               sign(t, jwt.SigningMethodRS256, rsaKey, claimsAt(now, nil)),
		"alg none":            noneTok,
		"tampered payload":    tampered,
		"wrong issuer":        sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, func(c *Claims) { c.Issuer = "someone-else" })),
		"no expiry":           sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, func(c *Claims) { c.ExpiresAt = nil })),
		"no issued-at":        sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, func(c *Claims) { c.IssuedAt = nil })),
		"missing user_id":     sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, func(c *Claims) { c.UserID = "" })),
		"username as user_id": sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, func(c *Claims) { c.UserID, c.Subject = "demo", "demo" })),
		"subject mismatch":    sign(t, jwt.SigningMethodHS256, []byte(goodSecret), claimsAt(now, func(c *Claims) { c.Subject = "aaaaaaaa-6a4d-4e9f-8b7a-1c2d3e4f5a6b" })),
		"garbage":             "not.a.jwt",
		"empty":               "",
		"two segments":        parts[0] + "." + parts[1],
	}
	for name, tok := range cases {
		if c, err := m.ValidateToken(tok); err == nil {
			t.Fatalf("%s accepted: %+v", name, c)
		}
	}
	if _, err := m.ValidateToken(valid); err != nil {
		t.Fatalf("control token rejected: %v", err)
	}
}

func TestSecretPolicy(t *testing.T) {
	bad := map[string]string{
		"empty":             "",
		"short":             "Ab1!xY2@zW3#",
		"reference default": "council-ai-secret-change-me-council-ai-secret",
		"placeholder":       "please-change-me-before-production-0123456789",
		"low variety":       strings.Repeat("ab", 20),
	}
	for name, s := range bad {
		if _, err := NewJWTManager(s, time.Hour); !errors.Is(err, ErrWeakSecret) {
			t.Fatalf("%s accepted: %v", name, err)
		}
	}
	if _, err := NewJWTManager(goodSecret, 0); err == nil {
		t.Fatal("zero expiration accepted")
	}
}

func TestCredentialValidation(t *testing.T) {
	for in, want := range map[string]string{"  Alice ": "alice", "bob_99": "bob_99", "x.y-z": "x.y-z"} {
		got, err := NormalizeUsername(in)
		if err != nil || got != want {
			t.Fatalf("%q: %q %v", in, got, err)
		}
	}
	for _, bad := range []string{"", "ab", strings.Repeat("a", 33), "has space", "_lead", "émile", "a/b", "admin;--"} {
		if _, err := NormalizeUsername(bad); !errors.Is(err, ErrInvalidInput) {
			t.Fatalf("%q accepted", bad)
		}
	}
	for _, bad := range []string{"", "short7!", strings.Repeat("p", 73)} {
		if err := ValidatePassword(bad); err == nil {
			t.Fatalf("password of %d bytes accepted", len(bad))
		}
	}
	hash, err := HashPassword("correct horse battery", 4)
	if err != nil || hash == "correct horse battery" || !strings.HasPrefix(hash, "$2") {
		t.Fatalf("%q %v", hash, err)
	}
	if !CheckPassword(hash, "correct horse battery") || CheckPassword(hash, "wrong password") || CheckPassword("", "anything") {
		t.Fatal("password check")
	}
}
