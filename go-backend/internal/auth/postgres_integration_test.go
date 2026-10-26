package auth

// Integration tests against a REAL PostgreSQL (POSTGRES_TEST_URL, e.g.
// postgres://user:pass@postgres:5432/db?sslmode=disable). Usernames are
// unique per run and removed afterwards.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func pgRepo(t *testing.T) *PostgresUserRepository {
	t.Helper()
	url := os.Getenv("POSTGRES_TEST_URL")
	if url == "" {
		t.Skip("POSTGRES_TEST_URL not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	r, err := NewPostgresUserRepository(ctx, PostgresConfig{URL: url, MaxConns: 8, MinConns: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func uname(t *testing.T, suffix string) string {
	return strings.ToLower(fmt.Sprintf("it%d%s", time.Now().UnixNano()%1e12, suffix))
}

func cleanupUsers(t *testing.T, r *PostgresUserRepository, prefix string) {
	t.Cleanup(func() {
		_, _ = r.pool.Exec(context.Background(), "DELETE FROM users WHERE username LIKE $1", prefix+"%")
	})
}

func TestIntegrationMigrationsAreIdempotentAndConcurrent(t *testing.T) {
	r := pgRepo(t)
	var wg sync.WaitGroup
	errs := make(chan error, 6)
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- r.Migrate(context.Background()) }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	var n int
	if err := r.pool.QueryRow(context.Background(), "SELECT count(*) FROM schema_migrations").Scan(&n); err != nil || n != len(migrations) {
		t.Fatalf("migrations recorded %d (%v)", n, err)
	}
}

func TestIntegrationCreateGetDuplicateAndHashOnly(t *testing.T) {
	r := pgRepo(t)
	name := uname(t, "alice")
	cleanupUsers(t, r, name)
	ctx := context.Background()
	hash, _ := HashPassword("correct horse battery", 4)
	u, err := r.CreateUser(ctx, name, hash)
	if err != nil || !IsUUID(u.ID) || u.Username != name || u.CreatedAt.IsZero() {
		t.Fatalf("%+v %v", u, err)
	}
	var stored string
	_ = r.pool.QueryRow(ctx, "SELECT password_hash FROM users WHERE id=$1", u.ID).Scan(&stored)
	if stored != hash || strings.Contains(stored, "correct horse") {
		t.Fatal("only the bcrypt hash may be stored")
	}
	got, err := r.GetUserByUsername(ctx, name)
	if err != nil || got.ID != u.ID || !CheckPassword(got.PasswordHash, "correct horse battery") {
		t.Fatalf("%+v %v", got, err)
	}
	if _, err := r.CreateUser(ctx, name, hash); !errors.Is(err, ErrUserAlreadyExists) {
		t.Fatalf("duplicate: %v", err)
	}
	if _, err := r.GetUserByUsername(ctx, name+"x"); !errors.Is(err, ErrUserNotFound) {
		t.Fatalf("missing user: %v", err)
	}
}

func TestIntegrationConcurrentRegistrationOneWinner(t *testing.T) {
	r := pgRepo(t)
	name := uname(t, "race")
	cleanupUsers(t, r, name)
	hash, _ := HashPassword("correct horse battery", 4)
	var ok, dup atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := r.CreateUser(context.Background(), name, hash)
			switch {
			case err == nil:
				ok.Add(1)
			case errors.Is(err, ErrUserAlreadyExists):
				dup.Add(1)
			default:
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if ok.Load() != 1 || dup.Load() != 19 {
		t.Fatalf("winners=%d duplicates=%d", ok.Load(), dup.Load())
	}
}

func TestIntegrationPersistenceAcrossPools(t *testing.T) {
	r := pgRepo(t)
	name := uname(t, "persist")
	cleanupUsers(t, r, name)
	hash, _ := HashPassword("correct horse battery", 4)
	u, _ := r.CreateUser(context.Background(), name, hash)
	r2 := pgRepo(t) // a new pool, as after a backend restart
	got, err := r2.GetUserByUsername(context.Background(), name)
	if err != nil || got.ID != u.ID {
		t.Fatalf("%v", err)
	}
	if st := r2.Stats(); st.MaxConns() != 8 {
		t.Fatalf("pool max conns %d", st.MaxConns())
	}
}

func TestIntegrationDocumentGrants(t *testing.T) {
	r := pgRepo(t)
	ctx := context.Background()
	prefix := uname(t, "doc")
	cleanupUsers(t, r, prefix)
	hash, _ := HashPassword("correct horse battery", 4)
	a, _ := r.CreateUser(ctx, prefix+"a", hash)
	b, _ := r.CreateUser(ctx, prefix+"b", hash)
	doc := "report_" + prefix
	if ok, _ := r.HasDocumentAccess(ctx, a.ID, doc); ok {
		t.Fatal("access before grant")
	}
	for i := 0; i < 2; i++ { // idempotent re-upload
		if err := r.GrantDocument(ctx, DocumentGrant{UserID: a.ID, DocID: doc, Filename: "r.pdf", ChunkCount: 3, OCRMethod: "direct_text"}); err != nil {
			t.Fatal(err)
		}
	}
	if ok, _ := r.HasDocumentAccess(ctx, a.ID, doc); !ok {
		t.Fatal("grant not visible")
	}
	if ok, _ := r.HasDocumentAccess(ctx, b.ID, doc); ok {
		t.Fatal("grant leaked to another user")
	}
	if err := r.GrantDocument(ctx, DocumentGrant{UserID: b.ID, DocID: doc, Filename: "same.pdf"}); err != nil {
		t.Fatal(err)
	}
	var n int
	_ = r.pool.QueryRow(ctx, "SELECT count(*) FROM document_access WHERE doc_id=$1", doc).Scan(&n)
	if n != 2 {
		t.Fatalf("expected one row per user, got %d", n)
	}
	if ok, err := r.HasDocumentAccess(ctx, "not-a-uuid", doc); ok || err != nil {
		t.Fatal("invalid user ids never have access")
	}
	if err := r.GrantDocument(ctx, DocumentGrant{UserID: "00000000-0000-4000-8000-000000000000", DocID: doc, Filename: "x"}); err == nil {
		t.Fatal("grant for a non-existent user accepted (foreign key)")
	}
}

func TestIntegrationCancellationAndOutage(t *testing.T) {
	r := pgRepo(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.GetUserByUsername(ctx, "anyone"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled: %v", err)
	}
	r.Close()
	if _, err := r.GetUserByUsername(context.Background(), "anyone"); err == nil || errors.Is(err, ErrUserNotFound) {
		t.Fatalf("closed pool must be an error, not 'not found': %v", err)
	}
	bad := strings.Replace(os.Getenv("POSTGRES_TEST_URL"), ":5432", ":6543", 1)
	dctx, dcancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer dcancel()
	_, err := NewPostgresUserRepository(dctx, PostgresConfig{URL: bad})
	if err == nil {
		t.Fatal("unreachable database accepted")
	}
	if pw := passwordFromURL(os.Getenv("POSTGRES_TEST_URL")); pw != "" && strings.Contains(err.Error(), pw) {
		t.Fatal("connection error leaks the password")
	}
}

func passwordFromURL(u string) string {
	at := strings.Index(u, "@")
	colon := strings.LastIndex(u[:max(at, 0)], ":")
	if at < 0 || colon < 0 {
		return ""
	}
	return u[colon+1 : at]
}

func TestConnectionStringRequiresCredentials(t *testing.T) {
	if _, err := (PostgresConfig{Host: "db", Database: "x"}).ConnectionString(); err == nil {
		t.Fatal("missing user/password must be an error (the reference defaulted to council_user/council_pass)")
	}
	s, err := PostgresConfig{Host: "db", Database: "app", User: "u", Password: "p@ss:w/rd"}.ConnectionString()
	if err != nil || !strings.Contains(s, "p%40ss%3Aw%2Frd") {
		t.Fatalf("%s %v", s, err)
	}
}
