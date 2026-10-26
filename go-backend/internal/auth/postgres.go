package auth

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresConfig holds connection and pool settings (reference fields).
// There are no default credentials: User and Password (or URL) must be set.
type PostgresConfig struct {
	URL               string
	Host              string
	Port              string
	User              string
	Password          string
	Database          string
	SSLMode           string
	MaxConns          int32
	MinConns          int32
	MaxConnIdleTime   time.Duration
	MaxConnLifetime   time.Duration
	HealthCheckPeriod time.Duration
}

// ConnectionString builds the connection URI.
func (c PostgresConfig) ConnectionString() (string, error) {
	if c.URL != "" {
		return c.URL, nil
	}
	if c.User == "" || c.Password == "" || c.Host == "" || c.Database == "" {
		return "", errors.New("postgres: host, database, user and password are required (no defaults)")
	}
	port, ssl := c.Port, c.SSLMode
	if port == "" {
		port = "5432"
	}
	if ssl == "" {
		ssl = "disable"
	}
	u := url.URL{Scheme: "postgres", User: url.UserPassword(c.User, c.Password), Host: c.Host + ":" + port, Path: "/" + c.Database, RawQuery: "sslmode=" + url.QueryEscape(ssl)}
	return u.String(), nil
}

// PostgresUserRepository implements UserRepository and DocumentAccess.
type PostgresUserRepository struct {
	pool *pgxpool.Pool
}

// migrations are applied in order, once, under an advisory lock so several
// backend instances can start concurrently.
var migrations = []struct {
	version int
	sql     string
}{
	{1, `CREATE TABLE IF NOT EXISTS users (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		username VARCHAR(255) UNIQUE NOT NULL,
		password_hash VARCHAR(255) NOT NULL,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);
	CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);`},
	{2, `CREATE TABLE IF NOT EXISTS document_access (
		user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		doc_id VARCHAR(64) NOT NULL,
		filename VARCHAR(255) NOT NULL,
		chunk_count INTEGER NOT NULL DEFAULT 0,
		ocr_method VARCHAR(32) NOT NULL DEFAULT '',
		first_uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
		last_uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
		PRIMARY KEY (user_id, doc_id)
	);`},
}

const migrationLockID = 744_210_9 // arbitrary constant for pg_advisory_xact_lock

// NewPostgresUserRepository opens a pgx pool and pings it (reference
// sizing defaults: 25 max / 5 min, 30m idle, 1h lifetime, 1m health check).
func NewPostgresUserRepository(ctx context.Context, cfg PostgresConfig) (*PostgresUserRepository, error) {
	connStr, err := cfg.ConnectionString()
	if err != nil {
		return nil, err
	}
	pc, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		return nil, errors.New("failed to parse postgres configuration") // never echo the URL (it holds the password)
	}
	pc.MaxConns, pc.MinConns = pick32(cfg.MaxConns, 25), pick32(cfg.MinConns, 5)
	pc.MaxConnIdleTime = pickDur(cfg.MaxConnIdleTime, 30*time.Minute)
	pc.MaxConnLifetime = pickDur(cfg.MaxConnLifetime, time.Hour)
	pc.HealthCheckPeriod = pickDur(cfg.HealthCheckPeriod, time.Minute)
	pool, err := pgxpool.NewWithConfig(ctx, pc)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize pgx pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to reach postgres: %w", err)
	}
	return &PostgresUserRepository{pool: pool}, nil
}

func pick32(v, d int32) int32 {
	if v > 0 {
		return v
	}
	return d
}

func pickDur(v, d time.Duration) time.Duration {
	if v > 0 {
		return v
	}
	return d
}

// Migrate applies pending schema migrations idempotently.
func (p *PostgresUserRepository) Migrate(ctx context.Context) error {
	tx, err := p.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("migration: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", migrationLockID); err != nil {
		return fmt.Errorf("migration lock: %w", err)
	}
	if _, err := tx.Exec(ctx, `CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`); err != nil {
		return fmt.Errorf("migration table: %w", err)
	}
	for _, m := range migrations {
		var exists bool
		if err := tx.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=$1)", m.version).Scan(&exists); err != nil {
			return fmt.Errorf("migration %d check: %w", m.version, err)
		}
		if exists {
			continue
		}
		if _, err := tx.Exec(ctx, m.sql); err != nil {
			return fmt.Errorf("migration %d: %w", m.version, err)
		}
		if _, err := tx.Exec(ctx, "INSERT INTO schema_migrations(version) VALUES ($1)", m.version); err != nil {
			return fmt.Errorf("migration %d record: %w", m.version, err)
		}
	}
	return tx.Commit(ctx)
}

// CreateUser inserts a user; a unique violation is ErrUserAlreadyExists.
func (p *PostgresUserRepository) CreateUser(ctx context.Context, username, passwordHash string) (*User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if username == "" || passwordHash == "" {
		return nil, ErrInvalidInput
	}
	var u User
	err := p.pool.QueryRow(ctx,
		`INSERT INTO users (username, password_hash) VALUES ($1, $2) RETURNING id::text, username, password_hash, created_at`,
		username, passwordHash).Scan(&u.ID, &u.Username, &u.PasswordHash, &u.CreatedAt)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return nil, ErrUserAlreadyExists
		}
		return nil, fmt.Errorf("failed to insert user: %w", err)
	}
	return &u, nil
}

// GetUserByUsername looks a user up by (normalised) username.
func (p *PostgresUserRepository) GetUserByUsername(ctx context.Context, username string) (*User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if username == "" {
		return nil, ErrInvalidInput
	}
	var u User
	err := p.pool.QueryRow(ctx,
		`SELECT id::text, username, password_hash, created_at FROM users WHERE username = $1`,
		username).Scan(&u.ID, &u.Username, &u.PasswordHash, &u.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrUserNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query user: %w", err)
	}
	return &u, nil
}

// GrantDocument records (or refreshes) a user's access to a document.
func (p *PostgresUserRepository) GrantDocument(ctx context.Context, g DocumentGrant) error {
	if !IsUUID(g.UserID) || g.DocID == "" {
		return ErrInvalidInput
	}
	_, err := p.pool.Exec(ctx, `
		INSERT INTO document_access (user_id, doc_id, filename, chunk_count, ocr_method)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (user_id, doc_id) DO UPDATE
		SET filename = EXCLUDED.filename, chunk_count = EXCLUDED.chunk_count,
		    ocr_method = EXCLUDED.ocr_method, last_uploaded_at = NOW()`,
		g.UserID, g.DocID, truncate(g.Filename, 255), g.ChunkCount, truncate(g.OCRMethod, 32))
	if err != nil {
		return fmt.Errorf("failed to record document access: %w", err)
	}
	return nil
}

// HasDocumentAccess reports whether the user has a grant for the document.
func (p *PostgresUserRepository) HasDocumentAccess(ctx context.Context, userID, docID string) (bool, error) {
	if !IsUUID(userID) || docID == "" {
		return false, nil
	}
	var ok bool
	err := p.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM document_access WHERE user_id = $1 AND doc_id = $2)`, userID, docID).Scan(&ok)
	if err != nil {
		return false, fmt.Errorf("failed to check document access: %w", err)
	}
	return ok, nil
}

// SeedUser creates a user if absent (used only by the explicit development
// demo-account option; never by default).
func (p *PostgresUserRepository) SeedUser(ctx context.Context, username, password string, cost int) error {
	if _, err := p.GetUserByUsername(ctx, username); err == nil {
		return nil
	} else if !errors.Is(err, ErrUserNotFound) {
		return err
	}
	hash, err := HashPassword(password, cost)
	if err != nil {
		return err
	}
	if _, err := p.CreateUser(ctx, username, hash); err != nil && !errors.Is(err, ErrUserAlreadyExists) {
		return err
	}
	return nil
}

// Stats exposes pool statistics.
func (p *PostgresUserRepository) Stats() *pgxpool.Stat { return p.pool.Stat() }

// Ping checks the pool.
func (p *PostgresUserRepository) Ping(ctx context.Context) error { return p.pool.Ping(ctx) }

// Close closes the pool.
func (p *PostgresUserRepository) Close() error {
	p.pool.Close()
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
