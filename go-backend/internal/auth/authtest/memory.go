// Package authtest provides in-memory user and document-access stores for
// unit tests only. The server binary never uses them: production requires
// PostgreSQL and fails to start without it (the reference silently fell
// back to an in-memory repository with a seeded demo user).
package authtest

import (
	"context"
	"crypto/rand"
	"fmt"
	"sync"
	"time"

	"example.com/councilai-starter/internal/auth"
)

// Repo implements auth.UserRepository and auth.DocumentAccess in memory.
type Repo struct {
	mu       sync.RWMutex
	users    map[string]*auth.User
	grants   map[string]bool // userID|docID
	Err      error           // injected failure for every call
	AllowAll bool            // grant every document (legacy unit tests that predate ownership)
}

func New() *Repo { return &Repo{users: map[string]*auth.User{}, grants: map[string]bool{}} }

func uuid() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func (r *Repo) CreateUser(ctx context.Context, username, hash string) (*auth.User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.Err != nil {
		return nil, r.Err
	}
	if username == "" || hash == "" {
		return nil, auth.ErrInvalidInput
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.users[username]; ok {
		return nil, auth.ErrUserAlreadyExists
	}
	u := &auth.User{ID: uuid(), Username: username, PasswordHash: hash, CreatedAt: time.Now().UTC()}
	r.users[username] = u
	c := *u
	return &c, nil
}

func (r *Repo) GetUserByUsername(ctx context.Context, username string) (*auth.User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.Err != nil {
		return nil, r.Err
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	u, ok := r.users[username]
	if !ok {
		return nil, auth.ErrUserNotFound
	}
	c := *u
	return &c, nil
}

func (r *Repo) GrantDocument(ctx context.Context, g auth.DocumentGrant) error {
	if r.Err != nil {
		return r.Err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.grants[g.UserID+"|"+g.DocID] = true
	return nil
}

func (r *Repo) HasDocumentAccess(ctx context.Context, userID, docID string) (bool, error) {
	if r.Err != nil {
		return false, r.Err
	}
	if r.AllowAll {
		return true, nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.grants[userID+"|"+docID], nil
}

func (r *Repo) Ping(ctx context.Context) error { return r.Err }
func (r *Repo) Close() error                   { return nil }
