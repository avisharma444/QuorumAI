package api

// End-to-end through the real HTTP router with REAL PostgreSQL (users,
// document grants) and REAL Redis (L1 cache, conversation memory, rate
// limiting). Only the LLM providers and the Python RAG service are faked.
// Requires POSTGRES_TEST_URL and REDIS_TEST_ADDR.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"example.com/councilai-starter/internal/api/middleware"
	"example.com/councilai-starter/internal/auth"
	"example.com/councilai-starter/internal/cache"
	"example.com/councilai-starter/internal/llm/llmtest"
	"example.com/councilai-starter/internal/memory"
)

func realEnv(t *testing.T, replies map[string]llmtest.Reply, rps float64, burst int) *env {
	t.Helper()
	pgURL, redisAddr := os.Getenv("POSTGRES_TEST_URL"), os.Getenv("REDIS_TEST_ADDR")
	if pgURL == "" || redisAddr == "" {
		t.Skip("POSTGRES_TEST_URL and REDIS_TEST_ADDR not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	repo, err := auth.NewPostgresUserRepository(ctx, auth.PostgresConfig{URL: pgURL, MaxConns: 8})
	if err != nil {
		t.Fatal(err)
	}
	if err := repo.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	client := cache.NewClient(redisAddr, "", 0)
	breaker := cache.NewCircuitBreaker("redis", cache.DefaultConfig())
	e := newEnv(t, replies)
	jm, _ := auth.NewJWTManager(testSecret, time.Hour)
	e.srv.JWT, e.srv.Users, e.srv.Docs, e.srv.BcryptCost = jm, repo, repo, 4
	e.srv.Cache = cache.NewRedisCacheWithBreaker(client, breaker, time.Minute)
	e.srv.Memory = memory.NewConversationStore(client, breaker, 10, time.Minute)
	prefix := fmt.Sprintf("e2e%d", time.Now().UnixNano())
	e.srv.UserLimiter, err = middleware.NewRateLimiter(client, breaker, rps, burst, prefix+"u", middleware.FailClosed)
	if err != nil {
		t.Fatal(err)
	}
	e.srv.AuthLimiter, _ = middleware.NewRateLimiter(client, breaker, 100, 100, prefix+"a", middleware.FailClosed)
	t.Cleanup(func() {
		bg := context.Background()
		for _, u := range e2eUsers {
			if usr, err := repo.GetUserByUsername(bg, u); err == nil {
				for _, pat := range []string{"councilai:v4:*" + usr.ID + "*", "councilai:v1:conv:" + usr.ID + ":*"} {
					if keys := client.Keys(bg, pat).Val(); len(keys) > 0 {
						client.Del(bg, keys...)
					}
				}
			}
		}
		repo.Close()
		client.Close()
	})
	return e
}

var e2eUsers []string

func e2eName(base string) string {
	n := fmt.Sprintf("%s%d", base, time.Now().UnixNano()%1e9)
	e2eUsers = append(e2eUsers, n)
	return n
}

func TestIntegrationEndToEndAuthOwnershipCacheAndMemory(t *testing.T) {
	e := realEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)}, 100, 100)
	tokA, uidA := register(t, e.srv, e2eName("alice"))
	tokB, _ := register(t, e.srv, e2eName("bob"))

	// Login against the real database returns the same UUID.
	rec := doJSON(t, e.srv, "POST", "/api/v1/login", "", `{"username":"`+e2eUsers[len(e2eUsers)-2]+`","password":"correct horse battery"}`)
	var login authResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &login)
	if rec.Code != 200 || login.UserID != uidA {
		t.Fatalf("login %d %s", rec.Code, rec.Body)
	}

	code, up := uploadAs(t, e, tokA, "z.md", []byte("Zogland facts."))
	if code != 200 {
		t.Fatalf("upload %d %v", code, up)
	}
	doc := up["doc_id"].(string)
	body := `{"question":"What is the capital of Zogland?","doc_id":"` + doc + `"}`
	if c, r, _ := queryAs(t, e, tokA, body); c != 200 || r.CacheHit {
		t.Fatalf("first %d %+v", c, r.Cache)
	}
	if _, r, _ := queryAs(t, e, tokA, body); !r.CacheHit || r.Cache.Tier != "l1_exact" {
		t.Fatalf("owner repeat should hit real Redis L1: %+v", r.Cache)
	}
	if c, _, _ := queryAs(t, e, tokB, body); c != 404 {
		t.Fatalf("bob read alice's document: %d", c)
	}

	// Multi-turn memory in real Redis.
	_, r1, _ := queryAs(t, e, tokA, `{"question":"Who rules Zogland?","session_id":"e2e"}`)
	_, r2, _ := queryAs(t, e, tokA, `{"question":"Since when?","session_id":"e2e"}`)
	if !r1.Session.Saved || !r2.Session.Saved || r2.Session.HistoryUsed != 2 || r2.Cache.L1 != "bypassed_conversation" {
		t.Fatalf("%+v %+v %+v", r1.Session, r2.Session, r2.Cache)
	}
	var hist struct {
		Turns []memory.Turn `json:"turns"`
	}
	_ = json.Unmarshal(doJSON(t, e.srv, "POST", "/api/v1/conversation/history", tokA, `{"session_id":"e2e"}`).Body.Bytes(), &hist)
	if len(hist.Turns) != 4 || hist.Turns[2].Content != "Since when?" {
		t.Fatalf("history %+v", hist.Turns)
	}
	_ = json.Unmarshal(doJSON(t, e.srv, "POST", "/api/v1/conversation/history", tokB, `{"session_id":"e2e"}`).Body.Bytes(), &hist)
	if len(hist.Turns) != 0 {
		t.Fatal("bob sees alice's session")
	}
	if !strings.Contains(e.llm.Prompt("member", "a"), "User: Who rules Zogland?") {
		t.Fatal("history not in prompt")
	}
}

func TestIntegrationEndToEndRateLimit(t *testing.T) {
	e := realEnv(t, map[string]llmtest.Reply{"route/router": route("direct", false), "direct/chair": {Text: "ok"}}, 0.5, 2)
	tokA, _ := register(t, e.srv, e2eName("carol"))
	tokB, _ := register(t, e.srv, e2eName("dave"))
	codes := []int{}
	for i := 0; i < 3; i++ {
		c, _, _ := queryAs(t, e, tokA, fmt.Sprintf(`{"question":"q%d"}`, i))
		codes = append(codes, c)
	}
	if fmt.Sprint(codes) != "[200 200 429]" {
		t.Fatalf("codes %v", codes)
	}
	calls := e.llm.Count("direct")
	if c, _, _ := queryAs(t, e, tokB, `{"question":"other user"}`); c != 200 {
		t.Fatal("another user must have their own quota")
	}
	if e.llm.Count("direct") != calls+1 {
		t.Fatal("the rejected request must not have reached a model")
	}
}
