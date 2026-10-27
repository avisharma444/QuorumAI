package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"example.com/councilai-starter/internal/llm"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"example.com/councilai-starter/internal/auth"
	"example.com/councilai-starter/internal/auth/authtest"
	"example.com/councilai-starter/internal/council"
	"example.com/councilai-starter/internal/llm/llmtest"
	"example.com/councilai-starter/internal/memory"
)

// fakeMemory is a test-only ConversationStore double.
type fakeMemory struct {
	mu      sync.Mutex
	turns   map[string][]memory.Turn
	GetErr  error
	SaveErr error
}

func newFakeMemory() *fakeMemory { return &fakeMemory{turns: map[string][]memory.Turn{}} }

func (m *fakeMemory) GetHistory(ctx context.Context, user, session string, limit int) ([]memory.Turn, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.GetErr != nil {
		return nil, m.GetErr
	}
	t := m.turns[user+"|"+session]
	if limit > 0 && len(t) > 2*limit {
		t = t[len(t)-2*limit:]
	}
	return append([]memory.Turn(nil), t...), nil
}

func (m *fakeMemory) AppendExchange(ctx context.Context, user, session string, u, a memory.Turn) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.SaveErr != nil {
		return m.SaveErr
	}
	m.turns[user+"|"+session] = append(m.turns[user+"|"+session], u, a)
	return nil
}

func (m *fakeMemory) Clear(ctx context.Context, user, session string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.turns, user+"|"+session)
	return nil
}

func doJSON(t *testing.T, srv *Server, method, path, token, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)
	return rec
}

func register(t *testing.T, srv *Server, name string) (token, userID string) {
	t.Helper()
	rec := doJSON(t, srv, "POST", "/api/v1/register", "", `{"username":"`+name+`","password":"correct horse battery"}`)
	if rec.Code != 201 {
		t.Fatalf("register %s: %d %s", name, rec.Code, rec.Body.String())
	}
	var out authResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return out.Token, out.UserID
}

// strictEnv: an env with strict document ownership and conversation memory.
func strictEnv(t *testing.T, replies map[string]llmtest.Reply) (*env, *authtest.Repo, *fakeMemory) {
	t.Helper()
	e := newEnv(t, replies)
	repo := withTestAuth(e.srv, false)
	mem := newFakeMemory()
	e.srv.Memory = mem
	return e, repo, mem
}

func uploadAs(t *testing.T, e *env, token, name string, content []byte) (int, map[string]any) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	part, _ := mw.CreateFormFile("file", name)
	_, _ = part.Write(content)
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/ingest", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	e.srv.Routes().ServeHTTP(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func queryAs(t *testing.T, e *env, token, body string) (int, council.QueryResult, string) {
	t.Helper()
	rec := doJSON(t, e.srv, "POST", "/api/v1/query", token, body)
	var out council.QueryResult
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out, rec.Body.String()
}

// ── Registration and login ──────────────────────────────────────────

func TestRegisterAndLogin(t *testing.T) {
	e, _, _ := strictEnv(t, nil)
	rec := doJSON(t, e.srv, "POST", "/api/v1/register", "", `{"username":"  Alice ","password":"correct horse battery"}`)
	var out authResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if rec.Code != 201 || out.Username != "alice" || len(out.UserID) != 36 || out.ExpiresIn != 3600 {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
	if claims, err := e.srv.JWT.ValidateToken(out.Token); err != nil || claims.UserID != out.UserID {
		t.Fatal("issued token invalid or not bound to the UUID")
	}
	if strings.Contains(rec.Body.String(), "$2") || strings.Contains(strings.ToLower(rec.Body.String()), "hash") {
		t.Fatal("password hash exposed")
	}
	if rec := doJSON(t, e.srv, "POST", "/api/v1/register", "", `{"username":"ALICE","password":"another password"}`); rec.Code != 409 {
		t.Fatalf("duplicate: %d", rec.Code)
	}
	rec = doJSON(t, e.srv, "POST", "/api/v1/login", "", `{"username":"alice","password":"correct horse battery"}`)
	var login authResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &login)
	if rec.Code != 200 || login.UserID != out.UserID {
		t.Fatalf("login: %d %s", rec.Code, rec.Body.String())
	}
	wrong := doJSON(t, e.srv, "POST", "/api/v1/login", "", `{"username":"alice","password":"wrong password!"}`)
	unknown := doJSON(t, e.srv, "POST", "/api/v1/login", "", `{"username":"nobody","password":"wrong password!"}`)
	if wrong.Code != 401 || unknown.Code != 401 || wrong.Body.String() != unknown.Body.String() {
		t.Fatalf("login failures must be identical: %d %s / %d %s", wrong.Code, wrong.Body, unknown.Code, unknown.Body)
	}
}

func TestRegisterValidation(t *testing.T) {
	e, repo, _ := strictEnv(t, nil)
	for body, want := range map[string]int{
		`{"username":"ab","password":"correct horse"}`:                    400,
		`{"username":"bob","password":"short"}`:                           400,
		`{"username":"bob","password":"` + strings.Repeat("x", 80) + `"}`: 400,
		`{"username":"bob","password":"correct horse","is_admin":true}`:   400,
		`not json`: 400,
		`{"username":"bob","password":"` + strings.Repeat("x", 5000) + `"}`: 400,
	} {
		if rec := doJSON(t, e.srv, "POST", "/api/v1/register", "", body); rec.Code != want {
			t.Fatalf("%.60s: %d", body, rec.Code)
		}
	}
	repo.Err = errors.New("connection refused")
	if rec := doJSON(t, e.srv, "POST", "/api/v1/register", "", `{"username":"carol","password":"correct horse"}`); rec.Code != 503 {
		t.Fatalf("database outage: %d", rec.Code)
	}
	if rec := doJSON(t, e.srv, "POST", "/api/v1/login", "", `{"username":"carol","password":"correct horse"}`); rec.Code != 503 {
		t.Fatalf("database outage on login: %d", rec.Code)
	}
}

// ── Protection ──────────────────────────────────────────────────────

func TestUnauthenticatedRequestsDoNothing(t *testing.T) {
	e, _, mem := strictEnv(t, nil)
	l1 := e.srv.Cache.(interface{ Keys() []string })
	for name, tok := range map[string]string{"none": "", "garbage": "not.a.token", "forged": "eyJhbGciOiJub25lIn0.eyJ1c2VyX2lkIjoieCJ9."} {
		for _, path := range []string{"/api/v1/query", "/api/v1/ingest", "/api/v1/conversation/history", "/api/v1/conversation/clear"} {
			if rec := doJSON(t, e.srv, "POST", path, tok, `{"question":"q","doc_id":"doc-1","session_id":"s"}`); rec.Code != 401 {
				t.Fatalf("%s %s: %d", name, path, rec.Code)
			}
		}
	}
	if len(e.llm.Calls()) != 0 || e.rag.count("/retrieve")+e.rag.count("/search")+e.rag.count("/embed")+e.rag.count("/ingest") != 0 {
		t.Fatalf("unauthenticated requests reached models or RAG: %v", e.llm.Calls())
	}
	if len(l1.Keys()) != 0 || len(mem.turns) != 0 {
		t.Fatal("unauthenticated requests touched cache or memory")
	}
}

// ── Document ownership ──────────────────────────────────────────────

func TestDocumentOwnership(t *testing.T) {
	e, repo, _ := strictEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	tokA, _ := register(t, e.srv, "alice")
	tokB, _ := register(t, e.srv, "bob")
	code, out := uploadAs(t, e, tokA, "a.md", []byte("A's document"))
	if code != 200 {
		t.Fatalf("upload %d %v", code, out)
	}
	docA := out["doc_id"].(string)
	if c, _, _ := queryAs(t, e, tokA, `{"question":"capital?","doc_id":"`+docA+`"}`); c != 200 {
		t.Fatalf("owner denied: %d", c)
	}
	retrievals := e.rag.count("/retrieve")
	c1, _, b1 := queryAs(t, e, tokB, `{"question":"capital?","doc_id":"`+docA+`"}`)
	c2, _, b2 := queryAs(t, e, tokB, `{"question":"capital?","doc_id":"never-uploaded-doc"}`)
	if c1 != 404 || c2 != 404 || b1 != b2 {
		t.Fatalf("unauthorized and unknown documents must look identical: %d %s / %d %s", c1, b1, c2, b2)
	}
	if e.rag.count("/retrieve") != retrievals {
		t.Fatal("unauthorized request reached retrieval")
	}
	// The same file uploaded by B: B gains its own grant; A keeps theirs.
	e.rag.mu.Lock()
	e.rag.ingestN = 0 // fake RAG returns doc-1 again, like content-addressed IDs
	e.rag.mu.Unlock()
	if code, out := uploadAs(t, e, tokB, "same.md", []byte("A's document")); code != 200 || out["doc_id"] != docA {
		t.Fatalf("%d %v", code, out)
	}
	if c, _, _ := queryAs(t, e, tokB, `{"question":"capital?","doc_id":"`+docA+`"}`); c != 200 {
		t.Fatal("B must access their own upload of the same content")
	}
	// Grant failure: the RAG service processed the file, but no access is granted.
	tokC, _ := register(t, e.srv, "carol")
	repo.Err = errors.New("db down")
	if code, _ := uploadAs(t, e, tokC, "c.md", []byte("x")); code != 503 {
		t.Fatalf("grant failure: %d", code)
	}
	repo.Err = nil
	e.rag.mu.Lock()
	n := e.rag.ingestN
	e.rag.mu.Unlock()
	if c, _, _ := queryAs(t, e, tokC, fmt.Sprintf(`{"question":"q","doc_id":"doc-%d"}`, n)); c != 404 {
		t.Fatal("failed grant must not give access")
	}
	repo.Err = errors.New("db down")
	if c, _, _ := queryAs(t, e, tokA, `{"question":"capital?","doc_id":"`+docA+`"}`); c != 503 {
		t.Fatal("authorization outage must fail closed")
	}
}

func TestLegacyDocumentsWithoutOwnerAreHidden(t *testing.T) {
	e, _, _ := strictEnv(t, nil)
	tok, _ := register(t, e.srv, "alice")
	// doc-1 exists in the RAG service (retrieve answers) but has no grant.
	if c, _, _ := queryAs(t, e, tok, `{"question":"q","doc_id":"doc-1"}`); c != 404 || e.rag.count("/retrieve") != 0 {
		t.Fatalf("legacy document exposed: %d", c)
	}
}

// ── User-scoped caching ─────────────────────────────────────────────

func TestCachesAreScopedPerUser(t *testing.T) {
	e, l1, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	repo := withTestAuth(e.srv, false)
	tokA, uidA := register(t, e.srv, "alice")
	tokB, uidB := register(t, e.srv, "bob")
	_ = repo.GrantDocument(context.Background(), grant(uidA, "shared-doc"))
	_ = repo.GrantDocument(context.Background(), grant(uidB, "shared-doc"))
	_, first, _ := queryAs(t, e, tokA, `{"question":"What is the capital?","doc_id":"shared-doc"}`)
	_, again, _ := queryAs(t, e, tokA, `{"question":"What is the capital?","doc_id":"shared-doc"}`)
	if first.CacheHit || !again.CacheHit {
		t.Fatal("owner should hit their own L1 entry")
	}
	// A's paraphrase hits A's own L2 entry (same fixed embedding)...
	if _, a2, _ := queryAs(t, e, tokA, `{"question":"paraphrase capital","doc_id":"shared-doc"}`); a2.Cache.Tier != "l2_semantic" {
		t.Fatalf("control: owner paraphrase should hit L2: %+v", a2.Cache)
	}
	// ...but B, with nothing stored yet, must miss both tiers.
	_, b1, _ := queryAs(t, e, tokB, `{"question":"paraphrase capital","doc_id":"shared-doc"}`)
	_, b2, _ := queryAs(t, e, tokB, `{"question":"What is the capital?","doc_id":"shared-doc"}`)
	if b1.CacheHit || b1.Cache.L2 != "miss" || b2.Cache.Tier == "l1_exact" {
		t.Fatalf("cache crossed users: %+v / %+v", b1.Cache, b2.Cache)
	}
	if l2.Len() != 2 {
		t.Fatalf("expected one L2 entry per user, got %d", l2.Len())
	}
	for _, k := range l1.Keys() {
		if strings.Contains(k, uidA) || strings.Contains(k, uidB) || strings.Contains(k, "public") {
			t.Fatalf("key leaks identity or uses the old public tenant: %s", k)
		}
	}
}

func TestL2ReuseIsOffByDefault(t *testing.T) {
	e, _, l2 := semEnv(t, map[string]llmtest.Reply{"route/router": route("council", true)})
	e.srv.SemanticReuse = false
	e.query(t, docQ("What is the capital?", ""))
	_, res, _ := e.query(t, docQ("paraphrase capital", ""))
	if res.CacheHit || res.Cache.L2 != "disabled_by_policy" || l2.Gets != 0 || l2.Len() != 0 {
		t.Fatalf("%+v gets=%d", res.Cache, l2.Gets)
	}
	if _, again, _ := e.query(t, docQ("What is the capital?", "")); !again.CacheHit {
		t.Fatal("L1 must stay on")
	}
}

// ── Conversations ───────────────────────────────────────────────────

func TestConversationContextAndCacheBypass(t *testing.T) {
	e, _, mem := strictEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	tok, uid := register(t, e.srv, "alice")
	_, r1, _ := queryAs(t, e, tok, `{"question":"Who founded Mirevale?","session_id":"chat1"}`)
	if r1.Session == nil || !r1.Session.Saved || r1.Session.HistoryUsed != 0 || len(r1.Cache.Stored) == 0 {
		t.Fatalf("first turn: %+v %+v", r1.Session, r1.Cache)
	}
	_, r2, _ := queryAs(t, e, tok, `{"question":"When was that?","session_id":"chat1"}`)
	if r2.Session.HistoryUsed != 2 || !r2.Session.Saved || r2.Cache.L1 != "bypassed_conversation" || len(r2.Cache.Stored) != 0 {
		t.Fatalf("follow-up: %+v %+v", r2.Session, r2.Cache)
	}
	p := e.llm.Prompt("member", "a")
	if !strings.Contains(p, "CONVERSATION SO FAR") || !strings.Contains(p, "User: Who founded Mirevale?") || !strings.Contains(p, "Previous response:\n"+r1.Answer) {
		t.Fatalf("history missing from member prompt:\n%s", p)
	}
	// The same question with no history (or other history) never reuses the contextual answer.
	_, fresh, _ := queryAs(t, e, tok, `{"question":"When was that?"}`)
	if fresh.CacheHit {
		t.Fatal("answer computed with history reused without it")
	}
	if len(mem.turns[uid+"|chat1"]) != 4 {
		t.Fatalf("turns %d", len(mem.turns[uid+"|chat1"]))
	}
}

func TestDirectModeSendsHistoryAsMessages(t *testing.T) {
	e, _, _ := strictEnv(t, map[string]llmtest.Reply{"route/router": route("direct", false), "direct/chair": {Text: "Paris."}})
	tok, _ := register(t, e.srv, "alice")
	queryAs(t, e, tok, `{"question":"Capital of France?","session_id":"d1"}`)
	queryAs(t, e, tok, `{"question":"And its population?","session_id":"d1"}`)
	msgs := e.llm.Options("direct", "chair").Messages
	roles := []string{}
	for _, m := range msgs {
		roles = append(roles, m.Role)
	}
	if strings.Join(roles, ",") != "system,user,assistant,user" || msgs[1].Content != "Capital of France?" || msgs[2].Content != "Paris." {
		t.Fatalf("messages %v", msgs)
	}
}

func TestConversationFailurePaths(t *testing.T) {
	e, _, mem := strictEnv(t, map[string]llmtest.Reply{"route/router": route("council", false), "member/a": {Err: errors.New("x")}, "member/b": {Err: errors.New("x")}, "member/c": {Err: errors.New("x")}})
	tok, uid := register(t, e.srv, "alice")
	if c, _, _ := queryAs(t, e, tok, `{"question":"q","session_id":"f1"}`); c != 502 || len(mem.turns[uid+"|f1"]) != 0 {
		t.Fatalf("failed request saved: %d %v", c, mem.turns)
	}
	e2, _, mem2 := strictEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	tok2, uid2 := register(t, e2.srv, "bob")
	mem2.GetErr = errors.New("redis down")
	_, res, _ := queryAs(t, e2, tok2, `{"question":"q","session_id":"f2"}`)
	if res.Session.Saved || res.Session.Error == "" || len(res.Cache.Stored) != 0 || res.Answer == "" {
		t.Fatalf("unknown history: %+v %+v", res.Session, res.Cache)
	}
	mem2.GetErr = nil
	mem2.SaveErr = errors.New("OOM")
	_, res, _ = queryAs(t, e2, tok2, `{"question":"q2","session_id":"f3"}`)
	if res.Session.Saved || !strings.Contains(res.Session.Error, "OOM") || len(mem2.turns[uid2+"|f3"]) != 0 {
		t.Fatalf("save failure: %+v", res.Session)
	}
	for _, bad := range []string{`"bad id!"`, `"` + strings.Repeat("s", 65) + `"`} {
		if c, _, _ := queryAs(t, e2, tok2, `{"question":"q","session_id":`+bad+`}`); c != 400 {
			t.Fatalf("session %s: %d", bad, c)
		}
	}
	e2.srv.Memory = nil
	if c, _, _ := queryAs(t, e2, tok2, `{"question":"q","session_id":"s"}`); c != 503 {
		t.Fatal("sessions without memory must be refused")
	}
}

func TestConversationEndpointsAreUserScoped(t *testing.T) {
	e, _, _ := strictEnv(t, map[string]llmtest.Reply{"route/router": route("council", false)})
	tokA, _ := register(t, e.srv, "alice")
	tokB, _ := register(t, e.srv, "bob")
	queryAs(t, e, tokA, `{"question":"secret question","session_id":"same"}`)
	var hist struct {
		Turns []memory.Turn `json:"turns"`
	}
	_ = json.Unmarshal(doJSON(t, e.srv, "POST", "/api/v1/conversation/history", tokB, `{"session_id":"same"}`).Body.Bytes(), &hist)
	if len(hist.Turns) != 0 {
		t.Fatal("B read A's session")
	}
	if rec := doJSON(t, e.srv, "POST", "/api/v1/conversation/clear", tokB, `{"session_id":"same"}`); rec.Code != 200 {
		t.Fatal(rec.Code)
	}
	_ = json.Unmarshal(doJSON(t, e.srv, "POST", "/api/v1/conversation/history", tokA, `{"session_id":"same"}`).Body.Bytes(), &hist)
	if len(hist.Turns) != 2 || hist.Turns[0].Content != "secret question" {
		t.Fatalf("B's clear affected A: %+v", hist.Turns)
	}
	doJSON(t, e.srv, "POST", "/api/v1/conversation/clear", tokA, `{"session_id":"same"}`)
	_ = json.Unmarshal(doJSON(t, e.srv, "POST", "/api/v1/conversation/history", tokA, `{"session_id":"same"}`).Body.Bytes(), &hist)
	if len(hist.Turns) != 0 {
		t.Fatal("clear failed")
	}
	if rec := doJSON(t, e.srv, "POST", "/api/v1/conversation/history", tokA, `{"session_id":"../x"}`); rec.Code != 400 {
		t.Fatal("invalid session id accepted")
	}
}

func TestCORSAllowlist(t *testing.T) {
	e, _, _ := strictEnv(t, nil)
	pre := func(origin string) http.Header {
		req := httptest.NewRequest("OPTIONS", "/api/v1/query", nil)
		req.Header.Set("Origin", origin)
		req.Header.Set("Access-Control-Request-Method", "POST")
		rec := httptest.NewRecorder()
		e.srv.Routes().ServeHTTP(rec, req)
		return rec.Header()
	}
	if h := pre("https://evil.example"); h.Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("CORS must be off by default")
	}
	e.srv.CORSOrigins = []string{"https://app.example"}
	if h := pre("https://app.example"); h.Get("Access-Control-Allow-Origin") != "https://app.example" || h.Get("Access-Control-Allow-Credentials") == "true" {
		t.Fatalf("allowed origin: %v", h)
	}
	if h := pre("https://evil.example"); h.Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("unlisted origin reflected")
	}
}

func TestHealthReportsPostgres(t *testing.T) {
	e, repo, _ := strictEnv(t, nil)
	var h map[string]any
	_ = json.Unmarshal(doJSON(t, e.srv, "GET", "/health", "", "").Body.Bytes(), &h)
	if h["postgres"] != "healthy" {
		t.Fatalf("%v", h)
	}
	repo.Err = errors.New("down")
	_ = json.Unmarshal(doJSON(t, e.srv, "GET", "/health", "", "").Body.Bytes(), &h)
	if h["postgres"] != "unhealthy" || h["status"] != "ok" {
		t.Fatalf("%v", h)
	}
}

func grant(userID, docID string) auth.DocumentGrant {
	return auth.DocumentGrant{UserID: userID, DocID: docID, Filename: docID + ".md"}
}

func TestHealthReportsLLMModeWithoutCallingModels(t *testing.T) {
	e := newEnv(t, nil)
	var h struct {
		LLM map[string]any `json:"llm"`
	}
	_ = json.Unmarshal(doJSON(t, e.srv, "GET", "/health", "", "").Body.Bytes(), &h)
	if h.LLM["mode"] != "live" || h.LLM["chairman"] != "chair" || len(e.llm.Calls()) != 0 {
		t.Fatalf("fake clients are not mock-prefixed: %v (calls %v)", h.LLM, e.llm.Calls())
	}
	m := council.New([]llm.LLMClient{&llm.MockClient{Name: "member-1"}}, &llm.MockClient{Name: "chairman"}, time.Second)
	e.srv.Council, e.srv.Router = m, nil
	_ = json.Unmarshal(doJSON(t, e.srv, "GET", "/health", "", "").Body.Bytes(), &h)
	if h.LLM["mode"] != "mock" {
		t.Fatalf("all-mock backend: %v", h.LLM)
	}
	e.srv.Council = council.New([]llm.LLMClient{&llm.MockClient{Name: "member-1"}, e.llm.Client("b")}, &llm.MockClient{Name: "chairman"}, time.Second)
	_ = json.Unmarshal(doJSON(t, e.srv, "GET", "/health", "", "").Body.Bytes(), &h)
	if h.LLM["mode"] != "mixed" {
		t.Fatalf("mixed backend: %v", h.LLM)
	}
}
