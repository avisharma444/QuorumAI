# Architecture and behaviour

This is the detailed reference for how QuorumAI works. The [README](../README.md)
gives the overview; [observability.md](observability.md) covers metrics and
tracing; [development.md](development.md) covers local setup and tests.

## Components

- **Go 1.23 control plane:** public `POST /api/v1/register`, `POST /api/v1/login`, `GET /health`; Bearer-token protected `POST /api/v1/query`, `POST /api/v1/ingest`, `POST /api/v1/conversation/history`, `POST /api/v1/conversation/clear`.
- **Authentication (PostgreSQL + JWT):** users in PostgreSQL 16 (bcrypt cost 12, UUID ids, versioned migrations under an advisory lock); HS256 JWTs (golang-jwt v5) that must carry issuer, expiry and issued-at and whose subject is the user UUID. The backend refuses to start without PostgreSQL or with a weak `JWT_SECRET`; there is no in-memory or "anonymous" fallback.
- **Per-user data:** documents are visible only to users who uploaded them (`document_access` grants; another user's document and a missing one return the same 404). L1/L2 cache keys and scopes include the user. Conversation memory lives in Redis per user and session (last 10 exchanges, 24 h).
- **Rate limiting:** Redis GCRA limiter (atomic Lua, Redis clock): 10 req/s burst 20 per user on protected routes, and 1 per 5 s burst 10 per client IP on register/login. 429 responses carry `Retry-After`.
- **Concurrent council:** configurable 3 LLM models, parallel candidate generation, parallel peer reviews, chairman synthesis, defensive fallbacks, optional reflection/revision.
- **Response streaming:** SSE events (`start`, `retrieved`, `candidate`, `review`, `final`; plus errors/fallbacks).
- **Python FastAPI RAG service (reference architecture):** document inspection → adaptive extraction (PyPDF2 text layer, pdfplumber layout/tables, Tesseract OCR for images and scanned PDFs) → layout-aware chunks (tables kept whole, headings attached, captions grouped) → `BAAI/bge-small-en-v1.5` embeddings (mean pooling, 384-D, CPU) → persistent **ChromaDB** collection per document → `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking of 3×k vector candidates. Endpoints: `/ingest`, `/retrieve`, `/retrieve-all`, `/embed`, `/search`, `/health` (liveness), `/ready` (models loaded).
- **Persistence:** Chroma data in the `chroma-data` volume; model weights in the `councilai-hf` volume (downloaded once). Re-uploading identical bytes replaces that document without ever losing the previous index. The new index is written to a staging collection; the swap keeps a backup, rolls back if the rename fails, and is completed or undone on startup after a crash.
- **Redis Stack caching (reference architecture):** L1 exact cache (JSON in Redis, 1 h TTL, hashed versioned keys), L2 semantic cache (RediSearch FLAT/COSINE/FLOAT32 384-D index `idx:semantic_cache`, KNN `FT.SEARCH`, 24 h TTL, threshold 0.85; **answer reuse off by default**, see below) and document summaries (`councilai:v4:doc_summary:<id>`), all behind one circuit breaker (3 failures → open, 10 s cooldown, 1 half-open probe, 2 successes → closed). Caching is an optimisation: when Redis is down nothing is cached in-process. Rate limiting fails **closed** by default (503 while Redis is unreachable, so model spending is never unmetered); `RATE_LIMIT_FAILURE_POLICY=open` restores the reference behaviour.
- **Observability:** Prometheus metrics (`client_golang`) on an internal-only port, Grafana dashboard (22 panels, provisioned), OpenTelemetry tracing in Go and Python with W3C `traceparent` propagation and an optional Jaeger backend, and k6 load-test profiles. See [observability.md](observability.md) and [performance.md](performance.md).
- **Streamlit UI:** register/login/logout, conversation sessions with history and clear, document upload, model stages, candidate/review inspection.
- **Safe default testing:** `MOCK_LLM=true` runs without model API keys, using clearly identified mock answers.
- **Tests and CI:** GitHub Actions runs Go race tests against real PostgreSQL and Redis Stack, the Python RAG suite (OCR plus the real BGE and reranker models), the evaluation harness tests, and repository checks. A two-service smoke test and HTTP benchmarks run locally.

## Retrieval notes

Embeddings are mean-pooled and not normalised (as in the reference), and Chroma uses L2 distance, so `vector_distance` is a raw distance, not a similarity. Cross-encoder `rerank_score` values are raw logits for ordering, not probabilities. Text-layer PDFs extracted with PyPDF2 produce one block per page, so their chunks can be large; PDFs with tables go through pdfplumber, which keeps paragraphs, tables and captions in page order. Multi-column reading order is not reconstructed. The starter's hashed-vector/SQLite retrieval survives only as a benchmark baseline in `python-rag/app/baselines/`.

## Authentication, ownership and conversations

- **Accounts:** usernames are 3–32 characters (`a-z 0-9 . _ -`,
  case-insensitive); passwords 8–72 bytes. A wrong password and an unknown
  user return the same 401 `invalid credentials` (bcrypt runs in both cases).
  When PostgreSQL is unavailable, register/login return 503.
- **Tokens:** HS256 only, issuer `council-ai`, `exp`/`iat` required, subject =
  user UUID. `JWT_SECRET` must be at least 32 bytes and not a placeholder
  (the reference default `council-ai-secret-change-me` is rejected).
  Changing the secret invalidates all issued tokens.
- **Documents:** a successful upload grants the uploader access. Uploading
  identical bytes again (same content-addressed `doc_id`) grants the second
  user access too, without exposing anything the first user did. If the grant
  cannot be recorded, the upload returns 503 and no access is given.
- **Conversations:** pass `session_id` (`A-Z a-z 0-9 _ -`, max 64). The last
  5 exchanges are included in the models' prompts (council members get them
  as text; direct mode sends them as chat messages). Answers produced with
  history are neither read from nor written to L1/L2, because the same
  question can mean something different in another conversation. An exchange
  is saved only after a successful answer (or cache hit), as one atomic Redis
  transaction; responses report `session.history_turns_used` and
  `session.saved`. If the history cannot be read, the request is answered
  without it, is not cached, and is not saved, and the response says so.
- **CORS** is off by default. `CORS_ALLOWED_ORIGINS` takes an explicit
  comma-separated list; `*` is rejected. Credentials are never allowed (the
  API uses Bearer headers, not cookies).
- **Logs** record method, route pattern, status, latency, request ID, trace
  ID and user UUID; never headers, bodies, query strings, tokens or passwords.

## How a query is routed

`strategy` defaults to `auto`: the configured **router model** reads the
question (and the attached document's summary) and returns JSON choosing
`direct`, `council` or `council_deep`, plus whether the document is needed.
`direct`, `council` and `council_deep` can also be sent explicitly as a
diagnostic override that skips the router.

- Uploading a document returns `preview_text` (a Beginning-Middle-End sample
  of its chunks) and a `summary` written by the **ingestion model**. The
  summary is kept in Redis for 1 hour.
- Questions without a document are grounded with live DuckDuckGo results
  (Python `/search`). Responses report `grounding` (`document`, `web` or
  `none`) and a `routing` block (strategy, reasoning, document use, errors).
- Router failures fall back to `council`, keeping any attached document. If a
  document's summary is missing, the document is still used.
- Mock mode (`MOCK_LLM=true`) routes with explicit markers in the question:
  `#route=direct|council|council_deep|invalid` and `#needs_doc=true|false`.
  Without markers the mock router picks `council`. Chairman and self-review
  paths can be scripted the same way: `#chair=invalid` (chairman fallback),
  `#reflect=needs_revision` (one revision), `#reflect=invalid`,
  `#revise=invalid`. Mock models never report confidence.

## Chairman synthesis and self-review

In `council` and `council_deep` the chairman returns validated JSON
(`answer`, `reasoning`, optional `source` and `confidence`). Invalid output
counts as a chairman failure: the top peer-ranked candidate is returned and
`degraded` says so. `council_deep` then runs one quality audit
(`quality`, `issues`, `suggestion`, `faithful`) and, only on
`needs_revision`, exactly one revision; failures keep the current answer and
are reported, never turned into a "good" audit. `confidence` is the model's
self-report (absent if it gave none), not a calibrated probability. Only
answers where every stage completed are cached.

## Model providers

Models are chosen in `config.yaml` (repository root), one entry per role:
three council `slots`, plus `chairman`, `router` and `ingestion`. Each entry
has a `provider` (`gemini`, `openrouter`, `nvidia-nim`, or `local` for vLLM)
and a `model`. Any entry can be overridden with environment variables such as
`COUNCIL_2_PROVIDER` / `COUNCIL_2_MODEL` or `CHAIRMAN_MODEL`.

API keys are read only from the environment. Edit `.env`:

```dotenv
MOCK_LLM=false
GEMINI_API_KEY=replace-me        # needed if any role uses gemini
OPENROUTER_API_KEY=replace-me    # needed if any role uses openrouter
NVIDIA_NIM_API_KEY=replace-me    # needed if any role uses nvidia-nim
```

Startup validates everything and fails with a list of every problem (missing
keys, unknown providers, bad durations). It never falls back to mock answers.
The old variables `MODEL_API_KEY`, `MODEL_BASE_URL`, `COUNCIL_MODELS`,
`CHAIR_MODEL` and `RAG_URL` are rejected with a pointer to their replacements.

Check provider pricing first. A complete council request makes 7+ model calls
(3 candidates + 3 peer reviews + chairman; more in deep mode). A member
whose candidate failed is not asked to review; it is listed in
`failed_candidates`, and the stream gets a `review_skipped` event. Responses
include provider-reported token usage per call and in total (`token_usage`).

**Self-hosted open-weight members.** A slot with provider `local` calls an
OpenAI-compatible vLLM server, either on your own machine or remotely with a
bearer key. Phi-4-mini on RunPod Serverless (one RTX 4090) has been run as the
third council member this way. Setup, key handling and the fallback when the
endpoint is down are in [runpod_vllm_deployment.md](runpod_vllm_deployment.md);
measurements are in [gpu_inference_performance.md](gpu_inference_performance.md).

## Caching: what is cached, and the semantic threshold

- **Lookup order:** L1 exact key → LLM router → L2 semantic lookup (only when
  the plan uses the attached document) → retrieval and council. L1 keys cover
  the user UUID, document, normalised question, auto vs
  override strategy, top_k, model/prompt fingerprint. L2 scope adds the
  resolved strategy and the embedding model, so paraphrases never reuse an
  answer produced under a different strategy, document or configuration.
- **Only complete answers are cached:** no router fallbacks, grounding
  failures, chairman/reflection/revision failures, single-model or
  no-review fallbacks, or cancelled requests. Web-grounded answers are never
  semantically cached.
- **Responses are truthful about cache hits:** `cache.tier`,
  `cache.inference_executed`, similarity/distance for L2, and the original
  generation's tokens in `cache.original_token_usage` (not counted again).
- **Measured threshold trade-off** (`benchmarks/cache_benchmark.py`: 25
  paraphrase pairs, 25 near-miss pairs, real BGE vectors; end-to-end through
  Redis matched these rates exactly):

  | similarity threshold | paraphrases reused | near-misses wrongly reused (with number guard) |
  |---|---|---|
  | 0.80 | 92% | 68% |
  | **0.85 (default)** | **52%** | **28%** |
  | 0.90 | 16% | 16% |
  | 0.95 | 4% | 0% |

  Pairs such as "features added / removed in version 3", "minimum / maximum
  age" and "enable / disable 2FA" score ≈0.93, above most real paraphrases,
  so no threshold separates them with these embeddings. A hit is also
  rejected when the two questions mention different numbers (2024 vs 2025).
  Because of this, **L2 answer reuse is disabled by default**
  (`SEMANTIC_CACHE_ENABLED=false`; responses show `cache.l2:
  "disabled_by_policy"`). L1 exact caching stays on. To evaluate L2, set
  `SEMANTIC_CACHE_ENABLED=true` (and consider raising
  `SEMANTIC_CACHE_THRESHOLD`; 0.95 here gave no false positives but little
  reuse). With authentication on, the end-to-end benchmark reproduced the
  same 52% / 28% rates.
