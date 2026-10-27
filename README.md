# QuorumAI

[![CI](https://github.com/avisharma444/QuorumAI/actions/workflows/ci.yml/badge.svg)](https://github.com/avisharma444/QuorumAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

QuorumAI answers questions with a **council of LLMs**:

1. Three models draft answers in parallel.
2. They peer-review one another anonymously.
3. A chairman model synthesises one validated answer.

Answers are grounded in your uploaded documents or the web. The system runs
as a Go control plane and a Python RAG service, with:

- two-tier Redis caching;
- per-user authentication and document ownership;
- Prometheus, Grafana and OpenTelemetry observability;
- an evaluation harness that measures what the council actually buys you.

**Status:** works end to end, with real-model benchmarks done. It is not
hardened for production (see [Limitations](#limitations)).

QuorumAI is an independent reimplementation of
[regular-life/CouncilAI](https://github.com/regular-life/CouncilAI), whose
README states the MIT license. It rebuilds that architecture with
correctness fixes, measurements and tests. For compatibility, internal
identifiers keep the `councilai` prefix: metric names, Redis keys, Docker
volumes, the Compose project and the Go module path.

## Measured results

All numbers come from runs in this repository. Every figure carries its
sample size, and the linked docs give method and confidence intervals.

| what | result | source |
|---|---|---|
| L1 exact cache, real models (4 models via OpenRouter, 12 questions × 3 repeats) | cold council p50 **14.3 s** → cache hit p50 **4.1 ms**; per-question median speedup **3,811×**; 0 model calls on all 36 hits | [real_model_performance.md](docs/real_model_performance.md) |
| council vs single model (same evidence, n = 12, exploratory) | equal accuracy; the direct call was about **5× faster and 6× cheaper** | [real_model_performance.md](docs/real_model_performance.md) |
| benchmark cost | 76 requests, 141 model calls, **$0.017** provider-reported (cap $2.00) | [real_model_performance.md](docs/real_model_performance.md) |
| cross-encoder reranking (n = 242) | Recall@1 **0.760 → 0.851**; but multi-hop all-evidence@3 **0.782 → 0.655** | [evaluation.md](docs/evaluation.md) |
| semantic (L2) cache safety, held-out pairs | at threshold 0.85, **65% false-positive rate** (true-positive rate 0.910), so L2 answer reuse is **off by default** | [evaluation.md](docs/evaluation.md) |
| load test, mock models (k6, 20 VUs) | L1-hit p95 **≈12 ms**, 0% application failures; mock latencies are not model latencies | [performance.md](docs/performance.md) |

## What works now

- **Go 1.23 control plane:** public `POST /api/v1/register`, `POST /api/v1/login`, `GET /health`; Bearer-token protected `POST /api/v1/query`, `POST /api/v1/ingest`, `POST /api/v1/conversation/history`, `POST /api/v1/conversation/clear`.
- **Authentication (PostgreSQL + JWT):** users in PostgreSQL 16 (bcrypt cost 12, UUID ids, versioned migrations under an advisory lock); HS256 JWTs (golang-jwt v5) that must carry issuer, expiry and issued-at and whose subject is the user UUID. The backend refuses to start without PostgreSQL or with a weak `JWT_SECRET`; there is no in-memory or "anonymous" fallback.
- **Per-user data:** documents are visible only to users who uploaded them (`document_access` grants; another user's document and a missing one return the same 404). L1/L2 cache keys and scopes include the user. Conversation memory lives in Redis per user and session (last 10 exchanges, 24 h).
- **Rate limiting:** Redis GCRA limiter (atomic Lua, Redis clock): 10 req/s burst 20 per user on protected routes, and 1 per 5 s burst 10 per client IP on register/login. 429 responses carry `Retry-After`.
- **Concurrent council:** configurable 3 LLM models, parallel candidate generation, parallel peer reviews, chairman synthesis, defensive fallbacks, optional reflection/revision.
- **Response streaming:** SSE events (`start`, `retrieved`, `candidate`, `review`, `final`; plus errors/fallbacks).
- **Python FastAPI RAG service (reference architecture):** document inspection → adaptive extraction (PyPDF2 text layer, pdfplumber layout/tables, Tesseract OCR for images and scanned PDFs) → layout-aware chunks (tables kept whole, headings attached, captions grouped) → `BAAI/bge-small-en-v1.5` embeddings (mean pooling, 384-D, CPU) → persistent **ChromaDB** collection per document → `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking of 3×k vector candidates. Endpoints: `/ingest`, `/retrieve`, `/retrieve-all`, `/embed`, `/search`, `/health` (liveness), `/ready` (models loaded).
- **Persistence:** Chroma data in the `chroma-data` volume; model weights in the `councilai-hf` volume (downloaded once). Re-uploading identical bytes replaces that document without ever losing the previous index. The new index is written to a staging collection; the swap keeps a backup, rolls back if the rename fails, and is completed or undone on startup after a crash.
- **Redis Stack caching (reference architecture):** L1 exact cache (JSON in Redis, 1 h TTL, hashed versioned keys), L2 semantic cache (RediSearch FLAT/COSINE/FLOAT32 384-D index `idx:semantic_cache`, KNN `FT.SEARCH`, 24 h TTL, threshold 0.85; **answer reuse off by default**, see below) and document summaries (`councilai:v4:doc_summary:<id>`), all behind one circuit breaker (3 failures → open, 10 s cooldown, 1 half-open probe, 2 successes → closed). Caching is an optimisation: when Redis is down nothing is cached in-process. Rate limiting fails **closed** by default (503 while Redis is unreachable, so model spending is never unmetered); `RATE_LIMIT_FAILURE_POLICY=open` restores the reference behaviour.
- **Observability:** Prometheus metrics (`client_golang`) on an internal-only port, Grafana dashboard (22 panels, provisioned), OpenTelemetry tracing in Go and Python with W3C `traceparent` propagation and an optional Jaeger backend, and k6 load-test profiles. See [Observability](#observability) and [docs/performance.md](docs/performance.md).
- **Streamlit UI:** register/login/logout, conversation sessions with history and clear, document upload, model stages, candidate/review inspection.
- **Safe default testing:** `MOCK_LLM=true` runs without model API keys, using clearly identified mock answers.
- **Tests and CI:** GitHub Actions runs Go race tests against real PostgreSQL and Redis Stack, the Python RAG suite (OCR plus the real BGE and reranker models), the evaluation harness tests, and repository checks. A two-service smoke test and HTTP benchmarks run locally.

### Limitations

- **Quality evidence is small.** The real-model comparison covers 12 easy, document-grounded questions; there the council matched a single model at about 5× the latency and 6× the cost. A larger, harder set is needed before claiming any quality gain.
- **L2 semantic answer reuse is unsafe** with these embeddings (near-miss questions such as "enable / disable 2FA" score as paraphrases), so it is off by default.
- **Reranking helps single-hop questions but hurts multi-hop evidence coverage** at k = 3.
- **No production operations:** no alerting rules, log aggregation, Python service metrics, provider retries or per-user spend budgets.
- **Accounts:** no roles or admin users, document sharing, document deletion API, token revocation (tokens stay valid until they expire) or password reset.
- **Legacy documents:** documents uploaded before authentication existed have no owner, so the API cannot reach them.

**Retrieval notes:** embeddings are mean-pooled and not normalised (as in the reference), and Chroma uses L2 distance, so `vector_distance` is a raw distance, not a similarity. Cross-encoder `rerank_score` values are raw logits for ordering, not probabilities. Text-layer PDFs extracted with PyPDF2 produce one block per page, so their chunks can be large; PDFs with tables go through pdfplumber, which keeps paragraphs, tables and captions in page order. Multi-column reading order is not reconstructed. The starter's hashed-vector/SQLite retrieval survives only as a benchmark baseline in `python-rag/app/baselines/`.

## Quick start: Docker on Mac M2

```bash
git clone https://github.com/avisharma444/QuorumAI.git && cd QuorumAI
cp .env.example .env
# Required secrets (never commit .env):
echo "JWT_SECRET=$(openssl rand -base64 48 | tr -d '\n')" >> .env
echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)" >> .env
# Make sure Docker Desktop is running.
docker compose up --build
```

Open:
- UI: <http://localhost:8501> (create an account on the first screen)
- Go backend: <http://localhost:8080/health>

PostgreSQL, Redis and the Python retrieval service have **no host ports**: they
are reachable only on the Docker network. The Python service has no
authentication of its own, so exposing it would bypass the Go backend's
document checks.

For a local demo account, set `APP_ENV=development` and
`AUTH_DEMO_PASSWORD=<your choice>`: the backend then ensures a user `demo`
with that password. The option is rejected outside development, and no
default password exists.

**Upgrading an existing `redis-data` volume:** Redis now uses an append-only
file. Redis ignores `dump.rdb` when AOF is enabled and no AOF exists yet, so
on the first start after this change cached answers and summaries would
appear empty (they are only cache data). To keep them, run once: start Redis
with `--appendonly no`, then `redis-cli CONFIG SET appendonly yes`, wait for
`aof_rewrite_in_progress:0`, and `redis-cli SHUTDOWN`.

**Hardware:** No NVIDIA GPU or vLLM server is needed. The default baseline uses CPU-only vectorization and fake LLM responses.

## Quick start: local development (no Docker)

Requires Go 1.23+, Python 3.11+. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
# CPU-only torch first, then the service (needs tesseract + poppler: brew install tesseract poppler)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r python-rag/requirements-dev.txt requests

# Terminal A (first start downloads ~220 MB of model weights into HF_HOME)
cd python-rag
CHROMA_DB_PATH=./chroma_db uvicorn app.main:app --reload --port 8000

# PostgreSQL and Redis Stack for local runs (example containers)
docker run -d --name pg -e POSTGRES_USER=councilai -e POSTGRES_PASSWORD=localdev -e POSTGRES_DB=councilai -p 127.0.0.1:5433:5432 postgres:16-alpine
docker run -d --name rs -p 127.0.0.1:6380:6379 redis/redis-stack-server:7.4.0-v1

# Terminal B, from project root
cd go-backend
MOCK_LLM=true RAG_SERVICE_URL=http://localhost:8000 REDIS_ADDR=localhost:6380 \
  DATABASE_URL='postgres://councilai:localdev@localhost:5433/councilai?sslmode=disable' \
  JWT_SECRET="$(openssl rand -base64 48)" go run ./cmd/server

# Optional terminal C
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

### Smoke test: register, upload document, query, repeat

```bash
TOKEN=$(curl -sS -X POST http://localhost:8080/api/v1/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"choose a password"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
# Later: same body to /api/v1/login. Tokens expire after JWT_EXPIRATION (24h).
curl -sS -H "Authorization: Bearer $TOKEN" -F 'file=@README.md' http://localhost:8080/api/v1/ingest
# Copy the returned doc_id into the next request:
curl -sS -X POST http://localhost:8080/api/v1/query -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What makes the council parallel?","doc_id":"DOC_ID","strategy":"council"}'

# Multi-turn: send the same session_id; earlier exchanges are given to the models.
curl -sS -X POST http://localhost:8080/api/v1/query -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"question":"And why does that matter?","session_id":"chat-1"}'
curl -sS -X POST http://localhost:8080/api/v1/conversation/history -H "Authorization: Bearer $TOKEN" -d '{"session_id":"chat-1"}'

# Or stream with no document:
curl -N -X POST http://localhost:8080/api/v1/query -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d '{"question":"Compare Go and Python concurrency","strategy":"council_deep"}'
```

### Authentication, ownership and conversations

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

### How a query is routed

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

### Chairman synthesis and self-review

In `council` and `council_deep` the chairman returns validated JSON
(`answer`, `reasoning`, optional `source` and `confidence`). Invalid output
counts as a chairman failure: the top peer-ranked candidate is returned and
`degraded` says so. `council_deep` then runs one quality audit
(`quality`, `issues`, `suggestion`, `faithful`) and, only on
`needs_revision`, exactly one revision; failures keep the current answer and
are reported, never turned into a "good" audit. `confidence` is the model's
self-report (absent if it gave none), not a calibrated probability. Only
answers where every stage completed are cached.

### Use real model providers

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

### Caching: what is cached, and the semantic threshold

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

## Evaluation

`evaluation/` holds a reproducible evaluation harness: a versioned dataset
built from SQuAD 2.0, HotpotQA and DROP (500 questions, dev/held-out
splits), a labelled question-pair set for semantic-cache safety (840 pairs),
retrieval and cache benchmarks on the real models, and an answer-quality
runner with statistics, cost accounting and a spend-limited real-model mode
that has not been run. Results: [docs/evaluation.md](docs/evaluation.md);
guide: [evaluation/README.md](evaluation/README.md).

## Observability

Metrics and tracing are always compiled in; the heavier pieces are optional
Compose profiles so the normal stack stays small:

```bash
# Prometheus (127.0.0.1:9091) + Grafana (127.0.0.1:3000). Set GRAFANA_ADMIN_PASSWORD in .env first.
docker compose --profile observability up -d
# Add Jaeger (127.0.0.1:16686) and export traces from Go and Python over OTLP/HTTP:
OTEL_TRACES_EXPORTER=otlp docker compose --profile observability --profile tracing up -d
```

- **Metrics** (`go-backend/internal/metrics`): the reference's six families
  (`councilai_request_count_total`, `councilai_request_latency_seconds`,
  `councilai_chairman_synthesis_count_total`, `councilai_llm_failure_count_total`,
  `councilai_cache_operations_total`, `councilai_council_response_seconds`) plus
  stage, model-request, token, router, cache-lookup/-write, circuit-breaker,
  RAG, dependency-error, rate-limit, conversation, in-flight and disconnect
  metrics. Labels are bounded: route *patterns* (`unmatched` for anything
  else), stage, provider family, result, level, strategy. No user, document
  or session IDs, questions, model names or raw paths.
  `/metrics` is served by promhttp on `METRICS_ADDR` (`:9464`), which is not
  published to the host; Prometheus scrapes it every 15 s. The public API
  port answers `/metrics` with 404.
- **Corrected reference behaviour:** the chairman counter is incremented
  (the reference declared but never incremented it); council time is observed
  once per run (the reference observed it twice); model failures of every
  stage are counted (not only candidates), timeouts count as failures and
  client cancellations do not; quantile panels aggregate `sum by (le)`.
- **Tracing** (`go-backend/internal/telemetry`, `python-rag/app/telemetry.py`):
  exporters `noop` (default), `stdout`, `otlp`; parent-based sampling
  (`OTEL_TRACES_SAMPLER_ARG`). One trace per request: `POST /api/v1/query` →
  rate-limit check, document access, L1 lookup, router, L2 lookup (embed),
  `rag POST /retrieve` → Python `POST /retrieve` → embed / vector search /
  rerank, `council.run` → candidates (one span per model, concurrent
  siblings) → peer review → chairman → reflection → revision, cache store,
  conversation save. Only stages that ran produce spans. Responses carry
  `X-Trace-ID`. Span attributes hold categories, counts and timings only:
  never questions, answers, document text, user IDs or tokens. Export runs
  in a background batch; an unreachable collector is logged (rate-limited)
  and counted in `councilai_telemetry_errors_total` without affecting
  requests. Python tracing is an extension: the reference traced Go only.
- **Grafana** uses the provisioned `Prometheus` data source and the
  provisioned dashboard (`monitoring/grafana/dashboards/councilai.json`,
  generated by `monitoring/build_dashboard.py`). There is no default
  password: Grafana refuses to start without `GRAFANA_ADMIN_PASSWORD`.
- **Load tests** (`tests/load/load_test.js`, k6): profiles `reference`
  (the original 0→5→20→0 VU stages), `cold`, `l1`, `l2` (needs
  `SEMANTIC_CACHE_ENABLED=true`), `direct`, `council`, `deep`, `multiuser`
  and `ratelimit`. Each run registers its own users with random credentials
  and uploads `tests/load/corpus.md`; rate limiting stays on and 429s are
  reported separately. Results: [docs/performance.md](docs/performance.md).

  ```bash
  docker run --rm --network councilai-reproduced_default -v "$PWD/tests/load:/scripts" \
    -e PROFILE=reference grafana/k6:0.54.0 run /scripts/load_test.js
  ```

### Tests and benchmarks

```bash
# Go tests, including race detector (integration tests skip without the services below)
(cd go-backend && go test -race ./...)
# Go tests against real PostgreSQL, Redis Stack and BGE embeddings (stack running;
# redis-plain / redis-auth are throwaway containers for the module-less and password cases)
docker run -d --rm --name redis-plain --network councilai-reproduced_default redis:7.4-alpine
docker run -d --rm --name redis-auth --network councilai-reproduced_default redis:7.4-alpine redis-server --requirepass secret
docker run --rm --network councilai-reproduced_default -v "$PWD/go-backend":/src -w /src \
  -e REDIS_TEST_ADDR=redis:6379 -e REDIS_PLAIN_TEST_ADDR=redis-plain:6379 -e REDIS_AUTH_TEST_ADDR=redis-auth:6379 \
  -e RAG_TEST_URL=http://python-rag:8000 \
  -e POSTGRES_TEST_URL="postgres://councilai:$(grep ^POSTGRES_PASSWORD= .env | cut -d= -f2-)@postgres:5432/councilai?sslmode=disable" \
  golang:1.23 go test -race ./...
docker stop redis-plain redis-auth
# Cache benchmarks: Redis-only latency (Go) and end-to-end / false-positive evaluation (Python)
docker run --rm --network councilai-reproduced_default -v "$PWD/go-backend":/src -w /src \
  -e CACHE_BENCH=1 -e REDIS_TEST_ADDR=redis:6379 golang:1.23 go test -run TestCacheLatencyReport -v ./internal/cache/
# (the end-to-end part needs the backend started with SEMANTIC_CACHE_ENABLED=true)
docker run --rm --network councilai-reproduced_default -v "$PWD/benchmarks:/b" councilai-rag-test python /b/cache_benchmark.py
# Python tests, including real OCR and real-model integration tests (Docker, no local installs)
(cd python-rag && docker build --target test -t councilai-rag-test . && docker run --rm -v councilai-hf:/models councilai-rag-test)
# Retrieval evaluation (40-chunk fictional corpus, 30 queries): hash baseline vs neural vs neural + rerank
docker run --rm -v councilai-hf:/models councilai-rag-test python -m benchmarks.rag_eval [--pdf]
# Running stack: register → upload → retrieve → council → cache → SSE (registers a throwaway user)
python tests/e2e_smoke.py
# Both API servers must be running for the HTTP benchmark
N=20 python benchmarks/benchmark_http.py
```

**Interpretation:** In mock mode, latency figures measure HTTP/orchestration/cache overhead without real model inference. They must **not** be reported as actual multi-LLM speedups. Query-quality comparisons require reference answers and a labeled dataset; a council is not automatically more accurate than a single model.

## Roadmap

**Done:**
- Go council with streaming
- layout-aware RAG with OCR, ChromaDB and reranking
- L1 and L2 Redis caches behind a circuit breaker
- PostgreSQL and JWT authentication with per-user documents, conversations and rate limiting
- metrics, tracing, dashboards and k6 load tests
- the evaluation harness, and a budget-guarded real-model benchmark

**Next:**
1. **Quality study that can tell council from direct:** a larger, harder held-out set; alias-aware scoring; blinded manual review; cost and latency per correct answer.
2. **Safer semantic reuse:** an answer-equivalence check (for example an entailment model) before L2 reuse, re-measured on the held-out pair set.
3. **Multi-hop retrieval:** keep evidence diversity when reranking (for example MMR, or reserving top vector hits), measured by all-evidence@k.
4. **Provider resilience and spend control:** bounded retries, fallback models, per-user token budgets.
5. **Real-model load test** under a fixed budget, to measure provider tail latency under concurrency.
6. **Operations:** alerting rules, Python service metrics, token revocation, a document deletion API.

[BUILD_PLAN.md](BUILD_PLAN.md) records the original phase plan and its
acceptance criteria.

## License

[MIT](LICENSE) © 2026 avisharma444. QuorumAI reimplements the architecture of
[regular-life/CouncilAI](https://github.com/regular-life/CouncilAI), which
its README lists as MIT.
