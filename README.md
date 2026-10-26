# CouncilAI — runnable learning build (Phase 1)

**Status:** Working end-to-end vertical slice, **not** the full `regular-life/CouncilAI` project and **not** production-ready.

Reference architecture: <https://github.com/regular-life/CouncilAI>. This is a new, intentionally smaller implementation intended for hands-on learning and progressively measurable engineering work.

## What works now

- **Go 1.23 control plane:** `POST /api/v1/query`, `POST /api/v1/ingest`, `GET /health`, `GET /metrics`.
- **Concurrent council:** configurable 3 LLM models, parallel candidate generation, parallel peer reviews, chairman synthesis, defensive fallbacks, optional reflection/revision.
- **Response streaming:** SSE events (`start`, `retrieved`, `candidate`, `review`, `final`; plus errors/fallbacks).
- **Python FastAPI RAG service (reference architecture):** document inspection → adaptive extraction (PyPDF2 text layer, pdfplumber layout/tables, Tesseract OCR for images and scanned PDFs) → layout-aware chunks (tables kept whole, headings attached, captions grouped) → `BAAI/bge-small-en-v1.5` embeddings (mean pooling, 384-D, CPU) → persistent **ChromaDB** collection per document → `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking of 3×k vector candidates. Endpoints: `/ingest`, `/retrieve`, `/retrieve-all`, `/embed`, `/search`, `/health` (liveness), `/ready` (models loaded).
- **Persistence:** Chroma data in the `chroma-data` volume; model weights in the `councilai-hf` volume (downloaded once). Re-uploading identical bytes replaces that document atomically.
- **Redis Stack caching (reference architecture):** L1 exact cache (JSON in Redis, 1 h TTL, hashed versioned keys), L2 semantic cache (RediSearch FLAT/COSINE/FLOAT32 384-D index `idx:semantic_cache`, KNN `FT.SEARCH`, 24 h TTL, threshold 0.85) and document summaries (`councilai:v3:doc_summary:<id>`), all behind one circuit breaker (3 failures → open, 10 s cooldown, 1 half-open probe, 2 successes → closed). Redis is an optimisation: when it is down, requests are answered normally and nothing is cached in-process.
- **Streamlit UI:** document upload, model stages, candidate/review inspection.
- **Safe default testing:** `MOCK_LLM=true` runs without model API keys, using clearly identified mock answers.
- **Tests:** Go unit tests (including parallel execution and SSE), Python ingestion/retrieval tests, an HTTP latency benchmark, and a real two-service smoke-test script.

### Deliberately NOT implemented yet

- JWT authentication and PostgreSQL user storage
- Prometheus/Grafana/OpenTelemetry instrumentation (the current `/metrics` is a JSON counter endpoint)
- Calibrated confidence scores, quality improvement claims, performance claims, or load-test evidence
- Multi-user authorization and protection of uploaded documents
- Provider retries/rate budgets, production deployment and hardening

**Retrieval notes:** embeddings are mean-pooled and not normalised (as in the reference), and Chroma uses L2 distance, so `vector_distance` is a raw distance, not a similarity. Cross-encoder `rerank_score` values are raw logits for ordering, not probabilities. Text-layer PDFs extracted with PyPDF2 produce one block per page, so their chunks can be large; PDFs with tables go through pdfplumber, which keeps paragraphs, tables and captions in page order. Multi-column reading order is not reconstructed. The starter's hashed-vector/SQLite retrieval survives only as a benchmark baseline in `python-rag/app/baselines/`.

## Quick start: Docker on Mac M2

```bash
cp .env.example .env
# Make sure Docker Desktop is running.
docker compose up --build
```

Open:
- UI: <http://localhost:8501>
- Go backend: <http://localhost:8080/health>
- Python retrieval: <http://localhost:8000/health>

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

# Terminal B, from project root
cd go-backend
MOCK_LLM=true RAG_SERVICE_URL=http://localhost:8000 go run ./cmd/server

# Optional terminal C
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

### Smoke test: upload document, query, repeat

```bash
curl -sS -F 'file=@README.md' http://localhost:8080/api/v1/ingest
# Copy the returned doc_id into the next request:
curl -sS -X POST http://localhost:8080/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What makes the council parallel?","doc_id":"DOC_ID","strategy":"council"}'

# Or test with no documents:
curl -N -X POST http://localhost:8080/api/v1/query \
  -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d '{"question":"Compare Go and Python concurrency","strategy":"council_deep"}'
```

### How a query is routed

`strategy` defaults to `auto`: the configured **router model** reads the
question (and the attached document's summary) and returns JSON choosing
`direct`, `council` or `council_deep`, plus whether the document is needed.
`direct`, `council` and `council_deep` can also be sent explicitly as a
diagnostic override that skips the router.

- Uploading a document returns `preview_text` (a Beginning-Middle-End sample
  of its chunks) and a `summary` written by the **ingestion model**. The
  summary is kept in the backend's in-memory cache for 1 hour (the reference
  uses Redis; summaries are lost on restart until the Redis phase).
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
(3 candidates + 3 peer reviews + chairman; more in deep mode). Responses
include provider-reported token usage per call and in total (`token_usage`).

### Caching: what is cached, and the semantic threshold

- **Lookup order:** L1 exact key → LLM router → L2 semantic lookup (only when
  the plan uses the attached document) → retrieval and council. L1 keys cover
  tenant ("public" until auth exists), document, normalised question, auto vs
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
  Treat the semantic cache as unsafe for high-stakes use at 0.85; raise
  `SEMANTIC_CACHE_THRESHOLD` (0.95 here gave no false positives but little
  reuse) or disable Redis.

### Tests and benchmarks

```bash
# Go tests, including race detector (integration tests need Redis Stack; see below)
(cd go-backend && go test -race ./...)
# Go tests against real Redis Stack + real BGE embeddings (stack running)
docker run --rm --network councilai-reproduced_default -v "$PWD/go-backend":/src -w /src \
  -e REDIS_TEST_ADDR=redis:6379 -e RAG_TEST_URL=http://python-rag:8000 golang:1.23 go test -race ./...
# Cache benchmarks: Redis-only latency (Go) and end-to-end / false-positive evaluation (Python)
docker run --rm --network councilai-reproduced_default -v "$PWD/go-backend":/src -w /src \
  -e CACHE_BENCH=1 -e REDIS_TEST_ADDR=redis:6379 golang:1.23 go test -run TestCacheLatencyReport -v ./internal/cache/
docker run --rm --network councilai-reproduced_default -v "$PWD/benchmarks:/b" councilai-rag-test python /b/cache_benchmark.py
# Python tests, including real OCR and real-model integration tests (Docker, no local installs)
(cd python-rag && docker build --target test -t councilai-rag-test . && docker run --rm -v councilai-hf:/models councilai-rag-test)
# Retrieval evaluation (40-chunk fictional corpus, 30 queries): hash baseline vs neural vs neural + rerank
docker run --rm -v councilai-hf:/models councilai-rag-test python -m benchmarks.rag_eval [--pdf]
# Both API servers must be running for the smoke test
python tests/e2e_smoke.py
# Both API servers must be running for the HTTP benchmark
N=20 python benchmarks/benchmark_http.py
```

**Interpretation:** In mock mode, latency figures measure HTTP/orchestration/cache overhead without real model inference. They must **not** be reported as actual multi-LLM speedups. Query-quality comparisons require reference answers and a labeled dataset; a council is not automatically more accurate than a single model.

## Next engineering milestones

1. **Phase 2: Actual semantic RAG.** Implement persistent ChromaDB + local MiniLM embeddings; evaluate recall@k/MRR and exact document-grounding failures against the lexical baseline.
2. **Phase 3: Shared caching.** Go Redis client, L1 exact + Redis Stack RediSearch VSS cache, document-scope isolation, circuit breaker, meaningful p50/p95 cache metrics.
3. **Phase 4: Hardening.** Auth, per-user document ownership, request cancellation, retries with bounded budgets, traces and Prometheus metrics, tests/load test.
4. **Phase 5: Model-quality experiment.** Curate held-out document questions, use identical retrieved evidence for single-model vs council, score with manual/source-grounded rubric and report cost/latency/accuracy tradeoffs.

See [BUILD_PLAN.md](BUILD_PLAN.md) for measurable acceptance criteria.
