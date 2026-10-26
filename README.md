# CouncilAI — runnable learning build (Phase 1)

**Status:** Working end-to-end vertical slice, **not** the full `regular-life/CouncilAI` project and **not** production-ready.

Reference architecture: <https://github.com/regular-life/CouncilAI>. This is a new, intentionally smaller implementation intended for hands-on learning and progressively measurable engineering work.

## What works now

- **Go 1.23 control plane:** `POST /api/v1/query`, `POST /api/v1/ingest`, `GET /health`, `GET /metrics`.
- **Concurrent council:** configurable 3 LLM models, parallel candidate generation, parallel peer reviews, chairman synthesis, defensive fallbacks, optional reflection/revision.
- **Response streaming:** SSE events (`start`, `retrieved`, `candidate`, `review`, `final`; plus errors/fallbacks).
- **Python FastAPI RAG service:** PDF/TXT/Markdown text extraction, overlapping per-page chunks, 384-D vectors, retrieval ranked by cosine similarity.
- **Persistent document store:** SQLite, so ingested documents survive service restarts.
- **Exact-query cache:** thread-safe Go in-process TTL cache (one-hour TTL) for repeated questions.
- **Streamlit UI:** document upload, model stages, candidate/review inspection.
- **Safe default testing:** `MOCK_LLM=true` runs without model API keys, using clearly identified mock answers.
- **Tests:** Go unit tests (including parallel execution and SSE), Python ingestion/retrieval tests, an HTTP latency benchmark, and a real two-service smoke-test script.

### Deliberately NOT implemented yet

- Redis Stack L1+L2 semantic cache and circuit breaker
- ChromaDB vector store, cross-encoder reranking, scanned-PDF OCR
- JWT authentication and PostgreSQL user storage
- Prometheus/Grafana/OpenTelemetry instrumentation (the current `/metrics` is a JSON counter endpoint)
- Calibrated confidence scores, quality improvement claims, performance claims, or load-test evidence
- Multi-user authorization and protection of uploaded documents
- OpenRouter/Gemini-specific structured response adapters, retries/rate budgets, production deployment and hardening

**Embeddings caveat:** The default `hash` backend uses *hashed lexical token vectors* (384 dimensions) to make local builds fast and testable; it is **not** genuine semantic embedding. Set `EMBEDDING_BACKEND=sentence-transformer` with the extra neural requirements to use a small 384-dimensional MiniLM embedding model. The default SQLite cosine store is **not ChromaDB**.

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
pip install -r python-rag/requirements.txt pytest requests

# Terminal A
cd python-rag
DATABASE_PATH=./documents.sqlite3 uvicorn app.main:app --reload --port 8000

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

Valid strategies: `auto`, `direct`, `council`, `council_deep`. `auto` currently uses **simple deterministic heuristics**; it is not an LLM router yet.

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

### Tests and benchmarks

```bash
# Go tests, including race detector
(cd go-backend && go test -race ./...)
# Python pure-core tests
(cd python-rag && python -m pytest -q tests)
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
