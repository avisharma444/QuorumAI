# Development and testing

## Local development (no Docker)

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

## Smoke test: register, upload document, query, repeat

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

## Tests and benchmarks

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

## Upgrading an existing `redis-data` volume

Redis now uses an append-only
file. Redis ignores `dump.rdb` when AOF is enabled and no AOF exists yet, so
on the first start after this change cached answers and summaries would
appear empty (they are only cache data). To keep them, run once: start Redis
with `--appendonly no`, then `redis-cli CONFIG SET appendonly yes`, wait for
`aof_rewrite_in_progress:0`, and `redis-cli SHUTDOWN`.
