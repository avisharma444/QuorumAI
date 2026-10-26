# Load-test and observability measurements (checkpoint 9)

All numbers come from real k6 runs against the running Docker stack and from
Prometheus scrapes of the Go backend. **Every model was a mock
(`MOCK_LLM=true`)**: mock models answer in well under a millisecond and
report no token usage. These figures therefore measure the HTTP layer,
authentication, rate limiting, Redis, PostgreSQL and the neural retrieval
service (BGE embedding, Chroma search, cross-encoder rerank on CPU). They
are **not** LLM inference latency, answer quality, or cost. No real-model
measurement has been made (it needs paid API calls).

Raw results are kept locally in `tests/load/results/` (git-ignored);
`python3 tests/load/summarize.py tests/load/results/<run>` regenerates the
tables below from them.

## Environment and method

| item | value |
|---|---|
| machine | Apple M2, 8 cores, 8 GB (MacBook Air) |
| Docker | 27.4.0, Linux VM aarch64, 8 vCPU, 3.83 GiB memory |
| load generator | k6 v0.54.0 in a container on the Compose network (`grafana/k6:0.54.0`) |
| models | mock (`MOCK_LLM=true`); no provider calls |
| corpus | `tests/load/corpus.md`, 1,993 bytes, one fictional handbook (2 chunks after layout chunking) |
| cache | L1 exact on; L2 semantic reuse **off** (default) except in the `l2` profile |
| rate limits | production defaults: 10 req/s, burst 20 per user; register/login 1 per 5 s, burst 10 per IP |
| tracing | off (noop) for the main series; one comparison series with OTLP export at 100% sampling |
| container limits | Postgres, Prometheus, Grafana, Jaeger 256 MiB each; others uncapped |
| warm-up | users and the document are created in `setup()`; the `l1` and `l2` profiles also ask every question once per user before measuring. Measurements exclude setup requests. |
| repetitions | 2–3 runs per profile (main series 22:24–22:40 UTC, 26 Oct 2026); 10 s pause between runs |

Each profile registers fresh users (random credentials, real login tokens),
uploads the corpus through the authenticated ingest endpoint, and queries
with that document only (no profile triggers live web search). HTTP 429 is
counted separately and never as an application failure.

## HTTP results (tracing off)

`p50/p95/p99` are over all measured queries; "cold" = no cache hit (retrieval
and mock models ran); "L1 hit" = served from the exact cache.

| profile | VUs | run | queries | req/s | p50 ms | p95 ms | p99 ms | cold p95 ms | L1-hit p95 ms | hit rate | app failures | thresholds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (0→5→20→0) | ≤20 | 1 | 952 | 16.4 | 2.9 | 180.9 | 444.5 | 446.2 | 5.4 | 94.7% | 0% | all pass |
| reference | ≤20 | 2 | 957 | 17.0 | 2.8 | 177.7 | 432.8 | 249.0 | 4.9 | 94.8% | 0% | all pass |
| reference | ≤20 | 3 | 948 | 16.7 | 3.1 | 200.9 | 449.8 | 472.3 | 7.5 | 94.7% | 0% | all pass |
| cold (auto strategy) | 5 | 1 | 178 | 4.1 | 595.5 | 952.4 | 1304.4 | 960.2 | – | 0% | 0% | all pass |
| cold | 5 | 2 | 173 | 4.1 | 650.0 | 935.8 | 1052.8 | 940.2 | – | 0% | 0% | all pass |
| l1 (warm) | 20 | 1 | 1580 | 24.3 | 6.0 | 18.4 | 455.8 | – | 11.7 | 100% | 0% | all pass |
| l1 | 20 | 2 | 1580 | 28.1 | 6.6 | 17.5 | 317.5 | – | 11.9 | 100% | 0% | all pass |
| direct | 5 | 1 | 175 | 4.1 | 566.4 | 1259.1 | 1472.5 | 1285.0 | – | 0% | 0% | all pass |
| direct | 5 | 2 | 190 | 4.5 | 511.1 | 959.2 | 1089.6 | 971.6 | – | 0% | 0% | all pass |
| council | 5 | 1 | 201 | 4.7 | 469.0 | 796.1 | 932.3 | 809.3 | – | 0% | 0% | all pass |
| council | 5 | 2 | 194 | 4.6 | 498.7 | 815.4 | 982.0 | 817.3 | – | 0% | 0% | all pass |
| council_deep | 5 | 1 | 193 | 4.6 | 523.0 | 813.4 | 969.6 | 834.6 | – | 0% | 0% | all pass |
| council_deep | 5 | 2 | 186 | 4.4 | 566.5 | 849.8 | 1123.6 | 853.1 | – | 0% | 0% | all pass |
| multiuser (10 users) | 10 | 1 | 714 | 15.5 | 2.7 | 467.6 | 1296.7 | 1393.9 | 4.7 | 93.0% | 0% | all pass |
| multiuser | 10 | 2 | 700 | 15.1 | 3.5 | 618.7 | 1899.7 | 2022.1 | 6.4 | 92.9% | 0% | all pass |

Status codes: every measured query above returned 200 (no 401, 403, 404,
429 or 5xx). `multiuser` recorded **0 cross-user cache hits** in both runs
(a hit on a user's first ask of a question would mean another user's answer
was reused).

**Rate limiter** (`ratelimit` profile, one user, 20 VUs without pauses,
20 s): 143,270 requests at 6,648 req/s, of which **190 were answered (200)
and 143,080 rejected with 429**. 190 = burst 20 + 10/s × ~17 s, i.e. the GCRA
limit held exactly under overload; no 5xx. Go CPU peaked at 225%.

**Semantic cache** (`l2`, benchmark-only `SEMANTIC_CACHE_ENABLED=true`, 5
VUs, 25 paraphrase queries per run, 2 runs): 25/25 L2 hits in both runs,
L2-hit p50 106.6 / 97.6 ms, p95 190.4 / 148.5 ms. The cost is mostly the
query embedding in the Python service. These five paraphrases are close
rewordings, so this shows the mechanism works; it says nothing about safety:
the checkpoint 7 benchmark measured 28% false reuse of near-miss questions
at the same threshold, which is why L2 reuse stays off by default.

### Repeatability

p95 spread across repeats: reference 177.7–200.9 ms (sd 12.6), cold
935.8–952.4, council 796.1–815.4, deep 813.4–849.8, l1 17.5–18.4, direct
959.2–1259.1 (sd 212), multiuser 467.6–618.7. Two or three runs are enough to
see the order of magnitude, not to claim small differences. The `direct`
and `multiuser` spreads are large because their latency is dominated by a
CPU-saturated Python service.

### Reference thresholds

The reference script's targets were error rate < 1%, p95 < 2 s, cache hit
rate > 70%, cache-hit p95 < 50 ms. They are applied only where they make
sense: the `reference` profile (repeated questions) meets all four; `l1`
meets error rate and L1-hit p95 < 50 ms (11.7–11.9 ms). Cold, strategy and
multi-user profiles are judged only on application error rate: a cold
pipeline is not expected to meet a warm-cache latency target.

## Where the time goes (Prometheus, main series window)

| measurement | p50 | p95 | source |
|---|---|---|---|
| RAG `/retrieve` (embed + vector search + rerank) | 580 ms | 1,398 ms | `councilai_rag_request_seconds` |
| RAG `/ingest` | 321 ms | 770 ms | `councilai_rag_request_seconds` |
| L1 lookup (Redis GET) | 0.56 ms | 2.5 ms | `councilai_cache_lookup_seconds` |
| each council stage and model call (mock) | < 1 ms | < 1 ms | `councilai_council_stage_seconds`, `councilai_llm_request_seconds` |
| council run total (mock), all strategies | < 2 ms | < 2 ms | `councilai_council_response_seconds` |

Every mock stage observation falls into the lowest histogram bucket
(0–1 ms); the quantile that Prometheus interpolates inside that bucket is not
meaningful, so it is reported as "< 1 ms". With mock models, **retrieval in
the Python service is effectively the whole cold-path latency**. A sample
trace (below) shows the split: `/retrieve` 194 ms = embed 25 ms + vector
search 4 ms + rerank 150 ms; the council took 1.5 ms.

Lookup outcomes over the main window (Prometheus `increase`, which
extrapolates at window edges, so counts are approximate): L1 hits ≈ 7,090,
L1 misses ≈ 1,847, L2 `disabled_by_policy` on every query.

## Resources (docker stats, 126 samples at 5 s during the main series)

| service | CPU avg % | CPU max % | memory avg MiB | memory max MiB |
|---|---|---|---|---|
| python-rag | 211.8 | 1024.6 | 679 | 1,147 |
| go-backend | 7.7 | 225.0 | 20 | 25 |
| redis | 1.5 | 43.8 | 16 | 22 |
| postgres | 1.0 | 14.2 | 27 | 31 |
| prometheus | 0.5 | 7.2 | 32 | 35 |
| grafana | 0.5 | 10.0 | 91 | 93 |
| jaeger (idle) | 0.2 | 8.0 | 23 | 25 |
| ui | 0.6 | 3.2 | 47 | 47 |

No container restarted or was OOM-killed during any run (`RestartCount=0`,
`OOMKilled=false` before and after). The Python service is the bottleneck:
it uses all 8 vCPUs under cold load, and its memory rose to 1.15 GiB.
Peak total memory stayed well below the 3.83 GiB VM.

## Tracing overhead (OTLP to Jaeger, 100% sampling)

| profile | p95 tracing off | p95 tracing on |
|---|---|---|
| reference | 177.7 – 200.9 ms (3 runs) | 187.5 – 228.9 ms (2 runs) |
| council (cold) | 796.1 – 815.4 ms (2 runs) | 899.5 – 1,003.3 ms (2 runs) |

Full-rate tracing added roughly 10–20% to the CPU-bound cold path (the
Python service records and exports spans on the same saturated CPUs) and
little to cached traffic. With only two runs per side, treat this as an
estimate. `councilai_telemetry_errors_total` stayed 0 and Jaeger used at
most 72 MiB.

## Sample trace

One auto-routed, document-grounded council request as stored in Jaeger
(trace `540aee8994393faf8983ca14c4c77ffc`; 23 spans, two services):

```
POST /api/v1/query  [go] 197.9 ms  route=/api/v1/query status=200 strategy=council tier=none
  ratelimit.check  [go] 0.2 ms
  auth.document_access  [go] 0.3 ms
  cache.l1_lookup  [go] 0.2 ms  result=miss
  router.plan  [go] 0.2 ms
    llm.router  [go] 0.1 ms
  rag POST /retrieve  [go] 195.0 ms  status=200
    POST /retrieve  [python] 194.2 ms
      rag.embed_query  [python] 24.9 ms
      rag.vector_search  [python] 4.4 ms
      rag.rerank  [python] 150.3 ms
  council.run  [go] 1.5 ms
    council.candidates  [go] 0.1 ms   (3 × llm.candidate, concurrent siblings)
    council.peer_review  [go] 0.3 ms  (3 × llm.review)
    council.chairman  [go] 0.1 ms     (llm.chairman)
  cache.store  [go] 0.7 ms
```

To inspect locally: start the stack with
`OTEL_TRACES_EXPORTER=otlp docker compose --profile observability --profile tracing up -d`,
send a request, take the `X-Trace-ID` response header, and open
`http://127.0.0.1:16686/trace/<id>`. Jaeger keeps traces in memory only.

## Limitations

- Mock models only: no statement here applies to real LLM latency, cost or
  quality. Token panels are empty because mocks report no usage.
- Single machine; the load generator shares the CPUs with the service it
  measures.
- One small document (two chunks), so reranking scores two candidates;
  bigger documents make retrieval slower.
- Two or three repetitions per profile; differences under ~10–20% between
  profiles are within the observed spread.
- Prometheus samples gauges every 15 s: short events (a 13 s breaker
  opening) can be missed by the state gauge; the transition counter
  records them.
