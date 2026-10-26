# CouncilAI roadmap and evaluation contract

## Design decisions

- Build a clean, explainable subset before recreating the full source repository.
- Go control plane for explicit fan-out/fan-in concurrency; Python microservice for PDF and embeddings.
- Separate mock LLM tests from real model-quality evaluations.
- Start with in-memory exact cache; migrate to Redis only when the core system is demonstrably correct.
- Use hosted inference on Mac M2 instead of trying to run an NVIDIA-vLLM container.

## Baselines to preserve

1. **Direct baseline:** query uses chair model only, no council and no peer review.
2. **Council baseline:** three models answer concurrently, then anonymous peer voting and chairman synthesis.
3. **RAG retrieval baseline:** hashed lexical vectors (no semantic guarantees), then MiniLM, then MiniLM + reranking.
4. **Cache baseline:** cold model execution vs exact query cache, then Redis exact and semantic tiers.

## Metric definitions (report both median and p95)

| Metric | Definition | Guardrail |
| --- | --- | --- |
| End-to-end request latency | client send → full final answer | separate cold and warm; include failed requests |
| First-stage event latency | request start → first SSE candidate | not "first LLM token"; candidates are emitted on completion |
| Cache hit rate | requests served without a new LLM call / eligible requests | report query distribution and hit tier |
| Retrieval recall@k | questions with at least one labeled relevant chunk in top k / questions | hold query set fixed across implementations |
| Grounded QA accuracy | correct, source-supported answers / labeled questions | evaluate direct and council on same evidence |
| Cost / successful answer | sum provider billed cost / accepted correct answers | include all candidate/reviewer/chair calls |
| Partial outage success | completed valid responses / injection fault runs | inject one, two, and all failed providers |

**Benchmark protocol:** Include hardware, model IDs, pricing date, corpus description, N, whether the cache was cold, and raw scripts/results. Re-run a baseline before quoting speedup values. A model-provided numeric confidence is not statistical calibration.

## Completion gates

- [x] Go backend compiles without third-party Go modules
- [x] Parallel fanout + peer voting + chairman fallback + deep reflection
- [x] Python document ingestion, retrieval, persisted metadata and vectors
- [x] In-process cache, SSE, basic metrics, Streamlit UI
- [x] Go/Python unit tests and a two-service end-to-end smoke run
- [ ] Neural embedding + Chroma persistence and reranker with retrieval evaluation
- [ ] Redis L1/L2, TTLs, invalidation and circuit breaker
- [ ] JWT auth + Postgres + per-document permissions
- [ ] Better cancellation/deadlines, prompt-injection defenses and secure public deployment
- [ ] Real-model benchmark + error analysis + reproducible cost and latency reports
