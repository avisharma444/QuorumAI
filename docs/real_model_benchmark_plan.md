# Real-model benchmark: plan and dry run (not yet executed)

Status: **prepared, awaiting the owner's approval.** No external inference has
been run. The results report (`docs/real_model_performance.md`) will be
written only from an approved live run.

## Procedure (`evaluation/councileval/cache_bench.py`)

12 seeded held-out questions from councileval-v1 (seed 11): 4 extractive
(SQuAD 2.0), 4 numeric (DROP), 2 multi-hop bridge and 2 comparison
(HotpotQA), each with its own real document (BGE + Chroma retrieval, top_k 4,
reranking on). One benchmark user; explicit strategies (no router):

| phase | per question | condition |
|---|---|---|
| A cold council | 1 | must be uncached, run 3 candidates + 3 reviews + chairman, not degraded, stored in L1 |
| B warm | 3 | only after an eligible cold answer; each must be `l1_exact`, `inference_executed=false`, and the Prometheus model-call counter must not move; one failure stops that question's repeats |
| C direct | 1 | different cache key, so uncached; same document and retrieval |
| D council_deep | 4 questions (one per category) | reflection and at most one revision |

L2 semantic reuse is disabled and checked on every response. The cache
workload is 12 cold + 36 exact repeats (75% repeats by construction): it
demonstrates the L1 mechanism, not organic traffic.

## Models (pinned, all via OpenRouter, non-reasoning)

| role | model | primary price (in/out per M) | highest provider price |
|---|---|---|---|
| member 1 | openai/gpt-4.1-nano | $0.10 / $0.40 | $0.11 / $0.44 |
| member 2 | meta-llama/llama-3.3-70b-instruct | $0.10 / $0.32 | $0.72 / $2.253 |
| member 3 | mistralai/mistral-small-3.2-24b-instruct | $0.075 / $0.20 | $0.10 / $0.30 |
| chairman, ingestion summary | openai/gpt-4o-mini | $0.15 / $0.60 | $0.165 / $0.66 |
| router (configured, not called) | openai/gpt-4.1-nano | — | — |

Prices: OpenRouter model and endpoint catalogue, 2026-10-27. All four
support `response_format` (JSON mode). OpenRouter reports token usage and the
charged cost (`usage.cost`) in every response; the adapter now records that
cost, so observed charges are reported separately from estimates.

## Dry run

| item | value |
|---|---|
| HTTP requests | 76 (12 ingest, 12 cold, 36 warm, 12 direct, 4 deep) |
| model calls (max) | 144: 12 summaries, 48 candidates, 48 reviews, 16 chairman, 12 direct, 4 reflection, ≤ 4 revision; warm 0 |
| worst-case cost | **$1.25** (every call at $0.72 / $2.253 per M, 2,048 output tokens per call, inputs at 3 chars/token) |
| likely cost | **≈ $0.04** (primary prices; candidate 350, review 250, chairman 450, direct 350 output tokens; inputs sized from the real retrieved chunks) |
| suggested caps | `--max-requests 76 --max-spend-usd 2.00` |

The budget guard refuses any request whose worst case would not fit under
the remaining cap, charges each finished request its OpenRouter-reported
cost (or its worst case when any call lacked usage), charges every ingestion
its worst case (the summarizer's usage is not returned by `/ingest`), and
charges an unexpected warm cache miss as a full council. The run aborts if
`/health` stops reporting exactly the pinned models.

## Validation without provider calls

The same procedure ran against the stack with mock models (`mode: offline`):
12/12 cold answers cacheable, 36/36 warm requests were L1 hits with zero
model calls according to Prometheus, L2 `disabled_by_policy` on every
response. Mock latencies validate the procedure only and are not reported as
results.
