# Real-model performance: L1 exact cache, council vs direct

Run `real-cache-bench-20261027T062112Z` was executed on 2026-10-27 between
06:21:12 and 06:26:56 UTC, after the owner approved the models and budget.
It completed with every planned request, no failures, no degraded answers
and no budget stop. Raw records, the run metadata and the computed summary
are in `evaluation/results/cache_bench/real-cache-bench-20261027T062112Z/`.
That directory is git-ignored because it contains full model outputs.
The plan and dry run are in [real_model_benchmark_plan.md](real_model_benchmark_plan.md).

## Configuration

| item | value |
|---|---|
| models (all via OpenRouter) | members `openai/gpt-4.1-nano`, `meta-llama/llama-3.3-70b-instruct`, `mistralai/mistral-small-3.2-24b-instruct`; chairman and ingestion summary `openai/gpt-4o-mini`; router configured as `openai/gpt-4.1-nano` but never called (explicit strategies) |
| identity checks | `/health` reported `mode: live` and exactly the pinned models before every question; every candidate's `served_model` equalled the requested id |
| questions | 12 held-out councileval-v1 items, seed 11: 4 extractive (SQuAD 2.0), 4 numeric (DROP), 2 bridge and 2 comparison (HotpotQA), each grounded in its own uploaded document |
| retrieval | BGE embeddings, Chroma, reranking on, `top_k` 4 |
| per question | cold `council` → 3 identical warm repeats → `direct`; `council_deep` on 4 predetermined questions |
| caches | L1 exact enabled. **L2 semantic reuse disabled** (`SEMANTIC_CACHE_ENABLED=false`); every one of the 64 query responses reported `l2: disabled_by_policy` |
| output caps | `max_tokens` on every call: 1024 (`LLM_MAX_OUTPUT_TOKENS`), reflection 500, ingestion 200 |
| limits | harness caps 76 requests and $2.00; the OpenRouter key had a provider-side limit of $2.00 (checked before the run) |
| host | Mac M2 Air 8 GB, Docker VM 8 vCPU / 3.8 GiB; requests sent one at a time from a container on the compose network |
| tracing | OTLP to Jaeger, sampling 1.0 |
| code | commit `2dc661a` plus the safety gates committed with this report (output caps, key-limit check, live preflight) |

The live preflight (first cold council plus its first warm repeat) passed
every check before the other 74 requests were sent:

- 3 candidates, 3 reviews and a parsed chairman JSON;
- usage and a reported cost on every call;
- no call over the output cap;
- Prometheus counted 7 model calls for the cold request and 0 for the warm repeat;
- the warm repeat was an identical L1 answer;
- L2 was disabled.

## Cold council vs warm L1 hit

| | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| cold council (uncached, 7 model calls) | 12 | 14,292 ms | 20,443 ms | 9,317 ms | 21,595 ms |
| warm, L1 exact hit (0 model calls) | 36 | 4.06 ms | 12.35 ms | 1.24 ms | 18.33 ms |
| direct (1 model call, uncached) | 12 | 2,861 ms | 4,851 ms | 1,506 ms | 5,021 ms |

Latency is client-side, end to end through the API (auth, rate limit,
retrieval, council, cache).

**Speedup (cold ÷ warm, same question, same user, same document):**
- median of the per-question speedups (cold ÷ median of its 3 warm requests): **3,811×**, with a range of 1,848× to 7,409× across the 12 questions;
- ratio of the medians: **3,518×**.

The warm-side numbers are a few milliseconds and sensitive to scheduling
noise. The speedup is better read as "a 9–22 s council becomes a 1–18 ms
cache read" than as a precise factor.

**Cache verification:** all 36 warm requests were `tier: l1_exact` with
`inference_executed: false`, and the Prometheus model-call counter moved by
exactly 0 on each. Every warm answer was byte-identical to its cold answer.
Every cold answer was cacheable (12/12). There were no warm failures.

**L1 hit rate:** 36 of 48 council requests = **75%**. This equals the repeat
share designed into the workload (3 repeats per cold question), so it shows
the mechanism works. It says nothing about organic hit rates, which depend on
how often real users repeat identical requests.

### Where the time goes (Jaeger)

Median stage durations over the 12 cold council requests:

| stage | median | range |
|---|---|---|
| rate limit + auth + L1 lookup | 1.3 + 3.0 + 0.2 ms | |
| retrieval (`rag POST /retrieve`, incl. embed 132 ms, vector search 7.6 ms) | 844 ms | 173–1,247 ms |
| candidates (3 in parallel) | 4,203 ms | 2,039–15,839 ms |
| peer review (3 in parallel) | 4,504 ms | 2,412–13,616 ms |
| chairman | 3,125 ms | 1,850–4,944 ms |
| L1 store | 5.6 ms | 2.6–7.5 ms |

The slowest parallel member sets each stage's time, so the per-member tail
dominates.

One question (`drop:48630807…`) traced end to end:

- **Cold, 14,069 ms server-side:**
  - candidates 6,345 ms; Mistral took 6,343 ms while the other two took about 2 s;
  - peer review 4,172 ms;
  - chairman 3,171 ms (882 tokens in, 115 out);
  - retrieval 304 ms.
- **Warm, 6.1 ms server-side:**
  - document access check 3.5 ms;
  - rate limit 1.6 ms;
  - L1 lookup 0.43 ms;
  - no retrieval and no model spans.

## Model usage and charges

All token counts and costs below come from the provider (OpenRouter
`usage` and `usage.cost`), not from estimates.

| | value |
|---|---|
| HTTP requests | 76 (12 ingest, 12 cold, 36 warm, 12 direct, 4 deep) |
| model calls on the query path | 129 = 84 cold (12 × 7) + 12 direct + 33 deep (3 × 8 + 1 × 9) |
| model calls on ingestion | 12 summaries |
| total model calls | **141** (approved maximum 144; only 1 of the 4 possible revisions ran) |
| query-path tokens | 86,372 prompt + 11,035 completion = 97,407 |
| query-path reported cost | **$0.015774** (sum of `usage.cost` in the responses) |
| key usage after the run (OpenRouter `/api/v1/key`) | **$0.016979** of the $2.00 limit |
| ingestion cost (difference) | $0.001205 for 12 summaries, derived because `/ingest` does not return usage |
| harness budget charge | $0.044509: reported cost for queries, worst case for each ingestion (fail closed) |
| dry-run estimates | likely ≈ $0.044, worst case $1.25 |

Per model, query path only:

| model | calls | prompt tokens | completion tokens | reported cost |
|---|---|---|---|---|
| openai/gpt-4.1-nano | 32 | 17,221 | 2,414 | $0.002616 |
| meta-llama/llama-3.3-70b-instruct | 32 | 17,515 | 2,330 | $0.004019 |
| mistralai/mistral-small-3.2-24b-instruct | 32 | 18,225 | 2,778 | $0.002106 |
| openai/gpt-4o-mini (chairman, direct, reflection, revision) | 33 | 33,411 | 3,513 | $0.007033 |

The chairman row is the request total minus the members' per-call usage. The
largest single-call outputs were:

| stage | max output tokens |
|---|---|
| candidate | 212 |
| review | 119 |
| chairman | 253 |
| revision | 127 |
| reflection | 100 |
| direct | 100 |

No call came near the 1,024-token cap.

### Avoided by the 36 warm hits

| | avoided |
|---|---|
| model calls | **252** (36 × 7) |
| prompt tokens | 167,589 |
| completion tokens | 22,824 |
| provider cost | **$0.030573**: each warm hit counted at the reported cost of its own cold council |

Without the cache, these 48 council requests would have cost about $0.0408
instead of the $0.0102 actually spent on the 12 cold councils.

## Exploratory quality: council vs direct

Each question was answered on the same document, with the same retrieved
evidence, by the cold council and by a single direct call to the chairman
model. The sample is small (n = 12), so this is exploratory, not a quality
claim.

| | automatic scorer | manual check | p50 latency | model calls | reported cost | cost per correct (automatic) |
|---|---|---|---|---|---|---|
| council | 11/12 | 12/12 | 14.3 s | 84 | $0.010191 | $0.00093 |
| direct | 11/12 | 12/12 | 2.9 s | 12 | $0.001620 | $0.00015 |
| council_deep | 3/4 | 4/4 | 13.3 s | 33 | $0.003963 | $0.00132 |

**Paired outcome (automatic scorer):** 0 council wins, 0 direct wins, 12 ties.

- In 11 questions both strategies were scored correct.
- In 1 question both were scored wrong, and that is a scorer false negative. The question was `hotpot:5a84de25…` ("Who had more pen names, George Orwell or Nelson DeMille?").
  - The gold answer is "Nelson Richard DeMille".
  - All three strategies answered "Nelson DeMille had more pen names…", which is correct.
  - The containment rule fails on the missing middle name.
  - The scorer also labels the direct answer a "false abstention" because it contains the phrase "no evidence". The answer did not abstain.
- I kept the automatic numbers as computed and did not change the scorer after seeing the results. The manual column is my reading of all 28 answers against the gold labels.
- EM and F1 are not reported. They are defined only for answers of ≤ 12 tokens, and almost every answer was a full sentence.

**Deep runs:**
- The reflection judged 3 of 4 answers "good".
- It sent 1 for revision (`drop:48630807…`), and the revision kept the correct answer, 365.
- All 4 were marked faithful.
- Reflection confidence is the model's self-assessment, not an independent check.

**Takeaway:** on these 12 document-grounded questions, the direct answer was
as accurate as the council. The council was about 5× slower and about 6.3×
more expensive. The council's extra cost buys nothing measurable at this
sample size and difficulty. The L1 cache removes that cost entirely for
exact repeats.

## Measured claims

These figures come from this run and can be quoted with the conditions
attached:

- With real models through OpenRouter, the Redis L1 exact cache served repeated council queries in 4.1 ms (p50) instead of 14.3 s.
  - That is a per-question median speedup of 3,811× (range 1,848–7,409×), over 12 questions × 3 repeats.
  - Prometheus confirmed zero model calls on all 36 hits.
- The hits avoided 252 model calls, 190k tokens and $0.031 of provider charges, from a workload that was 75% exact repeats by design.
- The experiment was budget-guarded: it was capped at $2.00 and actually spent $0.017 in total, as reported by the provider.
- The full four-model run took under 6 minutes: 76 requests, 141 model calls, a live preflight gate, and no failed or degraded requests.

## Limitations

- **Sample size.** 12 questions, one run, one machine. Latency percentiles over 12 cold requests are rough, and p95 is close to the maximum.
- **Synthetic hit rate.** The 75% hit rate is set by the design of 3 exact repeats per question. Organic hit rates are not measured.
- **Exact repeats only.** Same user, document, strategy and top_k. L2 semantic reuse stayed disabled, so the fuzzy-reuse behaviour (held-out false-positive rate 0.654 at threshold 0.85, see [evaluation.md](evaluation.md)) is not part of these numbers.
- **Provider variance.** Latency includes OpenRouter routing and upstream provider queueing, which change over time. Costs also depend on which upstream provider OpenRouter picked.
- **Derived ingestion cost.** Ingestion cost is the key's total usage minus the per-response costs. `/ingest` does not return the summarizer's usage.
- **Easy questions.** Both strategies answered every question correctly, so this sample cannot separate council from direct quality. A harder or larger set would be needed.
- **Scorer limits.** The automatic scorer under-counted one correct answer (aliases and full names). The manual check was done by the author without blinding.
