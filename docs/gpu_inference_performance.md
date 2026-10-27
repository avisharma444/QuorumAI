# GPU inference: Phi-4-mini on RunPod Serverless (vLLM)

Measured on 2026-10-27. **Model:** `microsoft/Phi-4-mini-instruct` (revision `cfbefac…`, bf16).

**Serving stack:**
- official worker `runpod/worker-v1-vllm:v2.28.0`, which builds vLLM 0.30.0 according to its Dockerfile at that tag (the running server does not report its version);
- **one NVIDIA RTX 4090**: Serverless endpoint `vb4t31nr8i9qtx`, max 1 worker;
- the client was a laptop in India (UTC+05:30), so every gateway number includes that network path.

Deployment, settings and run order: [runpod_vllm_deployment.md](runpod_vllm_deployment.md).
Raw records (git-ignored, kept locally):
- `evaluation/results/gpu_bench/runpod-phi4-bench-20261027T125837Z/`
- `evaluation/results/e2e_vllm/runpod-e2e-council-20261027T130856Z/`
- `evaluation/results/e2e_vllm/runpod-e2e-fallback-20261027T131548Z/`
- the GPU ledger `evaluation/results/gpu_budget/runpod-phi4-campaign.json`
- the provider re-check at 15:05 UTC (balance, billing, workers): `provider_recheck_20261027T150534Z.json` in the benchmark run directory

## Headline numbers

| measure | value |
|---|---|
| cold start: submit → first answer (async job, client) | **231.5 s**; RunPod `delayTime` 229.1 s, `executionTime` 0.444 s |
| second cold start (e2e warm-up, async job) | 177.2 s (`delayTime` 174.4 s) |
| warm request, 256 output tokens: latency p50 / p95 | **3.02 s / 3.14 s** (n = 20) |
| gateway TTFT (first streamed token) p50 / p95 | **0.661 s / 0.801 s** |
| worker-side 1-token execution p50 / p95 | 0.393 s / 0.430 s (n = 8), an upper bound on in-engine TTFT |
| gateway + network + queue overhead (p50 difference) | 0.269 s |
| decode speed per request, warm (p50) | **110 tokens/s** (105–114) |
| aggregate output throughput at concurrency 1 / 2 / 4 / 8 | **81 / 133 / 203 / 532 tokens/s** |
| QuorumAI council, end to end, Phi as member 3 (p50 / p95, n = 6) | **12.7 s / 17.9 s**, 6/6 correct, no degraded stage |
| RunPod spend, all GPU work (billing API and balance agree) | **$0.2013** |
| OpenRouter spend, all hosted calls (provider key usage) | **$0.007443** (see Cost) |

## Cold start

The first request is an **async job** (`/run`, then `/status` every 2 s). RunPod's synchronous routes stop waiting after about 90 s, so a first start can't be measured through them.

- **What was measured:** RunPod kept a *standby* worker record for the endpoint (`workersStandby: 1`, container stopped, $0/h), so the phase is labelled `startup_standby`. The job ran on a different RTX 4090 machine (`0jyg8ayb8i7t`) that RunPod reported as "Resumed" at 13:00:45.
- **Breakdown:** about 2 minutes passed from submission to a worker resuming, and about 1.8 minutes went to model download and load plus the first answer. Image layers may already have been on that host.
- **Sample size:** two measurements (231 s and 177 s) are two samples, not a distribution.
- **Validation:** before anything else ran, the first answer was checked: served model `microsoft/Phi-4-mini-instruct`, 8/8 tokens with usage reported, non-empty text.

## Warm streaming and TTFT

All requests used the same settings: 8 fixed prompts, `max_tokens` 256, temperature 0, seed 1234, `ignore_eos` (so every answer is exactly 256 tokens) and prefix caching off.

- **Token-level streaming confirmed:** the median request delivered 256 content chunks for 256 tokens, and the first-to-last chunk span covers 77% of total latency. RunPod's gateway forwards tokens as they are generated; it does not buffer them.
- **Two TTFTs, kept separate:**
  - *Gateway TTFT* (0.661 s p50) runs from the client sending the request to the first content chunk.
  - *Worker 1-token execution* (0.393 s p50) is RunPod's `executionTime` for a 1-token async job: time inside the worker, including its local proxy, vLLM prefill and one token.
  - The 0.27 s difference is gateway, network and queue overhead.
- **Not measured:** vLLM's own in-engine TTFT histogram is on its `/metrics` route, which the worker does not expose.

## Concurrency: does batching help?

16 requests per level, each 256 tokens, on one RTX 4090.

| concurrency | wall (s) | output tokens/s | vs c1 | latency p50 / p95 (s) | TTFT p50 / p95 (s) | decode/request p50 (tok/s) | GPU $ per 1M output tokens* |
|---|---|---|---|---|---|---|---|
| 1 | 50.5 | 81.2 | 1.00× | 3.06 / 3.15 | 0.70 / 0.78 | 108.7 | 3.76 |
| 2 | 30.7 | 133.3 | 1.64× | 3.20 / 5.46 | 0.74 / 3.07 | 104.3 | 2.29 |
| 4 | 20.2 | 202.5 | 2.49× | 3.85 / 8.34 | 1.37 / 5.91 | 104.2 | 1.51 |
| 8 | 7.7 | 532.4 | 6.56× | 3.35 / 3.41 | 0.84 / 0.88 | 102.2 | 0.57 |

\*Busy basis: $1.10/h × the level's wall time ÷ output tokens. Idle, cold start and queueing aren't included; the all-in cost is below.

What the measurements support:

- **Batching helps.** Per-request decode speed falls only from 109 to 102 tokens/s while 8 sequences run at once. At concurrency 8, aggregate throughput is 6.6× concurrency 1, which is 82% of linear.
- **The p95 tails at concurrency 2 and 4 come from RunPod's queue, not vLLM.** While those levels ran, some endpoint health samples showed every request queued and none in progress: 2 of 13 samples at concurrency 2 (`inQueue: 2, inProgress: 0`) and 2 of 8 at concurrency 4 (`inQueue: 4, inProgress: 0`). One worker was running throughout. At concurrency 1 and 8, no sample showed a queued request. Some requests waited 2–5 s at the gateway before dispatch, so their TTFT includes queueing. Those scaling figures (1.64×, 2.49×) are therefore lower bounds on what vLLM itself achieves.
- One run per level. No repetition, so no confidence intervals.

## QuorumAI end to end (Phi-4-mini on the GPU as council member 3)

**Setup:**
- Members `openai/gpt-4.1-nano` and `meta-llama/llama-3.3-70b-instruct` (OpenRouter), plus Phi-4-mini on RunPod; chairman `openai/gpt-4o-mini`.
- Strategy `council`, 6 unique document-grounded questions (seed 23), L2 semantic cache off.
- The worker was warmed first by the async job above.

| member | candidate latency p50 / p95 (ms) | review latency p50 / p95 (ms) | candidate output tokens p50 (max) |
|---|---|---|---|
| Phi-4-mini (RTX 4090, RunPod) | 2,101 / 8,456 | 1,679 / 1,736 | 126.5 (1,024) |
| gpt-4.1-nano (OpenRouter) | 2,399 / 2,632.5 | 2,798.5 / 8,358 | 84 (183) |
| llama-3.3-70b (OpenRouter) | 3,492.5 / 5,518.5 | 3,341 / 3,631 | 76.5 (105) |

- **Councils:** 6/6 answered correctly, every member answered, and no stage was degraded. End to end p50 12.7 s, p95 17.9 s; server-side p50 11.5 s.
- **Slowest candidate:** Llama-70B in 5 of 6 councils. Phi in 1: a 1,024-token answer that hit `LLM_MAX_OUTPUT_TOKENS`, about 98 tokens/s.
- **Phi reviews** were the fastest of the three in every council: 1.4–1.7 s.

## Failure behaviour (observed)

| case | what happened |
|---|---|
| invalid key | HTTP 401 at RunPod's gateway in 0.66 s; no worker involved |
| wrong model name | **HTTP 200 with no tokens and no usage** in 0.71 s. The worker marked the job failed (the endpoint's `failed` counter). The harness first counted this as a success; it is now classified `empty_stream`, and future runs keep the error body |
| prompt over `MAX_MODEL_LEN` | same as the wrong model name: HTTP 200, no tokens, 0.76 s |
| client timeout (2 s) | 152 tokens streamed, then the client stopped as configured |
| client cancel after 5 chunks | closed at 0.83 s; endpoint showed 1 job in progress at +1.8 s and 0 at +4.3 s. **Inconclusive:** the 256-token answer would have finished by about +2.2 s anyway, so this doesn't show whether the gateway stops generation on disconnect |
| QuorumAI with a wrong RunPod key | the council answered correctly in 12.2 s with the two hosted members; Phi's 401 is listed in `failed_candidates`, and Phi was **not** asked to review |
| QuorumAI client gives up after 3 s | backend request cancelled; over the next 20 s RunPod showed no Phi job queued or running, so nothing was left to purge |

## GPU telemetry

- **GPU type:** every worker that ran was an **RTX 4090**, the only type the endpoint allows. RunPod's worker listing confirms it, and at most one worker was `RUNNING` at any time. (The run summary's `max_workers_seen: 2` counts the stopped standby worker twice, because health reported it as both `idle` and `ready`; health's `running` count never exceeded 1.)
- **Utilisation and memory: not usable.** RunPod's GraphQL `runtime.gpus` for the running worker reported utilisation and memory as the *same* number in every sample: 0% in 22 samples, 100% in 16, 3% in 4. vLLM holds about 90% of VRAM for the worker's whole life, so a memory reading of 0% is not credible. The raw samples are kept in `records.jsonl` and no GPU-utilisation figure is reported.

## Cost

| item | amount | source |
|---|---|---|
| RunPod, everything (both cold starts, the benchmark, councils, idle time) | **$0.2013** | account balance $10.0000 → $9.7987, read with the Read Only key |
| RunPod billing API (re-checked 15:05 UTC) | **$0.2013** for 655.2 s billed | one record for the endpoint, hour bucket 13:00 UTC: $0.201252, `timeBilledMs` 655,235. It had no records at 13:16 UTC. It matches the balance drop to the cent |
| OpenRouter, all hosted calls (6 councils, the cancelled council, fallback, and any document-ingestion calls) | **$0.007443** | key usage $0.018964 (before) → $0.025824 (13:16:37 UTC, 28 s after the last run) → $0.026407 (15:05 UTC) |
| OpenRouter, cost reported in the responses | $0.005294 | councils $0.004824 + fallback $0.000470; the cancelled council and ingestion calls left no response cost |
| two earlier attempts that never got a worker | $0.00 | balance unchanged, no billing records, no workers in any state |

- **Per token, all in:** $0.2013 ÷ 21,520 generated benchmark tokens is about $9.35 per 1M tokens. That's an over-statement, because the same spend also covers the end-to-end councils and two cold starts.
- **Busy basis:** at concurrency 8 the busy-basis cost was $0.57 per 1M output tokens.
- **OpenRouter late amount:** $0.000583 of the $0.007443 appeared after 13:16:37. The stack ran in mock mode from then on, and nothing in mock mode calls OpenRouter, so this is most likely late-posted accounting for the campaign's own calls. That could not be confirmed from the key-usage figure alone.
- **Budget use:** 117 of the 120 approved GPU requests (including both failed attempts), $0.2013 of the $2.00 GPU cap, $0.0074 of the $0.25 OpenRouter cap.

## What went wrong before this run (kept for the record)

1. **Image pin.** `runpod/worker-v1-vllm:v2.29.0` was never published: its release build failed. The image was re-pinned to v2.28.0 by tag and digest.
2. **First attempt** (L4 / A5000 / RTX 3090): the job waited 1,200 s in the queue and no worker was ever created.
3. **Second attempt** (RTX 4090): no worker within the 8-minute allocation limit.
4. **Root cause of both:**
   - `allowedCudaVersions: ["13.0"]` is an exact set, and Serverless hosts advertised CUDA 13.2. No machine could match.
   - Fixed to `["13.0" … "13.4"]` through GraphQL `saveEndpoint`; REST rejects values above 13.0.
   - A fail-closed placement check (`product=SERVERLESS` catalog: GPU type and an available CUDA version must both match) now runs before any job is submitted.
5. **Standby workers.** RunPod keeps stopped standby workers on the endpoint that health reports as idle/ready. Scale-down checks accept them only when their container is `EXITED` and the account spends $0/h.

## Limits

- One GPU type, one region mix chosen by RunPod, one run per concurrency level, and two cold-start samples.
- Gateway numbers include network latency from India to RunPod.
- The cancel-propagation test was inconclusive (see above).
- In-engine vLLM TTFT is not measurable on this platform. GPU utilisation was reported but is not credible.
