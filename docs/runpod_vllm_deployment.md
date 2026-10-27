# Remote GPU serving: Phi-4-mini on RunPod Serverless (vLLM)

Status: **deployed and benchmarked 2026-10-27.**
- **Resources:** endpoint `vb4t31nr8i9qtx` (`quorumai-phi4-bench`) and template `fl6q5joxqg`, created by `deploy/runpod/deploy.sh`, later updated in place: RTX 4090 only, CUDA 13.0–13.4.
- **Results:** [gpu_inference_performance.md](gpu_inference_performance.md).
- **Left deployed:** with 0 minimum workers and no running worker. RunPod keeps a stopped, unbilled standby worker record.

This page is the deployment record and the runbook. It covers one open-weight model on one rented GPU, used as one QuorumAI council member. The QuorumAI backend stays on your machine; only inference runs on RunPod.

## What is deployed

One queue-based RunPod **Serverless** endpoint running the official vLLM
worker image. It is provisioned by script (`deploy/runpod/`, below); every
setting lives in the `endpoint` section of
`evaluation/config/experiments/runpod_phi4_bench.yaml`.

| setting | value | why |
|---|---|---|
| image | `runpod/worker-v1-vllm:v2.28.0` @ `sha256:6d8e4ff4…34571` | official worker, vLLM 0.30.0, released 2026-09-28. Pinned by tag **and** digest: deploy refuses if the tag moves. See "Image pin change" below |
| model | `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083` | 3.8 B parameters, 7.67 GB in bf16, MIT licence, not gated (no HF token) |
| GPU | exactly `NVIDIA GeForce RTX 4090` | RunPod Serverless "24 GB PRO", $1.10/h (runpod.io/pricing, 2026-10-27). Changed with the owner's approval from L4 / A5000 / RTX 3090 ($0.69/h): the first attempt never got a worker. One type keeps results comparable |
| GPUs per worker | 1 | |
| active (min) workers | **0** | nothing runs or bills while idle |
| max workers | **1** | caps spend: at most one GPU billed at any time |
| idle timeout | 30 s | keeps the worker between benchmark requests; scales to zero 30 s after the last one (default is 5 s) |
| execution timeout | 120 s | a stuck request is killed instead of billing for the 10-minute default |
| allowed CUDA versions | `13.0`, `13.1`, `13.2`, `13.3`, `13.4` | the image needs CUDA ≥ 13.0. RunPod treats the list as an **exact set**; `13.0` alone matched no Serverless host (they advertised 13.2). Set through GraphQL `saveEndpoint`, because REST rejects values above 13.0 |
| FlashBoot | on | faster restarts after scale-to-zero |
| container disk | 30 GB | holds the 7.7 GB weights downloaded at start plus caches; the 9.0 GB (compressed) image is stored outside it |
| network volume | none | weights download to the container disk on each cold start |
| data centres | any | no pinning |
| scaler | `QUEUE_DELAY`, 4 s | the REST default, written out; irrelevant with one worker |
| endpoint type | queue-based | the default for REST-created endpoints (no load balancer) |
| cached models | not used | kept out to avoid a second, unverified way of loading weights |

### Image pin change: v2.29.0 → v2.28.0

`runpod/worker-v1-vllm:v2.29.0` was never published.
- Its release build failed (GitHub Actions `release.yml`, 2026-10-07).
- The Docker registry returns 404 for that tag, while v2.28.0 and v2.30.1 resolve.
- An endpoint pinned to it could never start.

**Choice:** we pinned the nearest older published release, v2.28.0 (approved by you), instead of the 5-day-old v2.30.1.

**Fixes missing from v2.28.0, none of which affect this setup:**
- HF token masking (no token is used);
- a compile cache on network volumes (no volume);
- a longer default startup timeout: 1,200 s instead of 1,800 s. That is still above the benchmark's 1,200 s startup-job limit, so the worker never gives up first.

### Compatibility check (2026-10-27, from source, no deployment)

The pinned image and model revision were checked against the worker's and
vLLM's own source at the pinned tags. Nothing needed to change.

| check | result | evidence |
|---|---|---|
| image `v2.28.0` exists, linux/amd64 | yes | registry manifest: digest `sha256:6d8e4ff4…34571`, 9.0 GB compressed, 43 layers; release build succeeded |
| worker image `v2.28.0` contains vLLM 0.30.0 | yes | worker `Dockerfile` at tag v2.28.0: `ARG VLLM_VERSION=v0.30.0`; hub config `allowedCudaVersions: ["13.0"]` |
| vLLM 0.30.0 serves Phi-4-mini's architecture | yes | the model's config declares `Phi3ForCausalLM`; vLLM v0.30.0 `model_executor/models/registry.py` maps it to the native `phi3` implementation (no remote code) |
| pinned revision exists and is the current `main` | yes | Hugging Face API: revision `cfbefac…` resolves; `main` is the same commit; not gated |
| our env vars reach vLLM | yes | worker `args_builder.py`: `MODEL_NAME`→`--model`, `MODEL_REVISION`→`--revision`, and `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `DTYPE`, `ENABLE_PREFIX_CACHING` (emitted as `--no-enable-prefix-caching`) are known flags; `MAX_CONCURRENCY` is read by the worker itself (`main.py`) |
| GPUs | RTX 4090 (Ada, sm_89; the earlier L4 / A5000 / 3090 list was Ada and Ampere) | supported by CUDA 13 builds of vLLM. Hosts need a CUDA ≥ 13.0 driver; the allowed set is 13.0–13.4. Before any job, a fail-closed placement check reads the `product=SERVERLESS` catalog |
| async jobs | supported (v2.28.0) | worker `handler.py` accepts `{"openai_route": "/v1/chat/completions", "openai_input": {...}}` on `/run`; non-streaming bodies return the completion JSON once |

Two worker behaviours worth knowing, both from `main.py`:

These checks were repeated at v2.28.0: the same env flags, async input shape and
`return_aggregate_stream` are present.

- If the pinned revision ever stops resolving on Hugging Face, the worker
  drops the pin, logs a warning and loads the current `main` instead of
  failing. The pin is valid today; if `main` moves, a run would need the
  worker log to prove which revision was served.
- If vLLM runs out of memory at startup, the worker relaunches once with a
  smaller batch-token budget (`MAX_NUM_BATCHED_TOKENS=8192`). With a 7.7 GB
  model and `MAX_MODEL_LEN=4096` on 24 GB this is not expected. If it
  happens, the worker log says so.

Environment variables on the endpoint:

| variable | value |
|---|---|
| `MODEL_NAME` | `microsoft/Phi-4-mini-instruct` |
| `MODEL_REVISION` | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `MAX_MODEL_LEN` | `4096` (the model allows 131k; 4k keeps the KV cache small on 24 GB) |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `MAX_NUM_SEQS` | `16` (vLLM batch size limit; above the benchmark's highest concurrency, 8) |
| `MAX_CONCURRENCY` | `16` (worker-level concurrent jobs) |
| `DTYPE` | `bfloat16` |
| `ENABLE_PREFIX_CACHING` | `false` (repeated benchmark prompts would otherwise skip prefill and flatter TTFT) |

The worker exposes an OpenAI-compatible route:

```
https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions
```

RunPod authenticates callers at its gateway with a bearer API key. The vLLM
server inside the worker listens on loopback only. Its own `/health` route is
not reachable; endpoint health is `GET https://api.runpod.ai/v2/<ENDPOINT_ID>/health`.
That route returns job and worker counts and does not start a worker.

## Provisioning by script (`deploy/runpod/`)

`deploy/runpod/runpod_deploy.py` creates exactly the settings above.
- **runpodctl** (official CLI, v3.0.1 checked) handles everything it can express: balance, GPU catalog, template create/list/delete, endpoint list/update/health/delete.
- **REST v1 `POST /v1/endpoints`** is used for endpoint creation only. The CLI's `serverless create` accepts a single GPU pool and a *minimum* CUDA version. It cannot express the exact GPU-type list or `allowedCudaVersions: ["13.0"]`.

| command | network | effect |
|---|---|---|
| `plan` | none | prints the template command, the endpoint request body and the cost plan |
| `preflight` | read-only | checks the CLI version, key, balance vs the $2 GPU cap, GPU types in the catalog and their availability, image digest, model revision, and same-named resources (so you know whether deploy will create or reuse) |
| `deploy --authorize-deploy` | writes | runs preflight, then creates or reuses the template and the endpoint and verifies the live settings with the same check as `gpu_bench --check-endpoint`. Writes ids (no keys) to `deploy/runpod/state/` |
| `status` | read-only | live settings check plus worker and job counts |
| `cleanup --yes` | writes | sets workers to 0, waits until health shows none, deletes the endpoint, then the template |

- **Idempotent:** resources are found by exact name, so running deploy again reuses them instead of duplicating.
  - A same-named resource with different settings stops the script; it is never changed silently.
  - Cleanup with nothing left succeeds.
- **No inference and no cost:** deploy calls only management routes. A template plus an endpoint with 0 minimum workers starts nothing, and image pulls aren't billed.
- `deploy/runpod/deploy.sh` and `deploy/runpod/cleanup.sh` wrap these commands. Without a flag they only plan and check.

### Keys (create them in the console; put them in `.env` yourself)

Three keys, so that no process holds more access than it needs:

| `.env` name | RunPod permission | used by |
|---|---|---|
| `RUNPOD_MANAGEMENT_API_KEY` | *All* (creating templates and endpoints needs write access) | `deploy/runpod/` only. It is passed to runpodctl as that process's `RUNPOD_API_KEY` in a minimal environment, so the inference key below never reaches the CLI |
| `RUNPOD_API_KEY` | *Restricted*: Read/Write on **this endpoint only** | benchmark and QuorumAI. Create it **after** deploy, because it is scoped to the new endpoint's id |
| `RUNPOD_READONLY_API_KEY` | *Read Only* | `gpu_bench --check-endpoint`, GPU-type recording, billing |

`RUNPOD_ENDPOINT_ID` (not a secret) is printed by deploy and stored in `deploy/runpod/state/`.

Account checked on 2026-10-27 (read-only):
- balance $10.00, current spend $0/h, spend limit 80;
- no existing endpoints, templates or network volumes;
- L4 and RTX 3090 available ("Low" stock), RTX A5000 unavailable at that moment. These are Pod stock figures; serverless capacity is allocated separately.

## Backend configuration for the experiment

The production defaults in `config.yaml` and `MOCK_LLM=true` do not change.
For the experiment only, a marked block is added to `.env` and removed
afterwards:

```dotenv
MOCK_LLM=false
SEMANTIC_CACHE_ENABLED=false
COUNCIL_1_PROVIDER=openrouter   COUNCIL_1_MODEL=openai/gpt-4.1-nano
COUNCIL_2_PROVIDER=openrouter   COUNCIL_2_MODEL=meta-llama/llama-3.3-70b-instruct
COUNCIL_3_PROVIDER=local        COUNCIL_3_MODEL=microsoft/Phi-4-mini-instruct
VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_URL=https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions
VLLM_API_KEY=<the inference key, written literally: .env values are not interpolated>
CHAIRMAN_PROVIDER=openrouter    CHAIRMAN_MODEL=openai/gpt-4o-mini   # unchanged from the previous benchmark
ROUTER_PROVIDER=openrouter      ROUTER_MODEL=openai/gpt-4.1-nano    # not called: explicit strategy
INGESTION_PROVIDER=openrouter   INGESTION_MODEL=openai/gpt-4o-mini
```

How the vLLM adapter (`go-backend/internal/llm/vllm.go`) handles this:

- **Key handling:**
  - It sends `Authorization: Bearer <key>` only when a key is configured, from `VLLM_API_KEY` or the per-model `VLLM_<MODEL>_API_KEY`.
  - It refuses to send a key over plain HTTP, except to localhost.
  - Keys never appear in logs, errors or `/health`.
- **URL checks:**
  - A URL on `api.runpod.ai` must be the OpenAI chat route.
  - A RunPod URL without a key is a startup error.
- **Readiness:** a background check at startup calls RunPod's endpoint `/health`, which does not start a worker. It logs `ready=true (scaled to zero …)` or the failure, and never blocks or fails startup.
- **Timeouts and fallback:** a cold or broken endpoint is subject to the normal stage timeout (`STAGE_TIMEOUT`, 30 s), and the council continues with the two hosted members.
- **Failed members don't review:** a member whose candidate failed is not asked to peer-review.
  - It gets a `review_skipped` stream event and is listed in the result's `failed_candidates` with its error.
  - So a dead endpoint costs one stage timeout per council, not two, and receives one request, not two.
  - Reviews keep configuration order and each reviewer's configured slot. The chairman still sees "Reviewer 2/3" and the scoring and fallbacks are unchanged.
  - This deliberately differs from the reference, which asked every configured member.
  - Covered by `TestFailedMemberIsNotAskedToReview`, `TestSkippedReviewerStillFallsBackWhenRemainingReviewsFail` and `TestRunPodVLLMMemberTimeoutFallsBackToTwoMembers`.

## Cold start: asynchronous startup probe

RunPod's synchronous routes wait about 90 s by default (`/runsync`; the
OpenAI route is also synchronous). A first start can take longer:
- container start;
- downloading 7.7 GB of weights;
- loading them and capturing CUDA graphs.

So the first request is never a synchronous call:

- `gpu_bench` phase **startup** submits one small chat request as an async
  job (`POST /v2/<id>/run` with `openai_route`/`openai_input`) and polls
  `GET /v2/<id>/status/<job>` every 2 s for up to 20 minutes.
  - It records the status changes and RunPod's own `delayTime` (submission until a worker took the job: queue, start and model load) and `executionTime`.
  - On timeout it cancels the job (`POST /v2/<id>/cancel/<job>`), purges the queue and stops.
- The streamed benchmark phases start only after the startup job completes.
  Startup is reported on its own (`summary.json` → `startup`) and never mixed into warm latency or TTFT.
- `e2e_vllm` uses the same async job as its warm-up (council and cancel phases) before any council.

## Two different TTFTs

| name | what it measures | how |
|---|---|---|
| **gateway TTFT** (end to end) | request sent to the RunPod OpenAI route → first streamed content chunk at the client. Includes gateway, network, queueing, the worker's proxy, vLLM prefill and the first token | `gpu_bench` warm and concurrency phases, from the token stream |
| **worker-side 1-token time** | RunPod `executionTime` of an async job asking for `max_tokens=1`: time inside the worker (its local proxy + vLLM prefill + one token). Excludes gateway, network and queue. An **upper bound** on in-engine TTFT | `gpu_bench` phase **worker_timing** (8 jobs, one per prompt) |
| in-engine vLLM TTFT | vLLM's own `time_to_first_token` histogram | **not measurable here.** It lives on vLLM's `/metrics` route. vLLM listens on loopback inside the worker, and the worker's proxy only relays JSON, so the text metrics cannot be read. The report says so instead of estimating it |

The report shows the two measured numbers side by side, plus their difference: roughly the gateway, network and queue overhead.

## One GPU budget for every phase

`config/experiments/runpod_campaign.yaml` and `councileval/gpu_budget.py`
set one budget across all GPU work. Both scripts refuse to run without it:
- direct benchmarking and failure probes;
- startup and warm-up jobs;
- councils;
- fallback and cancellation runs.

Each run is a *session* in a shared ledger file, so the caps hold across processes.

- **Every request to the endpoint counts when it is sent.** That includes failure probes and requests the gateway rejects.
- **No retries:** neither the harness nor the backend retries model calls. The evaluation client retries only QuorumAI's own 429s, which are rejected before any model call.
- **Councils:** each council is charged 2 Phi calls (candidate + review).
- **Wall-time cap:** a request or council is refused if it could still be running at the session's wall cap.
- **Spend reservation:** a session reserves its worst case before sending anything. A crashed session keeps its reservation.

| session | GPU requests | wall cap | worst-case GPU $ |
|---|---|---|---|
| `gpu_bench`: 1 startup + 8 worker-timing + 20 warm + 4×16 concurrency + 5 failure probes | 98 | 2,400 s | 0.60 |
| `e2e_council`: 1 warm-up + 8 councils × 2 | 17 | 2,340 s | 0.59 |
| `e2e_fallback`: 1 council × 2 (wrong key, rejected at the gateway) | 2 | 300 s | 0.20 |
| `e2e_cancel`: 1 warm-up + 1 council × 2 | 3 | 1,200 s | 0.37 |
| **campaign caps** | **120** | **6,240 s** | **1.77 planned, $2.00 cap** |

Worst case per session = (wall cap + 600 s cold-start allowance + 120 s
execution timeout + 30 s idle) × the GPU price ($1.10/h for the RTX 4090; the table above was computed at the earlier $0.69/h and is superseded by `runpod_campaign.yaml`). With max workers = 1:
- a worker can still be loading after an abort;
- a job the gateway didn't cancel can run to the execution timeout;
- then the worker idles until the idle timeout.

## Run order (after approval)

0. **Deploy** (host, needs `RUNPOD_MANAGEMENT_API_KEY`):
   `deploy/runpod/deploy.sh --authorize-deploy`.
   Then create the restricted inference key for the printed endpoint id, and add it and `RUNPOD_ENDPOINT_ID` to `.env`.

The benchmark commands run from `evaluation/` in the test container on the
compose network. The keys come from `.env` as environment variables.

1. **Check the live settings** (read-only, Read Only key):
   `python -m councileval.gpu_bench --config config/experiments/runpod_phi4_bench.yaml --check-endpoint`.
   It stops if any of these differ from the approved deployment:
   - min workers ≠ 0 or max workers ≠ 1;
   - a GPU outside the approved list, or more than one GPU per worker;
   - CUDA versions other than 13.0;
   - an execution timeout above 120 s;
   - a different image or model env.
2. **Direct GPU benchmark:**
   `python -m councileval.gpu_bench --config config/experiments/runpod_phi4_bench.yaml --gpu-campaign config/experiments/runpod_campaign.yaml --authorize-paid-calls --max-requests 98 --max-wall-seconds 2400 --max-spend-usd 1.00`.
   Phases:
   - async startup;
   - 8 worker-timing jobs;
   - 20 warm streamed requests;
   - concurrency 1, 2, 4 and 8 (16 each);
   - 5 failure probes.

   Streamed requests produce 256 tokens with fixed settings. The run ends only after `/health` shows zero workers.
3. **QuorumAI end-to-end:** restart the backend with the block above, then
   run `python -m councileval.e2e_vllm --config config/experiments/runpod_e2e.yaml --phase council --gpu-campaign config/experiments/runpod_campaign.yaml --authorize-paid-calls --max-requests 8 --max-spend-usd 0.25`.
4. **Fallback:** restart the backend with a deliberately wrong
   `VLLM_API_KEY`, then run `--phase fallback --max-requests 1` (no warm-up).
5. **Cancellation:** restore the correct key, then run `--phase cancel --max-requests 1`.
   - It does an async warm-up.
   - The client gives up 3 s into a council.
   - RunPod `/health` is sampled for 20 s.
   - Anything still queued or running is purged.
6. **Billing:** `python -m councileval.gpu_bench --billing results/gpu_bench/<run>`
   reads the provider-side amount (it may lag by up to an hour).
7. **Restore mock mode and check zero workers.** Restore `MOCK_LLM=true`,
   remove the block, and confirm `/health` shows zero workers.
8. **Clean up:** `deploy/runpod/cleanup.sh --yes`. It scales to 0, waits for no workers, then deletes the endpoint and template. Re-running is harmless.

## Cost bounds

The budget plan is `config/experiments/runpod_campaign.yaml` (at $1.10/h):
- worst case $1.98 under the $2.00 GPU cap;
- the OpenRouter cap of $0.25 is split $0.165 / $0.0425 / $0.0425 across the three phases.

Measured spend is in [gpu_inference_performance.md](gpu_inference_performance.md#cost): **$0.2013** RunPod (billing API and balance agree) and **$0.0074** OpenRouter.

## Known limits

- **Provider quirks found while deploying.** Both are handled in code and covered by tests:
  - RunPod's edge rejects Python's default `Python-urllib` User-Agent with HTTP 403 "error code: 1010", so the scripts send their own User-Agent.
  - The endpoint's embedded template omits `imageName`, so the image is read from `GET /v1/templates/{id}`.
- **`workersStandby: 1` appears on the endpoint without being set.** The field is not in RunPod's API reference.
  - Right after deployment: health showed 0 workers in every state, account spend was $0/h, and the billing API had no records for the endpoint.
  - `workersMin` (the billed always-on count) is 0.

- **Token-level streaming is assumed, not verified.** RunPod documents
  streaming on the OpenAI route, but no source states whether the gateway
  forwards chunks immediately. The benchmark records chunk counts and timing
  per request, so a buffering gateway shows up in the data instead of
  inflating TTFT claims.
- **GPU utilisation and memory may not be exposed** for Serverless workers.
  The harness tries the read-only API and records "not exposed" when there is
  nothing to read.
- **The 90 s synchronous limit is documented for `/runsync`.** The OpenAI route's
  own limit is not documented, which is why startup uses async jobs.
- **Cold-start time depends on placement.** It varies with the host chosen,
  whether the image is already cached there, and FlashBoot. One measurement
  is one sample.
