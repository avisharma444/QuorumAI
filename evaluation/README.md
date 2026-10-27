# CouncilAI evaluation harness

Reproducible measurement of retrieval quality, semantic-cache safety and
(once real-model runs are approved) answer quality, latency and cost.

Everything runs inside the RAG test image (`councilai-rag-test`), which has
the real BGE embedding model, Chroma, the cross-encoder, `pyarrow` and the
test tools. Commands below are run from the repository root with the stack
up (`docker compose up -d`). Set this once:

```bash
EVAL='docker run --rm --network councilai-reproduced_default -v councilai-hf:/models
  -v "$PWD/evaluation:/srv/evaluation" -w /srv/evaluation -e PYTHONPATH=/srv:/srv/evaluation
  -e GIT_COMMIT='"$(git rev-parse HEAD)"' councilai-rag-test'
```

**No step in this guide calls a paid model API.** Real-model mode exists but
is gated (see the last section) and has never been run.

## 1. Data

| file | source | license | used for |
|---|---|---|---|
| `data/raw/squad_v2_dev.json` | rajpurkar.github.io (SQuAD 2.0 dev) | CC BY-SA 4.0 | extractive and unanswerable questions |
| `data/raw/hotpot_distractor_validation.parquet` | huggingface.co hotpotqa/hotpot_qa | CC BY-SA 4.0 | multi-hop (bridge, comparison) with gold supporting sentences |
| `data/raw/drop_dataset.zip` | AllenNLP (DROP) | CC BY-SA 4.0 | numeric, date and span reasoning |
| `data/raw/paws_wiki_labeled_final_test.parquet` | huggingface.co google-research-datasets/paws | "may be freely used for any purpose" (Google) | human-labelled paraphrase / adversarial pairs |
| `data/authored/cache_pairs_authored_v1.jsonl` | written for this project (in Git) | project | cache-equivalence pairs in ten categories |

Raw downloads and prepared datasets are git-ignored; their SHA-256 checksums
are pinned in `councileval/prepare.py` and recorded in each `manifest.json`.

```bash
mkdir -p evaluation/data/raw   # download the four files above into it (URLs in prepare.py)
$EVAL python -m councileval.prepare --raw data/raw --out data/councileval-v1
$EVAL python -m councileval.cache_pairs --raw data/raw --out data/cache-pairs-v1
```

**councileval-v1** (seed 13): 500 questions over 265 documents —
150 SQuAD extractive, 100 SQuAD unanswerable, 75 HotpotQA bridge, 75
HotpotQA comparison, 50 DROP numeric, 25 DROP temporal, 25 DROP span.
Each item has a stable id, source id, category, answer type, accepted
answers (copied from the source, never invented), answerability, document,
gold evidence sentences (SQuAD: minimal sentence window covering the gold
span; HotpotQA: supporting facts; DROP: none, so retrieval relevance is
"unknown"), split and license. Split by a hash of the id: `dev` 96 items
(debugging), `heldout` 404 items (reported results).

**cache-pairs-v1** (seed 29): 840 question/sentence pairs — 140 authored
(60 paraphrases; 80 near-misses: number change, negation, added/removed,
min/max, supported/unsupported, entity change, time window, comparison
direction, context-dependent follow-up), 400 rule-generated SQuAD near-misses
(number/year change, inserted negation, antonym swap, entity swap), 300 PAWS
(150 paraphrase, 150 adversarial). Labels: authored = by construction (not an
independent annotation), rules = meaning change by construction (outputs can
be ungrammatical), PAWS = human labels. Split by a hash of the id:
`explore` 311 (threshold and guard design), `heldout` 529 (reported).

## 2. Unit tests

```bash
$EVAL python -m pytest -q tests          # 49 tests, no services needed
```

## 3. Neural retrieval benchmark (real components)

```bash
$EVAL python -m councileval.retrieval_eval --split heldout --out results/retrieval
```

Each document goes through the application's inspector, text extraction,
layout chunker, BGE-small embeddings and a fresh Chroma store. Per question,
3·k vector candidates are fetched once and compared in vector order vs.
cross-encoder order (identical candidates). Relevance = chunk contains a gold
evidence sentence (after real chunking). Reports Recall@1/3/5, "all evidence
in top k" for multi-hop, MRR@5, nDCG@5 with paired bootstrap CIs, strata
(source, category, document size, single/multi-hop, chunk type), the share of
questions whose evidence is in the candidate pool, whether the top rerank
score separates answerable from unanswerable questions (AUROC), latencies,
indexing time and peak memory.

## 4. Semantic-cache correctness study (real BGE + real RediSearch)

```bash
docker run --rm --network councilai-reproduced_default -v "$PWD":/repo -v councilai-gomod:/go/pkg/mod \
  -w /repo/go-backend golang:1.23 go run ./cmd/cacheeval \
  -pairs ../evaluation/data/cache-pairs-v1/pairs.jsonl -out ../evaluation/results/cache/decisions.jsonl
$EVAL python -m councileval.cache_analysis
```

`cmd/cacheeval` embeds both questions with the RAG service's `/embed`
(BGE-small), stores q1 with the production `RedisSemanticCache` in its own
RediSearch index (dropped afterwards; the application's index is never
touched) and looks up q2 with the production `Get` at each threshold, which
includes the number guard. The analysis first verifies that every recorded
decision follows the threshold rule, then reports TPR/FPR/precision/F1 with
Wilson intervals, confusion matrices, per-source and per-category near-miss
reuse, a threshold chosen on `explore` (FPR ≤ 2%) evaluated on `heldout`,
and experimental guards (`councileval/guards.py`: polarity, opposite terms,
entities, role swap) as extra vetoes with paired bootstrap CIs. The guards
are **not** used by the application, and L2 answer reuse stays disabled by
default.

## 5. Answer-quality runs through the real API

```bash
$EVAL python -m councileval.runner --config config/experiments/offline_dev.yaml
$EVAL python -m councileval.report results/runs/<experiment_id>
$EVAL python -m councileval.blind  results/runs/<experiment_id> --n 40   # blind human packet
```

Configurations: `direct` (one model, the chairman, answers alone),
`council` (three candidates, anonymous peer review, chairman synthesis),
`council_deep` (+ reflection and at most one revision), `auto` (the router
decides). All are sent through `/api/v1/query` with the same document and
top_k; the retrieved chunks are recorded and the report checks that every
strategy saw identical evidence. Every repeat uses a fresh user, so no answer
can come from an earlier request's cache (cache hits are flagged anyway).
Strategy order is rotated per item (seeded).

Each run directory holds `metadata.json` (experiment id, git commit, dataset
version and item-id hash, backend model names, config, runtime, whether
external calls were enabled), `records.jsonl` (one line per request,
appended immediately), Prometheus snapshots before/after, `run_state.json`,
and after reporting `summary.json`, `scores.csv`, `report.md`.

**Resume:** rerun with `--resume results/runs/<experiment_id>`. Finished
(item, strategy, repeat) keys are skipped; a torn last line is redone. Ctrl-C
finishes the current request, writes the state and stops.

### Scoring (councileval/metrics.py, scoring.py)

* Normalization as in the official SQuAD script.
* Exact match and token F1 only for short responses (≤ 12 tokens); long
  responses get "not applicable", because the council writes prose.
* Automatic correctness: span → accepted answer contained (whole tokens) and
  no abstention; yes/no → first yes/no matches; number → a stated number
  equals the gold (integers exact, else 1%); date → every gold component
  present; unanswerable → the response abstains (documented phrase list).
  Containment is lenient (hedged lists can pass); it is reported as
  "automatic correctness", not as accuracy on its own.
* Error taxonomy: correct, retrieval miss, wrong despite correct retrieval
  (generation error / unsupported answer), false abstention, correct
  abstention, answered-unanswerable.
* Council stages: each candidate, the top peer-ranked candidate, the
  chairman synthesis (the pre-revision answer when a revision happened) and
  the final answer are scored separately; reflection is cross-tabulated with
  chairman correctness; revisions are counted as fixed / broke.
* Calibration (Brier, 10-bin ECE, reliability bins) uses only reported
  confidences; a missing confidence is excluded, never treated as 0. Mock
  models report none.
* Human rubric (blind packet, 1–5 each): factual correctness, completeness,
  relevance, unsupported claims (5 = none), clarity. Labels X1..Xn are
  shuffled per item; the key file is kept away from raters.

### Statistics (councileval/stats.py)

Paired comparisons use the same items: mean difference with a percentile
bootstrap CI (10,000 resamples, seed 20261027) and the exact McNemar test.
Proportions use 95% Wilson intervals. A difference is called an improvement
only when its whole CI is above 0. Offline (mock) reports print "not
interpreted" instead of verdicts.

## 6. Real-model mode (not run; requires the owner's explicit approval)

`mode: real` refuses to start unless **all** of these hold:

1. `--authorize-paid-calls` is given (a technical guard; it does not replace
   the project owner's approval);
2. `/health` reports `llm.mode == "live"` (checked before any query; offline
   mode conversely requires `mock`);
3. every configured model (as `/health` names it) has a **verified** price in
   `config/pricing.yaml` with a source URL and an effective date;
4. `--max-requests` ≥ 1 and `--max-spend-usd` > 0.

Before every request the `BudgetGuard` computes a worst-case cost (all calls
of the strategy, inputs estimated at 3 characters per token including all
candidate texts, `max_output_tokens_estimate` per call, priced at the most
expensive configured model) and stops if it could exceed either limit. After
a request it charges the reported usage at that price, or the worst case if
any call lacked reported usage. 5xx responses are not retried in real mode
(a failed request may already have been billed); 429s wait for Retry-After.
Error text stored in results is redacted (tokens, keys, passwords).

Cost reporting keeps three things apart: provider-reported tokens, the
estimated cost from verified prices, and unmeasured usage (calls without
reported tokens or models without a verified price). Mock runs report actual
spend 0 and "hypothetical real-model cost: unmeasured".
