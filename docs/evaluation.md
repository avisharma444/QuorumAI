# Evaluation results (checkpoint 10)

How to reproduce everything here: [evaluation/README.md](../evaluation/README.md).
Measured on 27 Oct 2026, Apple M2 (8 GB), Docker VM with 8 vCPU / 3.8 GiB.

**What is and is not measured.** Retrieval quality and semantic-cache safety
were measured with the real models (BGE-small embeddings, ChromaDB,
ms-marco-MiniLM cross-encoder, Redis Stack RediSearch through the production
Go cache code). Answer quality of the council versus a single model was
**not** measured: that needs paid provider calls, which were not authorized.
The answer-quality harness was run end to end with mock models to validate
the workflow; those runs are not accuracy results.

## 1. Datasets

| dataset | items | split | ground truth |
|---|---|---|---|
| councileval-v1 (SQuAD 2.0 dev, HotpotQA distractor, DROP dev; CC BY-SA 4.0) | 500 questions, 265 documents | 96 dev / 404 held-out | accepted answers and evidence copied from the sources |
| cache-pairs-v1 (authored, SQuAD rule edits, PAWS-Wiki) | 840 pairs (210 equivalent, 630 not) | 311 explore / 529 held-out | authored and rule labels by construction; PAWS human labels |

Categories: extractive 150, unanswerable 100, multi-hop bridge 75,
multi-hop comparison 75, numeric 50, temporal 25, span reasoning 25.

## 2. Neural retrieval: vector order vs cross-encoder reranking (held-out)

404 held-out questions, 221 documents, 2,095 chunks (median 11 per document,
max 49). Same 15 vector candidates per question in both configurations
(k = 5). Relevance is known for 242 questions (SQuAD answerable, HotpotQA);
DROP and unanswerable questions have no evidence labels.

| metric (n = 242) | vector | reranked | difference (95% paired bootstrap CI) |
|---|---|---|---|
| Recall@1 | 0.760 | 0.851 | **+0.091 (+0.033 to +0.149)** |
| Recall@3 | 0.921 | 0.926 | +0.004 (−0.029 to +0.037) |
| Recall@5 | 0.938 | 0.979 | +0.041 (+0.017 to +0.070) |
| MRR@5 | 0.834 | 0.898 | +0.063 (+0.027 to +0.100) |
| nDCG@5 | 0.832 | 0.879 | +0.047 (+0.016 to +0.079) |
| all evidence in top 3 (multi-hop, n = 119) | 0.782 | 0.655 | **−0.126 (−0.202 to −0.050)** |
| all evidence in top 5 (multi-hop, n = 119) | 0.866 | 0.832 | −0.034 (−0.092 to +0.025) |

Strata (Recall@1, vector → reranked, 95% CI of the difference):

| stratum | n | Recall@1 | note |
|---|---|---|---|
| SQuAD single-hop | 123 | 0.593 → 0.797 (+0.114 to +0.293) | reranking clearly helps |
| documents > 20 chunks | 76 | 0.553 → 0.776 (+0.105 to +0.342) | gains concentrate in large documents |
| documents 5–20 chunks | 166 | 0.855 → 0.886 (−0.030 to +0.090) | no reliable difference |
| HotpotQA bridge | 60 | 0.883 → 0.900 (−0.067 to +0.100) | all-evidence@3 0.617 → 0.467 (−0.283 to −0.017) |
| HotpotQA comparison | 59 | 0.983 → 0.915 (−0.153 to 0.000) | all-evidence@3 0.949 → 0.847 (−0.203 to −0.017) |

Findings:
- Reranking improves finding *one* relevant chunk, especially in long
  documents, but **hurts multi-hop questions**: it promotes the chunk most
  similar to the question and pushes the second supporting passage out of the
  top 3. This matches the earlier mixed small-corpus results; it is not
  "always better".
- The relevant chunk was in the 15-candidate pool for 98.8% of questions
  (95% CI 96.4–99.6%), so recall is not limited by the vector stage.
- The top cross-encoder score separates answerable from unanswerable SQuAD
  questions only weakly (AUROC 0.62; medians 2.48 vs 1.02): not usable alone
  as an abstention signal.
- All relevant chunks were narrative paragraphs. **Tables were not covered**
  by these datasets; the earlier 8-chunk PDF fixture remains the only table
  evidence.
- Latency per query: embedding p50 14 ms (p95 60), vector search 2 ms,
  **reranking p50 695 ms (p95 1,033)** for 15 candidates on CPU. Indexing
  1.13 s per document on average (p95 2.9 s), 250 s in total; peak process
  memory 1.77 GiB with both models loaded.

## 3. Semantic-cache correctness (held-out: 145 equivalent, 384 different pairs)

Decisions come from the production `RedisSemanticCache.Get` (RediSearch KNN,
cosine threshold, number guard) with real BGE-small vectors; every recorded
decision was verified against the threshold rule (0 mismatches) and RediSearch
similarities matched locally computed cosines.

| threshold | equivalent reused (TPR) | different questions reused (FPR) | precision | TP / FN / FP / TN |
|---|---|---|---|---|
| 0.80 | 0.979 (0.941–0.993) | 0.724 (0.677–0.766) | 0.338 | 142 / 3 / 278 / 106 |
| **0.85 (reference)** | 0.910 (0.853–0.947) | **0.654 (0.605–0.700)** | 0.345 | 132 / 13 / 251 / 133 |
| 0.90 | 0.807 (0.735–0.863) | 0.542 (0.492–0.591) | 0.360 | 117 / 28 / 208 / 176 |
| 0.95 | 0.648 (0.568–0.721) | 0.357 (0.310–0.406) | 0.407 | 94 / 51 / 137 / 247 |
| 0.98 | 0.503 (0.423–0.584) | 0.190 (0.154–0.232) | 0.500 | 73 / 72 / 73 / 311 |
| 0.99 | 0.366 (0.292–0.446) | 0.091 (0.066–0.124) | 0.602 | 53 / 92 / 35 / 349 |
| 0.995 | 0.159 (0.108–0.227) | 0.044 (0.028–0.070) | 0.575 | 23 / 122 / 17 / 367 |

(95% Wilson intervals.) The FPR depends strongly on the mix of near-misses:
by source at 0.85, authored pairs 0.553 (n = 47), SQuAD rule edits 0.541
(n = 244), PAWS adversarial 1.000 (n = 93). The earlier 25+25 benchmark's
28% at 0.85 used easier near-misses; this larger set includes negation,
word-order and role-swap pairs.

Near-miss reuse at 0.85 / 0.95 by category (held-out): negation 1.00 / 0.51
(n = 73), antonym 0.90 / 0.31 (49), PAWS word-swap 1.00 / 0.87 (93), entity
change 0.32 / 0.00 (66), comparison direction 1.00 / 1.00 (4), time window
0.67 / 0.00 (6), supported/unsupported 0.67 / 0.00 (6), min/max 0.29 / 0.00
(7), added/removed 0.50 / 0.00 (4), follow-up 1.00 / 0.00 (4), number and year
changes 0.00 / 0.00 (37 + 35, blocked by the number guard).

**Threshold selection** (explore split only, target FPR ≤ 2%): no threshold
up to 0.995 met the target with the production rule. With all four
experimental guards, 0.995 met it on explore; on held-out it gave TPR 0.103
(0.064–0.164) and FPR 0.021 (0.011–0.041), i.e. almost no reuse.

**Experimental guards at 0.85** (held-out, paired change vs production):

| added veto | TPR | FPR | ΔFPR (95% CI) | ΔTPR (95% CI) |
|---|---|---|---|---|
| polarity (negation parity) | 0.910 | 0.471 | −0.182 (−0.221 to −0.146) | 0.000 |
| opposite terms | 0.910 | 0.500 | −0.154 (−0.190 to −0.120) | 0.000 |
| named entities | 0.634 | 0.471 | −0.182 (−0.221 to −0.146) | −0.276 (−0.352 to −0.207) |
| role swap (same words, new order) | 0.869 | 0.622 | −0.031 (−0.049 to −0.016) | −0.041 (−0.076 to −0.014) |
| all four | 0.607 | 0.133 | −0.521 (−0.570 to −0.469) | −0.303 (−0.379 to −0.228) |

Conclusions:
- With BGE-small and the existing number guard, **semantic answer reuse is
  not safe at any threshold that still reuses a meaningful share of
  paraphrases**. L2 reuse stays disabled by default.
- Polarity and opposite-term vetoes cut false reuse by 15–18 points at no
  measured cost to true reuse on this set; even so, 47% of near-misses would
  still be reused at 0.85. The guards are lexical heuristics, designed with
  knowledge of the authored categories (which appear in both splits), so the
  held-out gain is likely optimistic for unseen phrasings. None is enabled in
  the application.
- Bug found and fixed: identical questions sometimes missed the semantic
  cache because RediSearch returned a cosine distance just below zero (float32
  rounding) and the hit rule rejected negative distances (23 of 50 identical
  random vectors missed before the fix, 0 after).

## 4. Answer-quality harness: offline workflow validation (mock models)

Dev split, 96 questions × {direct, council, council_deep, auto} = 384
requests through the real API (real retrieval, routing, cache isolation,
council workflow; mock answers).

- 384/384 requests succeeded; no cached answers (fresh user per run).
- Evidence identity: every one of the 96 items saw **identical retrieved
  chunks** under all four strategies, so strategy comparisons are not
  confounded by retrieval.
- Model calls per request (from responses, confirmed exactly by Prometheus
  deltas): direct 1, council 7, council_deep 8 (no mock revision), auto 8
  (router + council). Prometheus counted 864 candidate, 864 review, 288
  chairman, 96 direct, 96 router and 96 reflection calls.
- Token usage reported: 0 (mock models report none); actual spend $0;
  hypothetical real-model cost: unmeasured.
- The mock router sent every `auto` request to `council`; the router's
  real-model strategy choices are therefore unmeasured.
- The mock answers scored near zero, as they should. Mock `direct` answers
  repeat retrieved text and 9/96 passed the containment check; this
  demonstrates that containment rewards restating context, so real-model
  runs must report the error taxonomy and use the blind human rubric rather
  than containment alone.

## 5. Still unmeasured

Council vs single-model answer quality, the effect of reflection/revision,
router economics, confidence calibration, real token usage, cost per correct
answer and real-model latency. The runner, scoring, statistics, budget guard
and report are ready for a small, explicitly approved pilot (see the
checkpoint report for the proposed configuration and cost bound).
