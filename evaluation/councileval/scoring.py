"""Score recorded responses against the dataset's ground truth.

Correctness by answer type (all automatic, all documented in README):

  span    the response contains an accepted answer (normalized) and does not abstain
  yes_no  the first "yes"/"no" in the response equals the gold answer
  number  a number stated in the response equals the gold number (tolerance in metrics)
  date    every gold date component (day, month, year) appears in the response
  none    (unanswerable) the response abstains

Containment-style checks are lenient (a hedging answer that lists several
candidates can pass); they are labelled "automatic correctness", and the
blind human rubric exists for long-form judgments.

Error taxonomy for answerable items with evidence labels:
  correct | retrieval_miss (wrong; gold evidence not in retrieved chunks)
  | wrong_despite_retrieval (wrong and not abstaining although the gold
  evidence WAS retrieved: generation error / unsupported answer)
  | false_abstention.  Unanswerable: correct_abstention | answered_unanswerable.
"""
from __future__ import annotations

import re
from typing import Optional

from .metrics import abstained, answer_containment, exact_match, normalize, numeric_match, token_f1
from .schema import EvalItem

_YESNO = re.compile(r"\b(yes|no)\b", re.I)
_MONTHS = {m.lower() for m in ("January February March April May June July August September October November December").split()}


def yes_no_match(response: str, golds) -> Optional[bool]:
    m = _YESNO.search(response or "")
    if not m or not golds:
        return False if golds else None
    return m.group(1).lower() == golds[0].strip().lower()


def date_match(response: str, parts: dict) -> Optional[bool]:
    if not parts or not any(parts.values()):
        return None
    r = f" {normalize(response)} "
    for k in ("day", "month", "year"):
        v = (parts.get(k) or "").strip()
        if v and f" {normalize(v)} " not in r:
            return False
    return True


def is_correct(item: EvalItem, response: Optional[str]) -> Optional[bool]:
    """Automatic correctness of one answer text (None when not scorable)."""
    if response is None:
        return None
    if not item.answerable:
        return abstained(response)
    if abstained(response):
        return False
    if item.answer_type == "yes_no":
        return yes_no_match(response, item.answers)
    if item.answer_type == "number":
        return numeric_match(response, item.answers)
    if item.answer_type == "date":
        return date_match(response, item.metadata.get("date") or {})
    return answer_containment(response, item.answers)


def retrieval_status(item: EvalItem, chunks: list[dict]) -> dict:
    texts = [normalize(c.get("content", "")) for c in chunks]
    joined = " ".join(texts)
    if not item.evidence:
        return {"evidence_labels": False, "evidence_retrieved": None, "all_evidence_retrieved": None,
                "answer_in_context": any(normalize(a) in joined for a in item.answers) if item.answers else None}
    hits = [any(normalize(ev) in t for t in texts) for ev in item.evidence]
    return {"evidence_labels": True, "evidence_retrieved": any(hits), "all_evidence_retrieved": all(hits),
            "answer_in_context": any(normalize(a) in joined for a in item.answers) if item.answers else None}


def taxonomy(item: EvalItem, response: str, correct: Optional[bool], ret: dict) -> str:
    if not item.answerable:
        return "correct_abstention" if correct else "answered_unanswerable"
    if correct:
        return "correct"
    if abstained(response):
        return "false_abstention"
    if ret["evidence_retrieved"] is None:
        return "wrong_retrieval_unknown"
    return "wrong_despite_retrieval" if ret["evidence_retrieved"] else "retrieval_miss"


def model_calls(resp: dict, strategy: str) -> int:
    n = 0 if strategy != "auto" else int(bool((resp.get("routing") or {}).get("router_model")))
    if resp.get("strategy") == "direct":
        return n + 1
    n += len(resp.get("candidates") or []) + len(resp.get("reviews") or [])
    n += 1 if resp.get("synthesis") else 0
    refl = resp.get("reflection") or {}
    n += 1 if refl else 0
    n += 1 if refl.get("revision") else 0
    return n


def score_record(item: EvalItem, rec: dict) -> dict:
    out = {"item_id": item.id, "strategy": rec["strategy"], "repeat": rec["repeat"], "category": item.category,
           "answer_type": item.answer_type, "answerable": item.answerable, "status": rec["status"],
           "latency_ms": rec["latency_ms"], "trace_id": rec.get("trace_id", "")}
    resp = rec.get("response")
    if rec["status"] != 200 or not resp:
        out.update({"ok": False, "correct": None, "error": rec.get("error")})
        return out
    answer = resp.get("answer", "")
    correct = is_correct(item, answer)
    chunks = (resp.get("retrieval") or {}).get("chunks") or []
    ret = retrieval_status(item, chunks)
    cache = resp.get("cache") or {}
    refl = resp.get("reflection") or {}
    rev = refl.get("revision") or {}
    pre_revision = rev.get("previous_answer") if rev else None
    chairman_answer = pre_revision if pre_revision is not None else (answer if (resp.get("synthesis") or {}).get("status") == "synthesized" else None)
    cands = resp.get("candidates") or []
    cand_correct = [is_correct(item, c.get("answer")) if not c.get("error") else None for c in cands]
    selected = resp.get("selected")
    best = next((c for c in cands if c.get("label") == selected), None)
    out.update({
        "ok": True, "executed_strategy": resp.get("strategy"), "router_strategy": (resp.get("routing") or {}).get("strategy"),
        "router_fallback": (resp.get("routing") or {}).get("fallback"),
        "answer_chars": len(answer), "correct": correct, "abstained": abstained(answer),
        "exact_match": exact_match(answer, item.answers), "token_f1": token_f1(answer, item.answers),
        "taxonomy": taxonomy(item, answer, correct, ret), **ret,
        "evidence_fingerprint": "|".join(str(c.get("chunk_index")) for c in chunks),
        "grounding": resp.get("grounding"), "cache_tier": cache.get("tier"),
        "cache_contaminated": cache.get("tier") not in (None, "none"),
        "model_calls": model_calls(resp, rec["strategy"]), "token_usage": resp.get("token_usage"),
        "confidence": resp.get("confidence"), "confidence_source": resp.get("confidence_source"),
        "degraded": resp.get("degraded") or [], "source_kind": _source_kind(resp.get("source", "")),
        "duration_ms_server": resp.get("duration_ms"),
        # Council stage artifacts (None when the stage did not run)
        "candidates_n": len(cands), "candidates_failed": sum(1 for c in cands if c.get("error")),
        "candidates_correct": [c for c in cand_correct],
        "any_candidate_correct": any(c for c in cand_correct if c is not None) if cands else None,
        "mean_candidate_correct": (sum(1 for c in cand_correct if c) / len([c for c in cand_correct if c is not None])) if any(c is not None for c in cand_correct) else None,
        "best_ranked_correct": is_correct(item, best.get("answer")) if best else None,
        "chairman_correct": is_correct(item, chairman_answer) if chairman_answer is not None else None,
        "reflection_quality": refl.get("quality"), "reflection_faithful": refl.get("faithful"),
        "revision_status": rev.get("status"), "revised_correct": correct if rev.get("status") == "succeeded" else None,
        "candidate_latency_ms": [c.get("latency_ms") for c in cands], "review_latency_ms": [r.get("latency_ms") for r in resp.get("reviews") or []],
        "candidate_usage": [c.get("usage") for c in cands], "candidate_models": [c.get("model") for c in cands],
    })
    return out


def _source_kind(source: str) -> str:
    if source.startswith("chairman-revised:"):
        return "chairman_revised"
    if source.startswith("chairman:"):
        return "chairman"
    if source.startswith("direct:"):
        return "direct"
    if "fallback" in source or "single-response" in source:
        return "fallback"
    return "other"
