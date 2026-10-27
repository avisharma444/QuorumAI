"""Answer, retrieval and calibration metrics.

Normalization follows the official SQuAD evaluation script: lower-case,
remove punctuation, remove the articles a/an/the, collapse whitespace.

Which metric applies where (see evaluation/README.md):

* exact_match / token_f1: only for SHORT responses (<= SHORT_ANSWER_TOKENS
  normalized tokens). The council writes free-form prose; token F1 against a
  two-word gold span would mostly measure verbosity, so long responses get
  ``None`` (not applicable) instead of a misleading number.
* answer_containment: the normalized accepted answer appears as a contiguous
  token sequence in the normalized response. Lenient (a hedging answer that
  lists several candidates can pass), so it is reported as "contains the
  accepted answer", never as "accuracy" on its own.
* numeric_match: for number answers, some number stated in the response
  equals the gold number within tolerance (integers exact; otherwise 1%
  relative). Same hedging caveat as containment.
* abstained: the response declines to answer (documented phrase list).
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Optional, Sequence

SHORT_ANSWER_TOKENS = 12
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(prediction: str, golds: Sequence[str]) -> Optional[bool]:
    if not golds or len(tokens(prediction)) > SHORT_ANSWER_TOKENS:
        return None
    p = normalize(prediction)
    return any(p == normalize(g) for g in golds)


def _f1(pred: list[str], gold: list[str]) -> float:
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: Sequence[str]) -> Optional[float]:
    p = tokens(prediction)
    if not golds or len(p) > SHORT_ANSWER_TOKENS:
        return None
    return max(_f1(p, tokens(g)) for g in golds)


def answer_containment(response: str, golds: Sequence[str]) -> Optional[bool]:
    if not golds:
        return None
    r = f" {normalize(response)} "
    for g in golds:
        ng = normalize(g)
        if ng and f" {ng} " in r:
            return True
    return False


# Numbers: digits with optional thousands separators/decimals, or small
# number words (DROP answers are often written either way).
_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUM_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def parse_number(text: str) -> Optional[float]:
    t = text.strip().lower().replace(",", "")
    if t in _WORDS:
        return float(_WORDS[t])
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    found = [float(m.group(0).replace(",", "")) for m in _NUM_RE.finditer(text)]
    found += [float(_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in _WORDS]
    return found


def numbers_equal(a: float, b: float, rel_tol: float = 0.01) -> bool:
    if float(a).is_integer() and float(b).is_integer():
        return a == b
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def numeric_match(response: str, golds: Sequence[str], rel_tol: float = 0.01) -> Optional[bool]:
    gold_nums = [n for n in (parse_number(g) for g in golds) if n is not None]
    if not gold_nums:
        return None
    stated = numbers_in(response)
    return any(numbers_equal(s, g, rel_tol) for s in stated for g in gold_nums)


_ABSTAIN = re.compile(
    r"\b(not (?:mentioned|stated|specified|provided|included|contained|available|given|found)"
    r"|does not (?:say|state|mention|specify|contain|provide|include)"
    r"|doesn't (?:say|state|mention|specify|contain|provide|include)"
    r"|no (?:information|mention|data|evidence) (?:about|on|regarding|of|is)"
    r"|cannot (?:be )?(?:determine|determined|answer|answered|find|found)"
    r"|can't (?:determine|answer|find|tell)"
    r"|(?:is|are) (?:unknown|unanswerable)"
    r"|insufficient (?:information|evidence)"
    r"|i don't know|i do not know|unable to (?:determine|answer|find))\b",
    re.I,
)


def abstained(response: str) -> bool:
    """True when the response declines to answer. A heuristic: it can miss
    unusual phrasings and flag answers that discuss missing details."""
    return bool(_ABSTAIN.search(response or ""))


# ── Retrieval ──────────────────────────────────────────────────────────


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> Optional[float]:
    """1.0 if any relevant item is in the top k (None without labels)."""
    if not relevant:
        return None
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def all_recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> Optional[float]:
    """1.0 only if every relevant item is in the top k (multi-hop)."""
    if not relevant:
        return None
    return 1.0 if relevant <= set(ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> Optional[float]:
    if not relevant:
        return None
    for i, r in enumerate(ranked[:k], 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> Optional[float]:
    """Binary-relevance nDCG@k."""
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


# ── Calibration ────────────────────────────────────────────────────────


def _pairs(confidences: Iterable[Optional[float]], correct: Iterable[Optional[bool]]):
    """Only items with both a reported confidence and a known outcome; a
    missing confidence is excluded, never treated as 0."""
    return [(float(c), bool(y)) for c, y in zip(confidences, correct) if c is not None and y is not None]


def brier_score(confidences, correct) -> Optional[float]:
    p = _pairs(confidences, correct)
    if not p:
        return None
    return sum((c - y) ** 2 for c, y in p) / len(p)


def calibration_bins(confidences, correct, bins: int = 10) -> list[dict]:
    p = _pairs(confidences, correct)
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, y) for c, y in p if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if sel:
            out.append({"lo": lo, "hi": hi, "n": len(sel), "mean_confidence": sum(c for c, _ in sel) / len(sel),
                        "accuracy": sum(y for _, y in sel) / len(sel)})
    return out


def expected_calibration_error(confidences, correct, bins: int = 10) -> Optional[float]:
    p = _pairs(confidences, correct)
    if not p:
        return None
    return sum(b["n"] / len(p) * abs(b["accuracy"] - b["mean_confidence"]) for b in calibration_bins(confidences, correct, bins))


# ── Binary classification (cache equivalence) ─────────────────────────


def confusion(labels: Sequence[bool], predicted: Sequence[bool]) -> dict:
    if len(labels) != len(predicted):
        raise ValueError("labels and predictions differ in length")
    tp = sum(1 for y, p in zip(labels, predicted) if y and p)
    fp = sum(1 for y, p in zip(labels, predicted) if not y and p)
    fn = sum(1 for y, p in zip(labels, predicted) if y and not p)
    tn = sum(1 for y, p in zip(labels, predicted) if not y and not p)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision == 0 or recall == 0 else None)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": recall, "fpr": fp / (fp + tn) if fp + tn else None,
            "precision": precision, "recall": recall, "f1": f1}
