"""Experimental equivalence guards for semantic-cache hits (OFFLINE ONLY).

None of these is used by the application. They are evaluated against the
production rule (cosine threshold + number guard) on the same held-out
pairs. Each guard can only VETO a hit; it never creates one.

They are lexical heuristics with known blind spots (synonymous negation,
paraphrased antonyms, entities written in lower case). Passing them does
not make two questions equivalent.
"""
from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:[.,]\d+)*")
NEGATIONS = {"not", "no", "never", "without", "cannot", "neither", "nor", "none", "nothing", "nobody"}
# Pairs whose members ask opposite things. Matching is on whole words.
OPPOSITES = [
    ("first", "last"), ("most", "least"), ("most", "fewest"), ("largest", "smallest"), ("highest", "lowest"),
    ("earliest", "latest"), ("before", "after"), ("increase", "decrease"), ("increased", "decreased"),
    ("maximum", "minimum"), ("more", "less"), ("more", "fewer"), ("north", "south"), ("east", "west"),
    ("oldest", "youngest"), ("older", "younger"), ("longest", "shortest"), ("biggest", "smallest"),
    ("won", "lost"), ("start", "end"), ("began", "ended"), ("opened", "closed"), ("upper", "lower"),
    ("inside", "outside"), ("larger", "smaller"), ("higher", "lower"), ("major", "minor"), ("rise", "fall"),
    ("gain", "loss"), ("added", "removed"), ("joined", "left"), ("introduced", "discontinued"),
    ("created", "deleted"), ("raised", "cut"), ("granted", "revoked"), ("accepted", "rejected"),
    ("accept", "reject"), ("supported", "unsupported"), ("supports", "contradicts"), ("favour", "oppose"),
    ("favor", "oppose"), ("advantages", "disadvantages"), ("strengths", "weaknesses"), ("confirm", "refute"),
    ("confirmed", "refuted"), ("include", "exclude"), ("included", "excluded"), ("covered", "excluded"),
    ("safe", "unsafe"), ("allowed", "forbidden"), ("cheapest", "expensive"), ("closest", "farthest"),
    ("morning", "evening"), ("first", "second"), ("previous", "following"), ("bigger", "smaller"),
    ("tomorrow", "yesterday"), ("during", "after"), ("this", "last"), ("buyer", "seller"),
]
_OPP = {}
for a, b in OPPOSITES:
    _OPP.setdefault(a, set()).add(b)
    _OPP.setdefault(b, set()).add(a)
_QWORDS = {"what", "who", "whom", "which", "when", "where", "why", "how", "is", "are", "was", "were", "do", "does",
           "did", "can", "could", "should", "would", "will", "has", "have", "had", "in", "on", "at", "the", "a", "an"}


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _neg_count(ws: list[str]) -> int:
    return sum(1 for w in ws if w in NEGATIONS or w.endswith("n't"))


def polarity_differs(q1: str, q2: str) -> bool:
    """Veto when one question is negated and the other is not."""
    return (_neg_count(words(q1)) % 2) != (_neg_count(words(q2)) % 2)


def opposite_terms(q1: str, q2: str) -> bool:
    """Veto when a word in one question has an opposite in the other that
    the first question does not also contain."""
    w1, w2 = set(words(q1)), set(words(q2))
    for w in w1 - w2:
        if _OPP.get(w, set()) & (w2 - w1):
            return True
    return False


def entities(text: str) -> set[str]:
    """Capitalized words not at the start of the sentence, minus question words."""
    toks = _WORD.findall(text)
    out = set()
    for i, t in enumerate(toks):
        if i > 0 and t[:1].isupper() and t.lower() not in _QWORDS:
            out.add(t.lower())
    return out


def entities_differ(q1: str, q2: str) -> bool:
    return entities(q1) != entities(q2)


def role_swap(q1: str, q2: str) -> bool:
    """Same words in a different order (e.g. 'A faster than B' vs 'B faster
    than A'): veto, because word order then carries the meaning."""
    w1, w2 = words(q1), words(q2)
    return w1 != w2 and Counter(w1) == Counter(w2)


GUARDS = {
    "polarity": polarity_differs,
    "opposites": opposite_terms,
    "entities": entities_differ,
    "role_swap": role_swap,
}


def vetoed(q1: str, q2: str, names=tuple(GUARDS)) -> list[str]:
    return [n for n in names if GUARDS[n](q1, q2)]
