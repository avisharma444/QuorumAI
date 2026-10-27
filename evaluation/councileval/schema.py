"""Evaluation item schema and validation.

An item is one question with its ground truth, provenance and retrieval
labels. Ground truth comes only from the source dataset; when the source
does not support a judgment the field is None ("unknown"), never guessed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

CATEGORIES = {
    "extractive",        # answer is a span of one passage (SQuAD 2.0 answerable)
    "unanswerable",      # the document does not contain the answer (SQuAD 2.0 impossible)
    "multi_hop_bridge",  # HotpotQA bridge: needs two passages
    "multi_hop_comparison",  # HotpotQA comparison between two entities
    "numeric",           # DROP: number answer (counting, arithmetic)
    "temporal",          # DROP: date answer
    "span_reasoning",    # DROP: span answer requiring reasoning over the passage
}
ANSWER_TYPES = {"span", "number", "date", "yes_no", "none"}
SPLITS = {"dev", "heldout"}


class ValidationError(ValueError):
    pass


@dataclass
class EvalItem:
    id: str
    question: str
    source: str            # dataset name and split, e.g. "squad_v2/dev"
    source_id: str         # id in the source dataset
    category: str
    answer_type: str
    answers: list[str]     # accepted answers; empty only when answerable is False
    answerable: bool
    doc_id: str            # local document key (see documents.jsonl)
    evidence: list[str]    # gold supporting sentences (exact text from the source)
    split: str
    license: str
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str              # markdown-ish plain text as uploaded to the app
    source: str
    license: str


def validate_item(it: EvalItem) -> None:
    def bad(msg: str):
        raise ValidationError(f"{it.id or '<no id>'}: {msg}")

    if not it.id or not it.id.strip():
        bad("missing id")
    if not it.question or not it.question.strip():
        bad("empty question")
    if it.category not in CATEGORIES:
        bad(f"unknown category {it.category!r}")
    if it.answer_type not in ANSWER_TYPES:
        bad(f"unknown answer_type {it.answer_type!r}")
    if it.split not in SPLITS:
        bad(f"unknown split {it.split!r}")
    if not isinstance(it.answers, list) or any(not isinstance(a, str) for a in it.answers):
        bad("answers must be a list of strings")
    if it.answerable:
        if not it.answers or any(not a.strip() for a in it.answers):
            bad("answerable item needs non-empty accepted answers")
        if it.answer_type == "none":
            bad("answerable item cannot have answer_type 'none'")
    else:
        if it.answers:
            bad("unanswerable item must not list answers")
        if it.answer_type != "none":
            bad("unanswerable item must have answer_type 'none'")
    if it.answer_type == "number":
        from .metrics import parse_number
        if any(parse_number(a) is None for a in it.answers):
            bad("number item with a non-numeric accepted answer")
    if not it.doc_id:
        bad("missing doc_id")
    if not it.source or not it.license:
        bad("missing provenance (source/license)")


def validate_dataset(items: list[EvalItem], documents: dict[str, Document]) -> None:
    seen: set[str] = set()
    for it in items:
        validate_item(it)
        if it.id in seen:
            raise ValidationError(f"duplicate question id {it.id}")
        seen.add(it.id)
        if it.doc_id not in documents:
            raise ValidationError(f"{it.id}: document {it.doc_id} missing")
        doc = documents[it.doc_id]
        norm_doc = " ".join(doc.text.split())
        for ev in it.evidence:
            if " ".join(ev.split()) not in norm_doc:
                raise ValidationError(f"{it.id}: evidence sentence not found in its document")


def load_items(path: Path) -> list[EvalItem]:
    items = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"line {n}: malformed item: {exc}") from exc
    return items


def load_documents(path: Path) -> dict[str, Document]:
    docs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = Document(**json.loads(line))
            if d.doc_id in docs:
                raise ValidationError(f"duplicate document id {d.doc_id}")
            docs[d.doc_id] = d
    return docs


def load_dataset(directory: Path) -> tuple[list[EvalItem], dict[str, Document]]:
    directory = Path(directory)
    items = load_items(directory / "items.jsonl")
    docs = load_documents(directory / "documents.jsonl")
    validate_dataset(items, docs)
    return items, docs
