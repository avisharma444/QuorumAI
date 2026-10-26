"""Cross-encoder reranker (reference app/retrieval/reranker.py).

Uses sentence_transformers.CrossEncoder to score (query, passage) pairs
jointly. Scores are raw model logits useful for ordering, not probabilities.

Changes: models are cached per name (the reference kept one singleton and
ignored later model names), an empty candidate list never calls the model,
non-finite scores raise RerankError so the caller can fall back to vector
order, and ties keep the original (vector) order.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    pass


class Reranker:
    _cache: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        if model is not None:  # injected scorer (unit tests)
            self.model = model
            return
        with Reranker._lock:
            if model_name not in Reranker._cache:
                from sentence_transformers import CrossEncoder

                logger.info("Loading re-ranker model: %s", model_name)
                try:
                    Reranker._cache[model_name] = CrossEncoder(model_name, device="cpu")
                except Exception as exc:
                    raise RerankError(f"Cannot load reranker {model_name}: {exc}") from exc
        self.model = Reranker._cache[model_name]

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Return (original_index, score) pairs, best first, at most top_k."""
        if not documents:
            return []
        scores = self.model.predict([(query, doc) for doc in documents])
        scores = [float(s) for s in scores]
        if len(scores) != len(documents) or not all(math.isfinite(s) for s in scores):
            raise RerankError("Reranker returned missing or non-finite scores")
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in order[:top_k]]
