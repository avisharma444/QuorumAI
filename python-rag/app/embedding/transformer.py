"""Transformer embeddings (reference app/embedding/transformer.py).

Same algorithm as the reference: Hugging Face AutoTokenizer + AutoModel,
last_hidden_state mean-pooled with the attention mask, truncation at 512
tokens, eval mode, no gradients, and a class-level cache so each model loads
once per process. Like the reference, vectors are NOT L2-normalised, and the
same text encoding is used for queries and documents (no BGE query prefix).
Consequence: Euclidean distance in Chroma also reflects vector length, so
distances are comparable within one model but are not cosine similarities.

Changes: inputs are embedded in mini-batches (the reference ran every chunk
of a document through one forward pass), the output dimension is checked
against the configured value, non-finite outputs are an error, and the
LangChain Embeddings base class is not used (same two methods).
"""
from __future__ import annotations

import logging
import math
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class TransformerEmbeddings:
    _cache: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str, expected_dimension: int | None = 384, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._torch = torch
        with TransformerEmbeddings._lock:
            if model_name not in TransformerEmbeddings._cache:
                logger.info("Loading embedding model: %s", model_name)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                except Exception as exc:
                    raise EmbeddingError(f"Cannot load embedding model {model_name}: {exc}") from exc
                model.eval()
                TransformerEmbeddings._cache[model_name] = (tokenizer, model)
        self.tokenizer, self.model = TransformerEmbeddings._cache[model_name]
        self.dimension = int(self.model.config.hidden_size)
        if expected_dimension and self.dimension != expected_dimension:
            raise EmbeddingError(f"{model_name} produces {self.dimension}-dim vectors; configuration expects {expected_dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        torch = self._torch
        inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = (summed / counts).tolist()
        for vec in pooled:
            if len(vec) != self.dimension or not all(math.isfinite(v) for v in vec):
                raise EmbeddingError("Embedding model returned an invalid vector")
        return pooled
