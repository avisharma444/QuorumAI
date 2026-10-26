"""Persistent ChromaDB store (reference app/retrieval/chroma_store.py).

Kept from the reference: one Chroma collection per document, chunk text plus
doc_id / chunk_type / page_number / chunk_index / extra metadata, the
default L2 ("l2") space, and over-fetching 3 x top_k vector candidates for
the cross-encoder.

Changed (correctness):
* Native chromadb client instead of the LangChain wrapper. The wrapper's
  "relevance score" formula assumes unit-length vectors, which these mean-
  pooled embeddings are not; we report the raw L2 distance instead.
* Collection names are "doc_" + 40 hex chars of SHA-256(doc_id). The
  reference truncated "doc_<sanitised id>" to 63 characters, so long or
  differently punctuated IDs could share a collection.
* Unknown documents raise DocumentNotFound. The reference opened (and so
  created) an empty collection and reported success with no chunks.
* Re-ingesting a doc_id replaces it atomically: embeddings are computed
  first, written to a staging collection, and only then swapped in; a
  failure leaves the previous version untouched. The reference upserted by
  content-derived IDs, leaving stale chunks when content changed.
* Reranking also runs when there are fewer candidates than top_k (it still
  reorders them); the reference reranked only when it had to truncate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from app.models import Chunk, ChunkType

logger = logging.getLogger(__name__)

DISTANCE_METRIC = "l2"  # Chroma default, as in the reference
_STAGING = re.compile(r"^doc_[0-9a-f]{40}_s[0-9a-f]{8}$")


class DocumentNotFound(KeyError):
    pass


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    reranked: bool = False
    rerank_error: Optional[str] = None
    candidates: int = 0
    timings_ms: dict = field(default_factory=dict)


class ChromaStore:
    def __init__(self, persist_directory: str, embeddings, client=None, embedding_model: str = ""):
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.embedding_model = embedding_model or getattr(embeddings, "model_name", "")
        self.client = client or chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        self._cleanup_staging()

    # ── naming and lookup ───────────────────────────────────────────

    @staticmethod
    def collection_name(doc_id: str) -> str:
        if not doc_id:
            raise ValueError("doc_id is required")
        return "doc_" + hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:40]

    def _lock(self, doc_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[doc_id]

    def _names(self) -> set[str]:
        return {c if isinstance(c, str) else c.name for c in self.client.list_collections()}

    def _get(self, doc_id: str):
        name = self.collection_name(doc_id)
        if name not in self._names():
            raise DocumentNotFound(doc_id)
        return self.client.get_collection(name, embedding_function=None)

    def exists(self, doc_id: str) -> bool:
        return self.collection_name(doc_id) in self._names()

    def _cleanup_staging(self) -> None:
        for name in self._names():
            if _STAGING.match(name):
                logger.warning("Removing leftover staging collection %s", name)
                self.client.delete_collection(name)

    # ── write ───────────────────────────────────────────────────────

    def ingest(self, chunks: list[Chunk], doc_id: str) -> tuple[int, bool]:
        """Embed and store chunks; returns (count, replaced_existing)."""
        if not chunks:
            raise ValueError("No chunks to ingest")
        texts = [c.content for c in chunks]
        vectors = self.embeddings.embed_documents(texts)  # before any store change
        final = self.collection_name(doc_id)
        staging = f"{final}_s{uuid.uuid4().hex[:8]}"
        with self._lock(doc_id):
            col = self.client.create_collection(
                staging,
                metadata={"hnsw:space": DISTANCE_METRIC, "doc_id": doc_id, "embedding_model": self.embedding_model},
                embedding_function=None,
            )
            try:
                col.add(
                    ids=[f"{c.chunk_index:06d}" for c in chunks],
                    embeddings=vectors,
                    documents=texts,
                    metadatas=[self._to_metadata(c, doc_id) for c in chunks],
                )
            except Exception:
                self.client.delete_collection(staging)
                raise
            replaced = final in self._names()
            if replaced:
                self.client.delete_collection(final)
            col.modify(name=final)
        logger.info("Ingested %d chunks for doc_id=%s (replaced=%s)", len(chunks), doc_id, replaced)
        return len(chunks), replaced

    def delete(self, doc_id: str) -> bool:
        with self._lock(doc_id):
            if not self.exists(doc_id):
                return False
            self.client.delete_collection(self.collection_name(doc_id))
            return True

    # ── read ────────────────────────────────────────────────────────

    def retrieve(self, query: str, doc_id: str, top_k: int = 5, rerank: bool = True, reranker=None) -> RetrievalResult:
        import time

        col = self._get(doc_id)
        total = col.count()
        if total == 0:
            return RetrievalResult(chunks=[])
        fetch_k = min(top_k * 3, total)
        t0 = time.perf_counter()
        query_vec = self.embeddings.embed_query(query)
        t1 = time.perf_counter()
        res = col.query(query_embeddings=[query_vec], n_results=fetch_k, include=["documents", "metadatas", "distances"])
        t2 = time.perf_counter()
        candidates = []
        for rank, (text, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
            chunk = self._from_metadata(text, meta)
            chunk.metadata["vector_distance"] = round(float(dist), 6)
            chunk.metadata["vector_rank"] = rank
            candidates.append(chunk)
        result = RetrievalResult(chunks=candidates[:top_k], candidates=len(candidates))
        result.timings_ms = {"embed_query": round((t1 - t0) * 1000, 2), "vector_search": round((t2 - t1) * 1000, 2)}
        if rerank and len(candidates) >= 2:
            if reranker is None:
                result.rerank_error = "reranker unavailable"
            else:
                t3 = time.perf_counter()
                try:
                    order = reranker.rerank(query, [c.content for c in candidates], top_k=top_k)
                    result.chunks = []
                    for idx, score in order:
                        chunk = candidates[idx]
                        chunk.metadata["rerank_score"] = round(float(score), 6)
                        result.chunks.append(chunk)
                    result.reranked = True
                except Exception as exc:
                    result.rerank_error = f"{type(exc).__name__}: {exc}"
                result.timings_ms["rerank"] = round((time.perf_counter() - t3) * 1000, 2)
            if result.rerank_error:
                logger.warning("Re-ranking failed for doc_id=%s, using vector order: %s", doc_id, result.rerank_error)
        return result

    def get_all(self, doc_id: str) -> list[Chunk]:
        col = self._get(doc_id)
        res = col.get(include=["documents", "metadatas"])
        chunks = [self._from_metadata(t, m) for t, m in zip(res["documents"], res["metadatas"])]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    # ── metadata mapping (Chroma accepts only str/int/float/bool) ──

    @staticmethod
    def _to_metadata(chunk: Chunk, doc_id: str) -> dict:
        meta = {"doc_id": doc_id, "chunk_type": chunk.chunk_type.value, "page_number": chunk.page_number, "chunk_index": chunk.chunk_index}
        for key, value in chunk.metadata.items():
            if value is None:
                continue
            meta[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        return meta

    @staticmethod
    def _from_metadata(text: str, meta: dict) -> Chunk:
        meta = dict(meta or {})
        extra = {k: v for k, v in meta.items() if k not in ("chunk_type", "page_number", "chunk_index")}
        if isinstance(extra.get("bbox"), str):
            try:
                extra["bbox"] = json.loads(extra["bbox"])
            except ValueError:
                pass
        return Chunk(
            content=text,
            chunk_type=ChunkType(meta.get("chunk_type", "unknown")),
            page_number=int(meta.get("page_number", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
            metadata=extra,
        )
