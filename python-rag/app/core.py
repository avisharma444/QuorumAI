"""Document processing and zero-API-cost retrieval core.

Default: hashed lexical vectors (lightweight baseline, NOT semantic embeddings).
Set EMBEDDING_BACKEND=sentence-transformer to use all-MiniLM-L6-v2 (384-dim).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

VECTOR_DIM = 384


@dataclass(frozen=True)
class Chunk:
    page: int
    text: str


def extract_pages(data: bytes, filename: str) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(1, data.decode("utf-8", errors="replace"))]
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    raise ValueError("Unsupported file. Upload PDF, TXT, or Markdown.")


def chunk_pages(pages: list[tuple[int, str]], size: int = 850, overlap: int = 120) -> list[Chunk]:
    """Create bounded, overlapping windows without crossing page boundaries."""
    if size < 100 or overlap >= size or overlap < 0:
        raise ValueError("Invalid chunk size or overlap")
    chunks: list[Chunk] = []
    for page, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                split = text.rfind(" ", start + size // 2, end)
                if split > start:
                    end = split
            segment = text[start:end].strip()
            if segment:
                chunks.append(Chunk(page=page, text=segment))
            if end >= len(text):
                break
            next_start = max(start + 1, end - overlap)
            start = next_start
    return chunks


class Embedder:
    """Lazy loading optional neural model; deterministic lexical fallback for tests."""
    def __init__(self, backend: str | None = None):
        self.backend = backend or os.getenv("EMBEDDING_BACKEND", "hash")
        self._model = None
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "sentence-transformer":
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()
        if self.backend != "hash":
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {self.backend}")
        result = []
        for text in texts:
            vector = [0.0] * VECTOR_DIM
            for token in re.findall(r"[\w]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
                idx = int.from_bytes(digest, "big") % VECTOR_DIM
                vector[idx] += 1.0
            length = math.sqrt(sum(v * v for v in vector)) or 1.0
            result.append([v / length for v in vector])
        return result


class DocumentStore:
    """SQLite persistence for the initial vertical slice; replace with Chroma in phase 2."""
    def __init__(self, filename: str, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self.lock = threading.RLock()
        with self.lock, self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS chunks (doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, page INTEGER, content TEXT, vector TEXT, PRIMARY KEY(doc_id, chunk_index))")
            self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id)")

    def ingest(self, filename: str, data: bytes) -> dict:
        pages = extract_pages(data, filename)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No readable text found; scanned PDF OCR is a phase-2 feature.")
        vectors = self.embedder.embed([c.text for c in chunks])
        doc_id = uuid.uuid4().hex[:16]
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO documents(doc_id,name) VALUES (?,?)", (doc_id, filename))
            self.conn.executemany("INSERT INTO chunks(doc_id,chunk_index,page,content,vector) VALUES (?,?,?,?,?)", [
                (doc_id, i, c.page, c.text, json.dumps(vector)) for i, (c, vector) in enumerate(zip(chunks, vectors))
            ])
        return {"doc_id": doc_id, "filename": filename, "pages": len(pages), "chunks": len(chunks), "embedding_backend": self.embedder.backend}

    def retrieve(self, doc_id: str, question: str, top_k: int = 4) -> list[dict]:
        if not doc_id.strip():
            raise ValueError("doc_id required")
        if not question.strip():
            raise ValueError("question required")
        with self.lock:
            rows = self.conn.execute("SELECT chunk_index,page,content,vector FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
        if not rows:
            raise KeyError(doc_id)
        query_vector = self.embedder.embed([question])[0]
        ranked = []
        for index, page, text, vector_data in rows:
            vec = json.loads(vector_data)
            score = sum(a * b for a, b in zip(query_vector, vec))
            ranked.append({"chunk_index": index, "page": page, "text": text, "score": round(score, 5)})
        ranked.sort(key=lambda c: (-c["score"], c["chunk_index"]))
        return ranked[:max(1, min(top_k, 10))]
