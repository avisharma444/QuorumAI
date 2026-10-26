"""Model and store lifecycle.

The reference loaded both models synchronously at startup. Here they load
in a background thread so the process can answer liveness checks while
weights download, and /ready reports ready only once the embedding model,
reranker and Chroma store are all initialised. Load failures are recorded
and reported; nothing falls back to non-neural retrieval.
"""
from __future__ import annotations

import logging
import shutil
import threading

from app.config import Settings

logger = logging.getLogger(__name__)


class ServiceUnavailable(RuntimeError):
    pass


class RagServices:
    def __init__(self, settings: Settings, embedder=None, reranker=None, store=None, tesseract=None, autoload: bool = True):
        from app.ocr.tesseract import TesseractOCR

        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        self.tesseract = tesseract or TesseractOCR(dpi=settings.ocr_dpi, max_pages=settings.max_ocr_pages)
        self.autoload = autoload
        self.errors: dict[str, str] = {}
        self.loading = False
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return self.embedder is not None and self.reranker is not None and self.store is not None

    def start_background_load(self) -> None:
        if not self.autoload or self.ready or self._thread:
            return
        self.loading = True
        self._thread = threading.Thread(target=self.load, name="model-loader", daemon=True)
        self._thread.start()

    def load(self) -> None:
        from app.embedding.transformer import TransformerEmbeddings
        from app.retrieval.chroma_store import ChromaStore
        from app.retrieval.reranker import Reranker

        s = self.settings
        try:
            try:
                self.embedder = TransformerEmbeddings(s.embedding_model, s.embedding_dimension, s.embedding_batch_size)
                self.store = ChromaStore(s.chroma_db_path, self.embedder, embedding_model=s.embedding_model)
            except Exception as exc:
                self.errors["embedding"] = f"{type(exc).__name__}: {exc}"
                logger.error("Embedding model / store initialisation failed: %s", exc)
            try:
                self.reranker = Reranker(s.reranker_model)
            except Exception as exc:
                self.errors["reranker"] = f"{type(exc).__name__}: {exc}"
                logger.error("Reranker initialisation failed: %s", exc)
        finally:
            self.loading = False
        if self.ready:
            logger.info("RAG models ready: %s, %s", s.embedding_model, s.reranker_model)

    def require_store(self):
        if self.store is None or self.embedder is None:
            state = "loading" if self.loading else self.errors.get("embedding", "not initialised")
            raise ServiceUnavailable(f"Embedding model / vector store not ready: {state}")
        return self.store

    def status(self) -> dict:
        def model(name, obj, key):
            if obj is not None:
                return {"name": name, "status": "ready"}
            if key in self.errors:
                return {"name": name, "status": "failed", "error": self.errors[key]}
            return {"name": name, "status": "loading" if self.loading else "not_loaded"}

        s = self.settings
        tesseract = None
        try:
            import pytesseract

            tesseract = str(pytesseract.get_tesseract_version())
        except Exception:
            pass
        return {
            "ready": self.ready,
            "models": {
                "embedding": {**model(s.embedding_model, self.embedder, "embedding"), "dimension": s.embedding_dimension},
                "reranker": model(s.reranker_model, self.reranker, "reranker"),
            },
            "vector_store": {"type": "chromadb", "path": s.chroma_db_path, "status": "ready" if self.store else "unavailable"},
            "ocr": {"tesseract": tesseract, "poppler": shutil.which("pdftoppm") is not None},
        }
