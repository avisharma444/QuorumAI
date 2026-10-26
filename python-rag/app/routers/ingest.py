"""POST /ingest — inspect → OCR route → layout chunks → BME preview → embeddings → Chroma."""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.chunking.layout_chunker import chunk_document
from app.inspection.inspector import DocumentError, NoTextExtracted, inspect_document
from app.models import IngestResponse
from app.ocr.interface import ExtractionError
from app.ocr.router import route_ocr
from app.sampling import bme_preview
from app.services import RagServices, ServiceUnavailable
from app.state import get_services
from app.telemetry import span

logger = logging.getLogger(__name__)
router = APIRouter()

DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def make_doc_id(filename: str, data: bytes) -> str:
    """Deterministic ID: sanitised file stem + 16 hex chars of SHA-256 over the WHOLE file.

    The reference used MD5 of only the first 4096 bytes, so two files with the
    same name and the same first 4 KB (common for PDFs from one template)
    received the same ID. The same bytes always produce the same ID, so a
    re-upload replaces the earlier version instead of duplicating it.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).stem).strip("._-")[:40] or "doc"
    return f"{stem}_{hashlib.sha256(data).hexdigest()[:16]}"


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(file: UploadFile = File(...), doc_id: Optional[str] = None, svc: RagServices = Depends(get_services)):
    try:
        store = svc.require_store()
    except ServiceUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    limit = svc.settings.max_upload_bytes
    raw = file.file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(413, f"File exceeds {limit // (1024 * 1024)} MiB")
    filename = file.filename or "upload"
    if doc_id is not None and not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(400, "doc_id must match [A-Za-z0-9._-]{1,64}")
    doc_id = doc_id or make_doc_id(filename, raw)

    started = time.perf_counter()
    try:
        with span("rag.inspect", **{"rag.bytes": len(raw)}):
            metadata = inspect_document(raw, filename)
        with span("rag.extract") as sp:
            ocr = route_ocr(raw, filename, metadata, svc.tesseract)
            sp.set_attribute("rag.ocr_method", str(ocr.ocr_method))
    except NoTextExtracted as exc:
        raise HTTPException(422, str(exc)) from exc
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ExtractionError as exc:
        logger.exception("Extraction failed for %s", filename)
        raise HTTPException(500, str(exc)) from exc
    with span("rag.chunk") as sp:
        chunks = chunk_document(ocr)
        sp.set_attribute("rag.chunks", len(chunks))
    if not chunks:
        raise HTTPException(422, "No text could be extracted from the document")
    preview = bme_preview([c.content for c in chunks])
    try:
        with span("rag.store"):
            count, replaced = store.ingest(chunks, doc_id)
    except Exception as exc:
        logger.exception("Embedding/storage failed for %s", doc_id)
        raise HTTPException(500, f"Embedding or storage failed: {type(exc).__name__}: {exc}") from exc
    elapsed = time.perf_counter() - started
    return IngestResponse(
        doc_id=doc_id,
        chunk_count=count,
        metadata=ocr.metadata,
        preview_text=preview,
        message=f"Successfully ingested {count} chunks using {ocr.ocr_method} extraction in {elapsed:.2f}s",
        ocr_method=ocr.ocr_method,
        chunk_types=dict(Counter(c.chunk_type.value for c in chunks)),
        replaced=replaced,
        embedding_model=svc.settings.embedding_model,
    )
