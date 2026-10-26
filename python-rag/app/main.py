"""RAG HTTP service. Launch with uvicorn app.main:app --port 8000."""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import DocumentStore, Embedder

app = FastAPI(title="CouncilAI document retrieval", version="0.1.0")
store = DocumentStore(os.getenv("DATABASE_PATH", "./documents.sqlite3"), Embedder())


class RetrieveRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "python-rag", "embedding_backend": store.embedder.backend}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    raw = await file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 10 MiB")
    try:
        return store.ingest(file.filename or "upload.txt", raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Unable to parse file: {type(exc).__name__}") from exc


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    try:
        return {"doc_id": req.doc_id, "chunks": store.retrieve(req.doc_id, req.question, req.top_k)}
    except KeyError as exc:
        raise HTTPException(404, "Document not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
