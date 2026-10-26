"""RAG HTTP service (reference app/main.py). Run: uvicorn app.main:app --port 8000."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import state
from app.routers import embed, ingest, retrieve, retrieve_all, search
from app.telemetry import setup_tracing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.services.start_background_load()
    yield


app = FastAPI(title="CouncilAI RAG Service", version="1.0.0", lifespan=lifespan)
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(retrieve.router, tags=["Retrieval"])
app.include_router(retrieve_all.router, tags=["Retrieval"])
app.include_router(embed.router, tags=["Embedding"])
app.include_router(search.router, tags=["Web Search"])
setup_tracing(app)


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. See "ready" for model readiness."""
    return {"status": "ok", "service": "python-rag", **state.services.status()}


@app.get("/ready")
def ready():
    """Readiness: 200 only when the embedding model, reranker and Chroma store are loaded."""
    body = {"service": "python-rag", **state.services.status()}
    return JSONResponse(body, status_code=200 if body["ready"] else 503)
