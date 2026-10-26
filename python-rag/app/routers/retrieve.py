"""POST /retrieve — vector search over one document, then optional cross-encoder rerank."""
from fastapi import APIRouter, Depends, HTTPException

from app.models import RetrieveRequest, RetrieveResponse
from app.retrieval.chroma_store import DISTANCE_METRIC, DocumentNotFound
from app.services import RagServices, ServiceUnavailable
from app.state import get_services

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(req: RetrieveRequest, svc: RagServices = Depends(get_services)):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if not req.doc_id:
        # The reference fell back to a "default" collection that nothing writes to.
        raise HTTPException(400, "doc_id is required")
    try:
        store = svc.require_store()
        result = store.retrieve(req.question, req.doc_id, top_k=req.top_k, rerank=req.rerank, reranker=svc.reranker)
    except ServiceUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except DocumentNotFound as exc:
        raise HTTPException(404, f"Document {req.doc_id} not found") from exc
    return RetrieveResponse(
        chunks=result.chunks, question=req.question, doc_id=req.doc_id,
        reranked=result.reranked, rerank_error=result.rerank_error,
        candidates=result.candidates, distance_metric=DISTANCE_METRIC,
    )
