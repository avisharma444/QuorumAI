"""POST /retrieve-all — every chunk of a document in document order."""
from fastapi import APIRouter, Depends, HTTPException

from app.models import RetrieveAllRequest, RetrieveAllResponse
from app.retrieval.chroma_store import DocumentNotFound
from app.services import RagServices, ServiceUnavailable
from app.state import get_services

router = APIRouter()


@router.post("/retrieve-all", response_model=RetrieveAllResponse)
def retrieve_all_chunks(req: RetrieveAllRequest, svc: RagServices = Depends(get_services)):
    try:
        chunks = svc.require_store().get_all(req.doc_id)
    except ServiceUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except DocumentNotFound as exc:
        raise HTTPException(404, f"Document {req.doc_id} not found") from exc
    return RetrieveAllResponse(chunks=chunks, doc_id=req.doc_id, total_chunks=len(chunks))
