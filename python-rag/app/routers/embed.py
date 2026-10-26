"""POST /embed — one query embedding from the configured model (used later by the semantic cache)."""
from fastapi import APIRouter, Depends, HTTPException

from app.models import EmbedRequest, EmbedResponse
from app.services import RagServices, ServiceUnavailable
from app.state import get_services

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
def embed_text(req: EmbedRequest, svc: RagServices = Depends(get_services)):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    try:
        svc.require_store()
        vector = svc.embedder.embed_query(req.text)
    except ServiceUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return EmbedResponse(embedding=vector, model=svc.settings.embedding_model, dimension=len(vector))
