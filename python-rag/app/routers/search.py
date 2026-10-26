"""POST /search — live web-search grounding (unchanged behaviour from checkpoint 4)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.retrieval.search_agent import WebSearchAgent
from app.state import get_search_agent

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    max_results: int = Field(default=3, ge=1, le=10)


@router.post("/search")
def web_search(req: SearchRequest, agent: WebSearchAgent = Depends(get_search_agent)) -> dict:
    """Always 200 with a status ("ok" | "no_results" | "search_failed")."""
    return agent.search(req.query, req.max_results).as_dict()
