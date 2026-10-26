"""Process-wide service objects. Tests replace these before creating a TestClient."""
from __future__ import annotations

from app.config import get_settings
from app.retrieval.search_agent import WebSearchAgent
from app.services import RagServices

services: RagServices = RagServices(get_settings())
search_agent: WebSearchAgent = WebSearchAgent()


def get_services() -> RagServices:
    return services


def get_search_agent() -> WebSearchAgent:
    return search_agent
