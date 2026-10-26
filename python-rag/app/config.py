"""Settings for the RAG service: config.yaml (rag section) with environment overrides.

Mirrors the reference app/config.py (same keys, paths and env names) without
the pydantic-settings dependency.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CONFIG_PATHS = ["/app/config.yaml", "config.yaml", "../config.yaml", "../../config.yaml"]


class Settings(BaseModel):
    chroma_db_path: str = "/data/chroma_db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384  # verified against the loaded model
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_ocr_pages: int = 50
    ocr_dpi: int = 300  # reference value


@lru_cache()
def get_settings() -> Settings:
    rag: dict = {}
    for path in CONFIG_PATHS:
        if Path(path).exists():
            try:
                with open(path) as f:
                    rag = (yaml.safe_load(f) or {}).get("rag", {}) or {}
                logger.info("Loaded configuration from %s", path)
                break
            except (OSError, yaml.YAMLError) as exc:
                raise RuntimeError(f"Invalid config file {path}: {exc}") from exc
    env = os.getenv
    return Settings(
        chroma_db_path=env("CHROMA_DB_PATH", rag.get("chroma_db_path", "/data/chroma_db")),
        embedding_model=env("EMBEDDING_MODEL", rag.get("embedding_model", "BAAI/bge-small-en-v1.5")),
        embedding_dimension=int(env("EMBEDDING_DIMENSION", rag.get("embedding_dimension", 384))),
        embedding_batch_size=int(env("EMBEDDING_BATCH_SIZE", rag.get("embedding_batch_size", 32))),
        reranker_model=env("RERANKER_MODEL", rag.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        max_ocr_pages=int(env("MAX_OCR_PAGES", rag.get("max_ocr_pages", 50))),
        ocr_dpi=int(env("OCR_DPI", rag.get("ocr_dpi", 300))),
    )
