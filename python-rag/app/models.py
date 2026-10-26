"""Pydantic models for the RAG service (reference app/models.py).

Differences from the reference, each to avoid invented data:
* OCRBlock.confidence is optional: only Tesseract reports a real confidence;
  the reference hard-coded 1.0 / 0.95 / 0.9 / 0.8 for other extractors.
* OCRBlock.bbox is filled where pdfplumber provides positions.
* DocumentMetadata adds ocr_method and encrypted flags for diagnostics.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    HEADING = "heading"
    LIST = "list"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    has_text_layer: bool = False
    is_scanned: bool = False
    has_tables: bool = False
    is_multicolumn: bool = False
    page_count: int = 0
    file_type: str = ""
    file_name: str = ""
    ocr_method: str = ""


class Chunk(BaseModel):
    content: str
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    page_number: int = 0  # 1-based; 0 means unknown
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)


class OCRBlock(BaseModel):
    content: str
    block_type: ChunkType = ChunkType.PARAGRAPH
    page_number: int = 0
    bbox: Optional[list[float]] = None  # [x0, top, x1, bottom] in PDF points or pixels
    confidence: Optional[float] = None  # OCR engine confidence in [0, 1], if the engine reports one


class OCRResult(BaseModel):
    blocks: list[OCRBlock] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    ocr_method: str = ""


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    metadata: DocumentMetadata
    preview_text: str = ""
    message: str = "Document ingested successfully"
    # Extensions used by the Go backend and UI.
    ocr_method: str = ""
    chunk_types: dict[str, int] = Field(default_factory=dict)
    replaced: bool = False  # an earlier ingestion of the same doc_id was replaced
    embedding_model: str = ""


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    doc_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True


class RetrieveResponse(BaseModel):
    chunks: list[Chunk]
    question: str
    doc_id: Optional[str] = None
    # Diagnostics: whether cross-encoder scores ordered the chunks, and why not.
    reranked: bool = False
    rerank_error: Optional[str] = None
    candidates: int = 0
    distance_metric: str = "l2"


class RetrieveAllRequest(BaseModel):
    doc_id: str


class RetrieveAllResponse(BaseModel):
    chunks: list[Chunk]
    doc_id: str
    total_chunks: int


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str = ""
    dimension: int = 0
