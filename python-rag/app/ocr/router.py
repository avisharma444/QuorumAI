"""Adaptive extraction routing (reference app/ocr/router.py).

Reference decision table, unchanged:
  text layer and not scanned → layout_aware if tables else direct_text
  otherwise, tables          → layout_aware
  otherwise                  → tesseract
Plain .txt/.md files (an extension of the reference) go to plain_text.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from app.inspection.inspector import TEXT_EXTS, NoTextExtracted, open_pdf
from app.models import ChunkType, DocumentMetadata, OCRBlock, OCRResult
from app.ocr.interface import OCRBackend
from app.ocr.layout_aware import LayoutAwareOCR
from app.ocr.tesseract import TesseractOCR

logger = logging.getLogger(__name__)


class DirectTextExtractor(OCRBackend):
    """PyPDF2 text-layer extraction; paragraphs split on blank lines (reference)."""

    def name(self) -> str:
        return "direct_text"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        reader = open_pdf(file_bytes)
        blocks: list[OCRBlock] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Page %d text extraction failed: %s", page_num, exc)
                continue
            for para in text.split("\n\n"):
                if para.strip():
                    blocks.append(OCRBlock(content=para.strip(), block_type=ChunkType.PARAGRAPH, page_number=page_num))
        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(file_name=filename, file_type=".pdf", page_count=len(reader.pages)),
            ocr_method=self.name(),
        )


class PlainTextExtractor(OCRBackend):
    """Extension: UTF-8 text/Markdown. Blank lines separate blocks; '#' lines are headings."""

    def name(self) -> str:
        return "plain_text"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        text = file_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
        blocks: list[OCRBlock] = []
        for raw in re.split(r"\n\s*\n", text):
            block = raw.strip()
            if not block:
                continue
            kind = ChunkType.PARAGRAPH
            if block.startswith("#") and "\n" not in block:
                kind, block = ChunkType.HEADING, block.lstrip("#").strip()
            elif block.startswith(("- ", "* ", "• ")) or re.match(r"^\d+\.\s", block):
                kind = ChunkType.LIST
            if block:
                blocks.append(OCRBlock(content=re.sub(r"[ \t]+", " ", block), block_type=kind, page_number=1))
        ext = Path(filename).suffix.lower()
        return OCRResult(blocks=blocks, metadata=DocumentMetadata(file_name=filename, file_type=ext, page_count=1), ocr_method=self.name())


def choose_backend(filename: str, metadata: DocumentMetadata, tesseract: TesseractOCR | None = None) -> OCRBackend:
    tesseract = tesseract or TesseractOCR()
    if Path(filename).suffix.lower() in TEXT_EXTS:
        return PlainTextExtractor()
    if metadata.has_text_layer and not metadata.is_scanned:
        return LayoutAwareOCR(tesseract) if metadata.has_tables else DirectTextExtractor()
    if metadata.has_tables:
        return LayoutAwareOCR(tesseract)
    return tesseract


def route_ocr(file_bytes: bytes, filename: str, metadata: DocumentMetadata, tesseract: TesseractOCR | None = None) -> OCRResult:
    """Run the backend the inspection metadata selects.

    The returned metadata is the inspection metadata plus the method that
    actually ran (a layout_aware route on an image runs Tesseract).
    """
    backend = choose_backend(filename, metadata, tesseract)
    logger.info("Routing %s to %s", filename, backend.name())
    result = backend.process(file_bytes, filename)
    result.ocr_method = result.ocr_method or backend.name()
    merged = metadata.model_copy()
    merged.ocr_method = result.ocr_method
    if result.metadata.page_count:
        merged.page_count = result.metadata.page_count
    merged.has_tables = merged.has_tables or result.metadata.has_tables
    result.metadata = merged
    if not result.blocks:
        raise NoTextExtracted(f"No text could be extracted from the document (method: {result.ocr_method})")
    return result
