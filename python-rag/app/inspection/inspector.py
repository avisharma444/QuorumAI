"""Document inspection (reference app/inspection/inspector.py).

Produces DocumentMetadata that drives OCR routing: text layer, scanned,
tables, multicolumn, page count.

Kept from the reference: the text-coverage thresholds (a page "has text" if
it holds > 50 characters; coverage > 0.5 → text layer, < 0.3 → scanned), the
pipe/tab table heuristic, the multicolumn heuristic and the image aspect-ratio
table heuristic.

Changed:
* Empty, corrupt, encrypted and unsupported inputs raise DocumentError. The
  reference marked a corrupt PDF as "scanned" and sent it to OCR.
* Tables are also detected with pdfplumber's ruling-line table finder. The
  pipe/tab heuristic alone misses ordinary PDF tables (they rarely contain
  "|" or tab characters), which sent them to plain text extraction.
* .txt/.md are accepted as an extension of the reference (which supports
  only PDFs and images) and are labelled as plain text, not OCR.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from app.models import DocumentMetadata

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")
TEXT_EXTS = (".txt", ".md")
SUPPORTED_EXTS = (".pdf",) + IMAGE_EXTS + TEXT_EXTS
TABLE_SCAN_PAGES = 20  # pdfplumber table detection is bounded to the first pages


class DocumentError(ValueError):
    """The upload cannot be processed (empty, corrupt, encrypted, unsupported)."""


class NoTextExtracted(DocumentError):
    """The document opened, but the selected backend found no text (HTTP 422, as in the reference)."""


def inspect_document(file_bytes: bytes, filename: str) -> DocumentMetadata:
    if not file_bytes:
        raise DocumentError("Empty file")
    ext = Path(filename).suffix.lower()
    metadata = DocumentMetadata(file_name=filename, file_type=ext)
    if ext == ".pdf":
        return _inspect_pdf(file_bytes, metadata)
    if ext in IMAGE_EXTS:
        return _inspect_image(file_bytes, metadata)
    if ext in TEXT_EXTS:
        metadata.has_text_layer = True
        metadata.page_count = 1
        return metadata
    raise DocumentError(f"Unsupported file type {ext or '(none)'}; upload PDF, PNG, JPG, TIFF, BMP, TXT or MD")


def open_pdf(file_bytes: bytes) -> PdfReader:
    """Open a PDF with PyPDF2, raising DocumentError for unreadable input."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception as exc:  # unsupported encryption algorithm
                raise DocumentError(f"Encrypted PDF cannot be read: {exc}") from exc
            if not ok:
                raise DocumentError("Encrypted PDF requires a password")
        if len(reader.pages) == 0:
            raise DocumentError("PDF has no pages")
        return reader
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise DocumentError(f"Corrupt or unreadable PDF: {exc}") from exc


def _page_text(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception as exc:  # malformed content streams on one page
        logger.warning("Text extraction failed on a page: %s", exc)
        return ""


def _inspect_pdf(file_bytes: bytes, metadata: DocumentMetadata) -> DocumentMetadata:
    reader = open_pdf(file_bytes)
    metadata.page_count = len(reader.pages)
    texts = [_page_text(p) for p in reader.pages]

    total_chars = 0
    pages_with_text = 0
    table_indicators = 0
    for text in texts:
        chars = len(text.strip())
        total_chars += chars
        if chars > 50:
            pages_with_text += 1
        for line in text.split("\n"):
            if line.count("|") >= 2 or line.count("\t") >= 2:
                table_indicators += 1

    if pages_with_text > 0:
        coverage = pages_with_text / metadata.page_count
        metadata.has_text_layer = coverage > 0.5
        metadata.is_scanned = coverage < 0.3
    else:
        metadata.has_text_layer = False
        metadata.is_scanned = True

    metadata.has_tables = table_indicators >= 3
    if not metadata.has_tables and pages_with_text > 0:
        metadata.has_tables = _pdfplumber_finds_tables(file_bytes)

    if total_chars > 0 and pages_with_text > 0:
        avg_chars_per_page = total_chars / pages_with_text
        lines = [line for line in texts[0].split("\n") if line.strip()]
        if lines:
            avg_line_length = sum(len(line) for line in lines) / len(lines)
            metadata.is_multicolumn = avg_line_length < 60 and avg_chars_per_page > 500
    return metadata


def _pdfplumber_finds_tables(file_bytes: bytes) -> bool:
    import pdfplumber

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:TABLE_SCAN_PAGES]:
                for table in page.find_tables():
                    rows = table.extract()
                    if len(rows) >= 2 and max((len(r) for r in rows), default=0) >= 2:
                        return True
    except Exception as exc:
        logger.warning("pdfplumber table detection failed: %s", exc)
    return False


def _inspect_image(file_bytes: bytes, metadata: DocumentMetadata) -> DocumentMetadata:
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(file_bytes)) as img:
            width, height = img.size
            metadata.page_count = getattr(img, "n_frames", 1) or 1
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise DocumentError(f"Corrupt or unsupported image: {exc}") from exc
    metadata.is_scanned = True
    metadata.has_text_layer = False
    # Reference heuristic: wide images with structured content might be tables.
    metadata.has_tables = width > height * 1.5
    return metadata
