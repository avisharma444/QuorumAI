"""Layout-aware PDF extraction with pdfplumber (reference app/ocr/layout_aware.py).

Kept from the reference: tables extracted with pdfplumber and formatted as
markdown-style rows, table characters excluded from paragraph text, the
paragraph split rule (a line ending in "." followed by a capitalised line
starts a new paragraph) and the heading/caption/list heuristics.

Changed:
* Reading order: the reference appended all of a page's tables before its
  paragraphs. Here tables and paragraphs are merged by vertical position on
  the page, using pdfplumber's line and table coordinates, and each block
  records its bbox. Within a horizontal band the order is top-to-bottom,
  then left-to-right; true multi-column reading order is not reconstructed.
* Paragraph breaks also use vertical gaps and font-size changes between
  lines (pdfplumber exposes these; the reference relied on blank lines,
  which extract_text does not produce). This is what lets a heading be
  separated from the paragraph under it.
* Captions are checked before headings (the reference classified short
  captions such as "Table 2: Results" as headings).
* No invented confidence: pdfplumber reports none, so blocks carry None
  (the reference used fixed 0.9/0.95).
"""
from __future__ import annotations

import io
import logging
import statistics
from pathlib import Path

from app.models import ChunkType, DocumentMetadata, OCRBlock, OCRResult
from app.ocr.interface import ExtractionError, OCRBackend

logger = logging.getLogger(__name__)

CAPTION_PREFIXES = ("figure", "fig.", "fig ", "table", "chart", "diagram")


class LayoutAwareOCR(OCRBackend):
    def __init__(self, tesseract=None):
        self._tesseract = tesseract

    def name(self) -> str:
        return "layout_aware"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        ext = Path(filename).suffix.lower()
        if ext != ".pdf":
            # Reference: non-PDFs fall back to Tesseract. The result keeps
            # ocr_method "tesseract" so the method is reported truthfully.
            from app.ocr.tesseract import TesseractOCR

            return (self._tesseract or TesseractOCR()).process(file_bytes, filename)
        return self._process_pdf(file_bytes, filename)

    def _process_pdf(self, file_bytes: bytes, filename: str) -> OCRResult:
        import pdfplumber

        blocks: list[OCRBlock] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    blocks.extend(self._page_blocks(page, page_num))
        except Exception as exc:
            raise ExtractionError(f"Layout-aware extraction failed: {exc}") from exc
        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(
                file_name=filename, file_type=".pdf", page_count=page_count,
                has_tables=any(b.block_type == ChunkType.TABLE for b in blocks),
            ),
            ocr_method=self.name(),
        )

    def _page_blocks(self, page, page_num: int) -> list[OCRBlock]:
        positioned: list[tuple[float, float, OCRBlock]] = []
        table_bboxes = []
        for table in page.find_tables():
            text = format_table(table.extract())
            if text.strip():
                x0, top, x1, bottom = table.bbox
                table_bboxes.append(table.bbox)
                positioned.append((top, x0, OCRBlock(
                    content=text, block_type=ChunkType.TABLE, page_number=page_num,
                    bbox=[float(x0), float(top), float(x1), float(bottom)],
                )))

        def not_in_table(obj) -> bool:
            if obj.get("object_type") != "char":
                return True
            for tx0, ttop, tx1, tbottom in table_bboxes:
                if tx0 <= obj["x0"] <= tx1 and ttop <= obj["top"] <= tbottom:
                    return False
            return True

        text_page = page.filter(not_in_table) if table_bboxes else page
        lines = text_page.extract_text_lines(strip=True, return_chars=True)
        for para in group_lines(lines):
            positioned.append((para["top"], para["x0"], OCRBlock(
                content=para["text"], block_type=classify_block(para["text"]), page_number=page_num,
                bbox=[para["x0"], para["top"], para["x1"], para["bottom"]],
            )))
        positioned.sort(key=lambda p: (round(p[0], 1), p[1]))
        return [b for _, _, b in positioned]


def _line_size(line: dict) -> float:
    sizes = [c.get("size", 0) for c in line.get("chars", []) if c.get("text", "").strip()]
    return round(statistics.median(sizes), 1) if sizes else 0.0


def group_lines(lines: list[dict]) -> list[dict]:
    """Group text lines into paragraphs.

    A new paragraph starts when (reference rule) the previous line ends with
    "." and this line starts with a capital, or (layout signals) the vertical
    gap exceeds 1.5x the median line height, or the font size changes.
    """
    lines = [ln for ln in lines if ln.get("text", "").strip()]
    if not lines:
        return []
    heights = [ln["bottom"] - ln["top"] for ln in lines if ln["bottom"] > ln["top"]]
    median_h = statistics.median(heights) if heights else 10.0
    paras: list[dict] = []
    current: list[dict] = []

    def flush():
        if current:
            paras.append({
                "text": " ".join(ln["text"].strip() for ln in current),
                "top": float(min(ln["top"] for ln in current)),
                "bottom": float(max(ln["bottom"] for ln in current)),
                "x0": float(min(ln["x0"] for ln in current)),
                "x1": float(max(ln["x1"] for ln in current)),
            })

    for line in lines:
        text = line["text"].strip()
        if current:
            prev = current[-1]
            gap = line["top"] - prev["bottom"]
            sentence_break = prev["text"].rstrip().endswith(".") and text[0].isupper()
            layout_break = gap > 1.5 * median_h or abs(_line_size(line) - _line_size(prev)) > 0.5
            if sentence_break or layout_break:
                flush()
                current = []
        current.append(line)
    flush()
    return paras


def format_table(table: list[list]) -> str:
    """Markdown-style rows with a separator after the header (reference format)."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    if len(rows) > 1:
        rows.insert(1, "| " + " | ".join(["---"] * len(table[0])) + " |")
    return "\n".join(rows)


def classify_block(text: str) -> ChunkType:
    stripped = text.strip()
    if not stripped:
        return ChunkType.UNKNOWN
    if stripped.lower().startswith(CAPTION_PREFIXES):
        return ChunkType.CAPTION
    if stripped.startswith(("•", "-", "*", "1.", "2.", "3.")):
        return ChunkType.LIST
    if len(stripped) < 100 and not stripped.endswith(".") and stripped[0].isupper() and len(stripped.split()) <= 10:
        return ChunkType.HEADING
    return ChunkType.PARAGRAPH
