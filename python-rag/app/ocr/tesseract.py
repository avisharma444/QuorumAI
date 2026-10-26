"""Tesseract OCR backend for raster images and scanned PDFs (reference app/ocr/tesseract.py).

Kept from the reference: image_to_data, grouping words by Tesseract block,
averaging word confidences (/100), and 300 dpi rasterization with pdf2image.

Changed:
* Scanned PDFs use image_to_data per page too, so confidences are real; the
  reference used image_to_string with a fixed 0.8.
* No PyPDF2 fallback when pdf2image/Poppler is missing: that fallback returned
  embedded text labelled as OCR. Missing engines raise ExtractionError.
* Pages are rasterized one at a time (bounded memory) and capped.
* Multi-frame TIFFs are OCR'd page by page.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageSequence

from app.inspection.inspector import IMAGE_EXTS, DocumentError
from app.models import ChunkType, DocumentMetadata, OCRBlock, OCRResult
from app.ocr.interface import ExtractionError, OCRBackend

logger = logging.getLogger(__name__)


class TesseractOCR(OCRBackend):
    def __init__(self, dpi: int = 300, max_pages: int = 50):
        self.dpi = dpi
        self.max_pages = max_pages

    def name(self) -> str:
        return "tesseract"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        import pytesseract

        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise ExtractionError(f"Tesseract OCR engine is not available: {exc}") from exc

        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXTS:
            blocks, pages = self._process_image(file_bytes, pytesseract)
        elif ext == ".pdf":
            blocks, pages = self._process_pdf(file_bytes, pytesseract)
        else:
            raise ExtractionError(f"Tesseract cannot process {ext} files")
        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(file_name=filename, file_type=ext, page_count=pages),
            ocr_method=self.name(),
        )

    def _process_image(self, file_bytes: bytes, pytesseract) -> tuple[list[OCRBlock], int]:
        blocks: list[OCRBlock] = []
        with Image.open(io.BytesIO(file_bytes)) as img:
            frames = list(ImageSequence.Iterator(img))
            if len(frames) > self.max_pages:
                raise DocumentError(f"Image has {len(frames)} pages; the OCR limit is {self.max_pages}")
            for page_num, frame in enumerate(frames, start=1):
                blocks.extend(ocr_image_blocks(frame.convert("RGB"), page_num, pytesseract))
        return blocks, len(frames)

    def _process_pdf(self, file_bytes: bytes, pytesseract) -> tuple[list[OCRBlock], int]:
        try:
            from pdf2image import convert_from_bytes, pdfinfo_from_bytes
        except ImportError as exc:
            raise ExtractionError("pdf2image is not installed; scanned PDFs cannot be OCR'd") from exc
        try:
            pages = int(pdfinfo_from_bytes(file_bytes)["Pages"])
        except Exception as exc:  # Poppler missing or unreadable PDF
            raise ExtractionError(f"Cannot rasterize PDF (is Poppler installed?): {exc}") from exc
        if pages > self.max_pages:
            raise DocumentError(f"Scanned PDF has {pages} pages; the OCR limit is {self.max_pages}")
        blocks: list[OCRBlock] = []
        for page_num in range(1, pages + 1):
            images = convert_from_bytes(file_bytes, dpi=self.dpi, first_page=page_num, last_page=page_num)
            for img in images:
                blocks.extend(ocr_image_blocks(img.convert("RGB"), page_num, pytesseract))
                img.close()
        return blocks, pages


def ocr_image_blocks(img: Image.Image, page_num: int, pytesseract) -> list[OCRBlock]:
    """Group Tesseract words by block; confidence is the mean word confidence / 100."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    blocks: list[OCRBlock] = []
    words: list[str] = []
    confs: list[float] = []
    boxes: list[tuple[int, int, int, int]] = []
    current = None

    def flush():
        text = " ".join(words).strip()
        if text:
            x0 = min(b[0] for b in boxes)
            top = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes)
            bottom = max(b[3] for b in boxes)
            blocks.append(OCRBlock(
                content=text,
                block_type=ChunkType.PARAGRAPH,
                page_number=page_num,
                bbox=[float(x0), float(top), float(x1), float(bottom)],
                confidence=round(max(0.0, sum(confs) / len(confs) / 100.0), 4),
            ))

    for i in range(len(data["text"])):
        block_id = (data["page_num"][i], data["block_num"][i])
        if block_id != current:
            flush()
            words, confs, boxes = [], [], []
            current = block_id
        text = str(data["text"][i]).strip()
        conf = float(data["conf"][i])
        if text and conf > 0:  # reference: conf -1 marks non-word layout entries
            words.append(text)
            confs.append(conf)
            left, top = int(data["left"][i]), int(data["top"][i])
            boxes.append((left, top, left + int(data["width"][i]), top + int(data["height"][i])))
    flush()
    return blocks
