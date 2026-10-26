import io

import pytest

import fixtures as fx
from app.inspection.inspector import DocumentError, NoTextExtracted, inspect_document
from app.models import ChunkType, DocumentMetadata
from app.ocr.interface import ExtractionError
from app.ocr.layout_aware import LayoutAwareOCR, classify_block
from app.ocr.router import DirectTextExtractor, PlainTextExtractor, choose_backend, route_ocr
from app.ocr.tesseract import TesseractOCR


def run(data: bytes, name: str):
    meta = inspect_document(data, name)
    return meta, route_ocr(data, name, meta)


# ── Routing decision table (reference) ──────────────────────────────

@pytest.mark.parametrize("meta, name, expected", [
    (DocumentMetadata(has_text_layer=True), "a.pdf", "direct_text"),
    (DocumentMetadata(has_text_layer=True, has_tables=True), "a.pdf", "layout_aware"),
    (DocumentMetadata(is_scanned=True, has_tables=True), "a.pdf", "layout_aware"),
    (DocumentMetadata(is_scanned=True), "a.pdf", "tesseract"),
    (DocumentMetadata(), "a.pdf", "tesseract"),  # coverage between 0.3 and 0.5, no tables
    (DocumentMetadata(is_scanned=True), "a.png", "tesseract"),
    (DocumentMetadata(has_text_layer=True), "a.md", "plain_text"),
])
def test_choose_backend(meta, name, expected):
    assert choose_backend(name, meta).name() == expected


# ── Clean text-layer PDF ────────────────────────────────────────────

def test_clean_pdf_uses_direct_text():
    meta, result = run(fx.clean_pdf(), "policy.pdf")
    assert (meta.has_text_layer, meta.is_scanned, meta.has_tables, meta.page_count) == (True, False, False, 2)
    assert result.ocr_method == "direct_text" and result.metadata.ocr_method == "direct_text"
    text = {b.page_number: b.content for b in result.blocks}
    assert "Mirevale, a river city founded in 1842" in text[1].replace("\n", " ")
    assert "copper mines near Halvard" in text[2].replace("\n", " ")
    assert all(b.confidence is None for b in result.blocks), "text extraction is not OCR; no confidence"


# ── Table PDF ───────────────────────────────────────────────────────

def test_table_pdf_detected_by_layout_not_pipe_heuristic():
    data = fx.table_pdf()
    from PyPDF2 import PdfReader
    plain = PdfReader(io.BytesIO(data)).pages[0].extract_text()
    assert "|" not in plain, "the reference pipe heuristic alone cannot see this table"
    meta, result = run(data, "revenue.pdf")
    assert meta.has_tables and result.ocr_method == "layout_aware"
    table = [b for b in result.blocks if b.block_type == ChunkType.TABLE]
    assert len(table) == 1
    assert table[0].content == "| Region | Q1 | Q2 |\n| --- | --- | --- |\n| North | 120 | 135 |\n| South | 98 | 110 |"
    assert table[0].bbox and len(table[0].bbox) == 4


def test_table_keeps_reading_order():
    """The reference appended tables before paragraphs; text around the table must stay in place."""
    _, result = run(fx.table_pdf(), "revenue.pdf")
    kinds = [(b.block_type, b.content[:20]) for b in result.blocks]
    assert [k for k, _ in kinds] == [ChunkType.PARAGRAPH, ChunkType.TABLE, ChunkType.CAPTION, ChunkType.PARAGRAPH]
    assert result.blocks[0].content == fx.BEFORE_TABLE
    assert result.blocks[2].content == fx.TABLE_CAPTION
    assert result.blocks[3].content == fx.AFTER_TABLE
    tops = [b.bbox[1] for b in result.blocks]
    assert tops == sorted(tops)


def test_layout_separates_heading_from_paragraph():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    t = Table(fx.TABLE_ROWS)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build([
        Paragraph("Revenue Overview", styles["Heading1"]), Paragraph(fx.BEFORE_TABLE, styles["BodyText"]), Spacer(1, 12), t])
    _, result = run(buf.getvalue(), "h.pdf")
    assert [(b.block_type, b.content) for b in result.blocks[:2]] == [(ChunkType.HEADING, "Revenue Overview"), (ChunkType.PARAGRAPH, fx.BEFORE_TABLE)]


def test_caption_classified_before_heading():
    # Reference order made short captions headings.
    assert classify_block("Table 2 Quarterly results") == ChunkType.CAPTION
    assert classify_block("Quarterly Results") == ChunkType.HEADING
    assert classify_block("- first item") == ChunkType.LIST
    assert classify_block("A normal sentence that ends with a period.") == ChunkType.PARAGRAPH


# ── OCR ─────────────────────────────────────────────────────────────

@pytest.mark.ocr
def test_image_uses_tesseract_with_real_confidence():
    meta, result = run(fx.png_bytes(fx.text_image()), "invoice.png")
    assert meta.is_scanned and not meta.has_text_layer
    assert result.ocr_method == "tesseract"  # even though the wide image routes via layout_aware
    text = " ".join(b.content for b in result.blocks)
    assert "INVOICE 4471" in text and "1250" in text and "thirty days" in text
    assert all(0 < b.confidence <= 1 for b in result.blocks)
    assert all(b.bbox for b in result.blocks)


@pytest.mark.ocr
def test_scanned_pdf_ocr_per_page():
    data = fx.scanned_pdf([fx.OCR_LINES, ["Second page about rivers."]])
    meta, result = run(data, "scan.pdf")
    assert meta.is_scanned and meta.page_count == 2 and result.ocr_method == "tesseract"
    pages = {}
    for b in result.blocks:
        pages.setdefault(b.page_number, []).append(b.content)
    assert "INVOICE 4471" in " ".join(pages[1]) and "rivers" in " ".join(pages[2])
    assert len({b.confidence for b in result.blocks}) > 1 or result.blocks[0].confidence != 0.8, "confidence must come from Tesseract"


@pytest.mark.ocr
def test_scanned_pdf_page_limit():
    data = fx.scanned_pdf([["one"], ["two"], ["three"]])
    with pytest.raises(DocumentError, match="OCR limit"):
        TesseractOCR(max_pages=2).process(data, "big.pdf")


def test_missing_tesseract_is_an_error(monkeypatch):
    import pytesseract

    def boom():
        raise pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(pytesseract, "get_tesseract_version", boom)
    with pytest.raises(ExtractionError, match="Tesseract"):
        TesseractOCR().process(fx.png_bytes(fx.text_image()), "x.png")


@pytest.mark.ocr
def test_missing_pdf2image_never_falls_back_to_text_extraction(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pdf2image":
            raise ImportError("no pdf2image")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ExtractionError, match="pdf2image"):
        TesseractOCR().process(fx.clean_pdf(), "x.pdf")


# ── Errors ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("data, name, match", [
    (b"", "a.pdf", "Empty"),
    (fx.CORRUPT_PDF, "a.pdf", "Corrupt"),
    (b"not an image", "a.png", "Corrupt"),
    (b"PK\x03\x04", "a.docx", "Unsupported"),
    (b"data", "noextension", "Unsupported"),
])
def test_invalid_inputs_are_errors_not_scans(data, name, match):
    with pytest.raises(DocumentError, match=match):
        inspect_document(data, name)


def test_encrypted_pdf_is_an_error():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, encrypt="secret-password")
    c.drawString(100, 700, "Confidential text")
    c.save()
    with pytest.raises(DocumentError, match="(?i)encrypted|password"):
        inspect_document(buf.getvalue(), "locked.pdf")


@pytest.mark.ocr
def test_blank_pdf_reports_no_text():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.showPage()
    c.save()
    meta = inspect_document(buf.getvalue(), "blank.pdf")
    assert meta.is_scanned  # no text layer, so OCR is attempted...
    with pytest.raises(NoTextExtracted):  # ...and finds nothing: explicit error, not empty success
        route_ocr(buf.getvalue(), "blank.pdf", meta)


def test_plain_text_extension():
    data = b"# Zogland\n\nThe capital is Mirevale.\n\n- wool\n- copper\n"
    meta, result = run(data, "notes.md")
    assert result.ocr_method == "plain_text"
    assert [(b.block_type, b.content) for b in result.blocks] == [
        (ChunkType.HEADING, "Zogland"), (ChunkType.PARAGRAPH, "The capital is Mirevale."), (ChunkType.LIST, "- wool\n- copper")]


def test_direct_text_and_layout_reject_corrupt_pdf():
    with pytest.raises(DocumentError):
        DirectTextExtractor().process(fx.CORRUPT_PDF, "a.pdf")
    with pytest.raises(ExtractionError):
        LayoutAwareOCR().process(fx.CORRUPT_PDF, "a.pdf")
    assert PlainTextExtractor().name() == "plain_text"
