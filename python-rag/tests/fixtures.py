"""Deterministic document fixtures, generated at test time.

Text PDFs and the table PDF are drawn with reportlab (real text layers and
ruled tables); the image and scanned PDF are rendered with Pillow so they
contain pixels only and need OCR.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

HEADING = "Zogland Water Policy"
PARA_1 = ("The capital of Zogland is Mirevale, a river city founded in 1842. Mirevale sits where the Ardent "
          "and the Lune rivers meet, and its bridges are the oldest public works in the country.")
PARA_2 = ("Drinking water in Zogland is managed by the National Water Board. The board publishes quarterly "
          "reports on reservoir levels and sets the price of water for households and farms.")
PARA_3 = ("Zogland exports mostly wool and copper. The copper mines near Halvard employ about four thousand "
          "people and supply smelters across the region.")

BEFORE_TABLE = "Regional revenue grew steadily in the last fiscal year according to the treasury office."
AFTER_TABLE = "The northern region recovered after the drought that affected the previous fiscal year."
TABLE_ROWS = [["Region", "Q1", "Q2"], ["North", "120", "135"], ["South", "98", "110"]]
TABLE_CAPTION = "Table 1: Quarterly revenue by region in thousands"

OCR_LINES = ["INVOICE 4471", "Payment due within thirty days.", "Total amount: 1250 dollars."]


def text_pdf(pages: list[list[tuple[str, str]]]) -> bytes:
    """pages: list of [(style, text)] where style is 'h' (heading) or 'p' (paragraph)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = []
    for n, page in enumerate(pages):
        for style, text in page:
            story.append(Paragraph(text, styles["Heading1"] if style == "h" else styles["BodyText"]))
            story.append(Spacer(1, 12))
        if n < len(pages) - 1:
            story.append(PageBreak())
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()


def clean_pdf() -> bytes:
    return text_pdf([[("h", HEADING), ("p", PARA_1), ("p", PARA_2)], [("p", PARA_3)]])


def table_pdf() -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    table = Table(TABLE_ROWS)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    story = [
        Paragraph(BEFORE_TABLE, styles["BodyText"]), Spacer(1, 18),
        table, Spacer(1, 6),
        Paragraph(TABLE_CAPTION, styles["Italic"]), Spacer(1, 18),
        Paragraph(AFTER_TABLE, styles["BodyText"]),
    ]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()


def text_image(lines: list[str] = OCR_LINES, size: tuple[int, int] = (1400, 500)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=44)
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 110
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def scanned_pdf(pages: list[list[str]]) -> bytes:
    """A PDF whose pages are images only (no text layer)."""
    images = [text_image(lines, size=(1700, 900)) for lines in pages]
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", resolution=150, save_all=True, append_images=images[1:])
    return buf.getvalue()


CORRUPT_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> broken stream without xref"
