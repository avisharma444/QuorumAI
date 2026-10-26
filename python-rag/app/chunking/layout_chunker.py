"""Layout-aware chunking (reference app/chunking/layout_chunker.py).

Reference behaviour kept:
* tables become one chunk each and are never split (they may exceed MAX);
* a heading absorbs the following blocks up to the next heading or table;
* a caption joins an immediately preceding table, else stands alone;
* a paragraph shorter than MIN_CHUNK_CHARS absorbs following non-heading,
  non-table blocks while the total is below MAX_CHUNK_CHARS;
* text longer than MAX_CHUNK_CHARS splits at sentence boundaries;
* chunk_index counts chunks in document order.

Fixes:
* A single sentence longer than MAX_CHUNK_CHARS is split at word (then
  character) boundaries; the reference emitted it whole.
* page_number is the page where a chunk starts (the reference used the last
  block's page for heading groups); chunks that span pages record
  metadata["page_end"].
* Merged chunks record how many source blocks they contain, and OCR
  confidence is the mean of the blocks that reported one (never invented).
"""
from __future__ import annotations

import logging
import re

from app.models import Chunk, ChunkType, OCRBlock, OCRResult

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 50

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_document(ocr_result: OCRResult) -> list[Chunk]:
    blocks = ocr_result.blocks
    chunks: list[Chunk] = []
    method = ocr_result.ocr_method
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.block_type == ChunkType.TABLE:
            chunks.append(_make_chunk(block.content, ChunkType.TABLE, [block], method, len(chunks)))
            i += 1
        elif block.block_type == ChunkType.HEADING:
            j = i + 1
            while j < len(blocks) and blocks[j].block_type not in (ChunkType.HEADING, ChunkType.TABLE):
                j += 1
            group = blocks[i:j]
            body = "\n".join(b.content for b in group[1:])
            combined = block.content + "\n\n" + body if body else block.content
            for part in split_long_text(combined, MAX_CHUNK_CHARS):
                chunks.append(_make_chunk(part, ChunkType.PARAGRAPH, group, method, len(chunks), has_heading=True))
            i = j
        elif block.block_type == ChunkType.CAPTION:
            if chunks and chunks[-1].chunk_type == ChunkType.TABLE:
                prev = chunks[-1]
                prev.content += "\n\n" + block.content
                prev.metadata["has_caption"] = True
                prev.metadata["source_blocks"] += 1
                if block.page_number != prev.page_number:
                    prev.metadata["page_end"] = block.page_number
            else:
                chunks.append(_make_chunk(block.content, ChunkType.CAPTION, [block], method, len(chunks)))
            i += 1
        else:
            group = [block]
            j = i + 1
            if len(block.content) < MIN_CHUNK_CHARS:
                while (
                    j < len(blocks)
                    and blocks[j].block_type not in (ChunkType.HEADING, ChunkType.TABLE)
                    and sum(len(b.content) for b in group) < MAX_CHUNK_CHARS
                ):
                    group.append(blocks[j])
                    j += 1
            content = "\n".join(b.content for b in group)
            for part in split_long_text(content, MAX_CHUNK_CHARS):
                chunks.append(_make_chunk(part, block.block_type, group, method, len(chunks)))
            i = j
    logger.info("Chunked document into %d chunks", len(chunks))
    return chunks


def _make_chunk(content: str, kind: ChunkType, group: list[OCRBlock], method: str, index: int, has_heading: bool = False) -> Chunk:
    pages = [b.page_number for b in group if b.page_number]
    meta: dict = {"ocr_method": method, "source_blocks": len(group)}
    if has_heading:
        meta["has_heading"] = True
    if pages and max(pages) != min(pages):
        meta["page_end"] = max(pages)
    confs = [b.confidence for b in group if b.confidence is not None]
    if confs:
        meta["ocr_confidence"] = round(sum(confs) / len(confs), 4)
    if len(group) == 1 and group[0].bbox:
        meta["bbox"] = group[0].bbox
    return Chunk(content=content, chunk_type=kind, page_number=min(pages) if pages else 0, chunk_index=index, metadata=meta)


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split at sentence boundaries; oversized sentences split at words, then characters."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sentence in split_sentences(text):
        pieces.extend(_hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    out: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        added = len(piece) + (1 if current else 0)
        if current and length + added > max_chars:
            out.append(" ".join(current))
            current, length = [piece], len(piece)
        else:
            current.append(piece)
            length += added
    if current:
        out.append(" ".join(current))
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > max_chars:  # a "word" longer than the limit (URLs, base64, tables flattened)
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts
