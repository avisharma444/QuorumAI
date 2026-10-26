from app.chunking.layout_chunker import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, chunk_document, split_long_text
from app.models import ChunkType, OCRBlock, OCRResult
from app.sampling import bme_preview


def blocks(*items, method="layout_aware"):
    return OCRResult(blocks=[OCRBlock(content=c, block_type=t, page_number=p, confidence=conf) for t, c, p, conf in items], ocr_method=method)


P, H, T, C, L = ChunkType.PARAGRAPH, ChunkType.HEADING, ChunkType.TABLE, ChunkType.CAPTION, ChunkType.LIST
LONG_PARA = "This sentence is about Zogland water policy and it is long enough. " * 3


def test_empty_input():
    assert chunk_document(OCRResult()) == []


def test_heading_attaches_to_following_text_until_next_heading_or_table():
    chunks = chunk_document(blocks((H, "Water", 1, None), (P, LONG_PARA, 1, None), (P, "Second para about reservoirs. " * 3, 1, None),
                                   (H, "Mining", 1, None), (P, LONG_PARA, 1, None), (T, "| a | b |", 1, None)))
    assert [c.chunk_type for c in chunks] == [P, P, T]
    assert chunks[0].content.startswith("Water\n\n") and "reservoirs" in chunks[0].content and "Mining" not in chunks[0].content
    assert chunks[0].metadata["has_heading"] and chunks[0].metadata["source_blocks"] == 3
    assert chunks[1].content.startswith("Mining\n\n")


def test_table_is_never_split_even_above_max():
    big_table = "\n".join(f"| row {i} | {'x' * 40} |" for i in range(80))
    assert len(big_table) > MAX_CHUNK_CHARS
    chunks = chunk_document(blocks((T, big_table, 3, None)))
    assert len(chunks) == 1 and chunks[0].content == big_table and chunks[0].chunk_type == T and chunks[0].page_number == 3


def test_caption_attaches_to_preceding_table_else_standalone():
    chunks = chunk_document(blocks((T, "| a |", 1, None), (C, "Table 1: Revenue", 2, None)))
    assert chunks[0].content == "| a |\n\nTable 1: Revenue" and chunks[0].metadata["has_caption"]
    assert chunks[0].metadata["page_end"] == 2  # caption on the next page is recorded
    chunks = chunk_document(blocks((P, LONG_PARA, 1, None), (C, "Figure 2: Map", 2, None)))
    assert chunks[1].chunk_type == C and chunks[1].content == "Figure 2: Map" and chunks[1].page_number == 2


def test_short_paragraphs_merge_and_record_pages():
    chunks = chunk_document(blocks((P, "Short one.", 1, 0.9), (L, "- item", 1, 0.8), (P, "Also short.", 2, None), (H, "Next", 2, None)))
    assert len(chunks) == 2
    first = chunks[0]
    assert first.content == "Short one.\n- item\nAlso short."
    assert first.page_number == 1 and first.metadata["page_end"] == 2 and first.metadata["source_blocks"] == 3
    assert first.metadata["ocr_confidence"] == 0.85  # mean of reported confidences only


def test_long_paragraph_splits_at_sentences_within_bound():
    sentences = [f"Sentence number {i} talks about the Lune river crossing." for i in range(60)]
    chunks = chunk_document(blocks((P, " ".join(sentences), 4, None)))
    assert len(chunks) > 1 and all(len(c.content) <= MAX_CHUNK_CHARS for c in chunks)
    assert " ".join(c.content for c in chunks) == " ".join(sentences)
    assert all(c.page_number == 4 for c in chunks)


def test_very_long_sentence_respects_bound():
    sentence = " ".join(f"word{i}" for i in range(1200))  # one "sentence", ~8k chars
    parts = split_long_text(sentence, MAX_CHUNK_CHARS)
    assert len(parts) > 1 and all(len(p) <= MAX_CHUNK_CHARS for p in parts)
    assert " ".join(parts) == sentence
    giant = "x" * 4000
    parts = split_long_text(giant, MAX_CHUNK_CHARS)
    assert all(len(p) <= MAX_CHUNK_CHARS for p in parts) and "".join(parts) == giant


def test_chunk_indices_are_sequential_and_deterministic():
    items = [(H, "A", 1, None), (P, LONG_PARA * 12, 1, None), (T, "| t |", 2, None), (C, "Table 9", 2, None), (P, LONG_PARA, 3, None)]
    first, second = chunk_document(blocks(*items)), chunk_document(blocks(*items))
    assert [c.chunk_index for c in first] == list(range(len(first)))
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_min_chunk_constant_matches_reference():
    assert (MAX_CHUNK_CHARS, MIN_CHUNK_CHARS) == (1500, 50)


def test_bme_preview_over_layout_chunks():
    items = [(P, f"Paragraph {i} " + "body text. " * 12, i // 3 + 1, None) for i in range(12)]
    chunks = chunk_document(blocks(*items))
    assert len(chunks) == 12
    preview = bme_preview([c.content for c in chunks])
    picked = [0, 1, 2, 5, 6, 7, 9, 10, 11]
    assert preview == "\n".join(chunks[i].content for i in picked)
