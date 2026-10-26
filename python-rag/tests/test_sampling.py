import pytest

from app.core import DocumentStore, Embedder
from app.sampling import PREVIEW_CHAR_LIMIT, bme_indices, bme_preview


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, []),
        (1, [0]),
        (3, [0, 1, 2]),
        (9, list(range(9))),
        (10, [0, 1, 2, 4, 5, 6, 7, 8, 9]),
        (11, [0, 1, 2, 4, 5, 6, 8, 9, 10]),
        (100, [0, 1, 2, 49, 50, 51, 97, 98, 99]),
        (101, [0, 1, 2, 49, 50, 51, 98, 99, 100]),
    ],
)
def test_bme_indices(n, expected):
    assert bme_indices(n) == expected


def test_bme_matches_reference_slicing_for_many_sizes():
    for n in range(0, 300):
        chunks = [f"c{i}" for i in range(n)]
        if n <= 9:
            ref = chunks
        else:
            ref = chunks[:3] + chunks[n // 2 - 1 : n // 2 + 2] + chunks[-3:]
        assert [chunks[i] for i in bme_indices(n)] == ref
        idx = bme_indices(n)
        assert idx == sorted(idx) and len(set(idx)) == len(idx)  # ordered, no duplicates


def test_preview_joins_in_order_and_is_deterministic():
    chunks = [f"chunk-{i}" for i in range(12)]
    preview = bme_preview(chunks)
    assert preview == "\n".join(["chunk-0", "chunk-1", "chunk-2", "chunk-5", "chunk-6", "chunk-7", "chunk-9", "chunk-10", "chunk-11"])
    assert bme_preview(chunks) == preview


def test_preview_truncated_to_15000_characters():
    chunks = ["é" * 4000 for _ in range(9)]  # 36k characters (multi-byte) before truncation
    preview = bme_preview(chunks)
    assert PREVIEW_CHAR_LIMIT == 15_000
    assert len(preview) == 15_000
    assert preview == "\n".join(chunks)[:15_000]


def test_empty_document_preview():
    assert bme_preview([]) == ""


def test_negative_count_rejected():
    with pytest.raises(ValueError):
        bme_indices(-1)


def test_ingest_returns_preview_from_real_chunks(tmp_path):
    store = DocumentStore(str(tmp_path / "db.sqlite"), Embedder("hash"))
    text = " ".join(f"sentence{i} about topic{i % 7}." for i in range(3000))
    doc = store.ingest("long.txt", text.encode())
    assert doc["chunks"] == doc["chunk_count"] > 9
    rows = store.conn.execute("SELECT content FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc["doc_id"],)).fetchall()
    stored = [r[0] for r in rows]
    assert doc["preview_text"] == bme_preview(stored)
    assert doc["preview_text"].startswith(stored[0])
    assert doc["preview_text"].endswith(stored[-1])


def test_short_document_preview_is_whole_text(tmp_path):
    store = DocumentStore(str(tmp_path / "db.sqlite"), Embedder("hash"))
    doc = store.ingest("short.md", b"Zogland's capital is Mirevale.")
    assert doc["chunk_count"] == 1
    assert doc["preview_text"] == "Zogland's capital is Mirevale."


def test_empty_document_is_rejected(tmp_path):
    store = DocumentStore(str(tmp_path / "db.sqlite"), Embedder("hash"))
    with pytest.raises(ValueError):
        store.ingest("empty.txt", b"   \n  ")
