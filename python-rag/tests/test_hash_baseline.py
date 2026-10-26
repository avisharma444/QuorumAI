from app.baselines.hash_retrieval import DocumentStore, Embedder, chunk_pages, extract_pages


def test_chunking_page_boundary_and_overlap():
    chunks = chunk_pages([(1, "word " * 400), (2, "unique second page")], size=200, overlap=40)
    assert len(chunks) > 2
    assert chunks[-1].page == 2
    assert all(chunk.page == 1 for chunk in chunks[:-1])
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_persistent_retrieval(tmp_path):
    path = str(tmp_path / "db.sqlite")
    store = DocumentStore(path, Embedder("hash"))
    doc = store.ingest("notes.txt", b"Redis is a cache and key-value store.\n\nGo supports concurrent goroutines. " * 8)
    assert doc["chunks"] >= 1
    assert doc["embedding_backend"] == "hash"
    store.conn.close()
    reopened = DocumentStore(path, Embedder("hash"))
    found = reopened.retrieve(doc["doc_id"], "What is Redis?", top_k=2)
    assert found and "Redis" in found[0]["text"]


def test_non_existent_document(tmp_path):
    store = DocumentStore(str(tmp_path / "db.sqlite"), Embedder("hash"))
    try:
        store.retrieve("not-there", "some question")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_extract_text():
    assert extract_pages(b"hello", "a.md") == [(1, "hello")]
