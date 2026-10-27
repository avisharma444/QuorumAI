import math
import re

import pytest

from app.models import Chunk, ChunkType
from app.retrieval.chroma_store import ChromaStore, DocumentNotFound
from app.retrieval.reranker import Reranker, RerankError
from conftest import FakeCrossEncoder


def chunks(*texts, page=1, kind=ChunkType.PARAGRAPH):
    return [Chunk(content=t, chunk_type=kind, page_number=page, chunk_index=i, metadata={"ocr_method": "direct_text", "bbox": [1.0, 2.0, 3.0, 4.0]}) for i, t in enumerate(texts)]


DOC = ["Mirevale is the capital of Zogland.", "Copper mines near Halvard employ miners.", "The water board sets household prices.",
       "Wool exports grew this year.", "Bridges in Mirevale are very old."]


def test_ingest_retrieve_and_metadata(store):
    assert store.ingest(chunks(*DOC), "doc-a") == (5, False)
    res = store.retrieve("capital of Zogland", "doc-a", top_k=2, rerank=False)
    assert res.chunks[0].content == DOC[0] and len(res.chunks) == 2 and not res.reranked
    c = res.chunks[0]
    assert (c.chunk_type, c.page_number, c.chunk_index) == (ChunkType.PARAGRAPH, 1, 0)
    assert c.metadata["doc_id"] == "doc-a" and c.metadata["ocr_method"] == "direct_text" and c.metadata["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert c.metadata["vector_rank"] == 0 and math.isfinite(c.metadata["vector_distance"])
    assert res.candidates == 5  # min(3 * top_k, total)


def test_get_all_is_in_document_order(store):
    store.ingest(chunks(*DOC), "doc-a")
    assert [c.content for c in store.get_all("doc-a")] == DOC


def test_documents_are_isolated(store):
    store.ingest(chunks("Alpha document about tea."), "doc-a")
    store.ingest(chunks("Beta document about coffee."), "doc-b")
    assert [c.content for c in store.retrieve("coffee", "doc-a", top_k=5, rerank=False).chunks] == ["Alpha document about tea."]


def test_unknown_document_is_not_created(store):
    before = store._names()
    with pytest.raises(DocumentNotFound):
        store.retrieve("q", "missing", rerank=False)
    with pytest.raises(DocumentNotFound):
        store.get_all("missing")
    assert store._names() == before and not store.exists("missing")


def test_reingest_replaces_without_stale_chunks(store):
    store.ingest(chunks("old one", "old two", "old three"), "doc-a")
    assert store.ingest(chunks("new only"), "doc-a") == (1, True)
    assert [c.content for c in store.get_all("doc-a")] == ["new only"]
    assert len([n for n in store._names() if n.startswith("doc_")]) == 1


def test_failed_reingest_keeps_previous_version(store, monkeypatch):
    store.ingest(chunks(*DOC), "doc-a")
    monkeypatch.setattr(store.embeddings, "embed_documents", lambda texts: (_ for _ in ()).throw(RuntimeError("model crashed")))
    with pytest.raises(RuntimeError):
        store.ingest(chunks("replacement"), "doc-a")
    monkeypatch.undo()
    assert [c.content for c in store.get_all("doc-a")] == DOC


def test_failed_write_removes_staging_and_keeps_old(store, monkeypatch):
    store.ingest(chunks(*DOC), "doc-a")
    from chromadb.api.models.Collection import Collection

    def failing_add(self, *a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(Collection, "add", failing_add)
    with pytest.raises(RuntimeError, match="disk full"):
        store.ingest(chunks("replacement"), "doc-a")
    monkeypatch.undo()
    assert [c.content for c in store.get_all("doc-a")] == DOC
    assert not any("_s" in n for n in store._names())


def test_persists_across_client_restart(tmp_path, fake_embedder):
    path = str(tmp_path / "chroma")
    ChromaStore(path, fake_embedder).ingest(chunks(*DOC), "doc-a")
    reopened = ChromaStore(path, fake_embedder)
    assert reopened.retrieve("capital", "doc-a", top_k=1, rerank=False).chunks[0].content == DOC[0]


def test_leftover_staging_collections_are_cleaned(tmp_path, fake_embedder):
    s = ChromaStore(str(tmp_path / "c"), fake_embedder)
    s.client.create_collection("doc_" + "a" * 40 + "_s12345678", embedding_function=None)
    assert ChromaStore(str(tmp_path / "c"), fake_embedder)._names() == set()


def test_delete(store):
    store.ingest(chunks(*DOC), "doc-a")
    assert store.delete("doc-a") and not store.exists("doc-a") and not store.delete("doc-a")


def test_collection_names_are_safe_and_collision_free():
    ids = ["report.pdf_abc", "report_pdf_abc", "x" * 200 + "1", "x" * 200 + "2", "a"]
    names = [ChromaStore.collection_name(i) for i in ids]
    assert len(set(names)) == len(ids), "sanitising or truncation must not merge different IDs"
    for n in names:
        assert re.fullmatch(r"doc_[0-9a-f]{40}", n) and 3 <= len(n) <= 63
    with pytest.raises(ValueError):
        ChromaStore.collection_name("")


# ── Reranking (injected scorer) ─────────────────────────────────────

def test_rerank_reorders_and_keeps_metadata(store):
    store.ingest(chunks(*DOC), "doc-a")
    rr = Reranker("fake", model=FakeCrossEncoder())
    res = store.retrieve("old bridges mirevale", "doc-a", top_k=2, rerank=True, reranker=rr)
    assert res.reranked and res.rerank_error is None
    assert res.chunks[0].content == DOC[4]
    assert all("rerank_score" in c.metadata and "vector_distance" in c.metadata for c in res.chunks)


def test_rerank_disabled(store):
    store.ingest(chunks(*DOC), "doc-a")
    model = FakeCrossEncoder()
    res = store.retrieve("bridges", "doc-a", top_k=2, rerank=False, reranker=Reranker("fake", model=model))
    assert not res.reranked and model.calls == 0 and all("rerank_score" not in c.metadata for c in res.chunks)


def test_rerank_with_fewer_candidates_than_top_k(store):
    store.ingest(chunks("tea leaves", "coffee beans about bridges"), "doc-a")
    res = store.retrieve("bridges", "doc-a", top_k=5, rerank=True, reranker=Reranker("fake", model=FakeCrossEncoder()))
    assert res.reranked and len(res.chunks) == 2 and res.chunks[0].content == "coffee beans about bridges"


def test_single_candidate_skips_model(store):
    store.ingest(chunks("only chunk"), "doc-a")
    model = FakeCrossEncoder()
    res = store.retrieve("q", "doc-a", top_k=3, rerank=True, reranker=Reranker("fake", model=model))
    assert model.calls == 0 and not res.reranked and res.rerank_error is None


@pytest.mark.parametrize("model, expected", [
    (FakeCrossEncoder(error=RuntimeError("OOM")), "OOM"),
    (FakeCrossEncoder(scores=[float("nan")] * 5), "non-finite"),
    (FakeCrossEncoder(scores=[1.0]), "missing"),
])
def test_rerank_failure_falls_back_to_vector_order(store, model, expected):
    store.ingest(chunks(*DOC), "doc-a")
    vector = store.retrieve("capital", "doc-a", top_k=2, rerank=False)
    res = store.retrieve("capital", "doc-a", top_k=2, rerank=True, reranker=Reranker("fake", model=model))
    assert not res.reranked and expected in res.rerank_error
    assert [c.content for c in res.chunks] == [c.content for c in vector.chunks]
    assert all("rerank_score" not in c.metadata for c in res.chunks)


def test_missing_reranker_is_reported(store):
    store.ingest(chunks(*DOC), "doc-a")
    res = store.retrieve("capital", "doc-a", top_k=2, rerank=True, reranker=None)
    assert not res.reranked and res.rerank_error == "reranker unavailable"


def test_rerank_ties_keep_vector_order():
    rr = Reranker("fake", model=FakeCrossEncoder(scores=[1.0, 2.0, 2.0, 0.5]))
    assert [i for i, _ in rr.rerank("q", ["a", "b", "c", "d"], top_k=3)] == [1, 2, 0]
    assert Reranker("fake", model=FakeCrossEncoder()).rerank("q", [], 3) == []
    with pytest.raises(RerankError):
        Reranker("fake", model=FakeCrossEncoder(scores=[float("inf")])).rerank("q", ["a"], 1)


# ── failure-safe replacement: the previous index is never lost ─────

OLD, NEW = DOC, ["replacement text only"]


def _patch_modify(monkeypatch, fail_when):
    """Make Collection.modify(name=...) raise when fail_when(current, new) is true."""
    from chromadb.api.models.Collection import Collection

    real = Collection.modify

    def modify(self, *a, **k):
        if "name" in k and fail_when(self.name, k["name"]):
            raise RuntimeError("rename failed")
        return real(self, *a, **k)
    monkeypatch.setattr(Collection, "modify", modify)


def _contents(s, doc="doc-a"):
    return [c.content for c in s.get_all(doc)]


def test_staging_rename_failure_restores_previous_index(store, monkeypatch):
    store.ingest(chunks(*OLD), "doc-a")
    final = store.collection_name("doc-a")
    _patch_modify(monkeypatch, lambda cur, new: "_s" in cur and new == final)
    with pytest.raises(RuntimeError, match="rename failed"):
        store.ingest(chunks(*NEW), "doc-a")
    monkeypatch.undo()
    assert _contents(store) == OLD
    assert store._names() == {final}, "no staging or backup left behind"
    assert store.retrieve("capital", "doc-a", top_k=1, rerank=False).chunks[0].content == OLD[0]


def test_failed_rollback_still_serves_backup_and_restart_restores_it(tmp_path, fake_embedder, monkeypatch):
    path = str(tmp_path / "c")
    s = ChromaStore(path, fake_embedder)
    s.ingest(chunks(*OLD), "doc-a")
    final = s.collection_name("doc-a")
    # Both the swap-in and the rollback rename to the live name fail.
    _patch_modify(monkeypatch, lambda cur, new: new == final)
    with pytest.raises(RuntimeError):
        s.ingest(chunks(*NEW), "doc-a")
    assert s.exists("doc-a") and _contents(s) == OLD, "reads fall back to the backup"
    monkeypatch.undo()
    reopened = ChromaStore(path, fake_embedder)
    assert reopened._names() == {final} and _contents(reopened) == OLD


def test_failure_moving_live_index_aside_keeps_it(store, monkeypatch):
    store.ingest(chunks(*OLD), "doc-a")
    _patch_modify(monkeypatch, lambda cur, new: "_b" in new)
    with pytest.raises(RuntimeError):
        store.ingest(chunks(*NEW), "doc-a")
    monkeypatch.undo()
    assert _contents(store) == OLD and store._names() == {store.collection_name("doc-a")}


def test_crash_between_renames_is_rolled_back_at_startup(tmp_path, fake_embedder):
    path = str(tmp_path / "c")
    s = ChromaStore(path, fake_embedder)
    s.ingest(chunks(*OLD), "doc-a")
    final = s.collection_name("doc-a")
    # State after "live -> backup" but before "staging -> live": no live collection.
    s.client.get_collection(final, embedding_function=None).modify(name=final + "_b0000abcd")
    s.client.create_collection(final + "_s0000abcd", embedding_function=None)
    assert _contents(s) == OLD, "reads use the backup during the gap"
    reopened = ChromaStore(path, fake_embedder)
    assert reopened._names() == {final} and _contents(reopened) == OLD


def test_crash_before_backup_deletion_keeps_new_index(tmp_path, fake_embedder):
    path = str(tmp_path / "c")
    s = ChromaStore(path, fake_embedder)
    s.ingest(chunks(*OLD), "doc-a")
    final = s.collection_name("doc-a")
    s.client.get_collection(final, embedding_function=None).modify(name=final + "_b0000abcd")
    s.ingest(chunks(*NEW), "doc-a")  # finds only a backup: restores it, then swaps normally
    assert _contents(s) == NEW and s._names() == {final}
    # Swap finished but the backup was not deleted.
    old = s.client.create_collection(final + "_b0000ffff", embedding_function=None)
    old.add(ids=["x"], embeddings=[[0.0] * len(fake_embedder.embed_query("q"))], documents=["stale"])
    reopened = ChromaStore(path, fake_embedder)
    assert reopened._names() == {final} and _contents(reopened) == NEW


def test_backup_delete_failure_does_not_fail_the_ingest(store, monkeypatch):
    store.ingest(chunks(*OLD), "doc-a")
    real = store.client.delete_collection
    monkeypatch.setattr(store.client, "delete_collection", lambda name: (_ for _ in ()).throw(RuntimeError("busy")) if "_b" in name else real(name))
    assert store.ingest(chunks(*NEW), "doc-a") == (1, True)
    assert _contents(store) == NEW


def test_delete_also_removes_backups(store):
    store.ingest(chunks(*OLD), "doc-a")
    final = store.collection_name("doc-a")
    store.client.create_collection(final + "_b0000abcd", embedding_function=None)
    assert store.delete("doc-a") and store._names() == set() and not store.exists("doc-a")
