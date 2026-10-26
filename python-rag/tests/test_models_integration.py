"""Integration tests with the REAL configured models (BAAI/bge-small-en-v1.5 and
cross-encoder/ms-marco-MiniLM-L-6-v2). Weights download once into HF_HOME."""
import math

import pytest
from fastapi.testclient import TestClient

import fixtures as fx
from app import state
from app.config import Settings, get_settings
from app.embedding.transformer import EmbeddingError, TransformerEmbeddings
from app.retrieval.chroma_store import ChromaStore
from app.services import RagServices

pytestmark = pytest.mark.models


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_real_embeddings_shape_finite_repeatable(real_embedder):
    assert real_embedder.model_name == "BAAI/bge-small-en-v1.5" and real_embedder.dimension == 384
    vecs = real_embedder.embed_documents(["Mirevale is the capital of Zogland.", "Copper mining in Halvard."])
    assert len(vecs) == 2 and all(len(v) == 384 and all(math.isfinite(x) for x in v) for v in vecs)
    again = real_embedder.embed_query("Mirevale is the capital of Zogland.")
    assert max(abs(a - b) for a, b in zip(vecs[0], again)) < 1e-4  # repeatable; padding differs between batches
    assert real_embedder.embed_documents([]) == []
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) > 0.01, "reference behaviour: vectors are not normalised"


def test_model_is_loaded_once(real_embedder):
    s = get_settings()
    other = TransformerEmbeddings(s.embedding_model, s.embedding_dimension)
    assert other.model is real_embedder.model


def test_batching_matches_single_inference(real_embedder):
    texts = [f"Sentence {i} about rivers and bridges in Mirevale." for i in range(5)]
    small = TransformerEmbeddings(real_embedder.model_name, 384, batch_size=2)
    for a, b in zip(small.embed_documents(texts), real_embedder.embed_documents(texts)):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


def test_related_texts_are_closer(real_embedder):
    q, related, unrelated = real_embedder.embed_documents([
        "How do I reset my account password?",
        "Steps to change the password for your user account.",
        "The rainfall in the northern valley was unusually high this spring.",
    ])
    assert l2(q, related) < l2(q, unrelated)  # the metric Chroma uses
    assert cos(q, related) > cos(q, unrelated)


def test_wrong_dimension_and_missing_model_fail_clearly(real_embedder):
    with pytest.raises(EmbeddingError, match="384-dim"):
        TransformerEmbeddings(real_embedder.model_name, expected_dimension=768)
    with pytest.raises(EmbeddingError, match="Cannot load"):
        TransformerEmbeddings("/nonexistent/model-dir")


def test_real_cross_encoder_reorders_by_relevance(real_reranker):
    query = "Which city is the capital of Zogland?"
    docs = [
        "Wool exports from Zogland grew by four percent last year.",
        "Copper mines near Halvard employ about four thousand people.",
        "Mirevale, where the Ardent and Lune rivers meet, is the capital city of Zogland.",
    ]
    ranked = real_reranker.rerank(query, docs, top_k=3)
    assert ranked[0][0] == 2, f"observed order {ranked}"
    assert all(math.isfinite(s) for _, s in ranked)
    assert ranked[0][1] > ranked[1][1] >= ranked[2][1]


def test_real_store_retrieval_with_rerank(tmp_path, real_embedder, real_reranker):
    from app.chunking.layout_chunker import chunk_document
    from app.inspection.inspector import inspect_document
    from app.ocr.router import route_ocr

    data = fx.text_pdf([[("p", fx.PARA_1)], [("p", fx.PARA_2)], [("p", fx.PARA_3)]])
    chunks = chunk_document(route_ocr(data, "z.pdf", inspect_document(data, "z.pdf")))
    store = ChromaStore(str(tmp_path / "chroma"), real_embedder)
    store.ingest(chunks, "zog")
    res = store.retrieve("Who decides how much households pay for water?", "zog", top_k=1, reranker=real_reranker)
    assert res.reranked and "National Water Board" in res.chunks[0].content and res.chunks[0].page_number == 2


# ── End-to-end through the HTTP API with real OCR, embeddings, Chroma and reranker ──

@pytest.fixture(scope="module")
def real_client(tmp_path_factory, real_embedder, real_reranker):
    path = str(tmp_path_factory.mktemp("chroma"))
    s = get_settings()
    settings = Settings(**{**s.model_dump(), "chroma_db_path": path})
    svc = RagServices(settings, embedder=real_embedder, reranker=real_reranker, store=ChromaStore(path, real_embedder), autoload=False)
    old = state.services
    state.services = svc
    from app.main import app
    with TestClient(app) as c:
        yield c, path
    state.services = old


def ask(client, doc_id, question, top_k=2):
    r = client.post("/retrieve", json={"question": question, "doc_id": doc_id, "top_k": top_k})
    assert r.status_code == 200, r.text
    return r.json()


def test_e2e_searchable_pdf(real_client):
    client, _ = real_client
    body = client.post("/ingest", files={"file": ("policy.pdf", fx.clean_pdf())}).json()
    assert body["ocr_method"] == "direct_text" and body["embedding_model"] == "BAAI/bge-small-en-v1.5"
    res = ask(client, body["doc_id"], "What does the copper industry employ?", top_k=1)
    assert res["reranked"] and "Halvard" in res["chunks"][0]["content"] and res["chunks"][0]["page_number"] == 2


def test_e2e_table_pdf(real_client):
    client, _ = real_client
    body = client.post("/ingest", files={"file": ("revenue.pdf", fx.table_pdf())}).json()
    res = ask(client, body["doc_id"], "What was the North region revenue in Q2?", top_k=1)
    top = res["chunks"][0]
    assert top["chunk_type"] == "table" and "| North | 120 | 135 |" in top["content"]


@pytest.mark.ocr
def test_e2e_scanned_pdf_and_image(real_client):
    client, _ = real_client
    scan = client.post("/ingest", files={"file": ("scan.pdf", fx.scanned_pdf([fx.OCR_LINES, ["Rivers flood every spring in the valley."]]))}).json()
    assert scan["ocr_method"] == "tesseract" and scan["metadata"]["is_scanned"]
    res = ask(client, scan["doc_id"], "What is the total amount on the invoice?", top_k=1)
    assert "1250" in res["chunks"][0]["content"] and "ocr_confidence" in res["chunks"][0]["metadata"]
    img = client.post("/ingest", files={"file": ("invoice.png", fx.png_bytes(fx.text_image()))}).json()
    assert img["ocr_method"] == "tesseract"
    assert "INVOICE 4471" in ask(client, img["doc_id"], "invoice number", top_k=1)["chunks"][0]["content"]


def test_e2e_survives_service_restart(real_client, real_embedder, real_reranker):
    client, path = real_client
    doc = client.post("/ingest", files={"file": ("policy.pdf", fx.clean_pdf())}).json()["doc_id"]
    restarted = ChromaStore(path, real_embedder)  # a new client on the same directory, as after a restart
    res = restarted.retrieve("Where do the Ardent and Lune rivers meet?", doc, top_k=1, reranker=real_reranker)
    assert "Mirevale" in res.chunks[0].content


def test_e2e_question_not_in_document(real_client):
    client, _ = real_client
    doc = client.post("/ingest", files={"file": ("policy.pdf", fx.clean_pdf())}).json()["doc_id"]
    answerable = ask(client, doc, "What is the capital of Zogland?", top_k=1)["chunks"][0]["metadata"]["rerank_score"]
    unanswerable = ask(client, doc, "Who won the 1998 football world cup?", top_k=1)
    # Retrieval still returns the nearest chunk (it cannot return "nothing"), but the
    # cross-encoder scores it far lower; deciding "not in document" is the council's job.
    assert unanswerable["chunks"] and unanswerable["chunks"][0]["metadata"]["rerank_score"] < answerable
