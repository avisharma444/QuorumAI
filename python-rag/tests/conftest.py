import hashlib
import math
import re

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "models: loads the real embedding/reranker models (downloads once into HF_HOME)")
    config.addinivalue_line("markers", "ocr: needs the Tesseract and Poppler binaries")


class FakeEmbedder:
    """TEST-ONLY deterministic encoder (bag of hashed words, 384-d). Never used in production."""

    model_name = "test/fake-embedder"
    dimension = 384

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            v = [0.0] * 384
            for tok in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 384] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeCrossEncoder:
    """TEST-ONLY scorer: counts query words in the passage."""

    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.calls = scores, error, 0

    def predict(self, pairs):
        self.calls += 1
        if self.error:
            raise self.error
        if self.scores is not None:
            return self.scores
        return [float(sum(w in doc.lower() for w in q.lower().split())) for q, doc in pairs]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, fake_embedder):
    from app.retrieval.chroma_store import ChromaStore

    return ChromaStore(str(tmp_path / "chroma"), fake_embedder)


@pytest.fixture(scope="session")
def real_embedder():
    from app.config import get_settings
    from app.embedding.transformer import TransformerEmbeddings

    s = get_settings()
    return TransformerEmbeddings(s.embedding_model, s.embedding_dimension, s.embedding_batch_size)


@pytest.fixture(scope="session")
def real_reranker():
    from app.config import get_settings
    from app.retrieval.reranker import Reranker

    return Reranker(get_settings().reranker_model)
