import sys
import types

# ---------------------------------------------------------------------------
# Stub heavy third-party modules BEFORE importing the module under test
# ---------------------------------------------------------------------------

# ── fastembed ────────────────────────────────────────────────────────────────
fastembed_mod = types.ModuleType("fastembed")
class _FakeTextEmbedding:
    def __init__(self, model_name=None): pass
    def embed(self, chunks): return iter([[0.1] * 384 for _ in chunks])
class _FakeSparseTextEmbedding:
    def __init__(self, model_name=None): pass
    def embed(self, chunks): return iter([{"indices": [0], "values": [1.0]} for _ in chunks])
fastembed_mod.TextEmbedding = _FakeTextEmbedding
fastembed_mod.SparseTextEmbedding = _FakeSparseTextEmbedding
sys.modules["fastembed"] = fastembed_mod

# ── qdrant_client ────────────────────────────────────────────────────────────
qdrant_mod = types.ModuleType("qdrant_client")
models_mod = types.ModuleType("qdrant_client.models")

class _FakeVectorParams:
    def __init__(self, **kw): pass
class _FakeSparseVectorParams:
    def __init__(self, **kw): pass
class _FakeHnswConfigDiff:
    def __init__(self, **kw): pass
class _FakePointStruct:
    def __init__(self, **kw):
        self.__dict__.update(kw)
class _FakeDocument:
    def __init__(self, **kw): pass
class _FakeDistance:
    COSINE = "Cosine"
class _FakeModifier:
    IDF = "idf"

models_mod.VectorParams = _FakeVectorParams
models_mod.SparseVectorParams = _FakeSparseVectorParams
models_mod.HnswConfigDiff = _FakeHnswConfigDiff
models_mod.PointStruct = _FakePointStruct
models_mod.Document = _FakeDocument
models_mod.Distance = _FakeDistance
models_mod.Modifier = _FakeModifier
qdrant_mod.models = models_mod
sys.modules["qdrant_client"] = qdrant_mod
sys.modules["qdrant_client.models"] = models_mod

# ── sentence_transformers ────────────────────────────────────────────────────
st_mod = types.ModuleType("sentence_transformers")
class _FakeSentenceTransformer:
    def encode(self, chunks, prompt_name=None):
        import numpy as np
        return np.array([[0.2] * 384 for _ in chunks])
st_mod.SentenceTransformer = _FakeSentenceTransformer
sys.modules["sentence_transformers"] = st_mod

# ── mds.rag.vector_store ──────────────────────────────────────────────────────
mds_mod = types.ModuleType("mds")
mds_rag_mod = types.ModuleType("mds.rag")
mds_rag_vs_mod = types.ModuleType("mds.rag.vector_store")
mds_rag_vs_mod._create_qdrant_client = lambda: None   # replaced per-test
mds_mod.rag = mds_rag_mod
sys.modules["mds"] = mds_mod
sys.modules["mds.rag"] = mds_rag_mod
sys.modules["mds.rag.vector_store"] = mds_rag_vs_mod

# ── mds.rag.embeddings ────────────────────────────────────────────────────────
mds_rag_emb_mod = types.ModuleType("mds.rag.embeddings")
mds_rag_emb_mod.create_embedding_model = lambda model_id: _FakeTextEmbedding()
mds_rag_emb_mod.BAAI_MODEL_NAME = "baai"
mds_rag_emb_mod.MINILM_MODEL_NAME = "minilm"
mds_rag_emb_mod.HARRIER_MODEL_NAME = "harrier"
mds_rag_emb_mod.QWEN_MODEL_NAME = "qwen"
mds_rag_emb_mod.BM25_MODEL_NAME = "bm25"
mds_rag_emb_mod.RERANKER_MODEL_NAME = "reranker"
mds_rag_emb_mod.SEMANTIC_SEARCH_EMBEDDING_MODEL = "semantic"
sys.modules["mds.rag.embeddings"] = mds_rag_emb_mod

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
import sys
import importlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np

# Make sure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import qdrant_data_importer as sut


# ===========================================================================
# Helpers
# ===========================================================================

def _make_fake_client(collection_names: list[str] | None = None):
    """Return a MagicMock that mimics the minimal QdrantClient interface."""
    client = MagicMock()
    names = collection_names or []
    fake_col = MagicMock()
    fake_col.name = None          # overridden below
    collection_objects = []
    for n in names:
        c = MagicMock()
        c.name = n
        collection_objects.append(c)
    client.get_collections.return_value.collections = collection_objects
    client.count.return_value.count = 0
    return client


def _make_tmp_file(tmp_path, content: str = "word1 word2 word3") -> Path:
    p = tmp_path / "doc.txt"
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# load_document
# ===========================================================================

class TestLoadDocument:
    def test_returns_file_content(self, tmp_path):
        f = _make_tmp_file(tmp_path, "Hello World")
        assert sut.load_document(f) == "Hello World"

    def test_reads_utf8_correctly(self, tmp_path):
        content = "Ärzte Übergabe Öl"
        f = tmp_path / "uml.txt"
        f.write_text(content, encoding="utf-8")
        assert sut.load_document(f) == content


# ===========================================================================
# _split_into_chunks
# ===========================================================================

class TestSplitIntoChunks:
    def test_single_chunk_when_words_fewer_than_chunk_size(self, tmp_path):
        words = " ".join([f"word{i}" for i in range(10)])
        f = _make_tmp_file(tmp_path, words)
        chunks = sut._split_into_chunks(f, chunk_size=300)
        assert len(chunks) == 1
        assert chunks[0] == words

    def test_splits_into_correct_number_of_chunks(self, tmp_path):
        words = " ".join([f"word{i}" for i in range(600)])
        f = _make_tmp_file(tmp_path, words)
        chunks = sut._split_into_chunks(f, chunk_size=300)
        assert len(chunks) == 2

