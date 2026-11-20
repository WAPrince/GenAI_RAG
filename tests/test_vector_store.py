import sys
import types

# ---------------------------------------------------------------------------
# Stub heavy third-party modules BEFORE importing the module under test
# ---------------------------------------------------------------------------

# ── fastembed ────────────────────────────────────────────────────────────────
fastembed_mod = types.ModuleType("fastembed")

class _FakeTextEmbedding:
    def __init__(self, model_name=None): pass
    def embed(self, chunks):
        import numpy as np
        for _ in chunks:
            yield np.array([0.1] * 384)

class _FakeSparseTextEmbedding:
    def __init__(self, model_name=None): pass
    def embed(self, chunks):
        import numpy as np
        for _ in chunks:
            vec = types.SimpleNamespace(
                indices=np.array([0, 1]),
                values=np.array([0.5, 0.8]),
            )
            yield vec

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

class _FakePrefetch:
    def __init__(self, **kw): pass

class _FakeSparseVector:
    def __init__(self, **kw): pass

class _FakeFusion:
    RRF = "rrf"

class _FakeFusionQuery:
    def __init__(self, **kw): pass

class _FakeQdrantClient:
    pass

models_mod.VectorParams = _FakeVectorParams
models_mod.SparseVectorParams = _FakeSparseVectorParams
models_mod.HnswConfigDiff = _FakeHnswConfigDiff
models_mod.PointStruct = _FakePointStruct
models_mod.Document = _FakeDocument
models_mod.Distance = _FakeDistance
models_mod.Modifier = _FakeModifier
models_mod.Prefetch = _FakePrefetch
models_mod.SparseVector = _FakeSparseVector
models_mod.Fusion = _FakeFusion
models_mod.FusionQuery = _FakeFusionQuery
qdrant_mod.models = models_mod
qdrant_mod.QdrantClient = _FakeQdrantClient
sys.modules["qdrant_client"] = qdrant_mod
sys.modules["qdrant_client.models"] = models_mod

# ── sentence_transformers ────────────────────────────────────────────────────
st_mod = types.ModuleType("sentence_transformers")

class _FakeSentenceTransformer:
    def encode(self, text, prompt_name=None):
        import numpy as np
        if isinstance(text, list):
            return np.array([[0.2] * 384 for _ in text])
        return np.array([0.2] * 384)

class _FakeCrossEncoder:
    def __init__(self, model_name=None): pass
    def predict(self, pairs):
        return [0.9 - i * 0.1 for i in range(len(pairs))]

st_mod.SentenceTransformer = _FakeSentenceTransformer
st_mod.CrossEncoder = _FakeCrossEncoder
sys.modules["sentence_transformers"] = st_mod

# ── mlflow ────────────────────────────────────────────────────────────────────
mlflow_mod = types.ModuleType("mlflow")
mlflow_entities_mod = types.ModuleType("mlflow.entities")
class _FakeSpanType:
    TOOL = "tool"
mlflow_entities_mod.SpanType = _FakeSpanType
mlflow_mod.entities = mlflow_entities_mod
mlflow_mod.trace = lambda *a, **kw: (lambda f: f)  # no-op decorator
sys.modules["mlflow"] = mlflow_mod
sys.modules["mlflow.entities"] = mlflow_entities_mod

# ── mds.rag.embeddings ────────────────────────────────────────────────────────
mds_rag_emb_mod = types.ModuleType("mds.rag.embeddings")
mds_rag_emb_mod.create_embedding_model = lambda model_id: _FakeTextEmbedding()
mds_rag_emb_mod.BM25_MODEL_NAME = "bm25"
mds_rag_emb_mod.RERANKER_MODEL_NAME = "reranker"
mds_rag_emb_mod.SEMANTIC_SEARCH_EMBEDDING_MODEL = "semantic"
sys.modules["mds.rag.embeddings"] = mds_rag_emb_mod

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
import pytest
import importlib
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Das Problem ist ein **Modul-Caching-Konflikt**: `sys.modules["mds"]` und `sys.modules["mds.rag"]` werden als leere Stub-Module registriert,
# **bevor** Python die echten Pakete laden kann. Dadurch findet der Import `import mds.rag.vector_store as sut` zwar das Stub-Modul `mds.rag`,
# aber das echte `vector_store`-Modul wird nie daran gehängt.
#**Die Lösung:** Das echte `mds.rag.vector_store` muss nach dem `sys.path.insert` explizit mit geladen und direkt in `sys.modules`
# eingetragen werden – ohne die `mds`/`mds.rag` Parent-Module als leere Stubs vorab zu registrieren. `importlib`

# Load the real vector_store module via importlib and wire it into sys.modules
_vs_spec = importlib.util.spec_from_file_location(
    "mds.rag.vector_store",
    Path(__file__).parent.parent / "src" / "mds" / "rag" / "vector_store.py",
)
_vs_mod = importlib.util.module_from_spec(_vs_spec)

# Register parent packages so relative imports inside vector_store.py work
_mds = types.ModuleType("mds")
_mds_rag = types.ModuleType("mds.rag")
_mds_rag.embeddings = mds_rag_emb_mod
_mds.rag = _mds_rag
sys.modules.setdefault("mds", _mds)
sys.modules.setdefault("mds.rag", _mds_rag)
sys.modules["mds.rag.vector_store"] = _vs_mod

_vs_spec.loader.exec_module(_vs_mod)

sut = _vs_mod


# ===========================================================================
# Helpers
# ===========================================================================

def _make_point(score: float, text: str, source: str = "file.txt"):
    """Build a minimal fake Qdrant point object."""
    p = MagicMock()
    p.score = score
    p.payload = {"text": text, "source": source}
    return p


def _make_query_result(*points):
    """Wrap points in a fake query_points() return value."""
    r = MagicMock()
    r.points = list(points)
    return r


def _make_client(*points):
    """Return a fake QdrantClient whose query_points() returns the given points."""
    client = MagicMock()
    client.query_points.return_value = _make_query_result(*points)
    client.get_collection.return_value = MagicMock()
    return client


# ===========================================================================
# rerank
# ===========================================================================

class TestRerank:
    def test_returns_list(self):
        docs = [
            {"text": "Aspirin hilft bei Kopfschmerzen", "score": 0.5, "source": "a.txt"},
            {"text": "Ibuprofen hilft bei Fieber", "score": 0.4, "source": "b.txt"},
        ]
        result = sut.rerank(docs, "Kopfschmerzen Medikament")
        assert isinstance(result, list)

    def test_adds_rerank_score_to_every_item(self):
        docs = [
            {"text": "Text A", "score": 0.9, "source": "a.txt"},
            {"text": "Text B", "score": 0.3, "source": "b.txt"},
        ]
        result = sut.rerank(docs, "query")
        assert all("rerank_score" in item for item in result)

    def test_rerank_score_is_float(self):
        docs = [{"text": "Text A", "score": 0.5, "source": "a.txt"}]
        result = sut.rerank(docs, "query")
        assert isinstance(result[0]["rerank_score"], float)

    def test_result_sorted_descending_by_rerank_score(self):
        # _FakeCrossEncoder.predict returns [0.9, 0.8, 0.7] for 3 items,
        # so the first doc gets the highest score → order must stay as-is
        docs = [
            {"text": "Best match", "score": 0.1, "source": "a.txt"},
            {"text": "Medium match", "score": 0.5, "source": "b.txt"},
            {"text": "Weak match", "score": 0.9, "source": "c.txt"},
        ]
        result = sut.rerank(docs, "query")
        scores = [item["rerank_score"] for item in result]
        assert scores == sorted(scores, reverse=True)

    def test_preserves_all_original_fields(self):
        docs = [
            {"text": "Hallo Welt", "score": 0.7, "source": "orig.txt"},
        ]
        result = sut.rerank(docs, "query")
        assert result[0]["text"] == "Hallo Welt"
        assert result[0]["score"] == 0.7
        assert result[0]["source"] == "orig.txt"

    def test_empty_input_returns_empty_list(self):
        result = sut.rerank([], "query")
        assert result == []

    def test_single_item_is_returned(self):
        docs = [{"text": "Einziger Treffer", "score": 0.6, "source": "x.txt"}]
        result = sut.rerank(docs, "Treffer")
        assert len(result) == 1
        assert result[0]["text"] == "Einziger Treffer"

    def test_length_of_result_matches_input(self):
        docs = [
            {"text": f"Dokument {i}", "score": float(i) / 10, "source": f"{i}.txt"}
            for i in range(5)
        ]
        result = sut.rerank(docs, "query")
        assert len(result) == 5