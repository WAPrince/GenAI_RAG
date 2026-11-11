from mlflow.entities import SpanType
from qdrant_client import QdrantClient
from mds.rag.embeddings import create_embedding_model
import mlflow

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge"


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
    )

@mlflow.trace(name="qdrant_vector_search", span_type=SpanType.TOOL)
def qdrant_vector_search(
    query: str,
    limit: int = 3,
) -> list[dict]:
    embedding_model = create_embedding_model()

    query_embedding = next(
        embedding_model.embed([query])
    )

    client = create_qdrant_client()
    documents = []
    try:
        print(f" qdrant_client calls qdrant DB")
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=limit,
            with_payload=True,
        )
        for point in result.points:
            documents.append(
                {
                    "score": point.score,
                    "text": point.payload["text"],
                    "source": point.payload["source"],
                }
            )
    except Exception as e :
        print(f"qdrant_client exception: {e}")
    return documents