from fastembed import TextEmbedding
from mlflow.entities import SpanType
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from mds.rag.embeddings import create_embedding_model, BAAI_MODEL_NAME
import mlflow

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge"


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
    )

#@mlflow.trace(name="qdrant_vector_search", span_type=SpanType.TOOL)
def qdrant_vector_search(
    query: str,
    limit: int = 3,
    embedding_model_type :str = BAAI_MODEL_NAME
) -> list[dict]:


    embedding_model = create_embedding_model(embedding_model_type)
    query_embedding =None
    if(type(embedding_model) == SentenceTransformer):
        query_embedding  = embedding_model.encode(query, prompt_name="query")
    elif(type(embedding_model) == TextEmbedding):
        query_embedding = next(
            embedding_model.embed([query])
        )
    else:
        print("Query embedding failed!")
        return

    client = create_qdrant_client()
    documents = []
    try:
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