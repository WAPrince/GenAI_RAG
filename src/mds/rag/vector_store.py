from fastembed import TextEmbedding
from mlflow.entities import SpanType
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from mds.rag.embeddings import create_embedding_model, BAAI_MODEL_NAME
import mlflow

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "knowledge"





def rerank(retrieved_docs : list[str], query:str) -> list[dict]:
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    # 1. RICHTIG: Nur reine Text-Paare [Query, Text] für den Reranker bauen
    text_paare = [[query, item["text"]] for item in retrieved_docs]

    # 2. Neue Reranker-Scores berechnen
    rerank_scores = reranker.predict(text_paare)

    # 3. Neue Scores den ursprünglichen Objekten (inkl. 'source') wieder zuordnen
    for item, neuer_score in zip(retrieved_docs, rerank_scores):
        item["rerank_score"] = float(neuer_score)

    # 4. Nach dem neuen Score absteigend sortieren
    sortierte_treffer = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

    #for t in sortierte_treffer:
    #    print(f"Score: {t['rerank_score']:.4f} | Quelle: {t['source']} | Text: {t['text']}")

    return sortierte_treffer


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


    rerank(documents, query)

    return documents