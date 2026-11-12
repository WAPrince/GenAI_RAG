from pathlib import Path
from uuid import uuid4
from datetime import datetime

from fastembed import TextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from mds.rag.embeddings import create_embedding_model, BAAI_MODEL_NAME, HARRIER_MODEL_NAME
from mds.rag.vector_store import create_qdrant_client, COLLECTION_NAME

DATA_FILE = Path("../data/input/qdrant_Input__Nachrichten.txt") # 1,678 sec 18 KB


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start
        print(f"Ingestion took {duration.total_seconds()} seconds")
        return result
    return wrapper

def load_document(file :str ) -> str:
    return file.read_text(
        encoding="utf-8",
    )

# fixed chunk size => semantic search will affected ( completeness accuracy consistency costs )+
def _split_into_chunks(
    file: str,
    chunk_size: int = 50,#300,
) -> list[str]:
    text = load_document(file)
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    print(f"Document contains {len(text)} characters")
    print(f"Created {len(chunks)} chunks")
    return chunks

def _get_embeddings(embedding_model_id:str, chunks) -> list[dict]:
    data_embeddings = None
    embedding_model = create_embedding_model(embedding_model_id)
    if (type(embedding_model) == SentenceTransformer):
        data_embeddings = embedding_model.encode(chunks, prompt_name="query")
    elif (type(embedding_model) == TextEmbedding):
        data_embeddings = list(
            embedding_model.embed(chunks)
        )
    else:
        print("Query embedding failed!")
        return
    return data_embeddings

def _collection(client, vector_size):
    collections = client.get_collections()
    collection_exists = any(
        collection.name == COLLECTION_NAME
        for collection in collections.collections
    )
    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
    return

def _upsert(file, client, chunks, data_embeddings):
    points = []
    for chunk, embedding in zip(chunks, data_embeddings):
        print(f"chunk {chunk.index}")
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": str(file),
                },
            )
        )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )



@measure_time
def ingest(file : str, embedding_model_id : str) -> None:

    try:
        chunks = _split_into_chunks(file=file)

        data_embeddings =  _get_embeddings(embedding_model_id, chunks)

        client = create_qdrant_client()

        _collection(client, vector_size=len(data_embeddings[0]))

        _upsert(file,client,chunks, data_embeddings)

    except Exception as e:
        print(f"Exception ${e.str}")


if __name__ == "__main__":
    #vector size 1024
    ingest(DATA_FILE, HARRIER_MODEL_NAME)

    #vector size 364
    #ingest(DATA_FILE, BAAI_MODEL_NAME)