from pathlib import Path
from uuid import uuid4
from datetime import datetime

from fastembed import TextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from mds.rag.embeddings import create_embedding_model, BAAI_MODEL_NAME, HARRIER_MODEL_NAME
from mds.rag.vector_store import create_qdrant_client, COLLECTION_NAME

DATA_FILE = Path("../data/input/qdrant_Input__Nachrichten.txt") # 1,678 sec 18 KB

def load_document(file :str ) -> str:
    return file.read_text(
        encoding="utf-8",
    )

# fixed chunk size => semantic search will affected ( completeness accuracy consistency costs )+
def split_into_chunks(
    text: str,
    chunk_size: int = 50,#300,
) -> list[str]:
    words = text.split()

    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


def ingest(file : str, embedding_model_id : str) -> None:

    try:
        text = load_document(file)

        start = datetime.now()
        chunks = split_into_chunks(text)

        print(f"Document contains {len(text)} characters")
        print(f"Created {len(chunks)} chunks")

        embedding_model = create_embedding_model(embedding_model_id)
        data_embeddings = None
        if (type(embedding_model) == SentenceTransformer):
            data_embeddings = embedding_model.encode(chunks, prompt_name="query")
        elif (type(embedding_model) == TextEmbedding):
            data_embeddings = list(
                embedding_model.embed(chunks)
            )
        else:
            print("Query embedding failed!")
            return

        vector_size = len(data_embeddings[0])
        print(f"Vecor size {vector_size }")
        # load embeddings to Qdrant vector database in docker container
        client = create_qdrant_client()

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
        print(f"Upsert")
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        end = datetime.now()
        duration = end - start
        print(f"Einlesedauer in Sekunden: {duration.total_seconds()}")
        print(f"End")
    except Exception as e:
        print(f"Exception ${e.str}")

if __name__ == "__main__":
    #vector size 1024
    ingest(DATA_FILE, HARRIER_MODEL_NAME)

    #vector size 364
    #ingest(DATA_FILE, BAAI_MODEL_NAME)