from pycharmai.rag.vector_store import search_documents
from datetime import datetime

# Qdrant verwendet HNSW als zentralen ANN-Index. D
def main() -> None:
    query = "Was ist eine Kafka Partition?"


    start = datetime.now()
    documents = search_documents(
        query,
        limit=3,
    )
    end = datetime.now()
    duration = end - start

    print(f"Suchdauer in Sekunden: {duration.total_seconds()}")

    print("QUERY:")
    print(query)
    print()
    print("RESULTS:")
    print("=" * 70)

    for document in documents:
        print(f"Score: {document['score']:.4f}")
        print(f"Source: {document['source']}")
        print(document["text"])
        print("-" * 70)

#Wenn wir unser Projekt weiterbauen, wäre mit Qdrant ein HNSW-vs-Exact-vs-PQ Benchmark in Qdrant wesentlich sinnvoller als zu versuchen, IVF künstlich einzubauen.
if __name__ == "__main__":
    main()