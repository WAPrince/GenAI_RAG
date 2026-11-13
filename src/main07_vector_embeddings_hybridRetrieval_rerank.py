from mds.rag.embeddings import BAAI_MODEL_NAME, QWEN_MODEL_NAME, HARRIER_MODEL_NAME
from mds.rag.vector_store import qdrant_vector_search

def find_key(data, target_key="x"):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            # Search deeper if the value is another dict or list
            result = find_key(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_key(item, target_key)
            if result is not None:
                return result
    return None



def retrieveNonHybrid(query : str, limit : int, collection_name: str, embedding_model:str = BAAI_MODEL_NAME):
    retrievalResult:list[dict] =  qdrant_vector_search( query, limit, collection_name, embedding_model, False)
    if retrievalResult is not None and len(retrievalResult) > 0:
        print("Retrieval result:\n")
        for item in retrievalResult:
            print(f"SCORE: {find_key(item, "score")}")
            print(f"TEXT: {find_key(item,"text")}")
    else:
        print("Empty retrievalResult")

def retrieveHybrid(query : str, limit : int, collection_name,embedding_model:str = BAAI_MODEL_NAME) :
    qdrant_vector_search(query, limit, collection_name, embedding_model, True)


if __name__ == "__main__":
    """
        Execute search queries against Qdrant collections 'medicaments'
        The system prompt part contains the Qdrant collection that contains the desired answers
        The system prompt part contains the embedding model that should be applied to the query for the similiarity search (  it must be the same as the imported Qdrant collection data - otherwise vectors cannot be compared/nothing will be found )
        Choose to call a simple similarity search or an hybrid search
    """

    question:str="Welche Schmerzen können nach Einnahme der Medikamente auftreten?"

    #vector size 1024
    retrieveNonHybrid(question,10, "medicaments", HARRIER_MODEL_NAME)
    #retrieveHybrid(question, 10, "medicaments__dense_sparse",  HARRIER_MODEL_NAME)

    #Vector size 364
    #retrieveNonHybrid(question, 3, "medicaments",BAAI_MODEL_NAME)
    # retrieveHybrid(question, 3, "medicaments__dense_sparse", BAAI_MODEL_NAME)
