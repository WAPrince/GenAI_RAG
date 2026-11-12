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

def retrieveAndRerank(query : str, limit : int, embedding_model_type:str = BAAI_MODEL_NAME):

    retrievalResult:list[dict] =  qdrant_vector_search( query, limit, embedding_model_type)
    if retrievalResult is not None and len(retrievalResult) > 0:
        print("Retrieval result:\n")
        for item in retrievalResult:
            print(f"TEXT: {find_key(item,"text")}")
            print(f"SCORE: {find_key(item,"score")}")
    else:
        print("Empty retrievalResult")

if __name__ == "__main__":
    question:str="Was ist im Harz passiert?"
    print(f"Frage: {question}")

    #vector size 1024
    retrieveAndRerank(question,3, HARRIER_MODEL_NAME)

    #Vector size 364
    #retrieveAndRerank("Was ist im Harz passiert?", 3, BAAI_MODEL_NAME)