from mds.rag.embeddings import BAAI_MODEL_NAME, QWEN_MODEL_NAME, HARRIER_MODEL_NAME
from mds.rag.rerank import rerank
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
        for item in retrievalResult:

            print(f"retrievalResult TEXT: {find_key(item,"text")}")
            print(f"retrievalResult SCORE: {find_key(item,"score")}")

            # TODO
            #rerankResult:list[dict]=rerank( retrievalResult)
            #if rerankResult is not None and len(rerankResult) > 0:
            # print(f"Length rerankResult: {len(rerankResult)}")
            #else:
         #    print("Empty rerankResult")
    else:
        print("Empty retrievalResult")

if __name__ == "__main__":
    #vector size 1024
    retrieveAndRerank("Was ist im Harz passiert?",3, HARRIER_MODEL_NAME)

    #Vector size 364
    #retrieveAndRerank("Was ist im Harz passiert?", 3, BAAI_MODEL_NAME)