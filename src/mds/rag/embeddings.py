import torch
from fastembed import TextEmbedding
from numpy.f2py.auxfuncs import throw_error
from sentence_transformers import SentenceTransformer
from typing_extensions import Any


# vector size 364
#	• Einsatzgebiet: Der bewährte Spezialist für deutsche und mehrsprachige Dokumente mit Hybrid-Suche.
#	• Spezifikationen: 568M Parameter, 8.192 Tokens Kontext. Es beherrscht gleichzeitig semantische Suche (Dense), klassische Keyword-Suche (Sparse) und Multi-Vector-Retrieval in einem Modell.
BAAI_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# vector size 1024
HARRIER_MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"

# vector size ?
QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


def create_embedding_model(embedding_model_id:str) -> Any:
    if (embedding_model_id == BAAI_MODEL_NAME):
        return TextEmbedding(
            model_name=BAAI_MODEL_NAME,
        )
    elif (embedding_model_id == HARRIER_MODEL_NAME):
        return  SentenceTransformer(
            "microsoft/harrier-oss-v1-0.6b",
            model_kwargs={"dtype": torch.float32},#{"torch_dtype": "auto", "device_map": "auto"},
        )
    elif (embedding_model_id == QWEN_MODEL_NAME):
        return  SentenceTransformer(
            "Qwen/Qwen3-Embedding-8B",
            model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
        )
    else:
       throw_error(f"Embedding model {embedding_model_id} not supported!")
