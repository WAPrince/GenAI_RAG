import asyncio,os
import uvicorn
import uuid
import inspect
from fastmcp import FastMCP
#import mlflow
from mds.rag.embeddings import SEMANTIC_SEARCH_EMBEDDING_MODEL
from mds.rag.vector_store import qdrant_vector_search



mcp = FastMCP()
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))


#TRACKING_URI = "http://localhost:5000"
#os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI  # the MCP server subprocess reads this
#mlflow.set_tracking_uri(TRACKING_URI)
#mlflow.set_experiment("mcpserver-" + str(uuid.uuid4()))




@mcp.tool()
def search_in_vectordb(query: str, collection_name:str, limit: int = 3, semantic_search_embedding_model:str = SEMANTIC_SEARCH_EMBEDDING_MODEL, hybrid=True) -> str:
    """Search the qdrant vectordb ( knowledge base )  for information relevant to the query."""
    try:
        print(f"MCP server: search_in_vectordb : params query: {query}, semantic_search_embedding_model: {semantic_search_embedding_model}, hybrid: {hybrid}, collection: {collection_name}")
        documents = qdrant_vector_search(query, limit=limit, collection_name=collection_name, semantic_search_embedding_model=semantic_search_embedding_model, hybrid=hybrid)
    except Exception as e:
        print(f"search_in_vectordb Exception {e}")
        return "search_in_vectordb Exception "+{e}+"occurred!"

    if not documents:
        return "No relevant documents found!"

    result = []
    for document in documents:
        print(f"search_in_vectordb - document: {document}")
        if hybrid == False:
            result.append(
                    f"""
            Source: {document["source"]}
            Similarity: {document["score"]:.4f}
            {document["text"]}
            """.strip())
        else :
            payload = document.payload or {}
            result.append(
                f"""
                Source: {payload.get("source", "None")}
                Similarity: {document.score:.4f}
                {payload.get("text", "")}
                #""".strip()
                )

    return "\n\n---\n\n".join(result)


async def run_mcp_server():
    # run_http_async startet einen uvicorn-Server im Hintergrund.
    # Standardmäßig wird der moderne "streamable-http"-Transport verwendet.
    await mcp.run_http_async(
        host="127.0.0.1",
        port=3006,
        transport="streamable-http"
    )

if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["mcp.server.streamable_http"] = {"level": "CRITICAL"}
    #mlflow.anthropic.autolog()
    asyncio.run(run_mcp_server())