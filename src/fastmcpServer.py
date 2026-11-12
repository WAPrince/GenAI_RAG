import asyncio,os
import uvicorn
import uuid
import inspect
from fastmcp import FastMCP
import mlflow

from mds.rag.embeddings import BAAI_MODEL_NAME
from mds.rag.vector_store import qdrant_vector_search



mcp = FastMCP()
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))


TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI  # the MCP server subprocess reads this
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("mcpserver-" + str(uuid.uuid4()))



@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.tool()
def search_in_vectordb(query: str, limit: int = 3) -> str:
    """Search the qdrant vectordb ( knowledge base )  for information relevant to the query."""

    try:
        print(f"MCP server: call search_documents")
        documents = qdrant_vector_search(query, limit=limit, embedding_model_type=BAAI_MODEL_NAME)
    except Exception as e:
        print(f"Exception {e}")
        return "Exception "+{e}+"occurred!"
    if not documents:
        return "No relevant documents found!"

    result = []
    for document in documents:
        result.append(
            f"""
    Source: {document["source"]}
    Similarity: {document["score"]:.4f}
    {document["text"]}
    #""".strip())
    return "\n\n---\n\n".join(result)


async def run_mcp_server():
    # run_http_async startet einen uvicorn-Server im Hintergrund.
    # Standardmäßig wird der moderne "streamable-http"-Transport verwendet.
    await mcp.run_http_async(
        host="127.0.0.1",
        port=3005,
        transport="streamable-http"  # Optionen: "streamable-http", "sse" oder "http"
    )

if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["mcp.server.streamable_http"] = {"level": "CRITICAL"}
    mlflow.anthropic.autolog()
    asyncio.run(run_mcp_server())