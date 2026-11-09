import asyncio
import uvicorn
import inspect
from fastmcp import FastMCP
from pycharmai.rag.vector_store import search_documents


mcp = FastMCP()
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))


#MLflow
TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
mlflow.set_tracking_uri(TRACKING_URI)
#assert Version(mlflow.__version__) >= Version("3.15"), (
#    f"The MCP Registry needs MLflow >= 3.15 (found {mlflow.__version__})."
#)
#print(f"MLflow {mlflow.__version__} -> {TRACKING_URI}")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.tool()
def search_in_vectordb(query: str, limit: int = 3) -> str:
    """Search the qdrant vectordb ( knowledge base )  for information relevant to the query."""

    try:
        documents = search_documents(query, limit=limit)
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




async def main():
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
    asyncio.run(main())