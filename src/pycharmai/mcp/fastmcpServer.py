import mlflow
from mcp.server.fastmcp import FastMCP
from mlflow.entities import SpanType
from packaging.version import Version

from pycharmai.rag.vector_store import search_documents
import inspect

mcp = FastMCP("demo-server",port=3004)
print(inspect.signature(FastMCP))
print(inspect.signature(mcp.run))

assert Version(mlflow.__version__) >= Version("3.5.1"), (
    f"The MLflow MCP Server needs MLflow >= 3.5.1 (found {mlflow.__version__})."
)



@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.tool()
def search_knowledge(query: str, limit: int = 3) -> str:
    """Search the knowledge base for information relevant to the query."""

    documents = search_documents(query, limit=limit)

    if not documents:
        return "No relevant documents found."

    result = []

    for document in documents:
        result.append(
            f"""
Source: {document["source"]}
Similarity: {document["score"]:.4f}

{document["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(result)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")