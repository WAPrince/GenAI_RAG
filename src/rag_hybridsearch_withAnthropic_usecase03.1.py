import asyncio, os
import uuid
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.scorers import RelevanceToQuery, Correctness
from dotenv import load_dotenv
from mds.graph.state import AgentState
from mds.rag.mcp.fastmcpClient import MCP_SERVERS
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor



#TRACKING_URI:str = "http://localhost:5000"
MCP_URL:str = "http://localhost:3006/mcp"
MAX_TOOL_CALLS:int = 1



async def run_agent(prompt: str, qdrant_collection: str) -> None:

    load_dotenv()

    print(f" ANTHROPIC_API_KEY: {os.getenv("ANTHROPIC_API_KEY")}")
    print(f" LANGFUSE_PUBLIC_KEY: {os.getenv("LANGFUSE_PUBLIC_KEY")}")
    print(f" LANGFUSE_SECRET_KEY: {os.getenv("LANGFUSE_SECRET_KEY")}")
    print(f" LANGFUSE_BASE_URL: {os.getenv("LANGFUSE_BASE_URL")}")

    print(f" Is the MCP server running? )")
    print(f" Have documents from ./data/input/news and/or /data/input/medicaments been imported into the Qdrant collections? ( use qdrantDataImporter.py ))")

    # OpenTelemetry für Anthropic aktivieren
    AnthropicInstrumentor().instrument()
    langfuse_handler = CallbackHandler()

    ###########################################################################

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:

            await session.initialize()
            model = init_chat_model(
                model="claude-sonnet-4-6",
                model_provider="anthropic"
            )
            client = MultiServerMCPClient(MCP_SERVERS)
            async with client.session(qdrant_collection) as session:
                mcp_tools = await load_mcp_tools(session)
                model_with_tools = model.bind_tools(mcp_tools)

                async def call_model(state: AgentState):
                    response = await model_with_tools.ainvoke(state["messages"])
                    return {"messages": [response]}

                async def tool_middleware_node(state: AgentState):
                    current_count = state.get("tool_call_count", 0)
                    new_count = current_count + 1
                    print(f"\n[Middleware] Tool-Aufruf {new_count} von maximal {MAX_TOOL_CALLS} wird ausgeführt...")
                    tool_node_instance = ToolNode(mcp_tools)
                    result = await tool_node_instance.ainvoke(state)
                    return {**result, "tool_call_count": new_count}

                def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
                    last_message = state["messages"][-1]
                    current_count = state.get("tool_call_count", 0)

                    if last_message.tool_calls:
                        if current_count >= MAX_TOOL_CALLS:
                            print(
                                f"\n[Middleware] HINWEIS: Limit von {MAX_TOOL_CALLS} Tool-Aufrufen erreicht! Die Anfrage kann nicht weiter bearbeitet werden.")
                            return "__end__"
                        return "tools"
                    return "__end__"

                builder = StateGraph(AgentState)
                builder.add_node("call_model", call_model)
                builder.add_node("tools", tool_middleware_node)

                builder.add_edge(START, "call_model")
                builder.add_conditional_edges("call_model", should_continue)
                builder.add_edge("tools", "call_model")

                checkpointer = InMemorySaver()
                graph = builder.compile(checkpointer=checkpointer)

                config = {
                    "configurable": {
                        "thread_id": "console-session-"+str(uuid.uuid4()),
                        "callbacks": [langfuse_handler]
                    }
                }

                result = await graph.ainvoke(
                    {
                        "messages": [
                            ("user", prompt)
                        ]
                    },
                    config=config,
                )
                messages = result.get("messages", [])
                if messages:
                    print(messages[-1].content)

                get_client().flush()


async def main(embeddingModel:str = "BAAI", qdrantCollection:str = "news", hybrid:bool=True ) -> None:
    """
    The system prompt controls: embedding model ( the same embedding model Qdrant must be applied to data and query for similarity search to work )
                                qdrant collection / grounding data  ( Qdrant collections 'medicaments' and 'news' )
                                hybrid search with reranking or non-hybrid search without reranking
    The user prompt contains:   user question
    """

    # PROMPT 1
    # Non-hybrid search with BAAI data and query embeddings
    #await run_agent(
    #   "Verwende Embedding-Model 'BAAI', übergebe Collection-Name 'medicaments', führe keine hybride Suche aus und beantworte die Benutzerfrage: 'Welche Nebenwirkungen gibt es, wenn ich alle Medikamente einnehme?'",
    #    qdrant_collection="medicaments"
    #)

    # PROMPT 2
    # hybrid search with BAAI data and query embeddings and re-ranking
    await run_agent(
        "Verwende Embedding-Model 'BAAI', übergebe Collection-Name 'medicaments__dense_sparse', führe eine hybride Suche aus und beantworte die Benutzerfrage: 'Welche Nebenwirkungen gibt es, wenn ich alle Medikamente einnehme?'",
        qdrant_collection="medicaments__dense_sparse"
    )


if __name__ == "__main__":
    asyncio.run(main())
