import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from mds.graph.state import AgentState


load_dotenv()

# Checkpointer für In-Memory-Sitzungsverwaltung (pro thread_id)
memory = MemorySaver()

# MCP-Server-Konfiguration (hier können weitere MCP-Server eingetragen werden)
MCP_SERVERS = {
                "medicaments__dense_sparse": {
                    "transport": "streamable-http",
                    "url": "http://localhost:3006/mcp",
            }
    }

@asynccontextmanager
async def build_graph():
    """
    Initialisiert den MCP-Client, lädt die Tools und kompiliert den LangGraph.
    Wird als Async Context Manager genutzt, damit der MCP-Subprozess sauber beendet wird.
    """
    client = MultiServerMCPClient(MCP_SERVERS)
    async with client.session("medicaments__dense_sparse") as session:
        tools = await load_mcp_tools(session)

        # LLM mit den geladenen MCP-Tools verknüpfen
        llm = ChatAnthropic(model="claude-sonnet-4-6")
        llm_with_tools = llm.bind_tools(tools)

        # --- NODES ---
        async def agent_node(state: AgentState):
            system_prompt = SystemMessage(
                content="Du bist ein hilfreicher Assistent. Nutze die verfügbaren Werkzeuge, wenn nötig."
            )
            messages = [system_prompt] + state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        # Der vorgefertigte ToolNode führt automatisch alle angeforderten MCP-Tools aus
        mcp_tool_node = ToolNode(tools)

        # --- GRAPH AUFBAU ---
        builder = StateGraph(AgentState)

        builder.add_node("agent", agent_node)
        builder.add_node("tools", mcp_tool_node)

        # Einstiegspunkt
        builder.add_edge(START, "agent")

        # Bedingte Kante: Wenn das LLM ein Tool aufruft -> "tools", sonst -> END
        builder.add_conditional_edges("agent", tools_condition)

        # Nach der Tool-Ausführung geht es zurück zum Agenten zur Auswertung
        builder.add_edge("tools", "agent")

        # Kompilieren des Graphen mit Checkpointer für Chat-Historie
        graph = builder.compile(checkpointer=memory)
        yield graph
