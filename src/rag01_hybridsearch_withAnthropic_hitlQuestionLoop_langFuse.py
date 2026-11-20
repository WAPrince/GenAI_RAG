import asyncio, os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.tools import load_mcp_tools
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from transformers.trainer_callback import CallbackHandler
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI  # Oder ChatDatabricks(...)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from mds.rag.mcp.fastmcpClient import MCP_SERVERS
import warnings

# 1. State definieren
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    should_exit: bool


async def run_mcp_agent(prompt:str, qdrant_collection:str):

    load_dotenv()

    print(f" ANTHROPIC_API_KEY: {os.getenv("ANTHROPIC_API_KEY")}")
    print(f" LANGFUSE_PUBLIC_KEY: {os.getenv("LANGFUSE_PUBLIC_KEY")}")
    print(f" LANGFUSE_SECRET_KEY: {os.getenv("LANGFUSE_SECRET_KEY")}")
    print(f" LANGFUSE_BASE_URL: {os.getenv("LANGFUSE_BASE_URL")}")

    print(f" Is the MCP server running? )")
    print(
        f" Have documents from ./data/input/medicaments been imported into the Qdrant collections? ( use qdrant_data_importer.py ))")

    # OpenTelemetry für Anthropic aktivieren
    AnthropicInstrumentor().instrument()
    langfuse_handler = CallbackHandler()

    # MCP-Client starten und Tools dynamisch vom Server laden
    client = MultiServerMCPClient(MCP_SERVERS)

    model = init_chat_model(
        model="claude-sonnet-4-6",
        model_provider="anthropic"
    )

    async with client.session(qdrant_collection) as session:
            mcp_tools = await load_mcp_tools(session)
            model_with_tools = model.bind_tools(mcp_tools)

            tool_node = ToolNode(mcp_tools)

            # 3. Knoten (Nodes) definieren
            def human_input_node(state: AgentState):
                """Wartet auf die Eingabe des Nutzers und prüft auf Beendigung."""
                user_input = interrupt("Bitte geben Sie Ihre Frage ein (oder 'exit' zum Beenden):")

                # Prüfen, ob der Nutzer beenden möchte
                if user_input.strip().lower() in ["exit", "quit", "ende", "stop", "beenden"]:
                    return {"should_exit": True}

                # Löschbefehle für alle bisherigen Nachrichten (inkl. Tool-Calls) erzeugen
                delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

                # Alte Nachrichten entfernen und nur die neue Frage als frischen Kontext setzen
                return {
                    "messages": delete_messages + [HumanMessage(content=user_input)],
                    "should_exit": False,
                }

            async def agent_node(state: AgentState):
                """Das Modell überlegt, nutzt ggf. MCP-Tools oder formuliert die Antwort."""
                response = await  model_with_tools.ainvoke(state["messages"])
                return {"messages": [response]}

            # 4. Routing-Funktionen
            def route_after_human(state: AgentState) -> Literal["agent", "__end__"]:
                """Beendet den Graph bei Exit-Befehl, sonst weiter zum Agenten."""
                if state.get("should_exit"):
                    return END
                return "agent"

            def route_after_agent(state: AgentState) -> Literal["tools", "human_input"]:
                """Ruft MCP-Tools auf oder wartet auf die nächste Benutzeranweisung."""
                last_message = state["messages"][-1]
                if last_message.tool_calls:
                    return "tools"
                return "human_input"

            # 5. Graph zusammenbauen
            builder = StateGraph(AgentState)

            builder.add_node("human_input", human_input_node)
            builder.add_node("agent", agent_node)
            builder.add_node("tools", tool_node)

            # Startet direkt mit der Warte-Position auf den Benutzer
            builder.add_edge(START, "human_input")

            # Nach Nutzereingabe: Entweder Ende oder Agent
            builder.add_conditional_edges("human_input", route_after_human)

            # Nach Agent: Entweder MCP-Tool ausführen oder Antwort zeigen & auf Nutzer warten
            builder.add_conditional_edges("agent", route_after_agent)

            # Nach Tool-Ausführung: Ergebnis zurück an den Agenten zur Verarbeitung
            builder.add_edge("tools", "agent")

            # Mit Speicher (Checkpointer) kompilieren
            memory = MemorySaver()
            graph = builder.compile(checkpointer=memory)

            # 6. Interaktive Schleife für die Konsole / UI
            config = {"configurable": {"thread_id": "mcp-session-1"}}

            # Graph initial starten (läuft bis zum ersten interrupt in 'human_input')
            await graph.ainvoke({"messages": [prompt], "should_exit": False}, config)

            while True:
                # Aktuellen Zustand prüfen, ob der Graph bereits beendet wurde
                current_state = await graph.aget_state(config)
                if not current_state.next:
                    print("Sitzung beendet.")
                    break

                # Benutzereingabe abfragen
                user_text = input("\nDu: ")

                # Graph mit der Nutzereingabe fortsetzen (astream für asynchrone MCP-Tools)
                async for event in graph.astream(
                    Command(resume=user_text), config, stream_mode="updates"#values"
                ):
                    # Wir prüfen gezielt nur auf Updates aus dem Knoten 'agent'
                    if "agent" in event:
                        last_msg = event["agent"]["messages"][-1]
                        if last_msg.type == "ai" and not last_msg.tool_calls:
                            print(f"\nAssistent: {last_msg.content}")




if __name__ == "__main__":

    asyncio.run(run_mcp_agent( prompt="Verwende Embedding-Model 'BAAI', übergebe Collection-Name 'medicaments__dense_sparse', führe eine hybride Suche aus", qdrant_collection="medicaments__dense_sparse"))