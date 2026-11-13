import asyncio, os
import uuid
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
from mds.llm.bedrock import create_llm
from mds.rag.mcp.fastmcpClient import create_mcp_client
from mds.graph.state import AgentState

TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI  # the MCP server subprocess reads this
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("langgraph-" + str(uuid.uuid4()))


@mlflow.trace(name="run_agent", span_type=SpanType.AGENT)
async def run_agent(prompt: str) -> None:
    load_dotenv()
    print(f" Is the MLflow server running? ( .venv/Scripts/mlflow.exe server --port 5000  )")
    print(f" Is the MCP server running? )")
    print(f" Have documents from ./data/input/news been imported? )")
    print(f" AWS_REGION: {os.getenv("AWS_REGION")}")
    print(f" AWS_PROFILE: {os.getenv("AWS_PROFILE")}")

    mcp_url = "http://localhost:3005/mcp"

    ###########################################################################

    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            llm = create_llm()
            mcp_client = create_mcp_client()
            mcp_tools = await mcp_client.get_tools()
            tools = [*mcp_tools]
            llm_with_tools = llm.bind_tools(tools)
            tool_node = ToolNode(
                tools,
                handle_tool_errors=True,
            )

            @mlflow.trace(name="call_llm", span_type=SpanType.LLM)
            def call_model(state: AgentState) -> dict:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            def route_after_model(state: AgentState) -> str:
                last_message = state["messages"][-1]
                tool_calls = getattr(last_message, "tool_calls", [])
                if not tool_calls:
                    return "end"
                for tool_call in tool_calls:
                    if tool_call["name"] == "delete_file":
                        return "human_approval"
                return "tools"

            builder = StateGraph(AgentState)

            builder.add_node("call_model", call_model)
            builder.add_node("tools", tool_node)

            builder.add_edge(START, "call_model")
            builder.add_conditional_edges(
                "call_model",
                route_after_model,
                {
                    "end": END,
                    "tools": "tools",
                },
            )
            builder.add_edge("tools", "call_model")

            checkpointer = InMemorySaver()
            graph = builder.compile(
                checkpointer=checkpointer
            )

            config = {
                "configurable": {
                    "thread_id": "console-session-1"
                }
            }
            result = await graph.ainvoke(
                {
                    "messages": [
                        ("user", prompt)
                    ]
                },
                config=config
            )

            print_result(result)

            judges = [
                RelevanceToQuery(),
                Correctness()
            ]
            try:
                eval_results = mlflow.genai.evaluation(data=result, scorers=judges)
                eval_df = eval_results.tables["eval_results_table"]
                print(eval_df[["inputs", "outputs", "relevance_to_query/v1/score",
                               "relevance_to_query/v1/justification"]].head())
            except Exception as e:
                print(f"Exception: {e}")

def print_result(result: dict) -> None:
    print("\n========================================")
    print(" RESULT START")
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)
    print(" RESULT END\n")
    print("\n========================================")




async def main() -> None:
    """
    Execute search queries against Qdrant collections 'medicaments' and 'news'
    The system prompt part contains the Qdrant collection that contains the desired answers
    The system prompt part contains the embedding model that should be applied to the query for the similiarity search (  it is the same as the imported Qdrant collection data )
    """

    mlflow.anthropic.autolog()
    await run_agent(
        "Übergebe Collection-Name 'news'  und Embedding-Model 'HARRIER' an das MCP Tool! Was findest du zu Harz, Eastwick, Hannover? Rufe die Vektordatenbank auf!"

        #"Übergebe Collection-Name 'medicaments'  und Embedding-Model 'HARRIER' an das MCP Tool! Was findest du zu Amlodipin? Rufe die Vektordatenbank auf!"
    )

if __name__ == "__main__":
    asyncio.run(main())
