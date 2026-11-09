import asyncio,os
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.scorers import RelevanceToQuery, Correctness
from dotenv import load_dotenv
from pycharmai.graph.state import AgentState
from pycharmai.llm.bedrock import create_llm
from pycharmai.rag.mcp.fastmcpClient import create_mcp_client



TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI  # the MCP server subprocess reads this
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.set_experiment("mcp-server-demo-" + str(uuid.uuid4()))
print(f"Experiment: {experiment.name} (id={experiment.experiment_id})")


mlflow.trace(span_type=SpanType.AGENT)
async def run_agent(prompt: str) -> None:
    load_dotenv()
    print(f" Is the MLflow server running? ( .venv/Scripts/mlflow.exe server --port 5000  )")
    print(f" AWS_REGION: {os.getenv("AWS_REGION")}")
    print(f" AWS_PROFILE: {os.getenv("AWS_PROFILE")}")

    mcp_url = "http://localhost:3005/mcp"

    ###########################################################################

    async with streamablehttp_client(mcp_url) as (read, write, _):
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
                config=config,
            )
            print_result(result)

            judges = [
                RelevanceToQuery(),
                Correctness()
            ]
            #eval_results = mlflow.genai.evaluate(
            #    data=result[-1].content,  # Übergabe der gesammelten Trace-Objekte
            #    scorers=judges
            #)
            #print(eval_results.metrics)
            #eval_df = eval_results.tables["eval_results_table"]
            #print(eval_df[["inputs", "outputs", "relevance_to_query/v1/score", "relevance_to_query/v1/justification"]].head())


def print_result(result: dict) -> None:
    print("\n========================================")
    print(" RESULT")
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


async def main() -> None:
    mlflow.anthropic.autolog()
    await run_agent(
        # RAG-Szenario:
        "Wer ist im Harz gelandet?" # if not tool call detected in MLflow, change to a question not  yet nswered before ( not being in the cache )
    )

if __name__ == "__main__":
    asyncio.run(main())