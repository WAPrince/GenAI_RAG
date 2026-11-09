import asyncio
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.scorers import RelevanceToQuery, Correctness

from pycharmai.graph.graph import create_graph
import pandas as pd
import os





#mlflow.trace(span_type=SpanType.AGENT)
async def run_agent(prompt: str) -> None:
    load_dotenv()
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

            def guardrail_input(state: AgentState) -> AgentState:
                try:
                    result = bedrock_runtime.apply_guardrail(
                        guardrailIdentifier=GUARDRAIL_ID,
                        guardrailVersion=GUARDRAIL_VERSION,
                        source="INPUT",
                        content=[
                            {
                                "text": {
                                    "text": "",
                                    "qualifiers": ["query"]
                                }
                            }
                        ]
                    )
                    print(f"Guardrail result: {result}")
                except ClientError as e:
                    error = e.response["Error"]
                    print("AWS error code:", error["Code"])
                    print("AWS error message:", error["Message"])
                    print("Region:", bedrock_runtime.meta.region_name)
                    print("Guardrail ID:", GUARDRAIL_ID)
                    print("Version:", repr(GUARDRAIL_VERSION))

                # raise
                # blocked = result["action"] == "GUARDRAIL_INTERVENED"

                return state

            builder.add_node("guardrail_input", guardrail_input)
            builder.add_node("call_model", call_model)
            builder.add_node("tools", tool_node)

            builder.add_edge(START, "guardrail_input")
            builder.add_edge("guardrail_input", "call_model")
            builder.add_conditional_edges(
                "call_model",
                route_after_model,
                {
                    "end": END,
                    "tools": "tools",
                },
            )


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




def print_result(result: dict) -> None:
    print("\n========================================")
    print(" RESULT")
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


async def main() -> None:
    await run_agent(
        # RAG-Szenario:
        "Was ist im Harz passiert?"
    )

if __name__ == "__main__":
    asyncio.run(main())