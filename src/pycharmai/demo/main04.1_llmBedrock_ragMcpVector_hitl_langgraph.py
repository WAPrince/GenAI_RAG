import asyncio, os
from anthropic.types import deleted_file
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.types import Command

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from dotenv import load_dotenv
from pycharmai.graph.state import AgentState
from pycharmai.llm.bedrock import create_llm
from pycharmai.rag.mcp.fastmcpClient import create_mcp_client
from pycharmai.rag.tools.files import delete_file

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
            tools = [
                        delete_file,
                        *mcp_tools
                    ]
            llm_with_tools = llm.bind_tools(tools)
            tool_node = ToolNode(
                tools,
                handle_tool_errors=True,
            )

            def call_model(state: AgentState) -> dict:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            def human_approval(state: AgentState) -> dict:
                last_message = state["messages"][-1]
                tool_calls = getattr(last_message, "tool_calls", [])
                for tool_call in tool_calls:
                    if tool_call["name"] != "delete_file":
                        continue
                    filename = tool_call["args"]["filename"]
                    decision = interrupt(
                        {
                            "type": "approval_required",
                            "action": "delete_file",
                            "filename": filename,
                            "message": f"Do you really want to delete '{filename}'?",
                        }
                    )
                    if decision == "approve":
                        print("Decision: 'approve' selected")
                        return {"approval_status": "approved"}
                    else:
                        print("Decision: 'rejected' selected")
                        return {"approval_status": "rejected"}
                return {"approval_status": "approved"}

            def route_after_model(state: AgentState) -> str:
                last_message = state["messages"][-1]
                tool_calls = getattr(last_message, "tool_calls", [])
                if not tool_calls:
                    return "end"
                for tool_call in tool_calls:
                    if tool_call["name"] == "delete_file":
                        return "human_approval"
                return "tools"

            def route_after_approval(state: AgentState) -> str:
                if state.get("approval_status") == "approved":
                    print("approved routes to toolsapprove")
                    return "tools"
                else:
                    print("rejected routes to END")
                    return "end"

            builder = StateGraph(AgentState)

            builder.add_node("call_model", call_model)
            builder.add_node("human_approval", human_approval)
            builder.add_node("tools", tool_node)

            builder.add_edge(START, "call_model")
            builder.add_conditional_edges(
                "call_model",
                route_after_model,
                {
                    "end": END,
                    "human_approval": "human_approval",
                    "tools": "tools",
                },
            )
            builder.add_conditional_edges(
                "human_approval",
                route_after_approval,
                {
                    "tools": "tools",
                    "end": END,
                },
            )
            builder.add_edge("tools", "call_model")

            checkpointer = InMemorySaver()

            graph= builder.compile(
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

            ### Simple Interrupt Handling
            if "__interrupt__" not in result:
                print_result(result)
                return

            interrupts = result["__interrupt__"]

            for interrupt_info in interrupts:
                print("\n========================================")
                print(" HUMAN APPROVAL REQUIRED")
                print("========================================")
                print(interrupt_info.value)

            decision = input(
                "\nAktion ausführen? [approve/reject]: "
            ).strip().lower()

            while decision not in {"approve", "reject"}:
                decision = input(
                    "Bitte 'approve' oder 'reject' eingeben: "
                ).strip().lower()

            result = await graph.ainvoke(
                Command(resume=decision),
                config=config,
            )
            ### Simple Interrupt Handling END
            print_result(result)


def print_result(result: dict) -> None:
    print("\n========================================")
    print(" RESULT")
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


async def main() -> None:
    await run_agent(
        "Lösche die Datei data/news_min.txt!"
    )


if __name__ == "__main__":
    asyncio.run(main())
