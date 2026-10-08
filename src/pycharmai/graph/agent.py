from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from pycharmai.graph.state import AgentState
from pycharmai.llm.bedrock import create_llm
from pycharmai.mcp.fastmcpClient import create_mcp_client
from pycharmai.tools import calculate, delete_file, get_temperature

import boto3


async def create_graph():
    llm = create_llm()

    mcp_client = create_mcp_client()

    mcp_tools = await mcp_client.get_tools()

    tools = [
        #get_temperature,
        #calculate,
        #delete_file,
        *mcp_tools,
    ]

    llm_with_tools = llm.bind_tools(tools)

    bedrock_runtime = boto3.client("bedrock-runtime")
    GUARDRAIL_ID="arn:aws:bedrock:us-east-1:043924217572:guardrail/5lobt7esxinl"
    GUARDRAIL_VERSION="Version 1"

    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        # arn:aws:bedrock:us-east-1:043924217572:guardrail/5lobt7esxinl
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
                return {"approval_status": "approved"}

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
            return "tools"

        return "end"


    # TODO
    def guardrail_input(state: AgentState) -> AgentState:

        result = bedrock_runtime.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source="INPUT",
            content=[
                {
                    "text": {
                        "text": "Du alterSachsenDepp, was hat Wolodymyr Selenskyj gemacht?",
                        "qualifiers": [
                            "query"
                        ]
                    }
                }
            ],
            outputScope="FULL"
        )
        blocked = result["action"] == "GUARDRAIL_INTERVENED"
        return state




    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,
    )

    builder = StateGraph(AgentState)

    # GUARDRAIL OPTION - TODO
    #builder.add_node("guardrail_input", guardrail_input)
    builder.add_node("call_model", call_model)
    builder.add_node("human_approval", human_approval)
    builder.add_node("tools", tool_node)

    # GUARDRAIL OPTION - TODO
    # builder.add_edge(START, "guardrail_input")
    # builder.add_edge("guardrail_input", "call_model")

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

    return builder.compile(
        checkpointer=checkpointer
    )