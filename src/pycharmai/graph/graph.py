
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from pycharmai.graph.state import AgentState
from pycharmai.llm.bedrock import create_llm
from pycharmai.rag.mcp.fastmcpClient import create_mcp_client
from pycharmai.rag.tools.calculator import calculate
from pycharmai.rag.tools.files import delete_file
from pycharmai.rag.tools.weather import get_temperature

import os

#mlflow.langchain.autolog()

#@mlflow.trace(span_type=SpanType.CHAIN)
async def create_graph():


    load_dotenv()

    #GUARDRAIL_VERSION = "Version 1"
    #bedrock_runtime = boto3.client("bedrock-runtime")
    #GUARDRAIL_ID = "arn:aws:bedrock:us-east-1:043924217572:guardrail/6zi9vgqn29wn"

    llm = create_llm()
    mcp_client = create_mcp_client()
    mcp_tools = await mcp_client.get_tools()
    tools = [
        calculate,#pycharmai/rag/tools/calculator.py
        delete_file,#pycharmai/rag/tools/files.py
        get_temperature,#pycharmai/rag/tools/weather.py

        *mcp_tools,##pycharmai/rag/mcp/fastmcpServer.py
    ]
    llm_with_tools = llm.bind_tools(tools)



    #@mlflow.trace(span_type=SpanType.CHAT_MODEL)
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

        #raise
        #blocked = result["action"] == "GUARDRAIL_INTERVENED"

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
    #builder.add_edge(START, "guardrail_input")
    #builder.add_edge("guardrail_input", "call_model")

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