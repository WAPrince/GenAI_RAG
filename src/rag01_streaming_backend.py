import os
import sys
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from flask.cli import load_dotenv
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <--- NEU: MemorySaver importieren


load_dotenv()
#os.environ.setdefault("ANTHROPIC_API_KEY", "DEIN_ANTHROPIC_API_KEY")

agent_executor = None
mcp_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, mcp_client

    mcp_client = MultiServerMCPClient(
        {
            "medicaments__dense_sparse": {
            "transport": "streamable-http",
            "url": "http://localhost:3006/mcp",
            }
        }
    )

    tools = await mcp_client.get_tools()

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        streaming=True
    )

    # <--- NEU: In-Memory Checkpointer erstellen und an den Agenten binden
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# <--- NEU: thread_id im Request-Modell ergänzen
class ChatRequest(BaseModel):
    message: str
    thread_id: str


async def stream_langgraph_response(user_message: str, thread_id: str):
    inputs = {"messages": [("user", user_message)]}

    # <--- NEU: thread_id an LangGraph übergeben, damit der Verlauf zugeordnet wird
    config = {"configurable": {"thread_id": thread_id}}

    async for event in agent_executor.astream_events(inputs, config=config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield f"data: {json.dumps({'type': 'token', 'content': block['text']})}\n\n"
            elif isinstance(content, str) and content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        elif kind == "on_tool_start":
            tool_name = event["name"]
            yield f"data: {json.dumps({'type': 'tool', 'content': f'Nutze MCP-Tool: {tool_name}...'})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        stream_langgraph_response(request.message, request.thread_id),
        media_type="text/event-stream"
    )
