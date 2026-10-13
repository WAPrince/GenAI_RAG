import json
import os
import uuid
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY ist nicht gesetzt."
    )
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
SYSTEM_PROMPT = """Du bist ein hilfreicher IT-Assistent.
Antworte auf Deutsch, wenn der Benutzer Deutsch schreibt.
Erkläre technische Themen präzise und mit kurzen Beispielen.
"""


print("hallo")
llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
)


def call_model(state: MessagesState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm.invoke(messages)

    return {"messages": [response]}

print("hallo 1")
builder = StateGraph(MessagesState)
builder.add_node("assistant", call_model)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
print("hallo 2")
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
print("hallo 3")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = None

app = FastAPI(title="LangGraph Streaming Chat API")
print("hallo 4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("hallo 5")
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncIterator[str]:
        yield f"event: thread\ndata: {thread_id}\n\n"

        async for chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "assistant":
                continue

            content = chunk.content

            if isinstance(content, str) and content:
                yield f"event: token\ndata: {json.dumps(content)}\n\n"

        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
