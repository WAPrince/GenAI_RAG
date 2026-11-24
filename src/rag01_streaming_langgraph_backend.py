import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from mds.streaming.backend.graph import build_graph

# Globaler Speicher für die kompilierte Graph-Instanz
app_state = {}

print("S1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startet den LangGraph + MCP-Client beim Serverstart und räumt beim Beenden auf."""
    async with build_graph() as graph:
        app_state["graph"] = graph
        yield
    app_state.clear()

print("S2")
app = FastAPI(title="LangGraph MCP Chat API", lifespan=lifespan)
print("S3")

# CORS für Frontend-Anfragen (z.B. React, Vue, Next.js) freigeben
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion auf konkrete Frontend-Domain einschränken!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("S4")


# --- PYDANTIC SCHEMAS ---
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Nachricht des Nutzers")
    thread_id: str = Field(default="default-session", description="ID für den Chat-Verlauf")

class ChatResponse(BaseModel):
    response: str
    thread_id: str

# --- ENDPOINTS ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Klassischer Request/Response-Endpoint (wartet auf die komplette Antwort)."""
    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Graph nicht initialisiert.")

    config = {"configurable": {"thread_id": request.thread_id}}
    input_message = {"messages": [HumanMessage(content=request.message)]}

    try:
        result = await graph.ainvoke(input_message, config=config)
        final_message = result["messages"][-1].content
        return ChatResponse(response=final_message, thread_id=request.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Server-Sent Events (SSE) Endpoint für Live-Streaming von Tokens & Tool-Status."""
    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Graph nicht initialisiert.")

    config = {"configurable": {"thread_id": request.thread_id}}
    input_message = {"messages": [HumanMessage(content=request.message)]}

    async def event_generator():
        try:
            async for event in graph.astream_events(input_message, config=config, version="v2"):
                kind = event["event"]

                # 1. Text-Tokens vom LLM streamen
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        payload = json.dumps({"type": "token", "content": chunk.content})
                        yield f"data: {payload}\n\n"

                # 2. Frontend informieren, wenn ein MCP-Tool gestartet wird
                elif kind == "on_tool_start":
                    payload = json.dumps({"type": "tool_start", "tool": event["name"]})
                    yield f"data: {payload}\n\n"

                # 3. Frontend informieren, wenn das MCP-Tool fertig ist
                elif kind == "on_tool_end":
                    payload = json.dumps({"type": "tool_end", "tool": event["name"]})
                    yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
