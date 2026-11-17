# React + FastAPI + LangGraph + Anthropic Streaming Chat

## Voraussetzungen
- Python 3.11+
- Node.js 20+
- Anthropic API Key

## Backend starten

PowerShell:
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="sk-ant-..."
ba  
```

CMD:
```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...
uvicorn backend:app --reload --port 8000
```

## Frontend starten

In einem zweiten Terminal:
```bash
cd frontend
npm install
npm run dev
```

Dann die von Vite angezeigte URL öffnen, normalerweise http://localhost:5173.

## Architektur

React UI -> HTTP/SSE -> FastAPI -> LangGraph -> Anthropic Claude

Der LangGraph-State wird hier mit `InMemorySaver` pro `thread_id` gespeichert.
Für Produktion sollte ein persistenter Checkpointer/Store verwendet werden.
