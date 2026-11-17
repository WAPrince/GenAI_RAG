import { useRef, useState } from "react";

const API_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const textareaRef = useRef(null);

  async function sendMessage(event) {
    event?.preventDefault();

    const text = input.trim();
    if (!text || busy) return;

    setError("");
    setInput("");
    setBusy(true);

    setMessages((current) => [
      ...current,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          message: text,
          thread_id: threadId,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          const lines = rawEvent.split("\n");
          const eventLine = lines.find((line) => line.startsWith("event:"));
          const dataLine = lines.find((line) => line.startsWith("data:"));

          if (!eventLine || !dataLine) continue;

          const eventName = eventLine.slice(6).trim();
          const data = dataLine.slice(5).trim();

          if (eventName === "thread") {
            setThreadId(data);
          }

          if (eventName === "token") {
            const token = JSON.parse(data);

            setMessages((current) => {
              const copy = [...current];
              const last = copy.length - 1;

              copy[last] = {
                ...copy[last],
                content: copy[last].content + token,
              };

              return copy;
            });
          }
        }
      }
    } catch (err) {
      console.error(err);
      setError(
        "Verbindung zum Backend fehlgeschlagen. Läuft FastAPI auf Port 8000?"
      );

      setMessages((current) => {
        const copy = [...current];
        const last = copy.length - 1;

        if (copy[last]?.role === "assistant" && !copy[last].content) {
          copy[last] = {
            ...copy[last],
            content: "Fehler beim Abrufen der Antwort.",
          };
        }

        return copy;
      });
    } finally {
      setBusy(false);
      textareaRef.current?.focus();
    }
  }

  function newChat() {
    if (busy) return;
    setMessages([]);
    setThreadId(crypto.randomUUID());
    setError("");
    setInput("");
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="page">
      <main className="chat">
        <header className="header">
          <div>
            <h1>🤖 LangGraph Chat</h1>
            <p>React · FastAPI · LangGraph · Anthropic</p>
          </div>
          <button className="new-chat" onClick={newChat} disabled={busy}>
            Neuer Chat
          </button>
        </header>

        <section className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">💬</div>
              <h2>Wie kann ich helfen?</h2>
              <p>
                Die Antwort von Claude wird direkt während der Generierung
                angezeigt.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              className={`row ${message.role}`}
              key={`${index}-${message.role}`}
            >
              <div className="avatar">
                {message.role === "user" ? "Du" : "AI"}
              </div>
              <div className="bubble">
                {message.content}
                {busy &&
                  message.role === "assistant" &&
                  index === messages.length - 1 && (
                    <span className="cursor">▌</span>
                  )}
              </div>
            </div>
          ))}
        </section>

        {error && <div className="error">{error}</div>}

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Nachricht eingeben …"
            rows="1"
            disabled={busy}
          />
          <button
            className="send"
            type="submit"
            disabled={busy || !input.trim()}
          >
            {busy ? "…" : "Senden"}
          </button>
        </form>

        <footer>
          Enter = senden · Shift+Enter = neue Zeile · Thread:{" "}
          {threadId.slice(0, 8)}
        </footer>
      </main>
    </div>
  );
}

export default App;
