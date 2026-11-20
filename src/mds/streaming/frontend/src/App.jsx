import { useState } from "react";

export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [toolStatus, setToolStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // <--- NEU: Eindeutige Thread-ID pro Chat-Sitzung
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());

  // <--- NEU: Funktion zum Starten eines frischen Chats
  const startNewChat = () => {
    if (isLoading) return;
    setMessages([]);
    setThreadId(crypto.randomUUID());
    setToolStatus("");
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg, { role: "assistant", text: "" }]);
    setInput("");
    setIsLoading(true);
    setToolStatus("");

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // <--- NEU: thread_id im Body mitsenden
        body: JSON.stringify({ message: userMsg.text, thread_id: threadId }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const dataStr = line.replace("data: ", "").trim();
          if (dataStr === "[DONE]") break;

          const parsed = JSON.parse(dataStr);

          if (parsed.type === "tool") {
            setToolStatus(parsed.content);
          } else if (parsed.type === "token") {
            setToolStatus("");
            setMessages((prev) => {
              const updated = [...prev];
              const lastIndex = updated.length - 1;
              updated[lastIndex] = {
                ...updated[lastIndex],
                text: updated[lastIndex].text + parsed.content,
              };
              return updated;
            });
          }
        }
      }
    } catch (error) {
      console.error("Streaming-Fehler:", error);
    } finally {
      setIsLoading(false);
      setToolStatus("");
    }
  };

  return (
    <div style={{ maxWidth: 650, margin: "40px auto", fontFamily: "sans-serif" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Anthropic + LangGraph + MCP Chat</h2>
        <button
          onClick={startNewChat}
          disabled={isLoading}
          style={{
            padding: "6px 12px",
            borderRadius: 6,
            border: "1px solid #ccc",
            backgroundColor: "#fff",
            cursor: "pointer",
          }}
        >
          + Neuer Chat
        </button>
      </div>

      <div
        style={{
          border: "1px solid #ddd",
          borderRadius: 8,
          padding: 16,
          height: 400,
          overflowY: "auto",
          marginBottom: 16,
          backgroundColor: "#fafafa",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              marginBottom: 12,
              textAlign: msg.role === "user" ? "right" : "left",
            }}
          >
            <div
              style={{
                display: "inline-block",
                padding: "10px 14px",
                borderRadius: 12,
                backgroundColor: msg.role === "user" ? "#007bff" : "#e9ecef",
                color: msg.role === "user" ? "#fff" : "#000",
                whiteSpace: "pre-wrap",
                maxWidth: "80%",
              }}
            >
              {msg.text || (isLoading && idx === messages.length - 1 ? "..." : "")}
            </div>
          </div>
        ))}
        {toolStatus && (
          <div style={{ fontSize: 13, color: "#666", fontStyle: "italic" }}>
            ⚙️ {toolStatus}
          </div>
        )}
      </div>

      <form onSubmit={sendMessage} style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nachricht schreiben..."
          style={{ flex: 1, padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
        />
        <button
          type="submit"
          disabled={isLoading}
          style={{
            padding: "10px 20px",
            borderRadius: 6,
            border: "none",
            backgroundColor: "#007bff",
            color: "#fff",
            cursor: "pointer",
          }}
        >
          Senden
        </button>
      </form>
    </div>
  );
}
