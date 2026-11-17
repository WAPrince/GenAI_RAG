from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# 1. State Definition
class State(TypedDict):
    agent_1_result: str
    agent_2_result: str


# 2. Knoten für Agent 1 (mit erstem Interrupt)
def agent_1_node(state: State):
    print("\n🤖 [Agent 1]: Generiere ersten Vorschlag...")
    idea = "Konzept für ein Elektroauto"

    # 1. Interrupt: Wartet auf Freigabe für Agent 1
    user_decision = interrupt({
        "msg": f"Gefällt Ihnen die Idee '{idea}' von Agent 1? (ja/nein): "
    })

    print(f"-> Agent 1 hat Antwort erhalten: {user_decision}")
    return {"agent_1_result": f"{idea} ({user_decision})"}


# 3. Knoten für Agent 2 (mit zweitem Interrupt)
def agent_2_node(state: State):
    print("\n🤖 [Agent 2]: Baue auf dem Ergebnis von Agent 1 auf...")
    detail = f"Marketingkampagne für [{state['agent_1_result']}]"

    # 2. Interrupt: Wartet auf Freigabe für Agent 2
    user_decision = interrupt({
        "msg": f"Soll die Kampagne '{detail}' gestartet werden? (ja/nein): "
    })

    print(f"-> Agent 2 hat Antwort erhalten: {user_decision}")
    return {"agent_2_result": f"{detail} ({user_decision})"}


# 4. Graph-Konstruktion
builder = StateGraph(State)
builder.add_node("agent_1", agent_1_node)
builder.add_node("agent_2", agent_2_node)

builder.add_edge(START, "agent_1")
builder.add_edge("agent_1", "agent_2")
builder.add_edge("agent_2", END)

memory = MemorySaver()
agent = builder.compile(checkpointer=memory)

# --- 5. INTERAKTIVE STEUERUNG IN DER KONSOLE ---
config = {"configurable": {"thread_id": "zwei_agenten_session"}}

print("--- 🚀 STARTE WORKFLOW ---")


# Hilfsfunktion, um den Graphen bis zum nächsten Interrupt (oder Ende) zu streamen
def run_until_interrupt_or_end(input_cmd, config):
    stream = agent.stream(input_cmd, config, stream_mode="updates")
    for event in stream:
        # Sobald ein Interrupt-Signal kommt, stoppen wir das Streaming
        if "__interrupt__" in event:
            return True
    return False


# INITIALER START: Wir füttern den Graphen mit dem Start-State
has_interrupt = run_until_interrupt_or_end({"agent_1_result": "", "agent_2_result": ""}, config)

# Hauptschleife: Solange der Graph durch einen Interrupt pausiert ist, fragen wir den User
while has_interrupt:
    state_info = agent.get_state(config)

    if state_info.tasks:
        current_task = state_info.tasks[0] if isinstance(state_info.tasks, (list, tuple)) else state_info.tasks

        if hasattr(current_task, "interrupts") and current_task.interrupts:
            # Holen der Daten, die im aktuellen Knoten an interrupt() übergeben wurden
            interrupt_data = current_task.interrupts[0].value if isinstance(current_task.interrupts, (list,
                                                                                                      tuple)) else current_task.interrupts.value

            # Input über die Konsole einlesen (inkl. Validierung)
            konsolen_eingabe = ""
            while True:
                konsolen_eingabe = input(interrupt_data["msg"]).strip().lower()
                if konsolen_eingabe in ["ja", "nein"]:
                    break
                print("❌ Bitte exakt mit 'ja' oder 'nein' antworten.")

            print(f"\n--- Setze Workflow mit '{konsolen_eingabe}' fort ---")

            # Verpacken der Eingabe in das Command für die Fortsetzung
            resume_command = Command(resume=konsolen_eingabe)

            # Fortsetzen: Wir übergeben das Command an die Streaming-Funktion
            has_interrupt = run_until_interrupt_or_end(resume_command, config)
        else:
            break
    else:
        break

print("\n--- 🎉 WORKFLOW ERFOLGREICH BEENDET ---")
# Finalen Zustand ausgeben
print("Finaler State:", agent.get_state(config).values)
