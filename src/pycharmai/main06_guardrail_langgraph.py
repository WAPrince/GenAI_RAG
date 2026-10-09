import asyncio

import boto3
from botocore.exceptions import ClientError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph

from pycharmai.graph.state import AgentState

bedrock_runtime = boto3.client("bedrock-runtime")
GUARDRAIL_ID = "arn:aws:bedrock:us-east-1:043924217572:guardrail/wudwejrc0uij"
GUARDRAIL_VERSION = "1"
AWS_REGION = "us-east-1"
INSULTING_TEXT="hans@hansen.com"#Du Blödian"

def guardrail_input(state: AgentState) -> AgentState:

    guardrail_id=GUARDRAIL_ID
    guardrail_version=GUARDRAIL_VERSION
    region_name = AWS_REGION

    text_to_test=state["messages"][-1].content
    print(f"test to text: {text_to_test}")

    # Erstellt den Bedrock Runtime Client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )

    try:
        print(f"Teste Text gegen Guardrail <<{text_to_test}>>")

        # Aufruf der Amazon Bedrock Guardrail API
        response = bedrock_runtime.apply_guardrail(

            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",  # Kann 'INPUT' (Benutzereingabe) oder 'OUTPUT' (Modellantwort) sein
            content=[
                {
                    "text": {
                        "text": text_to_test
                    }
                }
            ]
        )

        # Auswertung der Antwort
        action = response.get("action")  # 'NONE' oder 'GUARDRAIL_INTERVENED'

        print("=== TEST ERGEBNIS ===")
        print(f"Eingegebener Text: '{text_to_test}'")
        print(f"Aktion der Guardrail: {action}")

        if action == "GUARDRAIL_INTERVENED":
            print("\n🚫 HINWEIS: Der Text wurde von der Guardrail blockiert!")

            # System-Antwort ausgeben (vordefinierte Guardrail-Nachricht)
            for output in response.get("outputs", []):
                print(f"Ausgabe-Nachricht: {output.get('text')}")

            # Details zu den verletzten Filtern anzeigen
            assessments = response.get("assessments", [])
            print("\nKlassifizierte Verstöße:")
            for assessment in assessments:
                # Prüfung auf Content Filter (Hate, Insults, Sexual, Violence)
                if "contentPolicy" in assessment:
                    for filter_type in assessment["contentPolicy"].get("filters", []):
                        if filter_type.get("action") == "BLOCKED":
                            print(
                                f" - [Inhaltsfilter] Typ: {filter_type.get('type')}, Konfidenz: {filter_type.get('confidence')}")

                # Prüfung auf blockierte Wörter / Phrasen
                if "wordPolicy" in assessment:
                    for custom_word in assessment["wordPolicy"].get("customWords", []):
                        if custom_word.get("action") == "BLOCKED":
                            print(f" - [Wortfilter] Blockiertes Wort erkannt: {custom_word.get('match')}")
                    for managed_word in assessment["wordPolicy"].get("managedWordLists", []):
                        if managed_word.get("action") == "BLOCKED":
                            print(f" - [Wortfilter] Profanität/Schimpfwort erkannt: {managed_word.get('match')}")

                # Prüfung auf sensible Daten (PII)
                if "sensitiveInformationPolicy" in assessment:
                    for pii in assessment["sensitiveInformationPolicy"].get("piiEntities", []):
                        if pii.get("action") == "BLOCKED":
                            print(f" - [PII Filter] Sensible Daten blockiert: {pii.get('type')}")

                # Prüfung auf blockierte Themen (Topic Deny List)
                if "topicPolicy" in assessment:
                    for topic in assessment["topicPolicy"].get("topics", []):
                        if topic.get("action") == "BLOCKED":
                            print(f" - [Themenfilter] Blockiertes Thema: {topic.get('name')}")
        else:
            print("\n✅ ROLLE BESTANDEN: Der Text ist sauber und wurde nicht blockiert.")

    except ClientError as e:
        print(f"Ein Fehler ist aufgetreten: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")

async def createGraph():

    builder = StateGraph(AgentState)

    builder.add_node("guardrail_input", guardrail_input)
    builder.add_edge(START, "guardrail_input")
    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )

async def graph() :
    graph = await createGraph()

    config = {
        "configurable": {
            "thread_id": "console-session-1"
        }
    }

    result = await graph.ainvoke(
        {
            "messages": [
                ("user", INSULTING_TEXT)
            ]
        },
        config=config,
    )

async def call():
    await graph()

if __name__ == "__main__":
    asyncio.run(call())

