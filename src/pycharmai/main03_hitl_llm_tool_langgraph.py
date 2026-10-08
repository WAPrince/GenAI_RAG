import asyncio

from langgraph.types import Command

from pycharmai.graph.agent import create_graph


async def run_agent(prompt: str) -> None:

    graph = await create_graph()

    config = {
        "configurable": {
            "thread_id": "console-session-1"
        }
    }

    result = await graph.ainvoke(
        {
            "messages": [
                ("user", prompt)
            ]
        },
        config=config,
    )

    ### Simple Interrupt Handling
    if "__interrupt__" not in result:
        print_result(result)
        return

    interrupts = result["__interrupt__"]

    for interrupt_info in interrupts:
        print("\n========================================")
        print(" HUMAN APPROVAL REQUIRED")
        print("========================================")
        print(interrupt_info.value)

    decision = input(
        "\nAktion ausführen? [approve/reject]: "
    ).strip().lower()

    while decision not in {"approve", "reject"}:
        decision = input(
            "Bitte 'approve' oder 'reject' eingeben: "
        ).strip().lower()

    result = await graph.ainvoke(
        Command(resume=decision),
        config=config,
    )
    ### Simple Interrupt Handling END
    print_result(result)


def print_result(result: dict) -> None:
    print("\n========================================")
    print(" RESULT")
    print("========================================")

    messages = result.get("messages", [])

    if messages:
        print(messages[-1].content)


async def main() -> None:
    await run_agent(
        # HITL scenario
        "Lösche Datei data/news_min.txt"
    )

if __name__ == "__main__":
    asyncio.run(main())