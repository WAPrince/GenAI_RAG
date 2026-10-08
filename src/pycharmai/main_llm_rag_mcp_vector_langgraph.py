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
        # RAG-Szenario:
        "Was ist im Harz passiert?"
    )


if __name__ == "__main__":
    asyncio.run(main())