import asyncio
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Correctness

from pycharmai.graph.agent import create_graph
import pandas as pd
import os


TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI  # the MCP server subprocess reads this
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.set_experiment("mcp-server-demo-16")
print(f"Experiment: {experiment.name} (id={experiment.experiment_id})")


mlflow.trace#(span_type=SpanType.AGENT)
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

    judges = [
        RelevanceToQuery(),
        Correctness()
    ]
    eval_results = mlflow.genai.evaluate(
        data=result[-1].content,  # Übergabe der gesammelten Trace-Objekte
        scorers=judges
    )
    print(eval_results.metrics)
    eval_df = eval_results.tables["eval_results_table"]
    print(eval_df[["inputs", "outputs", "relevance_to_query/v1/score", "relevance_to_query/v1/justification"]].head())

def print_result(result: dict) -> None:
    print("\n========================================")
    print(" RESULT")
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


async def main() -> None:
    mlflow.anthropic.autolog()
    await run_agent(
        # RAG-Szenario:
        "Was ist im Harz passiert?"
    )

if __name__ == "__main__":
    asyncio.run(main())