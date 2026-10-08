from __future__ import annotations

import asyncio
import random
import time

from dataclasses import dataclass
from enum import Enum
from typing import (
    AsyncIterator,
    Callable,
    Protocol,
    TypedDict,
)


# ============================================================
# 1. TypedDict
# ============================================================

class State(TypedDict):
    request: str
    results: dict[str, str]


# ============================================================
# 2. Enum
# ============================================================

class AgentType(Enum):
    ARCHITECT = "architect"
    SECURITY = "security"
    PERFORMANCE = "performance"


# ============================================================
# 3. Protocol
#
# Ähnlich wie ein Java Interface.
# ============================================================

class Agent(Protocol):

    name: str

    async def execute(self, state: State) -> str:
        ...


# ============================================================
# 4. Dataclass
# ============================================================

@dataclass(frozen=True)
class AgentResult:
    agent: str
    result: str
    duration: float


# ============================================================
# 5. Context Manager
# ============================================================

class Timer:

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.perf_counter() - self.start
        print(f"[END]   {self.name}: {duration:.2f}s")


# ============================================================
# 6. Decorator
# ============================================================

def log_execution(
        func: Callable
) -> Callable:

    async def wrapper(*args, **kwargs):

        name = func.__name__

        print(f"[CALL]  {name}")

        try:

            result = await func(*args, **kwargs)

            print(f"[DONE]  {name}")

            return result

        except Exception as e:

            print(f"[ERROR] {name}: {e}")

            raise

    return wrapper


# ============================================================
# 7. Konkrete Agents
# ============================================================

class ArchitectureAgent:

    name = AgentType.ARCHITECT.value

    @log_execution
    async def execute(self, state: State) -> str:

        await asyncio.sleep(random.uniform(1, 3))

        return (
            "Architecture analysis: "
            "Event-driven architecture with Kafka "
            "and clear bounded contexts."
        )


class SecurityAgent:

    name = AgentType.SECURITY.value

    @log_execution
    async def execute(self, state: State) -> str:

        await asyncio.sleep(random.uniform(1, 3))

        return (
            "Security analysis: "
            "Use OAuth2/OIDC, least privilege "
            "and zero-trust principles."
        )


class PerformanceAgent:

    name = AgentType.PERFORMANCE.value

    @log_execution
    async def execute(self, state: State) -> str:

        await asyncio.sleep(random.uniform(1, 3))

        return (
            "Performance analysis: "
            "Use asynchronous I/O, caching "
            "and horizontal scaling."
        )


# ============================================================
# 8. Event
# ============================================================

@dataclass(frozen=True)
class Event:

    agent: str
    result: str | None = None
    error: Exception | None = None


# ============================================================
# 9. Agent Runner
#
# Async Generator!
#
# yield erzeugt Events Stück für Stück.
# ============================================================

async def run_agent(
        agent: Agent,
        state: State
) -> AsyncIterator[Event]:

    try:

        with Timer(agent.name):

            result = await agent.execute(state)

        yield Event(
            agent=agent.name,
            result=result
        )

    except Exception as e:

        yield Event(
            agent=agent.name,
            error=e
        )


# ============================================================
# 10. Orchestrator
# ============================================================

class Orchestrator:

    def __init__(
            self,
            agents: list[Agent]
    ):

        self.agents = agents

    async def execute(
            self,
            state: State
    ) -> AsyncIterator[Event]:

        tasks = [
            asyncio.create_task(
                self._collect(agent, state)
            )
            for agent in self.agents
        ]

        # Ergebnisse kommen in der Reihenfolge zurück,
        # in der die Tasks tatsächlich fertig werden.
        for completed_task in asyncio.as_completed(tasks):

            events = await completed_task

            for event in events:

                yield event

    async def _collect(
            self,
            agent: Agent,
            state: State
    ) -> list[Event]:

        events = []

        async for event in run_agent(
                agent,
                state
        ):
            events.append(event)

        return events


# ============================================================
# 11. Aggregator
# ============================================================

async def aggregate(
        orchestrator: Orchestrator,
        state: State
):

    async for event in orchestrator.execute(state):

        if event.error:

            print(
                f"\n❌ {event.agent}: "
                f"{event.error}"
            )

            continue

        print(
            f"\n✅ {event.agent}:"
        )

        print(
            f"   {event.result}"
        )

        state["results"][event.agent] = event.result


# ============================================================
# 12. Main
# ============================================================

async def main():

    state: State = {

        "request":
            "Entwirf eine skalierbare "
            "Enterprise-AI-Plattform",

        "results": {}
    }

    agents: list[Agent] = [

        ArchitectureAgent(),

        SecurityAgent(),

        PerformanceAgent()
    ]

    orchestrator = Orchestrator(
        agents
    )

    await aggregate(
        orchestrator,
        state
    )

    print("\n==============================")

    print("FINAL STATE")

    print("==============================")

    for agent, result in state["results"].items():

        print(f"\n{agent}:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())