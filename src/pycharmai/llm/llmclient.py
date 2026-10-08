from typing import Protocol


class LLMClient(Protocol):
    def ask(self, prompt: str) -> str :
        ...