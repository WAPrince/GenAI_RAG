from anthropic import Anthropic
from pycharmai.llm import tools
from pycharmai.llm.tools import get_temperature
from pycharmai.config.settings import settings

tools = [
    {
        "name": "get_temperature",
        "description": "Return the current temperature for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city."
                }
            },
            "required": ["city"]
        }
    }
]


class AnthropicLLM:

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")

        self.client = Anthropic(
            api_key=settings.anthropic_api_key
        )

    def ask(self, prompt: str) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        while True:

            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                tools=tools,
                messages=messages,
            )

            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            tool_results = []
            final_text = ""

            for block in response.content:

                if block.type == "text":
                    final_text += block.text

                elif block.type == "tool_use":

                    result = get_temperature(
                        block.input["city"]
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            if not tool_results:
                return final_text

            messages.append({
                "role": "user",
                "content": tool_results,
            })





