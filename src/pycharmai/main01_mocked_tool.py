from pycharmai.llm.anthropic_llm import AnthropicLLM
from pycharmai.llm.llmclient import LLMClient

def answer_question(
    llm: LLMClient,
    question: str,
) -> str:
    return llm.ask(question)

if __name__ == "__main__":
    client = AnthropicLLM()
    answer=answer_question(client,"What's the temperature in Berlin?")
    print(answer)


