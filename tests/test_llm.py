from pycharmai.main import answer_question

class MockLLM:

    def ask(self, prompt: str) -> str:
        return "This is a test response."


def test_answer_question():
    llm = MockLLM()

    result = answer_question(
        llm,
        "Hello"
    )

    assert result == "This is a test response."
