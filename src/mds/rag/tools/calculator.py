from langchain_core.tools import tool


@tool
def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""

    try:
        # Nur als Lernbeispiel!
        result = eval(expression, {"__builtins__": {}}, {})

    except Exception as exc:
        raise ValueError(
            f"Invalid mathematical expression: {expression}"
        ) from exc

    if not isinstance(result, (int, float)):
        raise ValueError(
            "Expression did not produce a number"
        )

    return float(result)