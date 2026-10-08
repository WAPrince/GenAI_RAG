from langchain_core.tools import tool

@tool
def get_temperature(city: str) -> int:
    """Return the current temperature for a city."""

    print("getTemperatures")
    temperatures = {
        "Berlin": 18,
        "Munich": 16,
        "Hamburg": 15,
    }

    if city not in temperatures:
        raise ValueError(
            f"No temperature data available for {city}"
        )

    return temperatures[city]