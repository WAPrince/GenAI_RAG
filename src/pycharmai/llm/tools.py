def get_temperature(city: str) -> int:
    """Return the current temperature for a city."""

    temperatures = {
        "Berlin": 18,
        "Munich": 16,
        "Hamburg": 15,
    }

    return temperatures.get(city, 20)