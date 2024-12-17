from src.constants import Objects, SEP, Types


def get_valid_object():
    """Returns a mock valid object for testing."""
    return {
        Objects.ID: "building1",
        Types.HVAC: 'method1',
        f"{Types.HVAC}{SEP}{Objects.WEATHER}": Objects.WEATHER,
        Objects.WEATHER: Objects.WEATHER,
    }