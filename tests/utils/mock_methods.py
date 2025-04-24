import pandas as pd

from src.constants import Types, Objects, Columns, Keys
from src.core.base import TimeSeriesMethod
from src.utils.decorators import supported_types
from src.core.registry import Meta, register_method, method_registry


@supported_types(Types.HVAC)
class MockMethod1(TimeSeriesMethod, metaclass=Meta):
    """Mock HVAC Method for testing."""
    required_keys = []
    required_timeseries = {}
    dependencies = []

    def generate(self, obj, data, ts_type, **kwargs):
        return {"metric1": 1}, pd.DataFrame({Columns.DATETIME: [1, 2, 3], "Value": [10, 20, 30]})


@supported_types(Types.ELECTRICITY)
class MockMethod2(TimeSeriesMethod, metaclass=Meta):
    """Mock Electricity Method for testing."""
    required_keys = []
    required_timeseries = {}
    dependencies = []

    def generate(self, obj, data, ts_type, **kwargs):
        return {"metric2": 2}, pd.DataFrame({Columns.DATETIME: [1, 2, 3], "Value": [40, 50, 60]})


@supported_types(Types.HVAC)
class DummyMethod(TimeSeriesMethod, metaclass=Meta):
    """Dummy Method for testing other base.py utilities."""
    required_keys = {Objects.WEATHER: str}
    required_timeseries = {
        Objects.WEATHER: {
            Keys.COLUMNS: {Columns.TEMP_OUT: float},
            Keys.DTYPE: pd.DataFrame,
        }
    }
    dependencies = []

    def generate(self, obj, data, ts_type, **kwargs):
        pass


def register_mock_methods():
    """
    Fixture to register mock methods globally for all tests.
    Automatically invoked at the start of the pytest session.
    """
    # Check if the methods are already registered
    if "method1" in method_registry and "method2" in method_registry:
        return
    register_method("method1", MockMethod1)
    register_method("method2", MockMethod2)
    register_method("dummy", DummyMethod)
    print("Mock methods registered!")


# Optional standalone function to manually register methods if used outside pytest
def main():
    register_mock_methods()
    from src.core.registry import get_method

    # Validate registration
    print(get_method("method1"))  # Should output: <class 'MockMethod1'>
    print(get_method("method2"))  # Should output: <class 'MockMethod2'>
    print(get_method("dummy"))    # Should output: <class 'DummyMethod'>


if __name__ == "__main__":
    main()
