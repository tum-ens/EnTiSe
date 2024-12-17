import pytest
import pandas as pd
from unittest.mock import patch
from src.core.timeseries_generator import TimeSeriesGenerator
from src.core.registry import register_method, Meta
from src.core.base import TimeSeriesMethod
from src.constants import Objects, Keys, Columns, Types


# Mock TimeSeriesMethod for testing
class MockMethod(TimeSeriesMethod, metaclass=Meta):
    required_keys = {"key1": str}
    required_timeseries = {}

    def generate(self, obj, data, ts_type, **kwargs):
        return {"metric": 42}, pd.DataFrame({"col1": [1, 2, 3]})


# Fixtures for common test inputs
@pytest.fixture
def single_object():
    return {Objects.ID: "building1", Types.HVAC: "method1"}


@pytest.fixture
def multiple_objects():
    return [
        {Objects.ID: "building1", Types.HVAC: "method1"},
        {Objects.ID: "building2", Types.HVAC: "method2"},
    ]


@pytest.fixture
def invalid_object():
    return {Types.HVAC: "method1"}  # Missing 'id'


@pytest.fixture
def mock_method():
    from src.core.registry import register_method, method_registry

    class MockMethodTSGenerator:
        @classmethod
        def get_requirements(cls):
            return {"keys": {"key1": str}, "timeseries": {}}

    method_name = f"mock_method_{id(MockMethodTSGenerator)}"
    if method_name not in method_registry:
        register_method(method_name, MockMethodTSGenerator)
    return MockMethodTSGenerator


# Test cases for `add_objects`
def test_add_single_object(single_object):
    generator = TimeSeriesGenerator()
    generator.add_objects(single_object)
    assert len(generator.objects) == 1
    assert generator.objects.loc[0, Objects.ID] == "building1"


def test_add_multiple_objects(multiple_objects):
    generator = TimeSeriesGenerator()
    generator.add_objects(multiple_objects)
    assert len(generator.objects) == 2
    assert list(generator.objects[Objects.ID]) == ["building1", "building2"]


def test_add_invalid_input():
    generator = TimeSeriesGenerator()
    with pytest.raises(TypeError, match="Input must be a dictionary, list of dictionaries, or a pandas DataFrame."):
        generator.add_objects("invalid_input")


def test_missing_id_in_object(invalid_object):
    generator = TimeSeriesGenerator()
    with pytest.raises(ValueError, match="Each object in the list must have an id field."):
        generator.add_objects([invalid_object])


def test_missing_column_in_dataframe():
    generator = TimeSeriesGenerator()
    df = pd.DataFrame({Types.HVAC: ["method1"]})  # Missing 'id'
    with pytest.raises(ValueError, match="The 'objects' DataFrame must have an id column."):
        generator.add_objects(df)


# Test cases for `_generate` and `_process_object`
def test_empty_objects_error():
    generator = TimeSeriesGenerator()
    with pytest.raises(ValueError, match="No objects have been added for processing."):
        generator.generate({})


def test_generate_sequential(single_object):
    generator = TimeSeriesGenerator()
    generator.add_objects(single_object)
    data = {}

    with patch.object(generator, "_process_object", return_value={Objects.ID: "building1", "summary": {}, "timeseries": pd.DataFrame()}) as mock_process:
        summary, timeseries = generator.generate_sequential(data)
        assert mock_process.call_count == 1
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(timeseries, dict)


# Test cases for utility methods
def test_get_method_requirements():
    from src.core.registry import register_method

    class MockMethodTSGenerator:
        @classmethod
        def get_requirements(cls):
            return {"keys": {"key1": str}, "timeseries": {}}

    method_name = "mock_method_ts_generator"
    register_method(method_name, MockMethodTSGenerator)

    # Test specific method requirements
    requirements = TimeSeriesGenerator.get_method_requirements("mock_method_ts_generator")
    assert "mock_method_ts_generator" in requirements
    assert requirements["mock_method_ts_generator"]["keys"] == {"key1": str}

    # Test all methods
    all_requirements = TimeSeriesGenerator.get_method_requirements()
    assert "mock_method_ts_generator" in all_requirements



def test_log_warning():
    generator = TimeSeriesGenerator(raise_on_error=False)
    with patch.object(generator.logger, "warning") as mock_warning:
        generator._log_warning("Test warning")
        mock_warning.assert_called_once_with("Test warning")

    generator.raise_on_error = True
    with pytest.raises(RuntimeError, match="Test warning"):
        generator._log_warning("Test warning")
