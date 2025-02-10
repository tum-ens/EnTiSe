import pandas as pd
import pytest
from tests.utils.mock_methods import DummyMethod
from tests.fixtures.object_fixtures import get_valid_object
from tests.fixtures.timeseries_fixtures import get_valid_timeseries

from src.constants import Objects, Keys, SEP, Columns, Types
from src.core.base import TimeSeriesMethod


@pytest.fixture
def valid_object():
    return get_valid_object()


@pytest.fixture
def valid_timeseries():
    return get_valid_timeseries()


# Test Abstract Method
def test_generate_not_implemented():
    class IncompleteMethod(TimeSeriesMethod):
        pass  # No implementation of the abstract `generate` method

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteMethod"):
        IncompleteMethod()


# Test prepare_inputs
def test_prepare_inputs_valid_input(valid_object, valid_timeseries):
    method = DummyMethod()
    processed_obj = method.prepare_inputs(valid_object, valid_timeseries, Types.HVAC)
    assert Objects.WEATHER in processed_obj
    assert processed_obj[Objects.WEATHER] == valid_object[Objects.WEATHER]


def test_prepare_inputs_invalid_object(valid_object, valid_timeseries):
    method = DummyMethod()
    invalid_object = valid_object.copy()
    invalid_object.pop(Objects.WEATHER)
    invalid_object.pop(f'{Types.HVAC}{SEP}{Objects.WEATHER}')

    with pytest.raises(ValueError, match="Missing required keys"):
        method.prepare_inputs(invalid_object, valid_timeseries, Types.HVAC)


# Test resolve_column
def test_resolve_column_prefixed_column(valid_timeseries):
    column = Columns.TEMP_OUT
    ts_type = Types.HVAC
    prefixed_col = f"{ts_type}{SEP}{column}"
    valid_timeseries["weather"][prefixed_col] = valid_timeseries["weather"][column]

    resolved = DummyMethod.resolve_column("weather", column, ts_type, valid_timeseries)
    pd.testing.assert_series_equal(resolved, valid_timeseries["weather"][prefixed_col])


def test_resolve_column_shared_column(valid_timeseries):
    column = Columns.TEMP_OUT
    resolved = DummyMethod.resolve_column("weather", column, "hvac", valid_timeseries)
    pd.testing.assert_series_equal(resolved, valid_timeseries["weather"][column])


def test_resolve_column_missing_column(valid_timeseries):
    with pytest.raises(ValueError, match=r"Neither '.*' nor '.*' found in timeseries '.*'."):
        DummyMethod.resolve_column(Objects.WEATHER, Columns.TEMP_OUT, Types.HVAC, {Objects.WEATHER: pd.DataFrame()})


# Test get_relevant_objects
def test_get_relevant_objects(valid_object):
    method = DummyMethod()
    ts_type = "hvac"
    relevant_obj = method.get_relevant_objects(valid_object, ts_type)
    assert Objects.WEATHER in relevant_obj


def test_get_relevant_objects_shared_key(valid_object):
    method = DummyMethod()
    relevant_obj = method.get_relevant_objects(valid_object, None)
    assert relevant_obj == valid_object


# Test class methods
def test_get_requirements():
    requirements = DummyMethod.get_requirements()
    assert Keys.KEYS_REQUIRED in requirements
    assert Keys.TIMESERIES_REQUIRED in requirements
    assert requirements[Keys.KEYS_REQUIRED] == DummyMethod.required_keys
    assert requirements[Keys.TIMESERIES_REQUIRED] == DummyMethod.required_timeseries


def test_get_dependencies():
    assert DummyMethod.get_dependencies() == DummyMethod.dependencies


def test_get_available_outputs():
    assert DummyMethod.get_available_outputs() == DummyMethod.available_outputs
