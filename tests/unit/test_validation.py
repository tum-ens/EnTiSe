from copy import deepcopy
import pytest
import pandas as pd
from src.utils.validation import Validator
from src.constants import Keys, Columns, Objects
from tests.fixtures.object_fixtures import get_valid_object
from tests.fixtures.timeseries_fixtures import get_valid_timeseries


@pytest.fixture
def valid_object():
    return get_valid_object()


@pytest.fixture
def valid_timeseries():
    return get_valid_timeseries()


@pytest.fixture
def required_keys():
    """Fixture for required keys."""
    return {
        Objects.ID: str,
        Objects.WEATHER: str
    }


@pytest.fixture
def required_timeseries():
    """Fixture for required timeseries."""
    return {
        Objects.WEATHER: {
            Keys.COLS_REQUIRED: {
                Columns.DATETIME: pd.Timestamp,
                Columns.TEMP_OUT: float
            },
            Keys.DTYPE: pd.DataFrame
        }
    }


def test_validate_object_with_missing_keys(valid_object, required_keys):
    """Test validation with missing required keys."""
    invalid_object = deepcopy(valid_object)
    del invalid_object[Objects.WEATHER]

    with pytest.raises(ValueError, match="Missing required keys"):
        Validator.validate_object(invalid_object, required_keys, raise_dtype_errors=True)


def test_validate_object_with_invalid_types(valid_object, required_keys):
    """Test validation with incorrect types for keys."""
    invalid_object = valid_object.copy()
    invalid_object[Objects.ID] = 123  # Invalid type: int instead of str

    with pytest.raises(ValueError, match=r"Key '.*' must be of type '.*', got .*."):
        Validator.validate_object(invalid_object, required_keys, raise_dtype_errors=True)


def test_validate_object_with_union_types():
    obj = {
        Objects.WEATHER: "sunny",
        Objects.LOAD_BASE: 42.0,
    }
    required_keys = {
        Objects.WEATHER: str,
        Objects.LOAD_BASE: int | float,
    }
    Validator.validate_object(obj, required_keys, raise_dtype_errors=True)  # Should not raise

    invalid_obj = {
        Objects.WEATHER: "sunny",
        Objects.LOAD_BASE: "invalid_type",
    }
    with pytest.raises(ValueError, match=r"Key '.*' must be .*"):
        Validator.validate_object(invalid_obj, required_keys, raise_dtype_errors=True)


def test_validate_timeseries_with_missing_key(valid_timeseries, required_timeseries):
    """Test validation with missing timeseries key."""
    invalid_timeseries = valid_timeseries.copy()
    del invalid_timeseries[Objects.WEATHER]

    with pytest.raises(ValueError, match="Missing timeseries"):
        Validator.validate_timeseries(invalid_timeseries, required_timeseries, raise_dtype_errors=True)


def test_validate_timeseries_with_missing_column(valid_timeseries, required_timeseries):
    """Test validation with missing column in timeseries."""
    invalid_timeseries = valid_timeseries.copy()
    invalid_timeseries[Objects.WEATHER] = invalid_timeseries[Objects.WEATHER].drop(columns=[Columns.TEMP_OUT])

    with pytest.raises(ValueError, match=r"Timeseries '.*' is missing required column '.*'."):
        Validator.validate_timeseries(invalid_timeseries, required_timeseries, raise_dtype_errors=True)


def test_validate_timeseries_with_invalid_column_type(valid_timeseries, required_timeseries):
    """Test validation with incorrect column type."""
    invalid_timeseries = valid_timeseries.copy()
    invalid_timeseries[Objects.WEATHER][Columns.TEMP_OUT] = ["15", "10", "5"]  # Strings instead of floats

    with pytest.raises(ValueError, match=r"Column '.*' in timeseries '.*' must be of type .*"):
        Validator.validate_timeseries(invalid_timeseries, required_timeseries, raise_dtype_errors=True)


def test_validate_timeseries_with_invalid_dtype(valid_timeseries, required_timeseries):
    """Test validation with incorrect timeseries dtype."""
    invalid_timeseries = valid_timeseries.copy()
    invalid_timeseries[Objects.WEATHER] = invalid_timeseries[Objects.WEATHER].to_dict()  # Convert to dict

    with pytest.raises(ValueError, match=r"Timeseries '.*' must be of type .*"):
        Validator.validate_timeseries(invalid_timeseries, required_timeseries, raise_dtype_errors=True)


if __name__ == "__main__":
    pytest.main()
