import numpy as np
import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.methods.auxiliary.ventilation.selector import Ventilation


# Fixtures for minimal data setup
@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame(index=index)


@pytest.fixture
def timeseries_with_column():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame({"object_1": np.arange(24)}, index=index)


# Tests
def test_ventilation_inactive_strategy(minimal_weather):
    """Test the VentilationInactive strategy."""
    obj = {}
    data = {O.WEATHER: minimal_weather}
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    assert (result[O.VENTILATION] == 0).all()


def test_ventilation_constant_strategy(minimal_weather):
    """Test the VentilationConstant strategy."""
    obj = {O.VENTILATION: 100}
    data = {O.WEATHER: minimal_weather}
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    assert (result[O.VENTILATION] == 100).all()


def test_ventilation_timeseries_strategy(minimal_weather, timeseries_with_column):
    """Test the VentilationTimeSeries strategy."""
    obj = {O.ID: "object_1", O.VENTILATION_COL: "object_1", O.VENTILATION: O.VENTILATION}
    data = {
        O.VENTILATION: timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    expected = (
        np.arange(24) * 1 * 2.5 * 1.2 * 1000 / 3600
    )  # Apply the same transformation as in VentilationTimeSeries.run()
    assert np.allclose(result[O.VENTILATION], expected)


def test_ventilation_string_reference(minimal_weather, timeseries_with_column):
    """Test that a string reference to a time series is correctly handled."""
    obj = {O.VENTILATION: "ventilation_data"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    # This should raise an error because ventilation_col is not provided
    with pytest.raises(Warning):
        ventilation.generate(obj, data)


def test_ventilation_with_column_name(minimal_weather, timeseries_with_column):
    """Test that a column name is correctly used."""
    obj = {O.ID: "wrong_id", O.VENTILATION: "ventilation_data", O.VENTILATION_COL: "object_1"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    expected = (
        np.arange(24) * 1 * 2.5 * 1.2 * 1000 / 3600
    )  # Apply the same transformation as in VentilationTimeSeries.run()
    assert np.allclose(result[O.VENTILATION], expected)


def test_ventilation_with_object_id(minimal_weather, timeseries_with_column):
    """Test that the object ID is used as column name if ventilation_col is not provided."""
    obj = {O.ID: "object_1", O.VENTILATION: "ventilation_data"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    expected = (
        np.arange(24) * 1 * 2.5 * 1.2 * 1000 / 3600
    )  # Apply the same transformation as in VentilationTimeSeries.run()
    assert np.allclose(result[O.VENTILATION], expected)


def test_ventilation_column_not_found(minimal_weather, timeseries_with_column):
    """Test that an error is raised if the column is not found."""
    obj = {O.ID: "object_1", O.VENTILATION: "ventilation_data", O.VENTILATION_COL: "non_existent_column"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    with pytest.raises(Warning):
        ventilation.generate(obj, data)


def test_ventilation_timeseries_with_numeric_value(minimal_weather):
    """Test that VentilationTimeSeries falls back to VentilationConstant when given a numeric value."""
    # Create an object that would normally use VentilationTimeSeries
    obj = {O.ID: "object_1", O.VENTILATION_COL: "object_1", O.VENTILATION: 100}
    data = {O.WEATHER: minimal_weather}

    # Create a VentilationTimeSeries instance directly to test its behavior
    from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries

    ventilation_ts = VentilationTimeSeries()

    result = ventilation_ts.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    assert (result[O.VENTILATION] == 100).all()
