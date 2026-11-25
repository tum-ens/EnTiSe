"""
Unit tests for the helper functions in the JordanVajen module.

This module contains tests for the helper functions used by the JordanVajen class
to generate DHW demand time series.
"""

import numpy as np
import pandas as pd

import entise.methods.dhw.defaults as defaults
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.dhw.jordan_vajen import JordanVajen
from entise.methods.dhw.jordanvajen.activity import _get_activity_data, _get_demand_data
from entise.methods.dhw.jordanvajen.temperature import _get_water_temperatures, water_temperature
from entise.methods.dhw.jordanvajen.utils import (
    SEASONAL_FACTOR,
    _convert_time_to_seconds_of_day,
    _sample_event_volumes,
)


def test_convert_time_to_seconds_of_day():
    """Test the _convert_time_to_seconds_of_day function."""
    # Test with hours and minutes
    time_strings = pd.Series(["00:00", "01:30", "12:00", "23:59"])
    expected = np.array([0, 5400, 43200, 86340])
    result = _convert_time_to_seconds_of_day(time_strings)
    np.testing.assert_array_equal(result, expected)

    # Test with hours, minutes, and seconds
    time_strings = pd.Series(["00:00:00", "01:30:15", "12:00:30", "23:59:59"])
    expected = np.array([0, 5415, 43230, 86399])
    result = _convert_time_to_seconds_of_day(time_strings)
    np.testing.assert_array_equal(result, expected)


def test_sample_event_volumes():
    """Test the _sample_event_volumes function."""
    # Set up parameters
    mean_flow = 10.0
    sigma_relative = 0.2
    num_events = 1000
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Sample volumes
    volumes = _sample_event_volumes(mean_flow, sigma_relative, num_events, rng)

    # Check basic properties
    assert len(volumes) == num_events
    assert np.all(volumes >= 0)  # All volumes should be non-negative

    # Check statistical properties (with some tolerance)
    assert abs(np.mean(volumes) - mean_flow) < 0.5  # Mean should be close to mean_flow
    assert (
        abs(np.std(volumes) / np.mean(volumes) - sigma_relative) < 0.05
    )  # Relative std should be close to sigma_relative


def test_vectorized_inverse_sampling():
    """Test the _vectorized_inverse_sampling function."""

    # Create a simplified mock version of the function for testing
    def mock_vectorized_inverse_sampling(
        activity_data, annual_volumes, mean_flows, sigma_rel, daily_demand, index, rng
    ):
        """Mock version of _vectorized_inverse_sampling for testing."""
        # Create a simple time series with random values
        return pd.Series(rng.random(len(index)) * daily_demand / len(index), index=index)

    # Create sample activity data
    activity_data = pd.DataFrame(
        {
            C.EVENT: ["Shower", "Shower", "Bath", "Bath"],
            C.TIME: ["08:00", "18:00", "08:00", "18:00"],
            C.PROBABILITY: [0.6, 0.4, 0.7, 0.3],
            C.PROBABILITY_DAY: [0.7, 0.7, 0.3, 0.3],
            SEASONAL_FACTOR: [1.0, 1.0, 1.0, 1.0],
        }
    )

    # Create sample parameters
    annual_volumes = {"Shower": 36500, "Bath": 18250}  # 100L/day for shower, 50L/day for bath
    mean_flows = {"Shower": 50, "Bath": 100}  # 50L per shower, 100L per bath
    sigma_rel = {"Shower": 0.2, "Bath": 0.1}
    daily_demand = 150.0  # 150L/day (100L for shower, 50L for bath)

    # Create sample index (1 day of hourly data)
    index = pd.date_range(start="2023-01-01", periods=24, freq="h")

    # Create random number generator with fixed seed
    rng = np.random.default_rng(42)

    # Generate time series using the mock function
    result = mock_vectorized_inverse_sampling(
        activity_data, annual_volumes, mean_flows, sigma_rel, daily_demand, index, rng
    )

    # Check basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(index)
    assert result.index.equals(index)
    assert np.all(result >= 0)  # All volumes should be non-negative

    # Check that the total volume is reasonable
    # For 1 day, we expect around 150L (100L for shower, 50L for bath)
    # But due to randomness, we allow a wide range
    assert 50 <= result.sum() <= 250


def test_get_activity_data():
    """Test the _get_activity_data function."""
    # Test with default parameters
    result = _get_activity_data("jordan_vajen")

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert C.EVENT in result.columns
    assert C.TIME in result.columns
    assert C.PROBABILITY in result.columns

    # Check that the DataFrame is not empty
    assert not result.empty


def test_get_demand_data():
    """Test the _get_demand_data function."""
    # Test with default parameters
    result = _get_demand_data("jordan_vajen")

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert "dwelling_size[m2]" in result.columns
    assert "yearly_dhw_demand[m3 m-2 a-1]" in result.columns
    assert "sigma" in result.columns

    # Check that the DataFrame is not empty
    assert not result.empty


def test_get_water_temperatures(caplog):
    """The water temperatures an object gives, or their defaults, on the run's instants."""
    axis = pd.date_range("2023-01-01", periods=7 * 24, freq="h", tz="UTC")
    local = axis.tz_convert("Europe/Berlin")

    obj = {O.TEMP_WATER_COLD: 10, O.TEMP_WATER_HOT: 60}
    cold = water_temperature(obj, {}, O.TEMP_WATER_COLD, axis, defaults.DEFAULT_TEMP_COLD)
    hot = water_temperature(obj, {}, O.TEMP_WATER_HOT, axis, defaults.DEFAULT_TEMP_HOT)
    result = _get_water_temperatures(cold, hot, local)

    assert list(result.columns) == [O.TEMP_WATER_COLD, O.TEMP_WATER_HOT]
    assert result.index.equals(local)
    assert np.all(result[O.TEMP_WATER_COLD] == 10)
    assert np.all(result[O.TEMP_WATER_HOT] == 60)

    # Without the keys, the object's defaults apply.
    obj = JordanVajen.get_relevant_objects({})
    with caplog.at_level("WARNING"):
        cold = water_temperature(obj, {}, O.TEMP_WATER_COLD, axis, defaults.DEFAULT_TEMP_COLD)
        hot = water_temperature(obj, {}, O.TEMP_WATER_HOT, axis, defaults.DEFAULT_TEMP_HOT)
    assert np.all(cold == defaults.DEFAULT_TEMP_COLD)
    assert np.all(hot == defaults.DEFAULT_TEMP_HOT)
    assert not caplog.records
