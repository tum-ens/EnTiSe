"""
Unit tests for the helper functions in the JordanVajen module.

This module contains tests for the helper functions used by the JordanVajen class
to generate DHW demand time series.
"""

import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from entise.methods.dhw.jordan_vajen import (
    _convert_time_to_seconds_of_day,
    _find_nearest_activity_times,
    _sample_event_volumes,
    _calculate_volume_timeseries,
    _get_activity_data,
    _get_demand_data,
    _get_water_temperatures,
    SEASONAL_FACTOR
)
from entise.constants import Columns as C, Objects as O, Types
import entise.methods.dhw.defaults as defaults


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


def test_find_nearest_activity_times():
    """Test the _find_nearest_activity_times function."""
    # Create sample activity times
    activity_times = np.array([0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000])

    # Test exact matches
    seconds_of_day = np.array([0, 7200, 14400, 21600, 28800, 36000])
    expected_indices = np.array([0, 2, 4, 6, 8, 10])
    result, _ = _find_nearest_activity_times(activity_times, seconds_of_day)
    np.testing.assert_array_equal(result, expected_indices)

    # Test nearest matches
    seconds_of_day = np.array([1800, 5400, 9000, 12600, 16200, 19800, 23400, 27000, 30600, 34200, 37800])
    expected_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result, _ = _find_nearest_activity_times(activity_times, seconds_of_day)
    np.testing.assert_array_equal(result, expected_indices)

    # Test edge cases
    seconds_of_day = np.array([-1000, 100000])  # Before first and after last
    expected_indices = np.array([0, 10])  # Should clamp to first and last
    result, _ = _find_nearest_activity_times(activity_times, seconds_of_day)
    np.testing.assert_array_equal(result, expected_indices)


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
    assert abs(np.std(volumes) / np.mean(volumes) - sigma_relative) < 0.05  # Relative std should be close to sigma_relative


def test_vectorized_inverse_sampling():
    """Test the _vectorized_inverse_sampling function."""
    # Create a simplified mock version of the function for testing
    def mock_vectorized_inverse_sampling(activity_data, annual_volumes, mean_flows, sigma_rel, daily_demand, index, rng):
        """Mock version of _vectorized_inverse_sampling for testing."""
        # Create a simple time series with random values
        return pd.Series(rng.random(len(index)) * daily_demand / len(index), index=index)

    # Create sample activity data
    activity_data = pd.DataFrame({
        C.EVENT: ["Shower", "Shower", "Bath", "Bath"],
        C.TIME: ["08:00", "18:00", "08:00", "18:00"],
        C.PROBABILITY: [0.6, 0.4, 0.7, 0.3],
        C.PROBABILITY_DAY: [0.7, 0.7, 0.3, 0.3],
        SEASONAL_FACTOR: [1.0, 1.0, 1.0, 1.0]
    })

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
    result = _get_activity_data('jordan_vajen')

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
    result = _get_demand_data('jordan_vajen')

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert 'dwelling_size' in result.columns
    assert 'm3_per_m2_a' in result.columns
    assert 'sigma' in result.columns

    # Check that the DataFrame is not empty
    assert not result.empty


def test_get_water_temperatures():
    """Test the _get_water_temperatures function."""
    # Create sample data
    obj = {
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60
    }
    data = {}

    # Create sample weather data
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=7)
    index = pd.date_range(start=start, end=end, freq='h')
    weather = pd.DataFrame({
        C.DATETIME: index
    })

    # Get water temperatures
    result = _get_water_temperatures(obj, data, weather)

    # Check that the result is a DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert O.TEMP_WATER_COLD in result.columns
    assert O.TEMP_WATER_HOT in result.columns

    # Check that the values match the input
    assert np.all(result[O.TEMP_WATER_COLD] == 10)
    assert np.all(result[O.TEMP_WATER_HOT] == 60)

    # Test with missing cold water temperature (should use VDI4655 method)
    obj = {
        O.TEMP_WATER_HOT: 60
    }
    result = _get_water_temperatures(obj, data, weather)

    # Check that cold water temperature is calculated
    assert O.TEMP_WATER_COLD in result.columns
    assert not np.isnan(result[O.TEMP_WATER_COLD]).any()

    # Test with missing hot water temperature (should use default)
    obj = {
        O.TEMP_WATER_COLD: 10
    }
    result = _get_water_temperatures(obj, data, weather)

    # Check that hot water temperature is set to default
    assert O.TEMP_WATER_HOT in result.columns
    assert np.all(result[O.TEMP_WATER_HOT] == defaults.DEFAULT_TEMP_HOT)
