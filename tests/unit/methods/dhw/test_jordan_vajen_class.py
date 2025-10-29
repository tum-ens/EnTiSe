"""
Unit tests for the JordanVajen class.

This module contains tests for the JordanVajen class, which is used to generate
DHW demand time series based on the Jordan & Vajen methodology.
"""

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.dhw.jordan_vajen import JordanVajen


# Create sample weather data
def create_sample_weather(start_date="2023-01-01", days=7, freq="h"):
    """Create sample weather data for testing."""
    start = pd.Timestamp(start_date, tz="UTC")
    end = start + pd.Timedelta(days=days)
    index = pd.date_range(start=start, end=end, freq=freq)

    # Create a DataFrame with temperature data
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_OUT: np.random.normal(10, 5, len(index)),  # Random outdoor temperatures
        }
    )

    return weather


# Test basic functionality
def test_jordan_vajen_basic():
    """Test basic functionality of the JordanVajen class."""
    # Create sample data
    weather = create_sample_weather()
    obj = {
        O.ID: "test_obj",
        O.DWELLING_SIZE: 100,  # 100 m²
        O.TEMP_WATER_COLD: 10,  # 10°C
        O.TEMP_WATER_HOT: 60,  # 60°C
        O.DATETIMES: "weather",
        O.SEED: 42,  # Add a seed for reproducibility
    }
    data = {"weather": weather}

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series
    result = dhw.generate(obj, data)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check that the summary has the expected keys
    summary = result["summary"]
    assert f"{Types.DHW}:volume_total[l]" in summary
    assert f"{Types.DHW}:volume_avg[l]" in summary
    assert f"{Types.DHW}:volume_peak[l]" in summary
    assert f"{Types.DHW}:energy_total[Wh]" in summary
    assert f"{Types.DHW}:energy_avg[Wh]" in summary
    assert f"{Types.DHW}:energy_peak[Wh]" in summary
    assert f"{Types.DHW}:power_avg[W]" in summary
    assert f"{Types.DHW}:power_max[W]" in summary
    assert f"{Types.DHW}:power_min[W]" in summary

    # Check that the timeseries has the expected columns
    timeseries = result["timeseries"]
    assert f"{Types.DHW}:volume[l]" in timeseries.columns
    assert f"{Types.DHW}:energy[Wh]" in timeseries.columns
    assert f"{Types.DHW}:power[W]" in timeseries.columns

    # Check that the values are reasonable
    assert summary[f"{Types.DHW}:volume_total[l]"] > 0
    assert summary[f"{Types.DHW}:energy_total[Wh]"] > 0
    assert np.all(timeseries[f"{Types.DHW}:volume[l]"] >= 0)
    assert np.all(timeseries[f"{Types.DHW}:energy[Wh]"] >= 0)
    assert np.all(timeseries[f"{Types.DHW}:power[W]"] >= 0)


# Test reproducibility with different seeds
def test_jordan_vajen_reproducibility():
    """Test that the JordanVajen class produces reproducible results with the same seed."""
    # Create sample data
    weather = create_sample_weather()
    data = {"weather": weather}

    # Create two objects with the same seed
    obj1 = {
        O.ID: "test_obj1",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    obj2 = {
        O.ID: "test_obj2",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series for both objects
    result1 = dhw.generate(obj1, data)
    result2 = dhw.generate(obj2, data)

    # Check that the results are identical
    pd.testing.assert_series_equal(
        result1["timeseries"][f"{Types.DHW}:volume[l]"], result2["timeseries"][f"{Types.DHW}:volume[l]"]
    )
    pd.testing.assert_series_equal(
        result1["timeseries"][f"{Types.DHW}:energy[Wh]"], result2["timeseries"][f"{Types.DHW}:energy[Wh]"]
    )

    # Create a third object with a different seed
    obj3 = {
        O.ID: "test_obj3",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 43,  # Different seed
    }

    # Generate DHW demand time series for the third object
    result3 = dhw.generate(obj3, data)

    # Check that the results are different
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(
            result1["timeseries"][f"{Types.DHW}:volume[l]"], result3["timeseries"][f"{Types.DHW}:volume[l]"]
        )


# Test with different dwelling sizes
def test_jordan_vajen_dwelling_size():
    """Test that the JordanVajen class produces different results with different dwelling sizes."""
    # Create sample data
    weather = create_sample_weather()
    data = {"weather": weather}

    # Create objects with different dwelling sizes
    obj1 = {
        O.ID: "test_obj1",
        O.DWELLING_SIZE: 50,  # Small dwelling
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    obj2 = {
        O.ID: "test_obj2",
        O.DWELLING_SIZE: 200,  # Large dwelling
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series for both objects
    result1 = dhw.generate(obj1, data)
    result2 = dhw.generate(obj2, data)

    # Check that the results are different
    assert result1["summary"][f"{Types.DHW}:volume_total[l]"] < result2["summary"][f"{Types.DHW}:volume_total[l]"]
    assert result1["summary"][f"{Types.DHW}:energy_total[Wh]"] < result2["summary"][f"{Types.DHW}:energy_total[Wh]"]


# Test with different water temperatures
def test_jordan_vajen_water_temperatures():
    """Test that the JordanVajen class produces different results with different water temperatures."""
    # Create sample data
    weather = create_sample_weather()
    data = {"weather": weather}

    # Create objects with different water temperatures
    obj1 = {
        O.ID: "test_obj1",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 5,  # Colder water
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    obj2 = {
        O.ID: "test_obj2",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 15,  # Warmer water
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series for both objects
    result1 = dhw.generate(obj1, data)
    result2 = dhw.generate(obj2, data)

    # Check that the volume results are the same (water temperature doesn't affect volume)
    pd.testing.assert_series_equal(
        result1["timeseries"][f"{Types.DHW}:volume[l]"], result2["timeseries"][f"{Types.DHW}:volume[l]"]
    )

    # Check that the energy results are different (colder water requires more energy to heat)
    assert result1["summary"][f"{Types.DHW}:energy_total[Wh]"] > result2["summary"][f"{Types.DHW}:energy_total[Wh]"]


# Test with seasonal variation
def test_jordan_vajen_seasonal_variation():
    """Test that the JordanVajen class handles seasonal variation correctly."""
    # Create sample data for a full year
    weather = create_sample_weather(days=365)
    data = {"weather": weather}

    # Create objects with different seasonal variation
    obj1 = {
        O.ID: "test_obj1",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEASONAL_VARIATION: 0.0,  # No seasonal variation
        O.SEED: 42,
    }
    obj2 = {
        O.ID: "test_obj2",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEASONAL_VARIATION: 0.3,  # Strong seasonal variation
        O.SEED: 42,
    }

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series for both objects
    result1 = dhw.generate(obj1, data)
    result2 = dhw.generate(obj2, data)

    # Check that the results are different
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(
            result1["timeseries"][f"{Types.DHW}{SEP}volume[l]"], result2["timeseries"][f"{Types.DHW}{SEP}volume[l]"]
        )

    # Check seasonal pattern by comparing winter vs. summer demand
    # For the object with seasonal variation, winter demand should be higher than summer demand
    ts2 = result2["timeseries"][f"{Types.DHW}{SEP}volume[l]"]
    winter_demand = ts2["2023-01-01":"2023-03-31"].mean()  # Winter (Jan-Mar)
    summer_demand = ts2["2023-07-01":"2023-09-30"].mean()  # Summer (Jul-Sep)
    assert winter_demand > summer_demand


# Test edge cases
def test_jordan_vajen_edge_cases():
    """Test that the JordanVajen class handles edge cases correctly."""
    # Create sample data
    weather = create_sample_weather()
    data = {"weather": weather}

    # Test with very small dwelling
    obj_small = {
        O.ID: "test_obj_small",
        O.DWELLING_SIZE: 1,  # Very small dwelling
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    dhw = JordanVajen()
    result_small = dhw.generate(obj_small, data)
    assert result_small["summary"][f"{Types.DHW}{SEP}volume_total[l]"] > 0  # Should still produce some demand

    # Test with very large dwelling
    obj_large = {
        O.ID: "test_obj_large",
        O.DWELLING_SIZE: 1000,  # Very large dwelling
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    result_large = dhw.generate(obj_large, data)
    assert result_large["summary"][f"{Types.DHW}{SEP}volume_total[l]"] > 0  # Should produce demand

    # Test with equal water temperatures
    obj_equal_temp = {
        O.ID: "test_obj_equal_temp",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 40,
        O.TEMP_WATER_HOT: 40,  # Equal to cold water temperature
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    result_equal_temp = dhw.generate(obj_equal_temp, data)
    assert np.all(result_equal_temp["timeseries"][f"{Types.DHW}{SEP}energy[Wh]"] == 0)  # Energy should be zero


# Test error handling
def test_jordan_vajen_error_handling():
    """Test that the JordanVajen class handles errors correctly."""
    # Create sample data
    weather = create_sample_weather()
    data = {"weather": weather}
    dhw = JordanVajen()

    # Test with missing required parameter (dwelling_size)
    obj_missing_dwelling = {
        O.ID: "test_obj_missing_dwelling",
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather",
        O.SEED: 42,
    }
    with pytest.raises(KeyError):
        dhw.generate(obj_missing_dwelling, data)

    # Test with missing required parameter (datetimes)
    obj_missing_datetimes = {
        O.ID: "test_obj_missing_datetimes",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.SEED: 42,
    }
    with pytest.raises(KeyError):
        dhw.generate(obj_missing_datetimes, data)

    # Test with invalid datetimes reference
    obj_invalid_datetimes = {
        O.ID: "test_obj_invalid_datetimes",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "nonexistent_data",
        O.SEED: 42,
    }
    with pytest.raises(KeyError):
        dhw.generate(obj_invalid_datetimes, data)


# Test with different time resolutions
def test_jordan_vajen_time_resolution():
    """Test that the JordanVajen class works with different time resolutions."""
    # Create sample data with different time resolutions
    weather_hourly = create_sample_weather(freq="h")
    weather_15min = create_sample_weather(freq="15min")
    weather_daily = create_sample_weather(freq="d")

    data = {"weather_hourly": weather_hourly, "weather_15min": weather_15min, "weather_daily": weather_daily}

    # Create objects for different time resolutions
    obj_hourly = {
        O.ID: "test_obj_hourly",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather_hourly",
        O.SEED: 42,
    }
    obj_15min = {
        O.ID: "test_obj_15min",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather_15min",
        O.SEED: 42,
    }
    obj_daily = {
        O.ID: "test_obj_daily",
        O.DWELLING_SIZE: 100,
        O.TEMP_WATER_COLD: 10,
        O.TEMP_WATER_HOT: 60,
        O.DATETIMES: "weather_daily",
        O.SEED: 42,
    }

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series for all objects
    result_hourly = dhw.generate(obj_hourly, data)
    result_15min = dhw.generate(obj_15min, data)
    result_daily = dhw.generate(obj_daily, data)

    # Check that the results have the correct length
    assert len(result_hourly["timeseries"]) == len(weather_hourly)
    assert len(result_15min["timeseries"]) == len(weather_15min)
    assert len(result_daily["timeseries"]) == len(weather_daily)

    # Check that the total demand is similar across different resolutions
    # (allowing for some variation due to randomness)
    hourly_total = result_hourly["summary"][f"{Types.DHW}{SEP}volume_total[l]"]
    daily_total = result_daily["summary"][f"{Types.DHW}{SEP}volume_total[l]"]
    assert 0.5 * hourly_total <= daily_total <= 1.5 * hourly_total
