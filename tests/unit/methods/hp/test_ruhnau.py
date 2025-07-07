import numpy as np
import pandas as pd
import pytest

import entise.methods.hp.defaults as defs
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.methods.hp.ruhnau import Ruhnau


@pytest.fixture
def dummy_weather():
    """Create a dummy weather DataFrame for testing."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")

    # Create temperature data with daily variation
    air_temp = 5 + 5 * np.sin(np.linspace(0, 2 * np.pi, 24))  # Varies between 0 and 10°C
    soil_temp = 8 + 2 * np.sin(np.linspace(0, 2 * np.pi, 24))  # Varies between 6 and 10°C
    water_temp = np.full(24, 10)  # Constant 10°C

    return pd.DataFrame(
        {C.DATETIME: index, C.TEMP: air_temp, C.TEMP_SOIL: soil_temp, C.TEMP_WATER_GROUND: water_temp}, index=index
    )


@pytest.fixture
def dummy_inputs(dummy_weather):
    """Create dummy inputs for testing the Ruhnau class."""
    obj = {
        O.ID: "hp_system_1",
        O.HP_SOURCE: defs.HP_AIR,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: -1.0,
        O.TEMP_WATER: 50,
        O.CORRECTION_FACTOR: 0.85,
        O.WEATHER: "weather",
    }

    data = {"weather": dummy_weather}

    return obj, data


def test_ruhnau_initialization():
    """Test Ruhnau class initialization and attributes."""
    ruhnau = Ruhnau()

    # Check class attributes
    assert ruhnau.types == [Types.HP]
    assert ruhnau.name == "ruhnau"
    assert set(ruhnau.required_keys) == {O.WEATHER}
    assert set(ruhnau.optional_keys) == {
        O.HP_SOURCE,
        O.HP_SINK,
        O.TEMP_SINK,
        O.GRADIENT_SINK,
        O.TEMP_WATER,
        O.CORRECTION_FACTOR,
        O.HP_SYSTEM,
    }
    assert ruhnau.required_timeseries == [O.WEATHER]
    assert ruhnau.optional_timeseries == [O.HP_SYSTEM]

    # Check output definitions
    assert f"{Types.HP}_{Types.HEATING}_avg" in ruhnau.output_summary
    assert f"{Types.HP}_{Types.HEATING}_min" in ruhnau.output_summary
    assert f"{Types.HP}_{Types.HEATING}_max" in ruhnau.output_summary
    assert f"{Types.HP}_{Types.DHW}_avg" in ruhnau.output_summary
    assert f"{Types.HP}_{Types.DHW}_min" in ruhnau.output_summary
    assert f"{Types.HP}_{Types.DHW}_max" in ruhnau.output_summary
    assert f"{Types.HP}_{Types.HEATING}" in ruhnau.output_timeseries
    assert f"{Types.HP}_{Types.DHW}" in ruhnau.output_timeseries


def test_generate_method(dummy_inputs):
    """Test the generate method of the Ruhnau class."""
    obj, data = dummy_inputs
    ruhnau = Ruhnau()

    result = ruhnau.generate(obj, data)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check summary values
    summary = result["summary"]
    assert f"{Types.HP}_{Types.HEATING}_avg" in summary
    assert f"{Types.HP}_{Types.HEATING}_min" in summary
    assert f"{Types.HP}_{Types.HEATING}_max" in summary
    assert f"{Types.HP}_{Types.DHW}_avg" in summary
    assert f"{Types.HP}_{Types.DHW}_min" in summary
    assert f"{Types.HP}_{Types.DHW}_max" in summary

    # Check that COP values are positive and within reasonable range
    assert summary[f"{Types.HP}_{Types.HEATING}_avg"] > 0
    assert summary[f"{Types.HP}_{Types.HEATING}_avg"] < 10  # Reasonable upper bound
    assert summary[f"{Types.HP}_{Types.DHW}_avg"] > 0
    assert summary[f"{Types.HP}_{Types.DHW}_avg"] < 10  # Reasonable upper bound

    # Check timeseries values
    ts = result["timeseries"]
    assert f"{Types.HP}_{Types.HEATING}" in ts.columns
    assert f"{Types.HP}_{Types.DHW}" in ts.columns
    assert len(ts) == len(data["weather"])
    assert (ts > 0).all().all()  # All COP values should be positive


def test_different_heat_pump_sources(dummy_weather):
    """Test COP calculation for different heat pump sources."""
    ruhnau = Ruhnau()

    # Test air source heat pump
    air_obj = {
        O.ID: "air_hp",
        O.HP_SOURCE: defs.HP_AIR,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: -1.0,
        O.TEMP_WATER: 50,
        O.WEATHER: "weather",
    }

    # Test ground source heat pump
    ground_obj = {
        O.ID: "ground_hp",
        O.HP_SOURCE: defs.HP_SOIL,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: -1.0,
        O.TEMP_WATER: 50,
        O.WEATHER: "weather",
    }

    # Test water source heat pump
    water_obj = {
        O.ID: "water_hp",
        O.HP_SOURCE: defs.HP_WATER,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: -1.0,
        O.TEMP_WATER: 50,
        O.WEATHER: "weather",
    }

    data = {"weather": dummy_weather}

    # Generate results for each heat pump type
    air_result = ruhnau.generate(air_obj, data)
    ground_result = ruhnau.generate(ground_obj, data)
    water_result = ruhnau.generate(water_obj, data)

    # Check that all results have the expected structure
    for result in [air_result, ground_result, water_result]:
        assert "summary" in result
        assert "timeseries" in result

    # Check that ground and water source heat pumps have higher COP than air source
    # This is expected due to more stable source temperatures
    assert (
        ground_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
        > air_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
    )
    assert (
        water_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
        > air_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
    )


def test_different_heat_sink_types(dummy_weather):
    """Test COP calculation for different heat sink types."""
    ruhnau = Ruhnau()

    # Test floor heating (low temperature)
    floor_obj = {
        O.ID: "floor_heating",
        O.HP_SOURCE: defs.HP_AIR,
        O.HP_SINK: defs.FLOOR,
        O.TEMP_SINK: 30,
        O.GRADIENT_SINK: -0.5,
        O.TEMP_WATER: 50,
        O.WEATHER: "weather",
    }

    # Test radiator heating (medium temperature)
    radiator_obj = {
        O.ID: "radiator_heating",
        O.HP_SOURCE: defs.HP_AIR,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: -1.0,
        O.TEMP_WATER: 50,
        O.WEATHER: "weather",
    }

    data = {"weather": dummy_weather}

    # Generate results for each sink type
    floor_result = ruhnau.generate(floor_obj, data)
    radiator_result = ruhnau.generate(radiator_obj, data)

    # Check that all results have the expected structure
    for result in [floor_result, radiator_result]:
        assert "summary" in result
        assert "timeseries" in result

    # Check that floor heating (lower temperature) has higher COP than radiator heating
    assert (
        floor_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
        > radiator_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
    )


def test_custom_coefficients(dummy_inputs):
    """Test COP calculation with custom coefficients."""
    obj, data = dummy_inputs
    ruhnau = Ruhnau()

    # Generate result with default coefficients
    default_result = ruhnau.generate(obj, data)

    # Generate result with custom coefficients
    custom_coefficients = {
        "a": 0.0008,  # Modified coefficient (default for air is 0.0005)
        "b": -0.12,  # Modified coefficient (default for air is -0.09)
        "c": 7.0,  # Modified coefficient (default for air is 6.08)
    }

    # Create a copy of the object and add cop_coefficients directly to it
    custom_obj = obj.copy()
    custom_obj["cop_coefficients"] = custom_coefficients

    custom_result = ruhnau.generate(custom_obj, data)

    # Check that both results have the expected structure
    assert "summary" in default_result
    assert "timeseries" in default_result
    assert "summary" in custom_result
    assert "timeseries" in custom_result

    # Check that the custom coefficients produce different COP values
    assert (
        custom_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
        != default_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
    )
    assert (
        custom_result["summary"][f"{Types.HP}_{Types.DHW}_avg"]
        != default_result["summary"][f"{Types.HP}_{Types.DHW}_avg"]
    )


def test_missing_weather_data():
    """Test that an error is raised when weather data is missing."""
    obj = {O.ID: "hp_system_error", O.HP_SOURCE: defs.HP_AIR, O.HP_SINK: defs.RADIATOR, O.WEATHER: "missing_weather"}

    data = {}  # No weather data

    ruhnau = Ruhnau()

    # Check that an error is raised
    # The actual error is an AttributeError when trying to access columns of None
    with pytest.raises((ValueError, AttributeError)):
        ruhnau.generate(obj, data)


def test_default_values(dummy_weather):
    """Test that default values are applied correctly."""
    obj = {
        O.ID: "hp_system_defaults",
        O.WEATHER: "weather",
        # No other parameters specified
    }

    data = {"weather": dummy_weather}

    ruhnau = Ruhnau()
    result = ruhnau.generate(obj, data)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check that COP values are calculated (indicating defaults were applied)
    assert result["summary"][f"{Types.HP}_{Types.HEATING}_avg"] > 0
    assert result["summary"][f"{Types.HP}_{Types.DHW}_avg"] > 0


def test_temperature_relationship(dummy_weather):
    """Test that COP values decrease as temperature difference increases."""
    # Create a modified weather dataset with varying temperatures
    modified_weather = dummy_weather.copy()

    # Create a range of temperatures from -10 to 20°C
    temperatures = np.linspace(-10, 20, len(modified_weather))
    modified_weather[C.TEMP] = temperatures

    obj = {
        O.ID: "hp_system_temp_test",
        O.HP_SOURCE: defs.HP_AIR,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,  # Fixed sink temperature
        O.GRADIENT_SINK: 0,  # No gradient
        O.WEATHER: "weather",
    }

    data = {"weather": modified_weather}

    ruhnau = Ruhnau()
    result = ruhnau.generate(obj, data)

    # Get the COP time series
    cop_series = result["timeseries"][f"{Types.HP}_{Types.HEATING}"]

    # Check that COP increases as outdoor temperature increases (smaller temperature difference)
    # We can check this by comparing the first half (colder) with the second half (warmer)
    first_half_avg = cop_series.iloc[: len(cop_series) // 2].mean()
    second_half_avg = cop_series.iloc[len(cop_series) // 2 :].mean()

    assert second_half_avg > first_half_avg


def test_reference_values():
    """Test COP calculation against reference values from the Ruhnau paper."""
    # Create a weather dataset with specific temperatures
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")

    # Test cases from paper (approximate values):
    # 1. Air source heat pump, outdoor temp = 0°C, radiator (40°C) -> COP ≈ 3
    # 2. Ground source heat pump, soil temp = 10°C, floor heating (30°C) -> COP ≈ 5
    # 3. Water source heat pump, water temp = 10°C, radiator (40°C) -> COP ≈ 4

    weather_data = pd.DataFrame(
        {C.DATETIME: index, C.TEMP: [0, 0, 0], C.TEMP_SOIL: [10, 10, 10], C.TEMP_WATER_GROUND: [10, 10, 10]},
        index=index,
    )

    data = {"weather": weather_data}

    ruhnau = Ruhnau()

    # Test air source heat pump
    air_obj = {
        O.ID: "air_reference",
        O.HP_SOURCE: defs.HP_AIR,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: 0,
        O.WEATHER: "weather",
    }

    # Test ground source heat pump
    ground_obj = {
        O.ID: "ground_reference",
        O.HP_SOURCE: defs.HP_SOIL,
        O.HP_SINK: defs.FLOOR,
        O.TEMP_SINK: 30,
        O.GRADIENT_SINK: 0,
        O.WEATHER: "weather",
    }

    # Test water source heat pump
    water_obj = {
        O.ID: "water_reference",
        O.HP_SOURCE: defs.HP_WATER,
        O.HP_SINK: defs.RADIATOR,
        O.TEMP_SINK: 40,
        O.GRADIENT_SINK: 0,
        O.WEATHER: "weather",
    }

    # Generate results
    air_result = ruhnau.generate(air_obj, data)
    ground_result = ruhnau.generate(ground_obj, data)
    water_result = ruhnau.generate(water_obj, data)

    # Check against reference values (with some tolerance)
    air_cop = air_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
    ground_cop = ground_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]
    water_cop = water_result["summary"][f"{Types.HP}_{Types.HEATING}_avg"]

    # Print actual values for debugging
    print(f"Air COP: {air_cop}, Ground COP: {ground_cop}, Water COP: {water_cop}")

    assert 2.5 < air_cop < 3.5  # Air source COP should be around 3
    assert 4.5 < ground_cop < 7.0  # Ground source COP should be around 5-6
    assert 3.5 < water_cop < 5.5  # Water source COP should be around 4-5
