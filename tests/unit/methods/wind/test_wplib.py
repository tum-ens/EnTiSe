import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.methods.wind.wplib import WPLib, calculate_timeseries, get_input_data, process_weather_data


@pytest.fixture
def dummy_weather():
    """Create a dummy weather DataFrame for testing."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            "temperature_2m": np.full(24, 15),  # 15°C
            "surface_pressure": np.full(24, 1013.25),  # 1013.25 hPa
            "wind_speed_100m": np.concatenate([np.full(6, 2), np.linspace(2, 15, 12), np.full(6, 5)]),  # m/s
            "wind_direction_100m": np.full(24, 180),  # degrees
            "roughness_length": np.full(24, 0.15),  # roughness length
        },
        index=index,
    )


@pytest.fixture
def dummy_inputs(dummy_weather):
    """Create dummy inputs for testing the WindLib class."""
    obj = {
        O.ID: "wind_system_1",
        O.POWER: 5000,  # 5 kW system
        "turbine_type": "SWT130/3600",
        "hub_height": 135,
    }

    data = {O.WEATHER: dummy_weather}

    return obj, data


def test_windlib_initialization():
    """Test WindLib class initialization and attributes."""
    windlib = WPLib()

    # Check class attributes
    assert windlib.types == [Types.WIND]
    assert windlib.name == "wplib"
    assert set(windlib.required_keys) == {O.WEATHER}
    assert set(windlib.optional_keys) == {O.POWER, O.TURBINE_TYPE, O.HUB_HEIGHT, O.WIND_MODEL}
    assert windlib.required_timeseries == [O.WEATHER]
    assert windlib.optional_timeseries == [O.WIND_MODEL]

    # Check output definitions
    assert f"{C.GENERATION}_{Types.WIND}" in windlib.output_summary
    assert f"{O.GEN_MAX}_{Types.WIND}" in windlib.output_summary
    assert f"{C.FLH}_{Types.WIND}" in windlib.output_summary
    assert f"{C.GENERATION}_{Types.WIND}" in windlib.output_timeseries


def test_process_weather_data(dummy_weather):
    """Test the process_weather_data function."""
    processed = process_weather_data(dummy_weather)

    # Check that the output has a multi-index for columns
    assert isinstance(processed.columns, pd.MultiIndex)

    # Check that required variables are present
    assert ("temperature", 2) in processed.columns
    assert ("pressure", 0) in processed.columns
    assert ("wind_speed", 100) in processed.columns
    assert ("wind_direction", 100) in processed.columns
    assert ("roughness_length", 0) in processed.columns

    # Check that temperature is converted to Kelvin
    assert processed[("temperature", 2)].iloc[0] > 273  # Should be around 288K (15°C + 273.15)

    # Check that pressure is converted to Pa
    assert processed[("pressure", 0)].iloc[0] > 100000  # Should be around 101325 Pa


def test_get_input_data(dummy_inputs):
    """Test the get_input_data function."""
    obj, data = dummy_inputs
    obj_out, data_out = get_input_data(obj, data)

    # Check that required fields are present
    assert O.ID in obj_out
    assert O.POWER in obj_out
    assert O.TURBINE_TYPE in obj_out
    assert O.HUB_HEIGHT in obj_out

    # Check that data fields are present
    assert O.WEATHER in data_out

    # Check that weather data has been processed correctly
    weather = data_out[O.WEATHER]
    assert isinstance(weather.columns, pd.MultiIndex)
    assert ("wind_speed", 100) in weather.columns
    assert ("wind_direction", 100) in weather.columns


def test_get_input_data_defaults():
    """Test that get_input_data applies default values correctly."""
    obj = {}

    data = {
        O.WEATHER: pd.DataFrame(
            {
                C.DATETIME: pd.date_range("2025-01-01", periods=24, freq="h"),
                "temperature_2m": np.full(24, 15),
                "surface_pressure": np.full(24, 1013.25),
                "wind_speed_100m": np.full(24, 5),
                "wind_direction_100m": np.full(24, 180),
            }
        )
    }

    obj_out, data_out = get_input_data(obj, data)

    # Check that default values are applied
    assert obj_out["turbine_type"] == "SWT130/3600"
    assert obj_out["hub_height"] == 135
    assert obj_out[O.POWER] == 1


def test_get_input_data_missing_weather():
    """Test that get_input_data raises an exception when weather data is missing."""
    obj = {}

    data = {}  # No weather data

    with pytest.raises(Exception, match=f"{O.WEATHER} not available"):
        get_input_data(obj, data)


def test_calculate_timeseries(dummy_inputs):
    """Test the calculate_timeseries function."""
    obj, data = dummy_inputs
    obj_out, data_out = get_input_data(obj, data)

    ts = calculate_timeseries(obj_out, data_out)

    # Check that the output is a DataFrame
    assert isinstance(ts, pd.DataFrame)

    # Check that the output has the expected shape
    assert len(ts) == len(data[O.WEATHER])

    # Check that the output has non-negative values
    assert (ts >= 0).all().all()

    # Check that there's some generation during high wind periods
    high_wind = data[O.WEATHER]["wind_speed_100m"] > 10
    assert ts.loc[high_wind].sum().sum() > 0


def test_generate_method(dummy_inputs):
    """Test the generate method of the WindLib class."""
    obj, data = dummy_inputs
    windlib = WPLib()

    result = windlib.generate(obj, data)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check summary values
    summary = result["summary"]
    assert f"{C.GENERATION}_{Types.WIND}" in summary
    assert f"{O.GEN_MAX}_{Types.WIND}" in summary
    assert f"{C.FLH}_{Types.WIND}" in summary

    # Check that generation values are non-negative
    assert summary[f"{C.GENERATION}_{Types.WIND}"].item() >= 0
    assert summary[f"{O.GEN_MAX}_{Types.WIND}"].item() >= 0
    assert summary[f"{C.FLH}_{Types.WIND}"].item() >= 0

    # Check timeseries values
    ts = result["timeseries"]
    assert f"{C.POWER}_{Types.WIND}" in ts.columns
    assert len(ts) == len(data[O.WEATHER])
    assert (ts >= 0).all().all()


def test_generate_with_kwargs():
    """Test the generate method with keyword arguments."""
    # Create test data
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            "temperature_2m": np.full(24, 15),
            "surface_pressure": np.full(24, 1013.25),
            "wind_speed_100m": np.full(24, 10),
            "wind_direction_100m": np.full(24, 180),
            "roughness_length": np.full(24, 0.15),
        },
        index=index,
    )

    windlib = WPLib()

    # Call generate with keyword arguments
    result = windlib.generate(data={"weather": weather}, power=5000, turbine_type="V164/8000", hub_height=140)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check that generation occurs
    summary = result["summary"]
    assert summary[f"{C.GENERATION}_{Types.WIND}"].item() > 0
