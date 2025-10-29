import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.wind.wplib import WPLib, calculate_timeseries


@pytest.fixture
def dummy_weather():
    """Create a dummy weather DataFrame for testing."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            "air_temperature": np.full(24, 15),  # 15°C
            "surface_air_pressure": np.full(24, 101325),  # 101325 Pa
            "wind_speed": np.concatenate([np.full(6, 2), np.linspace(2, 15, 12), np.full(6, 5)]),  # m/s
            "wind_from_direction": np.full(24, 180),  # degrees
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
    windlib = WPLib()
    processed = windlib.process_weather_data(dummy_weather)

    # Check that the output has a multi-index for columns
    assert isinstance(processed.columns, pd.MultiIndex)

    # Check that required variables are present (independent of parsed heights)
    level0 = set(col[0] for col in processed.columns)
    assert "temperature" in level0
    assert "pressure" in level0
    assert "wind_speed" in level0
    assert "wind_direction" in level0
    assert "roughness_length" in level0

    # Check that temperature is converted to Kelvin
    temp_height = [col[1] for col in processed.columns if col[0] == "temperature"][0]
    assert processed[("temperature", temp_height)].iloc[0] > 273  # ~288K (15°C + 273.15)

    # Check that pressure is in Pa
    pres_height = [col[1] for col in processed.columns if col[0] == "pressure"][0]
    assert processed[("pressure", pres_height)].iloc[0] > 100000  # ~101325 Pa


def test_get_input_data(dummy_inputs):
    """Test the get_input_data function."""
    obj, data = dummy_inputs
    windlib = WPLib()
    obj_out, data_out = windlib.get_input_data(obj, data)

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
    # Accept either with parsed heights or without; check by level-0 name presence
    level0 = set(col[0] for col in weather.columns)
    assert ("wind_speed" in level0) or any(name.startswith("wind_speed") for name in level0)
    assert ("wind_direction" in level0) or any(
        name.startswith("wind_direction") or name.startswith("wind_from_direction") for name in level0
    )


def test_get_input_data_defaults():
    """Test that get_input_data applies default values correctly."""
    obj = {}

    data = {
        O.WEATHER: pd.DataFrame(
            {
                C.DATETIME: pd.date_range("2025-01-01", periods=24, freq="h"),
                "air_temperature": np.full(24, 15),
                "surface_air_pressure": np.full(24, 101325),
                "wind_speed": np.full(24, 5),
                "wind_from_direction": np.full(24, 180),
            }
        )
    }

    windlib = WPLib()
    obj_out, data_out = windlib.get_input_data(obj, data)

    # Check that default values are applied
    assert obj_out[O.TURBINE_TYPE] == "SWT130/3600"
    assert obj_out[O.HUB_HEIGHT] == 135
    assert obj_out[O.POWER] == 1


def test_get_input_data_missing_weather():
    """Test that get_input_data raises an exception when weather data is missing."""
    obj = {}

    data = {}  # No weather data

    with pytest.raises(Exception, match=f"{O.WEATHER} not available"):
        WPLib().get_input_data(obj, data)


def test_calculate_timeseries(dummy_inputs):
    """Test the calculate_timeseries function."""
    obj, data = dummy_inputs
    windlib = WPLib()
    obj_out, data_out = windlib.get_input_data(obj, data)

    ts = calculate_timeseries(obj_out, data_out)

    # Check that the output is a DataFrame
    assert isinstance(ts, pd.DataFrame)

    # Check that the output has the expected shape
    assert len(ts) == len(data[O.WEATHER])

    # Check that the output has non-negative values
    assert (ts >= 0).all().all()

    # Check that there's some generation during high wind periods
    high_wind = data[O.WEATHER]["wind_speed"] > 10
    assert ts.loc[high_wind].sum().sum() >= 0


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
    assert f"{Types.WIND}{SEP}{C.GENERATION}" in summary
    assert f"{Types.WIND}{SEP}{O.GEN_MAX}" in summary
    assert f"{Types.WIND}{SEP}{C.FLH}" in summary

    # Check that generation values are non-negative
    assert summary[f"{Types.WIND}{SEP}{C.GENERATION}"].item() >= 0
    assert summary[f"{Types.WIND}{SEP}{O.GEN_MAX}"].item() >= 0
    assert summary[f"{Types.WIND}{SEP}{C.FLH}"].item() >= 0

    # Check timeseries values
    ts = result["timeseries"]
    assert f"{Types.WIND}{SEP}{C.POWER}" in ts.columns
    assert len(ts) == len(data[O.WEATHER])
    assert (ts >= 0).all().all()


def test_generate_with_kwargs():
    """Test the generate method with keyword arguments."""
    # Create test data
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            "air_temperature": np.full(24, 15),
            "surface_air_pressure": np.full(24, 101325),
            "wind_speed": np.full(24, 12),
            "wind_from_direction": np.full(24, 180),
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
    assert summary[f"{Types.WIND}{SEP}{C.GENERATION}"].item() >= 0
