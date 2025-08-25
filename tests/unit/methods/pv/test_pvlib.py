import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.pv.pvlib import PVLib, calculate_timeseries


@pytest.fixture
def dummy_weather():
    """Create a dummy weather DataFrame for testing."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.SOLAR_GHI: np.concatenate([np.zeros(6), np.linspace(0, 800, 12), np.zeros(6)]),
            C.SOLAR_DHI: np.concatenate([np.zeros(6), np.linspace(0, 200, 12), np.zeros(6)]),
            C.SOLAR_DNI: np.concatenate([np.zeros(6), np.linspace(0, 600, 12), np.zeros(6)]),
            C.TEMP_OUT: np.concatenate([np.full(6, 10), np.linspace(10, 25, 12), np.full(6, 15)]),
        },
        index=index,
    )


@pytest.fixture
def dummy_pv_arrays():
    """Create dummy PV array parameters for testing."""
    return {
        "module_parameters": {"pdc0": 1, "gamma_pdc": -0.004},
        "temperature_model_parameters": {"a": -3.56, "b": -0.075, "deltaT": 3},
    }


@pytest.fixture
def dummy_pv_inverter():
    """Create dummy PV inverter parameters for testing."""
    return {"pdc0": 3}


@pytest.fixture
def dummy_inputs(dummy_weather, dummy_pv_arrays, dummy_pv_inverter):
    """Create dummy inputs for testing the PVLib class."""
    obj = {
        O.ID: "pv_system_1",
        O.LAT: 48.1,
        O.LON: 11.6,
        O.ALTITUDE: 520,
        O.AZIMUTH: 180,  # South-facing
        O.TILT: 30,
        O.POWER: 5000,  # 5 kW system
        O.PV_ARRAYS: "pv_arrays",
        O.PV_INVERTER: "pv_inverter",
    }

    data = {O.WEATHER: dummy_weather, "pv_arrays": dummy_pv_arrays, "pv_inverter": dummy_pv_inverter}

    return obj, data


def test_pvlib_initialization():
    """Test PVLib class initialization and attributes."""
    pvlib = PVLib()

    # Check class attributes
    assert pvlib.types == [Types.PV]
    assert pvlib.name == "pvlib"
    assert set(pvlib.required_keys) == {O.LAT, O.LON, O.WEATHER}
    assert set(pvlib.optional_keys) == {O.POWER, O.AZIMUTH, O.TILT, O.ALTITUDE, O.PV_ARRAYS, O.PV_INVERTER}
    assert pvlib.required_timeseries == [O.WEATHER]
    assert pvlib.optional_timeseries == [O.PV_ARRAYS]

    # Check output definitions
    assert f"{C.GENERATION}_{Types.PV}" in pvlib.output_summary
    assert f"{O.GEN_MAX}_{Types.PV}" in pvlib.output_summary
    assert f"{C.FLH}_{Types.PV}" in pvlib.output_summary
    assert f"{C.GENERATION}_{Types.PV}" in pvlib.output_timeseries


def test_get_input_data(dummy_inputs):
    """Test the get_input_data function."""
    obj, data = dummy_inputs
    pvlib = PVLib()
    obj_out, data_out = pvlib._get_input_data(obj, data)

    # Check that required fields are present
    assert O.ID in obj_out
    assert O.LAT in obj_out
    assert O.LON in obj_out
    assert O.ALTITUDE in obj_out
    assert O.AZIMUTH in obj_out
    assert O.TILT in obj_out
    assert O.POWER in obj_out
    assert O.PV_ARRAYS in obj_out
    assert O.PV_INVERTER in obj_out

    # Check that data fields are present
    assert O.WEATHER in data_out
    assert O.PV_ARRAYS in data_out
    assert O.PV_INVERTER in data_out

    # Check that weather data has been processed correctly
    weather = data_out[O.WEATHER]
    assert "ghi" in weather.columns
    assert "dni" in weather.columns
    assert "dhi" in weather.columns
    assert weather.index.tz is not None  # Check that timezone is set


def test_get_input_data_defaults():
    """Test that get_input_data applies default values correctly."""
    obj = {O.LAT: 48.1, O.LON: 11.6}

    data = {
        O.WEATHER: pd.DataFrame(
            {
                C.DATETIME: pd.date_range("2025-01-01", periods=24, freq="h"),
                C.SOLAR_GHI: np.zeros(24),
                C.SOLAR_DHI: np.zeros(24),
                C.SOLAR_DNI: np.zeros(24),
            }
        )
    }

    pvlib = PVLib()
    obj_out, data_out = pvlib._get_input_data(obj, data)

    # Check that default values are applied
    assert obj_out[O.AZIMUTH] == 0
    assert obj_out[O.TILT] == 0
    assert obj_out[O.POWER] == 1
    assert obj_out[O.ALTITUDE] is not None  # Should be looked up
    assert O.PV_ARRAYS in data_out
    assert O.PV_INVERTER in data_out


def test_get_input_data_missing_weather():
    """Test that get_input_data raises an exception when weather data is missing."""
    obj = {O.LAT: 48.1, O.LON: 11.6}

    data = {}  # No weather data

    with pytest.raises(Exception, match=f"{O.WEATHER} not available"):
        PVLib()._get_input_data(obj, data)


def test_calculate_timeseries(dummy_inputs):
    """Test the calculate_timeseries function."""
    obj, data = dummy_inputs
    pvlib = PVLib()
    obj_out, data_out = pvlib._get_input_data(obj, data)

    ts = calculate_timeseries(obj_out, data_out)

    # Check that the output is a DataFrame
    assert isinstance(ts, pd.DataFrame)

    # Check that the output has the expected shape
    assert len(ts) == len(data[O.WEATHER])

    # Check that the output has non-negative values
    assert (ts >= 0).all().all()

    # Check that there's some generation during daylight hours
    daytime = data[O.WEATHER][C.SOLAR_GHI] > 0
    assert ts.loc[daytime].sum().sum() > 0


def test_generate_method(dummy_inputs):
    """Test the generate method of the PVLib class."""
    obj, data = dummy_inputs
    pvlib = PVLib()

    result = pvlib.generate(obj, data)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check summary values
    summary = result["summary"]
    assert f"{Types.PV}{SEP}{C.GENERATION}" in summary
    assert f"{Types.PV}{SEP}{O.GEN_MAX}" in summary
    assert f"{Types.PV}{SEP}{C.FLH}" in summary

    # Check that generation values are non-negative
    assert summary[f"{Types.PV}{SEP}{C.GENERATION}"].item() >= 0
    assert summary[f"{Types.PV}{SEP}{O.GEN_MAX}"].item() >= 0
    assert summary[f"{Types.PV}{SEP}{C.FLH}"].item() >= 0

    # Check timeseries values
    ts = result["timeseries"]
    assert f"{Types.PV}{SEP}{C.POWER}" in ts.columns
    assert len(ts) == len(data[O.WEATHER])
    assert (ts >= 0).all().all()


def test_edge_cases():
    """Test edge cases like extreme values."""
    # Test with very high tilt
    obj = {
        O.ID: "pv_system_edge",
        O.LAT: 48.1,
        O.LON: 11.6,
        O.TILT: 90,  # Vertical panels
        O.AZIMUTH: 180,
        O.POWER: 5000,
    }

    # Create minimal weather data
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.SOLAR_GHI: np.full(24, 500),
            C.SOLAR_DHI: np.full(24, 100),
            C.SOLAR_DNI: np.full(24, 400),
            C.TEMP_OUT: np.full(24, 20),
        },
        index=index,
    )

    data = {O.WEATHER: weather}

    pvlib = PVLib()
    result = pvlib.generate(obj, data, Types.PV)

    # Check that the result has the expected structure
    assert "summary" in result
    assert "timeseries" in result

    # Check that generation occurs even with extreme tilt
    summary = result["summary"]
    assert summary[f"{Types.PV}{SEP}{C.GENERATION}"].item() > 0


def test_warning_for_invalid_azimuth(caplog):
    """Test that a warning is logged for invalid azimuth values."""
    obj = {
        O.ID: "pv_system_warning",
        O.LAT: 48.1,
        O.LON: 11.6,
        O.AZIMUTH: 400,  # Invalid azimuth
        O.POWER: 5000,
    }

    # Create minimal weather data
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.SOLAR_GHI: np.full(24, 500),
            C.SOLAR_DHI: np.full(24, 100),
            C.SOLAR_DNI: np.full(24, 400),
            C.TEMP_OUT: np.full(24, 20),
        },
        index=index,
    )

    data = {O.WEATHER: weather}

    # Test that a warning is logged
    caplog.clear()
    pvlib = PVLib()
    pvlib.generate(obj, data, Types.PV)

    # Check that the warning was logged
    assert "Azimuth value 400 outside normal range [0-360]" in caplog.text
