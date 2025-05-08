import pandas as pd
import numpy as np
import pytest
from entise.methods.hvac.R1C1 import R1C1
from entise.constants import Objects as O, Columns as C, Types

@pytest.fixture
def dummy_inputs():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame({
        C.DATETIME: index,
        C.TEMP_OUT: np.full(24, 0.0),
        C.SOLAR_GHI: np.full(24, 100.0),
        C.SOLAR_DHI: np.full(24, 20.0),
        C.SOLAR_DNI: np.full(24, 80.0)
    }, index=index)

    windows = pd.DataFrame([{
        O.ID: "obj1", "area": 10.0, "transmittance": 0.7, "shading": 1.0,
        "tilt": 90.0, "orientation": 180.0
    }])

    internal_gains = pd.DataFrame({
        "obj1": np.arange(24)
    }, index=pd.date_range("2025-01-01", periods=24, freq="h"))

    obj = {
        O.ID: "obj1",
        O.CAPACITANCE: 1e5,
        O.RESISTANCE: 2.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.POWER_HEATING: 3000.0,
        O.POWER_COOLING: 3000.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL_COL: "obj1",  # Specifies the column in the timeseries
        O.GAINS_INTERNAL: "internal_gains",  # Points to timeseries in `data`
    }

    data = {
        O.WEATHER: weather,
        O.WINDOWS: windows,
        "internal_gains": internal_gains
    }

    return obj, data

def test_r1c1_outputs(dummy_inputs):
    obj, data = dummy_inputs
    r1c1 = R1C1()
    result = r1c1.generate(obj, data, Types.HVAC)

    assert "timeseries" in result
    ts = result["timeseries"]
    assert all(col in ts.columns for col in [C.TEMP_IN, f"{C.LOAD}_{Types.HEATING}", f"{C.LOAD}_{Types.COOLING}"])
    assert len(ts) == 24

@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({
        C.DATETIME: index,
        C.TEMP_OUT: np.zeros(24),
        C.SOLAR_GHI: np.full(24, 100.0),
        C.SOLAR_DHI: np.full(24, 20.0),
        C.SOLAR_DNI: np.full(24, 80.0)
    }, index=index)

@pytest.fixture
def dummy_windows():
    return pd.DataFrame([{
        O.ID: "obj1", "area": 10.0, "transmittance": 0.7, "shading": 1.0,
        "tilt": 90.0, "orientation": 180.0
    }])

def test_r1c1_constant_internal_gains(minimal_weather, dummy_windows):
    obj = {
        O.CAPACITANCE: 1e5,
        O.RESISTANCE: 2.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.POWER_HEATING: 3000.0,
        O.POWER_COOLING: 3000.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 100,  # Numeric → triggers InternalConstant
    }
    data = {
        O.WEATHER: minimal_weather,
        O.WINDOWS: dummy_windows,
    }

    r1c1 = R1C1()
    result = r1c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result
    assert isinstance(result["timeseries"], pd.DataFrame)
    assert C.TEMP_IN in result["timeseries"]
