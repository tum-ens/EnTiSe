import pytest
import pandas as pd
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.constants import Objects as O

@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame(index=index)

@pytest.fixture
def dummy_windows():
    return pd.DataFrame([{"area": 1.0, "transmittance": 0.9, "shading": 1.0, "tilt": 90, "orientation": 180}])

def test_inactive_strategy_used(minimal_weather):
    sg = SolarGains()
    obj = {}
    data = {O.WEATHER: minimal_weather}

    result = sg.generate(obj, data)
    assert isinstance(result, pd.DataFrame)
    assert (result[O.GAINS_SOLAR] == 0).all()

def test_pvlib_strategy_used(minimal_weather, dummy_windows):
    sg = SolarGains()
    obj = {O.LAT: 48.1, O.LON: 11.6}
    data = {
        O.WEATHER: minimal_weather.assign(solar_ghi=500, solar_dhi=100, solar_dni=400),
        O.WINDOWS: dummy_windows,
    }

    result = sg.generate(obj, data)
    assert isinstance(result, pd.DataFrame)
    assert result[O.GAINS_SOLAR].ge(0).all()
    assert len(result) == 24
