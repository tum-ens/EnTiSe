import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.auxiliary.solar.selector import SolarGains


@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(index=index)


@pytest.fixture
def dummy_windows():
    return pd.DataFrame(
        [
            {
                O.ID: 42,
                C.AREA: 1.0,
                C.G_VALUE: 0.9,
                C.SHADING: 1.0,
                C.TILT: 90,
                C.ORIENTATION: 180,
            }
        ]
    )


def test_inactive_strategy_used(minimal_weather):
    sg = SolarGains()
    obj = {}
    data = {O.WEATHER: minimal_weather}

    result = sg.generate(obj, data)
    assert isinstance(result, pd.DataFrame)
    assert (result[O.GAINS_SOLAR] == 0).all()


def test_pvlib_strategy_used(minimal_weather, dummy_windows):
    sg = SolarGains()
    obj = {O.ID: 42, O.LAT: 48.1, O.LON: 11.6}
    data = {
        O.WEATHER: minimal_weather.assign(**{C.SOLAR_GHI: 500, C.SOLAR_DHI: 100, C.SOLAR_DNI: 400}),
        O.WINDOWS: dummy_windows,
    }

    result = sg.generate(obj, data)
    assert isinstance(result, pd.DataFrame)
    assert result[O.GAINS_SOLAR].ge(0).all()
    assert len(result) == 24
