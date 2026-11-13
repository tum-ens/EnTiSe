"""The solar-gain caches must not make an object's gains depend on what ran before it."""

import numpy as np
import pandas as pd
import pvlib
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.auxiliary.solar import strategies
from entise.methods.auxiliary.solar.strategies import SolarGainsPVLib

# Two positions inside the same 0.1 degree cell (both round to 48.1 N, 11.6 E).
POSITION_A = (48.11, 11.56)
POSITION_B = (48.14, 11.64)


@pytest.fixture(autouse=True)
def _empty_solar_cache():
    strategies.clear_solar_cache()
    yield
    strategies.clear_solar_cache()


@pytest.fixture
def windows():
    return pd.DataFrame(
        [
            {O.ID: 1, C.AREA: 4.0, C.G_VALUE: 0.6, C.SHADING: 1.0, C.TILT: 90, C.ORIENTATION: 180},
            {O.ID: 1, C.AREA: 2.0, C.G_VALUE: 0.6, C.SHADING: 1.0, C.TILT: 90, C.ORIENTATION: 90},
        ]
    )


def _weather(ghi, dhi, dni):
    index = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame({C.SOLAR_GHI: ghi, C.SOLAR_DHI: dhi, C.SOLAR_DNI: dni}, index=index)


@pytest.fixture
def weather():
    hours = np.arange(48) % 24
    ghi = np.clip(800.0 * np.sin((hours - 5) / 14 * np.pi), 0.0, None)
    return _weather(ghi, 0.2 * ghi, 0.7 * ghi)


def _gains(weather, windows, position):
    latitude, longitude = position
    result = SolarGainsPVLib().run(weather=weather, windows=windows, latitude=latitude, longitude=longitude)
    return result[O.GAINS_SOLAR].to_numpy()


def test_gains_do_not_depend_on_which_object_ran_first(weather, windows):
    a_first = _gains(weather, windows, POSITION_A)
    b_second = _gains(weather, windows, POSITION_B)

    strategies.clear_solar_cache()
    b_first = _gains(weather, windows, POSITION_B)
    a_second = _gains(weather, windows, POSITION_A)

    np.testing.assert_array_equal(b_second, b_first)
    np.testing.assert_array_equal(a_first, a_second)


def test_gains_match_an_uncached_run(weather, windows):
    _gains(weather, windows, POSITION_A)
    cached = _gains(weather, windows, POSITION_B)

    strategies.clear_solar_cache()
    uncached = _gains(weather, windows, POSITION_B)

    np.testing.assert_array_equal(cached, uncached)


def test_weather_frames_with_equal_summary_do_not_share_gains(weather, windows):
    """Same time grid and same mean GHI, but different direct and diffuse irradiance."""
    other = _weather(weather[C.SOLAR_GHI].to_numpy(), 0.6 * weather[C.SOLAR_GHI], 0.1 * weather[C.SOLAR_GHI])

    _gains(weather, windows, POSITION_A)
    after_first_frame = _gains(other, windows, POSITION_A)

    strategies.clear_solar_cache()
    alone = _gains(other, windows, POSITION_A)

    np.testing.assert_array_equal(after_first_frame, alone)


def test_same_weather_frame_reuses_the_cached_irradiance(weather, windows, mocker):
    spy = mocker.spy(pvlib.irradiance, "get_total_irradiance")

    _gains(weather, windows, POSITION_A)
    calls_after_first_object = spy.call_count
    _gains(weather, windows, POSITION_B)

    assert calls_after_first_object == 2
    assert spy.call_count == calls_after_first_object
