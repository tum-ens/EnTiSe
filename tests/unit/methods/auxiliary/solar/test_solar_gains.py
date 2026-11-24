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


def _solar_position_methods(monkeypatch, mode, minimal_weather, dummy_windows):
    """Return the pvlib solar-position methods that one solar-gains call uses under an accelerator mode."""
    import pvlib

    from entise.perf import set_accelerator

    methods = []
    original = pvlib.location.Location.get_solarposition

    def spy(self, times, *args, method="nrel_numpy", **kwargs):
        methods.append(method)
        return original(self, times, *args, method="nrel_numpy", **kwargs)

    monkeypatch.setattr(pvlib.location.Location, "get_solarposition", spy)
    set_accelerator(mode)
    try:
        # A new frame, so the solar-position cache of an earlier call is not used.
        weather = minimal_weather.assign(**{C.SOLAR_GHI: 500, C.SOLAR_DHI: 100, C.SOLAR_DNI: 400})
        obj = {O.ID: 42, O.LAT: 48.1, O.LON: 11.6}
        SolarGains().generate(obj, {O.WEATHER: weather, O.WINDOWS: dummy_windows})
    finally:
        set_accelerator("auto")
    return methods


def test_the_solar_position_follows_the_accelerator_setting(monkeypatch, minimal_weather, dummy_windows):
    # Without numba, or with the accelerator off, pvlib's numba solver only warns and falls back.
    assert _solar_position_methods(monkeypatch, "none", minimal_weather, dummy_windows) == ["nrel_numpy"]


def test_the_solar_position_uses_numba_when_the_accelerator_does(monkeypatch, minimal_weather, dummy_windows):
    pytest.importorskip("numba")
    assert _solar_position_methods(monkeypatch, "numba", minimal_weather, dummy_windows) == ["nrel_numba"]
