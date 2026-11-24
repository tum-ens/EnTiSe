import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.methods.hvac.R1C0 import R1C0


@pytest.fixture
def dummy_inputs():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(24, 0.0),
            C.SOLAR_GHI: np.full(24, 100.0),
            C.SOLAR_DHI: np.full(24, 20.0),
            C.SOLAR_DNI: np.full(24, 80.0),
        },
        index=index,
    )

    windows = pd.DataFrame(
        [{O.ID: "obj1", C.AREA: 10.0, C.G_VALUE: 0.7, C.SHADING: 1.0, C.TILT: 90.0, C.ORIENTATION: 180.0}]
    )

    internal_gains = pd.DataFrame({"obj1": np.arange(24)}, index=pd.date_range("2025-01-01", periods=24, freq="h"))

    obj = {
        O.ID: "obj1",
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
        O.WEATHER: f"{O.WEATHER}_dummy",
    }

    data = {f"{O.WEATHER}_dummy": weather, O.WINDOWS: windows, "internal_gains": internal_gains}

    return obj, data


def test_r1c0_outputs(dummy_inputs):
    obj, data = dummy_inputs
    r1c0 = R1C0()
    result = r1c0.generate(obj, data, Types.HVAC)

    assert "timeseries" in result
    ts = result["timeseries"]
    assert all(
        col in ts.columns for col in [C.TEMP_IN, f"{Types.HEATING}{SEP}{C.LOAD}[W]", f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    )
    assert len(ts) == 24


@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.zeros(24),
            C.SOLAR_GHI: np.full(24, 100.0),
            C.SOLAR_DHI: np.full(24, 20.0),
            C.SOLAR_DNI: np.full(24, 80.0),
        },
        index=index,
    )


@pytest.fixture
def dummy_windows():
    return pd.DataFrame(
        [{O.ID: "obj1", "area": 10.0, "transmittance": 0.7, "shading": 1.0, O.TILT: 90.0, O.ORIENTATION: 180.0}]
    )


def test_r1c0_constant_internal_gains(minimal_weather, dummy_windows):
    obj = {
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

    r1c0 = R1C0()
    result = r1c0.generate(obj, data, Types.HVAC)

    assert K.SUMMARY in result
    assert K.TIMESERIES in result
    assert isinstance(result[K.TIMESERIES], pd.DataFrame)
    assert C.TEMP_IN in result[K.TIMESERIES]


def _air_change_heating_load(minimal_weather, **geometry):
    """Mean heating load of a 1R0C zone at 0 °C outside whose ventilation is 0.5 air changes per hour."""
    vent = pd.DataFrame({"rate[1/h]": np.full(24, 0.5)}, index=minimal_weather.index)
    obj = {
        O.ID: "obj1",
        O.RESISTANCE: 1.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.POWER_HEATING: 1e6,
        O.VENTILATION: "vent",
        O.VENTILATION_COL: "rate[1/h]",
        **geometry,
    }
    data = {O.WEATHER: minimal_weather, "vent": vent}
    result = R1C0().generate(obj, data, Types.HVAC)
    return result[K.TIMESERIES][f"{Types.HEATING}{SEP}{C.LOAD}[W]"].mean()


@pytest.mark.parametrize("area", [100.0, 200.0])
def test_r1c0_air_changes_use_the_zone_volume(minimal_weather, area):
    """Ventilation in air changes per hour scales with the object's area and height."""
    h_ve = 0.5 * area * 2.6 * 1.2 * 1000 / 3600
    expected = 20.0 * (1.0 + h_ve)

    load = _air_change_heating_load(minimal_weather, **{O.AREA: area, O.HEIGHT: 2.6})

    assert load == pytest.approx(expected, abs=1.0)


def test_r1c0_air_changes_without_geometry_use_the_default_volume(minimal_weather):
    """Without area and height, the default zone volume applies."""
    from entise.constants import Constants as Const

    h_ve = 0.5 * Const.DEFAULT_AREA.value * Const.DEFAULT_HEIGHT.value * 1.2 * 1000 / 3600
    assert _air_change_heating_load(minimal_weather) == pytest.approx(20.0 * (1.0 + h_ve), abs=1.0)


def _band_run(temp_air: float, gains: float, **keys) -> pd.DataFrame:
    """Run 1R0C for 24 constant hours with H = 100 W/K, no ventilation and an indoor band of 20-25 °C."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame({C.DATETIME: index, C.TEMP_AIR: np.full(24, temp_air)}, index=index)
    obj = {
        O.ID: "band",
        O.RESISTANCE: 0.01,
        O.VENTILATION: 0.0,
        O.GAINS_INTERNAL: gains,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        **keys,
    }
    return R1C0().generate(obj, {O.WEATHER: weather})[K.TIMESERIES]


HEAT = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
COOL = f"{Types.COOLING}{SEP}{C.LOAD}[W]"


def test_r1c0_floats_inside_the_band_without_load():
    """Between the minimum and maximum, the zone takes its free-running temperature T_out + gains / H."""
    ts = _band_run(15.0, 600.0)

    assert ts[C.TEMP_IN].tolist() == pytest.approx([21.0] * 24)
    assert (ts[HEAT] == 0).all() and (ts[COOL] == 0).all()


def test_r1c0_cools_down_to_the_maximum_whatever_the_outdoor_temperature():
    """Gains that push the zone above the maximum are cooled away, also when it is cooler outside."""
    ts = _band_run(22.0, 800.0)

    assert ts[COOL].tolist() == [500] * 24
    assert ts[C.TEMP_IN].tolist() == pytest.approx([25.0] * 24)
    assert (ts[HEAT] == 0).all()


@pytest.mark.parametrize("init", [20.0, 22.0])
def test_r1c0_heats_up_to_the_minimum_and_ignores_the_initial_temperature(init):
    """Heating holds the minimum; 1R0C has no thermal mass, so the initial temperature has no effect."""
    ts = _band_run(0.0, 0.0, **{O.TEMP_INIT: init})

    assert ts[HEAT].tolist() == [2000] * 24
    assert ts[C.TEMP_IN].tolist() == pytest.approx([20.0] * 24)


def test_r1c0_capped_or_switched_off_plant_leaves_the_zone_outside_the_band():
    """With too little power, or with the plant off, the zone ends up where the available power leaves it."""
    capped = _band_run(0.0, 0.0, **{O.POWER_HEATING: 1000.0})
    off = _band_run(22.0, 800.0, **{O.ACTIVE_COOLING: False})

    assert capped[HEAT].tolist() == [1000] * 24
    assert capped[C.TEMP_IN].tolist() == pytest.approx([10.0] * 24)
    assert (off[COOL] == 0).all()
    assert off[C.TEMP_IN].tolist() == pytest.approx([30.0] * 24)
