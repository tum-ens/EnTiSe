"""The physical methods read time through the run's time layer.

Each test checks structure only: instants, labels, row counts, which instant a
calculation sees, or an energy identity.
"""

import json
import logging
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core import _time
from entise.core.generator import Generator

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, os.pardir, os.pardir, "examples")
RC_EXAMPLES = ["hvac_1r0c", "hvac_1r1c", "hvac_5r1c", "hvac_7r2c"]


def _weather(first: int = 0, rows: int = 48) -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"))
    return weather.iloc[first : first + rows].reset_index(drop=True)


def _instants(weather: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(weather[C.DATETIME], utc=True)).as_unit("us")


def _example(example: str) -> tuple[dict, dict]:
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    data = {}
    folder = os.path.join(EXAMPLES, example, "data")
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            stem, ext = os.path.splitext(name)
            path = os.path.join(folder, name)
            if ext == ".csv":
                data[stem] = pd.read_csv(path)
            elif ext == ".json":
                with open(path, encoding="utf-8") as handle:
                    data[stem] = json.load(handle)
    return obj, data


def _run(objects: list[dict], data: dict, **options):
    generator = Generator(raise_on_error=True)
    generator.add_objects(objects)
    return generator.generate(data, workers=1, show_progress=False, **options)


def _first_ts(timeseries: dict, obj: dict, ts_type: str) -> pd.DataFrame:
    return timeseries[obj[O.ID]][ts_type]


# -- RC models ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("example", RC_EXAMPLES)
def test_rc_rows_are_intervals_starting_at_the_initial_temperature(example):
    obj, data = _example(example)
    weather = _weather(2000, 72)

    summary, timeseries = _run([obj], {**data, O.WEATHER: weather})

    hvac = _first_ts(timeseries, obj, Types.HVAC)
    assert len(hvac) == len(weather)
    assert hvac[C.TEMP_IN].iloc[0] == pytest.approx(float(obj[O.TEMP_INIT]))

    # One integration rule: summary energy equals the sum of mean power times the step,
    # up to the rounding of each row to whole watts.
    for kind in (Types.HEATING, Types.COOLING):
        energy = hvac[f"{kind}{SEP}{C.LOAD}[W]"].sum() * 1.0  # hourly steps
        assert summary.loc[obj[O.ID], f"{kind}{SEP}{C.DEMAND}[Wh]"] == pytest.approx(energy, abs=0.5 * len(hvac))


@pytest.mark.parametrize("example", RC_EXAMPLES)
def test_the_first_interval_is_simulated(example):
    obj, data = _example(example)
    weather = _weather(0, 24)
    # Starting below the minimum temperature needs heating in the very first interval.
    obj = {**obj, O.TEMP_INIT: float(obj[O.TEMP_MIN]) - 5.0}

    _, timeseries = _run([obj], {**data, O.WEATHER: weather})

    assert _first_ts(timeseries, obj, Types.HVAC)[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[0] > 0


@pytest.mark.parametrize("example", ["hvac_5r1c", "hvac_7r2c"])
def test_a_gains_table_read_with_read_csv_matches_the_same_table_indexed_by_time(example):
    obj, data = _example(example)
    weather = _weather(0, 48)
    gains = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "residential": np.linspace(100, 600, len(weather))})
    obj = {**obj, O.GAINS_INTERNAL: "gains", O.GAINS_INTERNAL_COL: "residential"}
    indexed = gains.set_index(pd.DatetimeIndex(pd.to_datetime(gains[C.DATETIME], utc=True)))

    _, as_read = _run([obj], {**data, O.WEATHER: weather, "gains": gains})
    _, by_time = _run([obj], {**data, O.WEATHER: weather, "gains": indexed})

    first, second = _first_ts(as_read, obj, Types.HVAC), _first_ts(by_time, obj, Types.HVAC)
    assert not first[C.TEMP_IN].isna().any()
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("example", ["hvac_1r0c", "hvac_1r1c"])
def test_shuffled_gains_align_by_instant(example):
    obj, data = _example(example)
    weather = _weather(0, 48)
    gains = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "residential": np.linspace(100, 600, len(weather))})
    obj = {**obj, O.GAINS_INTERNAL: "gains", O.GAINS_INTERNAL_COL: "residential"}

    _, ordered = _run([obj], {**data, O.WEATHER: weather, "gains": gains})
    _, shuffled = _run([obj], {**data, O.WEATHER: weather, "gains": gains.sample(frac=1.0, random_state=3)})

    pd.testing.assert_frame_equal(_first_ts(ordered, obj, Types.HVAC), _first_ts(shuffled, obj, Types.HVAC))


@pytest.mark.parametrize("example", ["hvac_1r0c", "hvac_1r1c", "hvac_5r1c", "hvac_7r2c"])
def test_gains_that_do_not_cover_the_weather_raise(example):
    obj, data = _example(example)
    weather = _weather(0, 48)
    later = pd.to_datetime(weather[C.DATETIME], utc=True) + pd.DateOffset(years=1)
    gains = pd.DataFrame({C.DATETIME: later, "residential": 300.0})
    obj = {**obj, O.GAINS_INTERNAL: "gains", O.GAINS_INTERNAL_COL: "residential"}

    with pytest.raises(ValueError, match="'gains'.*does not cover"):
        _run([obj], {**data, O.WEATHER: weather, "gains": gains})


# -- solar geometry ---------------------------------------------------------------------------------------------------


def _record_solar_positions():
    import pvlib

    seen = []
    original = pvlib.location.Location.get_solarposition

    def spy(self, times, *args, **kwargs):
        seen.append(pd.DatetimeIndex(times).tz_convert("UTC"))
        return original(self, times, *args, **kwargs)

    return patch.object(pvlib.location.Location, "get_solarposition", spy), seen


@pytest.mark.parametrize("example", ["pv_pvlib", "hvac_1r1c"])
def test_solar_position_is_taken_at_the_interval_midpoint(example):
    pytest.importorskip("pvlib")
    obj, data = _example(example)
    weather = _weather(4000, 24)
    from entise.methods.auxiliary.solar.strategies import clear_solar_cache

    clear_solar_cache()
    spy, seen = _record_solar_positions()
    with spy:
        _run([obj], {**data, O.WEATHER: weather})

    midpoints = (_instants(weather) + pd.Timedelta(1800, unit="s")).as_unit("us")
    assert seen
    assert all(times.as_unit("us").equals(midpoints) for times in seen)


# -- weather key ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("example, ts_type", [("pv_pvlib", Types.PV), ("wind_wplib", Types.WIND)])
def test_pv_and_wind_use_the_objects_weather(example, ts_type):
    obj, data = _example(example)
    weather = _weather(4000, 24)
    obj = {**obj, O.WEATHER: "site_b"}

    _, timeseries = _run([obj], {**data, "site_b": weather})

    assert len(_first_ts(timeseries, obj, ts_type)) == len(weather)


# -- output index -----------------------------------------------------------------------------------------------------

ALL_EXAMPLES = [
    ("hvac_1r0c", Types.HVAC),
    ("hvac_1r1c", Types.HVAC),
    ("hvac_5r1c", Types.HVAC),
    ("hvac_7r2c", Types.HVAC),
    ("hp_ruhnau", Types.HP),
    ("pv_pvlib", Types.PV),
    ("wind_wplib", Types.WIND),
]


@pytest.mark.parametrize("example, ts_type", ALL_EXAMPLES)
@pytest.mark.parametrize("utc", [False, True], ids=["site zone", "utc"])
def test_outputs_carry_the_weather_instants_in_the_site_zone_or_utc(example, ts_type, utc):
    obj, data = _example(example)
    # 2022-03-27 crosses the change to summer time, so the input offsets are mixed.
    weather = _weather(2030, 24)
    obj = {**obj, O.TIMEZONE: "Europe/Berlin"}

    _, timeseries = _run([obj], {**data, O.WEATHER: weather}, utc=utc)

    index = _first_ts(timeseries, obj, ts_type).index
    assert isinstance(index, pd.DatetimeIndex)
    assert str(index.tz) == ("UTC" if utc else "Europe/Berlin")
    assert index.tz_convert("UTC").as_unit("us").equals(_instants(weather))


# -- Ruhnau -----------------------------------------------------------------------------------------------------------


def test_ruhnau_soil_proxy_averages_24_hours_at_any_step():
    from entise.methods.hp.ruhnau import Ruhnau

    index = pd.date_range("2022-01-01", periods=4 * 72, freq="15min", tz="UTC")
    step_at = index[4 * 24]
    air = pd.Series(np.where(index < step_at, 0.0, 10.0), index=index)
    weather = pd.DataFrame({C.TEMP_AIR: air}, index=index)

    soil = Ruhnau._add_missing_temperature_columns(weather.copy())[C.TEMP_SOIL]

    settled = soil.iloc[-1]
    assert soil[step_at + pd.Timedelta(12 * 3600, unit="s")] < settled
    assert soil[step_at + pd.Timedelta(24 * 3600, unit="s")] == pytest.approx(settled)


# -- irradiance labels ------------------------------------------------------------------------------------------------


def test_the_shipped_weather_is_labelled_at_the_interval_start():
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"))
    table, _ = _time.normalize_table(weather, table="weather")

    offset = _time.check_irradiance_labels(table, 11.06, table="weather")

    assert abs(offset) < 0.25


def test_end_labelled_irradiance_is_reported_once_per_run(caplog):
    obj, data = _example("pv_pvlib")
    weather = _weather(4000, 24 * 15)
    radiation = [C.SOLAR_GHI, C.SOLAR_DNI, C.SOLAR_DHI]
    end_labelled = weather.assign(**{c: weather[c].shift(1).fillna(0.0) for c in radiation})
    objects = [obj, {**obj, O.ID: "second"}]

    with caplog.at_level(logging.WARNING, logger=_time.__name__):
        _run(objects, {**data, O.WEATHER: end_labelled})

    warnings = [r for r in caplog.records if "Irradiance in table 'weather'" in r.getMessage()]
    assert len(warnings) == 1
