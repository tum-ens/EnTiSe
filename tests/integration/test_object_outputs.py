"""All time series of one object share one index, and their energy matches the summary."""

import logging
import os

import numpy as np
import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.core.generator import Generator

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
BERLIN = "Europe/Berlin"


def _weather() -> pd.DataFrame:
    # Two days around the change to summer time.
    return pd.read_csv(os.path.join(EXAMPLES, "common_data", "weather.csv")).iloc[2030:2078].reset_index(drop=True)


def _data() -> dict:
    weather = _weather()
    trips = pd.DataFrame({"datetime": weather["datetime"], "mobility:load[W]": np.arange(len(weather), dtype=float)})
    windows = pd.read_csv(os.path.join(EXAMPLES, "hvac_1r1c", "data", "windows.csv"))
    return {"weather": weather, "windows": windows, "trips": trips}


HOUSE = {
    O.ID: "SFH_low",
    O.TIMEZONE: BERLIN,
    O.WEATHER: "weather",
    O.DATETIMES: "weather",
    "pv": "pvlib",
    "latitude[degree]": 49.72,
    "longitude[degree]": 11.06,
    "pv:power[W]": 7100,
    "azimuth[degree]": 180,
    "tilt[degree]": 30,
    "electricity": "demandlib_electricity",
    "electricity:demand[kWh]": 3002,
    "hvac": "1R1C",
    "resistance[K W-1]": 0.005952381,
    "capacitance[J K-1]": 11200000,
    "ventilation[W K-1]": 61,
    "init_temperature[C]": 20,
    "min_temperature[C]": 20,
    "max_temperature[C]": 25,
    "area[m2]": 140,
    "height[m]": 2.6,
    "windows": "windows",
    "gains_internal[W]": 150,
    "dhw": "JordanVajen",
    "dwelling_size[m2]": 100,
    "seed": 1,
    "hp": "ruhnau",
    "hp_source": "air",
    "hp_sink": "radiator",
    "wind": "wplib",
    "wind:power[W]": 5000000,
    "turbine_type": "AD116/5000",
    "hub_height[m]": 102,
    "mobility": "file",
    "filename": "trips",
}
HEAT = {
    O.ID: "MFH",
    O.TIMEZONE: BERLIN,
    O.WEATHER: "weather",
    "heating": "demandlib_heat",
    "demand[kWh]": 5578,
    "building_type": "MFH",
    "building_class": 1,
    "wind_class": 0,
}


def _run(objects, *, utc=False):
    generator = Generator(raise_on_error=True)
    generator.add_objects(objects)
    return generator.generate(_data(), workers=1, show_progress=False, utc=utc)


@pytest.mark.parametrize("utc", [False, True], ids=["site zone", "utc"])
def test_every_time_series_of_an_object_shares_one_index(utc):
    _, timeseries = _run([HOUSE, HEAT], utc=utc)
    expected = pd.DatetimeIndex(pd.to_datetime(_weather()["datetime"], utc=True)).as_unit("us")
    zone = "UTC" if utc else BERLIN

    for obj_id, frames in timeseries.items():
        assert len(frames) > 1 or obj_id == "MFH"
        for ts_type, frame in frames.items():
            assert str(frame.index.tz) == zone, (obj_id, ts_type)
            assert frame.index.tz_convert("UTC").equals(expected), (obj_id, ts_type)


# (object, type, power or energy column, summary energy key)
ENERGY = [
    ("SFH_low", "pv", "pv:power[W]", "pv:generation[Wh]"),
    ("SFH_low", "wind", "wind:power[W]", "wind:generation[Wh]"),
    ("SFH_low", "electricity", "electricity:load[W]", "electricity:demand[Wh]"),
    ("SFH_low", "hvac", "heating:load[W]", "heating:demand[Wh]"),
    ("SFH_low", "hvac", "cooling:load[W]", "cooling:demand[Wh]"),
    ("SFH_low", "dhw", "dhw:energy[Wh]", "dhw:energy_total[Wh]"),
    ("SFH_low", "dhw", "dhw:power[W]", "dhw:energy_total[Wh]"),
    ("MFH", "heating", "heating:load[W]", "heating:demand[Wh]"),
]


@pytest.mark.parametrize("obj_id, ts_type, column, key", ENERGY, ids=[f"{t}:{c}" for _, t, c, _ in ENERGY])
def test_the_summary_energy_is_the_sum_over_the_series(obj_id, ts_type, column, key):
    summary, timeseries = _run([HOUSE, HEAT])
    series = timeseries[obj_id][ts_type][column]
    # The example weather is hourly, so a step is one hour for W and Wh columns alike.
    reported = float(np.asarray(summary.loc[obj_id, key]).ravel()[0])

    # Rounding to whole watts or watt-hours per row bounds the difference.
    assert abs(float(series.sum()) - reported) <= len(series)


def test_a_seed_stays_an_integer_when_other_objects_have_no_seed():
    alone, _ = _run([HOUSE])
    mixed, _ = _run([HOUSE, HEAT])

    assert alone.loc["SFH_low", "dhw:volume_total[l]"] == mixed.loc["SFH_low", "dhw:volume_total[l]"]


def test_a_finer_internal_gains_table_is_averaged_to_the_weather_step(caplog):
    weather = _weather().iloc[:6]
    instants = pd.to_datetime(weather["datetime"], utc=True)
    quarter_hours = pd.date_range(instants.iloc[0], periods=24, freq="15min")
    values = np.arange(24, dtype=float) * 10.0
    fine = pd.DataFrame({"datetime": quarter_hours, "residential": values})
    hourly = pd.DataFrame({"datetime": instants.to_numpy(), "residential": values.reshape(6, 4).mean(axis=1)})
    house = {
        O.ID: "SFH_low",
        O.TIMEZONE: BERLIN,
        O.WEATHER: "weather",
        "hvac": "1R1C",
        "resistance[K W-1]": 0.005952381,
        "capacitance[J K-1]": 11200000,
        "ventilation[W K-1]": 61,
        "gains_internal[W]": "gains",
        "gains_internal_column": "residential",
    }

    def run(gains):
        generator = Generator(raise_on_error=True)
        generator.add_objects([house])
        return generator.generate({"weather": weather, "gains": gains}, workers=1, show_progress=False)

    with caplog.at_level(logging.WARNING, logger="entise.core._time"):
        fine_summary, fine_series = run(fine)
    hourly_summary, hourly_series = run(hourly)

    pd.testing.assert_frame_equal(fine_series["SFH_low"]["hvac"], hourly_series["SFH_low"]["hvac"])
    pd.testing.assert_frame_equal(fine_summary, hourly_summary)
    assert any("'gains'" in r.getMessage() and "averag" in r.getMessage() for r in caplog.records)


def test_summary_keys_written_by_two_types_are_kept_for_both(caplog):
    hvac_only = {
        key: value
        for key, value in HOUSE.items()
        if key.split(":")[0] not in {"pv", "electricity", "dhw", "hp", "wind", "mobility"}
    }
    heat_keys = {"heating": "demandlib_heat", "heating:demand[kWh]": 5578, "building_type": "EFH"}
    heat_keys.update({"building_class": 1, "wind_class": 0})
    both = {**hvac_only, **heat_keys, O.ID: "both"}
    heat_only = {key: value for key, value in both.items() if key != "hvac"} | {O.ID: "heat_only"}

    with caplog.at_level(logging.WARNING, logger="entise.core.generator"):
        summary, _ = _run([both, {**hvac_only, O.ID: "hvac_only"}, heat_only])

    combined = summary.loc["both"]
    assert combined["hvac:heating:demand[Wh]"] == summary.loc["hvac_only", "heating:demand[Wh]"]
    assert combined["heating:heating:demand[Wh]"] == summary.loc["heat_only", "heating:demand[Wh]"]
    assert pd.isna(combined["heating:demand[Wh]"])
    # Keys only one type writes keep their names.
    assert combined["cooling:demand[Wh]"] == summary.loc["hvac_only", "cooling:demand[Wh]"]
    assert "hvac:cooling:demand[Wh]" not in summary.columns
    assert any("'both'" in r.getMessage() and "heating:demand[Wh]" in r.getMessage() for r in caplog.records)
