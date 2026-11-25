"""An object that names an input table gets that table (#143).

The table names here differ from the fixed keys (``pv_inverter`` and the water
temperature keys), so a lookup by the fixed key cannot pass the tests.
"""

import logging
import os
import re

import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.generator import Generator
from entise.methods.dhw.defaults import DEFAULT_TEMP_COLD, DEFAULT_TEMP_HOT
from entise.methods.dhw.jordan_vajen import JordanVajen
from entise.methods.pv.pvlib import PVLib

HERE = os.path.dirname(__file__)
PV_POWER = f"{Types.PV}{SEP}{C.POWER}"
COLD = f"{Types.DHW}{SEP}{O.TEMP_WATER_COLD}"
HOT = f"{Types.DHW}{SEP}{O.TEMP_WATER_HOT}"
SMALL = {"pdc0": 0.3}
LARGE = {"pdc0": 3}


@pytest.fixture(scope="module")
def weather() -> pd.DataFrame:
    """Two summer days of the example weather."""
    frame = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"))
    return frame.iloc[24 * 160 : 24 * 162].reset_index(drop=True)


@pytest.fixture(scope="module")
def instants(weather) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(weather[C.DATETIME], utc=True))


def _run(objects: list[dict], data: dict) -> dict:
    generator = Generator(raise_on_error=True)
    generator.add_objects(objects)
    return generator.generate(data, workers=1, show_progress=False)[1]


def _pv(object_id: str, **keys) -> dict:
    return {
        O.ID: object_id,
        Types.PV: "pvlib",
        O.WEATHER: O.WEATHER,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.ALTITUDE: 520,
        O.AZIMUTH: 180,
        O.TILT: 30,
        O.POWER: 1000,
        O.TIMEZONE: "UTC",
        **keys,
    }


def _dhw(object_id: str, **keys) -> dict:
    return {
        O.ID: object_id,
        Types.DHW: "jordanvajen",
        O.DATETIMES: O.WEATHER,
        O.DWELLING_SIZE: 100,
        O.SEED: 42,
        O.TIMEZONE: "UTC",
        **keys,
    }


# PV inverter


def test_pv_objects_in_one_run_get_the_inverter_they_name(weather):
    data = {O.WEATHER: weather, "inv_small": SMALL, "inv_large": LARGE}
    objects = [_pv("small", **{O.PV_INVERTER: "inv_small"}), _pv("large", **{O.PV_INVERTER: "inv_large"})]

    series = _run(objects, data)

    small = series["small"][Types.PV][PV_POWER]
    large = series["large"][Types.PV][PV_POWER]
    assert small.max() < large.max()
    for object_id, inverter in (("small", SMALL), ("large", LARGE)):
        alone = _run([_pv(object_id)], {O.WEATHER: weather, O.PV_INVERTER: inverter})
        pd.testing.assert_series_equal(series[object_id][Types.PV][PV_POWER], alone[object_id][Types.PV][PV_POWER])


def test_pv_uses_the_fixed_inverter_key_when_the_object_names_none(weather, caplog):
    named = _run([_pv("pv", **{O.PV_INVERTER: "inv_small"})], {O.WEATHER: weather, "inv_small": SMALL})
    default = _run([_pv("pv")], {O.WEATHER: weather})

    with caplog.at_level(logging.WARNING, logger="entise.methods.pv"):
        fixed = _run([_pv("pv")], {O.WEATHER: weather, O.PV_INVERTER: SMALL})

    pd.testing.assert_frame_equal(fixed["pv"][Types.PV], named["pv"][Types.PV])
    assert fixed["pv"][Types.PV][PV_POWER].max() < default["pv"][Types.PV][PV_POWER].max()
    assert not [record for record in caplog.records if record.name.startswith("entise.methods.pv")]


def test_pv_inverter_given_as_a_keyword(weather):
    obj = _pv("pv")
    fixed = PVLib().generate(obj, {O.WEATHER: weather, O.PV_INVERTER: SMALL})
    keyword = PVLib().generate(obj, {O.WEATHER: weather}, pv_inverter=SMALL)

    pd.testing.assert_frame_equal(keyword["timeseries"], fixed["timeseries"])


@pytest.mark.parametrize("key", [O.PV_INVERTER, O.PV_ARRAYS])
def test_pv_raises_for_a_named_table_missing_from_the_data(weather, key):
    message = f"Object 'pv' names the {key} table 'missing', which is not in the data."
    with pytest.raises(ValueError, match=re.escape(message)):
        PVLib().generate(_pv("pv", **{key: "missing"}), {O.WEATHER: weather})


# DHW water temperatures


def test_dhw_objects_in_one_run_get_the_water_temperatures_they_name(weather, instants):
    cold = pd.Series(2.0, index=instants)
    hot = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "temperature[C]": 55.0})
    data = {O.WEATHER: weather, "cold_tbl": cold, "hot_tbl": hot}
    objects = [
        _dhw("cold", **{O.TEMP_WATER_COLD: "cold_tbl"}),
        _dhw("hot", **{O.TEMP_WATER_HOT: "hot_tbl"}),
        _dhw("scalars", **{O.TEMP_WATER_COLD: 5, O.TEMP_WATER_HOT: 45}),
    ]

    series = _run(objects, data)

    assert (series["cold"][Types.DHW][COLD] == 2.0).all()
    assert (series["cold"][Types.DHW][HOT] == DEFAULT_TEMP_HOT).all()
    assert (series["hot"][Types.DHW][COLD] == DEFAULT_TEMP_COLD).all()
    assert (series["hot"][Types.DHW][HOT] == 55.0).all()
    assert (series["scalars"][Types.DHW][COLD] == 5).all()
    assert (series["scalars"][Types.DHW][HOT] == 45).all()


def test_dhw_reads_a_named_column_of_a_water_temperature_table(weather):
    table = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "mains[C]": 7.0, "other[C]": 99.0})
    obj = _dhw("d", **{O.TEMP_WATER_COLD: "water|mains[C]"})

    result = JordanVajen().generate(obj, {O.WEATHER: weather, "water": table})

    assert (result["timeseries"][COLD] == 7.0).all()


def test_dhw_water_temperature_given_as_a_keyword_table(weather):
    table = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "temperature[C]": 3.0})

    result = JordanVajen().generate(_dhw("d"), {O.WEATHER: weather}, temp_water_cold=table)

    assert (result["timeseries"][COLD] == 3.0).all()


def test_dhw_energy_follows_the_named_water_temperature(weather, instants):
    data = {O.WEATHER: weather, "cold_tbl": pd.Series(2.0, index=instants)}
    named = JordanVajen().generate(_dhw("d", **{O.TEMP_WATER_COLD: "cold_tbl"}), data)
    scalar = JordanVajen().generate(_dhw("d", **{O.TEMP_WATER_COLD: 2.0}), data)

    pd.testing.assert_frame_equal(named["timeseries"], scalar["timeseries"])


def test_dhw_raises_for_a_water_temperature_series_off_the_run(weather, instants):
    shifted = pd.Series(2.0, index=instants + pd.Timedelta(hours=1))
    obj = _dhw("d", **{O.TEMP_WATER_COLD: "cold_tbl"})

    with pytest.raises(ValueError, match="does not match"):
        JordanVajen().generate(obj, {O.WEATHER: weather, "cold_tbl": shifted})


@pytest.mark.parametrize("key", [O.TEMP_WATER_COLD, O.TEMP_WATER_HOT])
def test_dhw_raises_for_a_named_table_missing_from_the_data(weather, key):
    message = f"Object 'd' names the {key} table 'missing', which is not in the data."
    with pytest.raises(ValueError, match=re.escape(message)):
        JordanVajen().generate(_dhw("d", **{key: "missing"}), {O.WEATHER: weather})


def test_dhw_ignores_an_unnamed_table_under_the_water_temperature_key(weather, instants):
    data = {O.WEATHER: weather, O.TEMP_WATER_COLD: pd.Series(2.0, index=instants)}

    result = JordanVajen().generate(_dhw("d"), data)

    assert (result["timeseries"][COLD] == DEFAULT_TEMP_COLD).all()


def test_dhw_reads_a_water_temperature_number_given_as_text(weather):
    result = JordanVajen().generate(_dhw("d", **{O.TEMP_WATER_COLD: "4"}), {O.WEATHER: weather})

    assert (result["timeseries"][COLD] == 4.0).all()
