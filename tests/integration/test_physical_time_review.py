"""Follow-up checks on the physical methods' time handling."""

import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core import _time
from entise.methods.hp.ruhnau import Ruhnau
from entise.methods.hvac.R1C1 import R1C1
from entise.methods.pv.pvlib import PVLib
from entise.methods.wind.wplib import WPLib

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, os.pardir, os.pardir, "examples")
HEAT = f"{Types.HEATING}{SEP}{C.LOAD}[W]"


def _weather(first: int = 0, rows: int = 24) -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"))
    return weather.iloc[first : first + rows].reset_index(drop=True)


# -- table|column references ------------------------------------------------------------------------------------------


def _house(power_heating) -> dict:
    return {
        O.ID: "house",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 30.0,
        O.POWER_HEATING: power_heating,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: O.WEATHER,
    }


def _caps(weather: pd.DataFrame) -> pd.DataFrame:
    # The heater is off for the first half and unlimited for the second; rows are shuffled.
    half = len(weather) // 2
    caps = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "ph": [0.0] * half + [1e9] * (len(weather) - half)})
    return caps.sample(frac=1.0, random_state=7)


@pytest.mark.parametrize("form", ["table|column", "key_COL"])
def test_a_power_cap_from_a_table_aligns_by_instant(form):
    weather = _weather(0, 24).assign(**{"air_temperature[C]@2m": -10.0})
    caps = _caps(weather)
    obj = _house("caps|ph") if form == "table|column" else {**_house("caps"), f"{O.POWER_HEATING}_COL": "ph"}

    heat = R1C1().generate(obj, {O.WEATHER: weather, "caps": caps}, Types.HVAC)["timeseries"][HEAT].to_numpy()

    assert heat[:12].max() == 0
    assert heat[12:].min() > 0


# -- direct calls read each table once --------------------------------------------------------------------------------


def _example(example: str) -> tuple[dict, dict]:
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    data = {}
    folder = os.path.join(EXAMPLES, example, "data")
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            stem, ext = os.path.splitext(name)
            if ext == ".json":
                with open(os.path.join(folder, name), encoding="utf-8") as handle:
                    data[stem] = json.load(handle)
    return obj, data


@pytest.mark.parametrize(
    "example, method, ts_type",
    [("hp_ruhnau", Ruhnau, Types.HP), ("pv_pvlib", PVLib, Types.PV), ("wind_wplib", WPLib, Types.WIND)],
)
def test_a_direct_call_reads_naive_weather_once(example, method, ts_type, caplog):
    obj, data = _example(example)
    weather = _weather(4000, 24)
    weather[C.DATETIME] = weather[C.DATETIME].str.slice(0, 19)  # naive local text

    with caplog.at_level(logging.WARNING, logger=_time.__name__):
        method().generate(obj, {**data, O.WEATHER: weather}, ts_type=ts_type)

    naive = [r for r in caplog.records if "without a UTC offset" in r.getMessage()]
    assert len(naive) == 1


# -- warnings ---------------------------------------------------------------------------------------------------------


def test_pv_emits_no_setting_with_copy_warning():
    warning = getattr(pd.errors, "SettingWithCopyWarning", None)
    if warning is None:
        pytest.skip("pandas without SettingWithCopyWarning")
    obj, data = _example("pv_pvlib")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PVLib().generate(obj, {**data, O.WEATHER: _weather(4000, 24)}, ts_type=Types.PV)

    assert not [w for w in caught if issubclass(w.category, warning)]


def test_numeric_text_is_still_a_value():
    weather = _weather(0, 24).assign(**{"air_temperature[C]@2m": -10.0})

    heat = R1C1().generate(_house("2000"), {O.WEATHER: weather}, Types.HVAC)["timeseries"][HEAT]

    assert np.isclose(heat.max(), 2000)
