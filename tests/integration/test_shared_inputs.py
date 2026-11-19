"""Objects in one run share input frames and the process without affecting each other."""

import os
import warnings

import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.generator import Generator

HERE = os.path.dirname(__file__)


def _weather(start: int, rows: int) -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"))
    return weather.iloc[start : start + rows].reset_index(drop=True)


def _hvac_inputs() -> tuple[dict, dict]:
    folder = os.path.join(HERE, "hvac_1r1c")
    obj = pd.read_csv(os.path.join(folder, "objects.csv")).iloc[0].to_dict()
    data = {
        name: pd.read_csv(os.path.join(folder, "data", f"{name}.csv"))
        for name in ("windows", "internal_gains", "ventilation")
    }
    return obj, data


def _run(objects: list[dict], data: dict):
    generator = Generator(raise_on_error=True)
    generator.add_objects(objects)
    return generator.generate(data, workers=1, show_progress=False)


def test_heat_pump_and_wind_leave_the_shared_weather_unchanged():
    weather = _weather(0, 48)
    original = weather.copy()
    heat_pump = {
        O.ID: "hp",
        Types.HP: "ruhnau",
        O.WEATHER: O.WEATHER,
        O.HP_SOURCE: "air",
        O.HP_SINK: "radiator",
    }
    wind = {
        O.ID: "wind",
        Types.WIND: "wplib",
        O.WEATHER: O.WEATHER,
        O.POWER: 5_000_000,
        O.TURBINE_TYPE: "AD116/5000",
        O.HUB_HEIGHT: 102,
    }

    _, timeseries = _run([heat_pump, wind], {O.WEATHER: weather})

    pd.testing.assert_frame_equal(weather, original)
    assert len(timeseries["wind"][Types.WIND]) == len(weather)


def test_solar_gains_accept_weather_in_a_named_time_zone():
    obj, data = _hvac_inputs()
    weather = _weather(0, 48)
    instants = pd.to_datetime(weather[C.DATETIME], utc=True)
    in_utc = weather.assign(**{C.DATETIME: instants})
    in_berlin = weather.assign(**{C.DATETIME: instants.dt.tz_convert("Europe/Berlin")})

    _, utc_run = _run([obj], {**data, O.WEATHER: in_utc})
    _, berlin_run = _run([obj], {**data, O.WEATHER: in_berlin})

    utc_hvac = utc_run[obj[O.ID]][Types.HVAC]
    berlin_hvac = berlin_run[obj[O.ID]][Types.HVAC]
    assert berlin_hvac.index.tz_convert("UTC").equals(utc_hvac.index.tz_convert("UTC"))
    pd.testing.assert_frame_equal(berlin_hvac.reset_index(drop=True), utc_hvac.reset_index(drop=True))


def test_electricity_leaves_the_warning_filters_of_the_process_unchanged():
    electricity = {
        O.ID: "electricity",
        Types.ELECTRICITY: "demandlib_electricity",
        O.DATETIMES: O.WEATHER,
        O.DEMAND_KWH: 3000,
    }
    weather = _weather(2030, 24)

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        before = list(warnings.filters)
        _run([electricity], {O.WEATHER: weather})
        after = list(warnings.filters)

    assert after == before


@pytest.mark.filterwarnings("default")
def test_hvac_after_electricity_parses_weather_across_daylight_saving():
    obj, data = _hvac_inputs()
    electricity = {
        O.ID: "electricity",
        Types.ELECTRICITY: "demandlib_electricity",
        O.DATETIMES: O.WEATHER,
        O.DEMAND_KWH: 3000,
    }
    # 2022-03-27, the change to summer time, mixes +01:00 and +02:00 labels.
    weather = _weather(2030, 24)
    assert weather[C.DATETIME].str[-6:].nunique() == 2

    _, timeseries = _run([electricity, obj], {**data, O.WEATHER: weather})

    assert len(timeseries[obj[O.ID]][Types.HVAC]) == len(weather)
