"""Weather for the RC models, read through the run's time layer."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import RunTime, regular_step, run_time
from entise.core.base import Method


def _strip_heights(table: pd.DataFrame) -> pd.DataFrame:
    return Method._strip_weather_height(table)


def hvac_weather(obj: dict, data: dict, method_type: str) -> tuple[RunTime, str, pd.DataFrame]:
    """Return the run, the weather key and the object's weather.

    The weather is indexed by UTC instants, one row per interval that starts at
    its label, with column heights stripped. It is prepared once per run and
    shared between objects, so it must not be changed in place.

    Raises:
        ValueError: If the weather is missing or has an irregular time step.
    """
    run = run_time(data)
    key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
    if key not in run:
        raise ValueError(f"Weather table '{key}' is not in the data.")
    weather = run.derived(key, "hvac", _strip_heights)
    regular_step(weather.index, table=key)
    return run, key, weather


def step_seconds(weather: pd.DataFrame) -> float:
    """Return the regular time step of the weather in seconds."""
    return float(regular_step(weather.index, table=O.WEATHER).total_seconds())


def check_window_irradiance(
    run: RunTime, key: str, weather: pd.DataFrame, obj: dict, windows: Optional[pd.DataFrame], method_type: str
) -> None:
    """Warn once per run if the irradiance used for window gains looks end-labelled."""
    longitude = Method.get_with_method_backup(obj, O.LON, method_type)
    if windows is None or longitude is None or C.SOLAR_GHI not in weather.columns:
        return
    run.check_irradiance(key, float(longitude))
