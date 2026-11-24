"""Align an auxiliary input table with the weather the model runs on."""

from typing import Optional

import pandas as pd

from entise.constants import Objects as O
from entise.core._time import run_time


def align_to_weather(data: dict, key) -> Optional[pd.DataFrame]:
    """Return table ``key`` aligned with ``data["weather"]``, or ``None`` if there is no such table.

    A table with timestamps is aligned by instant and must cover the weather.
    A table without timestamps must have one row per weather step. Both rules
    come from :meth:`entise.core._time.RunTime.align`. Without a weather frame,
    the table is returned as it is.
    """
    if not isinstance(key, str) or key not in data:
        return None
    weather = data.get(O.WEATHER)
    if weather is None:
        return data[key]
    axis = weather.index
    if not isinstance(axis, pd.DatetimeIndex) or axis.tz is None:
        axis = run_time({O.WEATHER: weather}).table(O.WEATHER).index
    return run_time(data).align(key, axis)
