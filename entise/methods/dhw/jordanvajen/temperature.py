"""
Water temperature handling functions for the Jordan & Vajen DHW method.

This module contains functions for handling water temperature data
used by the Jordan & Vajen DHW method.
"""

import logging
from typing import Any, Dict

import pandas as pd

from entise.constants import Objects as O
from entise.core.base import Method
from entise.core.utils import resolve_ts_or_scalar_by_id

logger = logging.getLogger(__name__)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def water_temperature(obj: Dict[str, Any], data: Dict[str, Any], key: str, axis: pd.DatetimeIndex) -> pd.Series:
    """Return the water temperature that the object gives for ``key``, on the run's instants.

    The object gives a number (also as text), a Series on the run's instants, or the
    name of a table in ``data``:

    - a Series stored under the name;
    - ``"table|column"``;
    - ``"table"``, which reads the column named by the object's ``id``, or the table's only column.

    Tables follow the time contract: a table with timestamps is aligned by instant,
    and a table without them is used row by row.

    Args:
        obj: The object's resolved keys; ``key`` has its default if the object gives none.
        data: The input tables of the run.
        key: ``O.TEMP_WATER_COLD`` or ``O.TEMP_WATER_HOT``.
        axis: The run's instants.

    Raises:
        ValueError: If the object names a table that is not in ``data``, or a Series
            does not match the run's instants.
    """
    value = obj.get(key)
    if isinstance(value, str) and not _is_number(value):
        name = value.split("|", 1)[0].strip()
        table = Method.named_table({**obj, key: name}, data, key)
        if isinstance(table, pd.Series):
            obj = {**obj, key: table}
    return resolve_ts_or_scalar_by_id(obj, data, key, axis)


def _get_water_temperatures(cold: pd.Series, hot: pd.Series, index: pd.Index) -> pd.DataFrame:
    """Return the cold and hot water temperatures as one table on ``index``.

    Args:
        cold: Cold water temperature on the run's instants, in °C.
        hot: Hot water temperature on the run's instants, in °C.
        index: The index of the output, in the same order as the run's instants.

    Returns:
        pd.DataFrame: Columns ``O.TEMP_WATER_COLD`` and ``O.TEMP_WATER_HOT``.
    """
    df = pd.DataFrame(
        {
            O.TEMP_WATER_COLD: pd.Series(cold.to_numpy(), index=index, dtype=cold.dtype),
            O.TEMP_WATER_HOT: pd.Series(hot.to_numpy(), index=index, dtype=hot.dtype),
        },
        index=index,
    )

    if (df[O.TEMP_WATER_HOT] <= df[O.TEMP_WATER_COLD]).any():
        logger.warning("Hot water temperature is not greater than cold water temperature for some timestamps")

    logger.info(
        f"Water temperatures: cold={df[O.TEMP_WATER_COLD].mean():.1f}°C, hot={df[O.TEMP_WATER_HOT].mean():.1f}°C"
    )

    return df
