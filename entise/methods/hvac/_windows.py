"""Resolve the windows table an HVAC object names."""

from typing import Optional

import pandas as pd

from entise.constants import Objects as O
from entise.core.base import Method


def windows_table(obj: dict, data: dict, method_type: str) -> Optional[pd.DataFrame]:
    """Return the windows table the object names, or None if it names none.

    Raises:
        ValueError: If the object names a table that is not in ``data``, or if its windows would
            get no solar gains: the object has no ``latitude[degree]`` and ``longitude[degree]``,
            or the table has no row with its ``id``. Without these checks the solar gains would
            silently be zero. Objects whose solar gains are switched off
            (``active_gains_solar`` False) or supplied (``gains_solar[W]``) are not checked.
    """
    key = Method.get_with_method_backup(obj, O.WINDOWS, method_type)
    if not isinstance(key, str) or not key.strip():
        return None
    table = Method.get_with_backup(data, key)
    if table is None:
        raise ValueError(f"Object '{obj.get(O.ID)}' names the windows table '{key}', which is not in the data.")
    _check_windows_get_solar_gains(obj, table, key, method_type)
    return table


def _check_windows_get_solar_gains(obj: dict, table: pd.DataFrame, key: str, method_type: str) -> None:
    """Raise if the windows an object names would get no calculated solar gains."""
    if not bool(Method.get_with_method_backup(obj, O.ACTIVE_GAINS_SOLAR, method_type, True)):
        return
    if Method.get_with_method_backup(obj, O.GAINS_SOLAR, method_type) is not None:
        return

    object_id = Method.get_with_backup(obj, O.ID)
    missing = [name for name in (O.LAT, O.LON) if Method.get_with_method_backup(obj, name, method_type) is None]
    if missing:
        raise ValueError(
            f"Object '{object_id}' names the windows table '{key}' but has no {' or '.join(missing)}, "
            f"so its windows would get no solar gains. Add the location, remove '{O.WINDOWS}', "
            f"or set '{O.ACTIVE_GAINS_SOLAR}' to False."
        )
    if O.ID not in table.columns:
        raise ValueError(f"The windows table '{key}' has no '{O.ID}' column to match its rows to objects.")
    if not (table[O.ID] == object_id).any():
        ids = [value.item() if hasattr(value, "item") else value for value in pd.unique(table[O.ID])[:3]]
        examples = ", ".join(f"{value!r} ({type(value).__name__})" for value in ids)
        raise ValueError(
            f"Object {object_id!r} ({type(object_id).__name__}) has no rows in the windows table '{key}', "
            f"so its windows would get no solar gains. The table's ids include {examples}."
        )
