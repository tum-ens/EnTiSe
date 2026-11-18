"""Resolve the windows table an HVAC object names."""

from typing import Optional

import pandas as pd

from entise.constants import Objects as O
from entise.core.base import Method


def windows_table(obj: dict, data: dict, method_type: str) -> Optional[pd.DataFrame]:
    """Return the windows table the object names, or None if it names none.

    Raises:
        ValueError: If the object names a table that is not in ``data``. Without
            this check the solar gains would silently be zero.
    """
    key = Method.get_with_method_backup(obj, O.WINDOWS, method_type)
    if not isinstance(key, str) or not key.strip():
        return None
    table = Method.get_with_backup(data, key)
    if table is None:
        raise ValueError(f"Object '{obj.get(O.ID)}' names the windows table '{key}', which is not in the data.")
    return table
