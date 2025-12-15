import numpy as np
import pandas as pd


def resolve_table_and_column(obj: dict, key: str):
    """Resolve table and column from the given object.
    Either in the new format 'table|column' or the old format with separate keys."""
    val = obj.get(key)

    # New pattern: "table|column"
    if isinstance(val, str) and "|" in val:
        table_key, col = val.split("|", 1)
        return table_key.strip(), col.strip()

    # Legacy: key="table", key+"_COL"="column"
    table_key = val if isinstance(val, str) else None
    col = obj.get(f"{key}_COL", None)

    if table_key is None or col is None:
        raise KeyError(
            f"Could not resolve table and column for key '{key}'. "
            f"Either use 'table|column' or provide both '{key}' and '{key}_COL'."
        )

    return table_key, col


def resolve_ts_or_scalar(obj: dict, data: dict, key: str, index, default=None) -> pd.Series:
    """
    Resolve:
      - scalar → constant Series
      - "table|column"
      - table + separate '{key}_COL'
    Always aligned to index.
    """
    if key not in obj or obj[key] is None:
        if default is not None:
            return pd.Series(default, index=index)
        raise KeyError(f"Key '{key}' not found in object and no default provided.")

    val = obj[key]

    # Scalar handling (Python + NumPy types + bool)
    if np.isscalar(val):
        return pd.Series(val, index=index)

    # Table/column reference
    if isinstance(val, str):
        table_key, col = resolve_table_and_column(obj, key)

        df = data.get(table_key)
        if df is None:
            raise KeyError(f"Data table '{table_key}' not found for key '{key}'.")

        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in table '{table_key}'.")

        return df[col].reindex(index)

    raise TypeError(f"Unsupported type for key '{key}': {type(val)}.")
