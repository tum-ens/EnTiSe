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
      - pd.Series → validated against target index
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

    # Raw pd.Series — must be aligned to the target index.
    if isinstance(val, pd.Series):
        if not val.index.equals(index):
            raise ValueError(f"Series for '{key}' does not match the target index.")
        return val

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


def align_auxiliary_series(series, index, key: str, method: str) -> pd.Series:
    """Align an auxiliary strategy's output onto a model's own index.

    Auxiliary strategies (``VentilationTimeSeries``, ``InternalTimeSeries``,
    ...) carry through whatever index the user's table had. Multiplying such a
    series against one built on the model index makes pandas align on the
    *union* of the two, which silently yields a longer, NaN-padded frame
    instead of an error — see issue #106. Solvers then read the first ``n``
    positions and never notice, while the latent-cooling post-pass broadcasts
    the over-long array against the ``n``-length weather arrays and fails three
    frames away from the cause.

    Args:
        series: Auxiliary output. A one-column DataFrame is squeezed first.
        index: The model's target index (the weather index).
        key: Object key the series came from, for the error message.
        method: Calling method name, for the error message.

    Returns:
        A Series carrying `index`, aligned by label when the auxiliary produced
        a DatetimeIndex and positionally otherwise (which is how R1C1 consumes
        the same auxiliaries).

    Raises:
        ValueError: If the series cannot be aligned unambiguously — a length
            mismatch on the positional path, or timestamps missing from a
            label-indexed table.
    """
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis=1)
    series = pd.Series(series)

    if series.index.equals(index):
        return series

    # Label-based alignment when the auxiliary carried real timestamps: the
    # mapping is unambiguous, so a gap is a genuine input error rather than
    # something to pad with NaN.
    if isinstance(series.index, pd.DatetimeIndex) and isinstance(index, pd.DatetimeIndex):
        aligned = series.reindex(index)
        if aligned.isna().any() and not series.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(
                f"[{method}] '{key}' does not cover the model index: {missing} of {len(index)} "
                f"timestamps are missing from the provided timeseries "
                f"(first: {index[aligned.isna()][0]}). Provide a series covering the full "
                f"weather index."
            )
        return aligned

    # Positional alignment for index-less tables (e.g. a CSV read with the
    # default RangeIndex), matching how R1C1 consumes the same auxiliaries.
    if len(series) != len(index):
        raise ValueError(
            f"[{method}] '{key}' has {len(series)} steps but the model index has {len(index)}. "
            f"Provide a series of matching length, or one indexed by the weather timestamps "
            f"so it can be aligned by label."
        )
    return pd.Series(series.to_numpy(), index=index, name=series.name)
