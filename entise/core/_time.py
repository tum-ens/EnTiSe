"""Read time tables as UTC instants, once per generation run.

Every table that carries timestamps, in a ``datetime`` column or in its index,
is normalized to a tz-aware UTC ``DatetimeIndex`` in ``datetime64[us]``:

- Typed datetime values, and ISO 8601 strings with a UTC offset or ``Z``,
  keep their instants. Other text formats and numbers are rejected.
- Naive timestamps are read as UTC. The run logs one warning per table that
  shows how to localize them.
- Labels must be present and unique; rows are sorted by instant. Errors name
  the table and the timestamp.

Each label marks the start of its interval. Interval quantities such as
irradiance are means over ``[t, t + step)``. Instantaneous quantities such as
air temperature are values at ``t``.

:class:`RunTime` holds the normalized tables of one run. ``Generator.generate``
builds it once and passes it to the methods under :data:`RUN_TIME_KEY`.
:func:`run_time` builds one for direct ``Method.generate`` calls.

This module imports only pandas, numpy and the standard library, so a method
that uses it still imports only its own extra.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from entise.constants import Columns as C
from entise.constants import Objects as O

logger = logging.getLogger(__name__)

RUN_TIME_KEY = "_entise_run_time"
UNIT = "us"
UTC = ZoneInfo("UTC")

# A UTC offset or "Z" that follows a time of day, so the "-01" of a date such as "2022-01-01" is not one.
_OFFSET = re.compile(r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[zZ]|[+-]\d{2}(?::?\d{2})?)$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_MIN_DAYS_FOR_LABEL_CHECK = 14


# -- parsing ----------------------------------------------------------------------------------------------------------


def _is_aware(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_OFFSET.search(value.strip()))
    return getattr(value, "tzinfo", None) is not None


def parse_instants(values: pd.Series | pd.Index, *, table: str) -> tuple[pd.DatetimeIndex, bool]:
    """Return the instants of ``values`` as a UTC index and whether they were naive.

    Raises:
        ValueError: If a timestamp is missing or unreadable, or if naive and
            tz-aware timestamps are mixed. The message names the table.
    """
    series = pd.Series(np.asarray(values, dtype=object) if isinstance(values, pd.Index) else values)
    series = series.reset_index(drop=True)
    if series.empty:
        raise ValueError(f"Table '{table}' has no timestamps.")
    missing = series.isna().to_numpy()
    if missing.any():
        raise ValueError(f"Table '{table}' has a missing timestamp at row {int(np.argmax(missing))}.")

    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(
            f"Table '{table}' has numbers as timestamps. Convert them first, for example "
            "`pd.to_datetime(values, unit='s', utc=True)` for seconds since 1970."
        )
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        axis, naive = pd.DatetimeIndex(series).tz_convert(UTC), False
    elif pd.api.types.is_datetime64_dtype(series.dtype):
        axis, naive = pd.DatetimeIndex(series).tz_localize(UTC), True
    else:
        aware = series.map(_is_aware).to_numpy(dtype=bool)
        if aware.all() != aware.any():
            first_naive = series[~aware].iloc[0]
            raise ValueError(
                f"Table '{table}' mixes timestamps with and without a UTC offset, "
                f"for example '{first_naive}' at row {int(np.argmax(~aware))}."
            )
        naive = not aware.any()
        try:
            axis = pd.DatetimeIndex(pd.to_datetime(series, utc=not naive, format="ISO8601"))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                f"Table '{table}' has timestamps that are not ISO 8601, for example '{series.iloc[0]}'. "
                "Write them as '2022-07-01 12:00:00+02:00', or pass typed datetime values."
            ) from exc
        if axis.tz is None:
            axis = axis.tz_localize(UTC)
        else:
            axis = axis.tz_convert(UTC)

    return pd.DatetimeIndex(axis, name=C.DATETIME).as_unit(UNIT), naive


def _index_has_time(index: pd.Index) -> bool:
    if isinstance(index, pd.DatetimeIndex) or index.name == C.DATETIME:
        return True
    if isinstance(index, pd.RangeIndex) or index.empty:
        return False
    first = index[0]
    return isinstance(first, date) or (isinstance(first, str) and bool(_ISO_DATE.match(first)))


def _check_unique(axis: pd.DatetimeIndex, *, table: str, naive: bool) -> None:
    duplicated = axis.duplicated()
    if duplicated.any():
        stamp = axis[duplicated][0]
        hint = (
            " If the table holds local times, localize them before passing it, for example "
            "`df['datetime'] = df['datetime'].dt.tz_localize('Europe/Berlin', ambiguous=\"infer\")`."
            if naive
            else ""
        )
        raise ValueError(f"Table '{table}' repeats the timestamp {stamp:%Y-%m-%d %H:%M:%S%z}.{hint}")


def normalize_table(frame: pd.DataFrame, *, table: str) -> tuple[pd.DataFrame | None, bool]:
    """Return a copy of ``frame`` indexed by its UTC instants, and whether they were naive.

    Returns ``(None, False)`` for a table without timestamps. The ``datetime``
    column is dropped from the copy, rows are sorted by instant, and ``frame``
    is not changed.

    Raises:
        ValueError: If a timestamp is missing, unreadable or repeated, or if the
            ``datetime`` column and the index disagree.
    """
    has_column = C.DATETIME in frame.columns
    has_index = _index_has_time(frame.index)
    if not has_column and not has_index:
        return None, False

    if has_column:
        axis, naive = parse_instants(frame[C.DATETIME], table=table)
        if has_index:
            index_axis, _ = parse_instants(frame.index, table=table)
            if not index_axis.equals(axis):
                position = int(np.argmax(index_axis != axis))
                raise ValueError(
                    f"Table '{table}' has a datetime column and index that identify different instants, "
                    f"first at row {position}."
                )
    else:
        axis, naive = parse_instants(frame.index, table=table)

    _check_unique(axis, table=table, naive=naive)
    normalized = frame.drop(columns=C.DATETIME) if has_column else frame.copy()
    normalized.index = axis
    if not axis.is_monotonic_increasing:
        normalized = normalized.sort_index(kind="stable")
    return normalized, naive


def regular_step(axis: pd.DatetimeIndex, *, table: str) -> pd.Timedelta:
    """Return the elapsed step of ``axis``.

    Raises:
        ValueError: If the axis has fewer than two labels or its step changes.
    """
    if len(axis) < 2:
        raise ValueError(f"Table '{table}' needs at least two timestamps to define a time step.")
    steps = axis[1:] - axis[:-1]
    changed = steps != steps[0]
    if changed.any():
        position = int(np.argmax(changed))
        raise ValueError(
            f"Table '{table}' needs a regular time step, but the step changes from {steps[0]} "
            f"to {steps[position]} at {axis[position + 1]:%Y-%m-%d %H:%M:%S%z}."
        )
    return pd.Timedelta(steps[0])


# -- site zone --------------------------------------------------------------------------------------------------------


def site_zone(obj: Mapping[str, Any]) -> ZoneInfo:
    """Return the object's time zone from its ``timezone`` key, or UTC if it has none.

    Raises:
        ValueError: If the key holds something other than an IANA time zone name.
    """
    value = obj.get(O.TIMEZONE)
    if isinstance(value, str):
        name = value.strip()
        if not name:
            return UTC
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            pass
    elif value is None or pd.isna(value):
        return UTC
    raise ValueError(
        f"Object '{obj.get(O.ID)}' has the time zone {value!r}. Use an IANA name such as 'Europe/Berlin' or 'UTC'."
    )


# -- run time ---------------------------------------------------------------------------------------------------------


class RunTime:
    """The time tables of one generation run, each read once.

    Tables are normalized when the run starts. A table that cannot be read
    raises when an object uses it, so one broken table does not stop the
    objects that do not need it.
    """

    def __init__(self, data: Mapping[str, Any], *, utc: bool = False):
        self.utc = utc
        self._frames: dict[str, pd.DataFrame] = {}
        self._tables: dict[str, pd.DataFrame] = {}
        self._errors: dict[str, str] = {}
        self._positional_warned: set[str] = set()
        self._derived: dict[tuple[str, str], Any] = {}
        self._label_checks: set[tuple[str, float]] = set()
        for key, frame in data.items():
            if key == RUN_TIME_KEY or not isinstance(frame, pd.DataFrame):
                continue
            self._frames[key] = frame
            try:
                table, naive = normalize_table(frame, table=key)
            except Exception as exc:  # noqa: BLE001 - reported when an object uses the table
                message = str(exc)
                self._errors[key] = message if f"'{key}'" in message else f"Table '{key}' cannot be read: {message}"
                continue
            if table is None:
                continue
            self._tables[key] = table
            if naive:
                logger.warning(
                    "Table '%s' has timestamps without a UTC offset; EnTiSe reads them as UTC. "
                    "If they are local times, localize them before passing the table, for example "
                    "`df['datetime'] = df['datetime'].dt.tz_localize('Europe/Berlin', ambiguous=\"infer\")`.",
                    key,
                )

    def __contains__(self, key: object) -> bool:
        return key in self._frames

    def derived(self, key: str, name: str, build: Callable[[pd.DataFrame], Any]) -> Any:
        """Return ``build(table(key))``, built once per run and process under ``name``.

        Methods use it for preparation that every object repeats, such as
        stripping column heights. The result is shared between objects, so do
        not change it in place.
        """
        cache_key = (key, name)
        if cache_key not in self._derived:
            self._derived[cache_key] = build(self.table(key))
        return self._derived[cache_key]

    def check_irradiance(self, key: str, longitude: float) -> None:
        """Run :func:`check_irradiance_labels` on table ``key`` once per run and location cell."""
        check = (key, round(float(longitude), 1))
        if check in self._label_checks:
            return
        self._label_checks.add(check)
        check_irradiance_labels(self.table(key), float(longitude), table=key)

    def table(self, key: str) -> pd.DataFrame:
        """Return the normalized table ``key``.

        Each call returns a copy, so a method may change it.

        Raises:
            ValueError: If the table cannot be read, has no timestamps, or is missing.
        """
        if key in self._errors:
            raise ValueError(self._errors[key])
        if key in self._tables:
            return self._tables[key].copy()
        if key in self._frames:
            raise ValueError(f"Table '{key}' has no timestamps.")
        raise ValueError(f"Table '{key}' is not in the data.")

    def align(self, key: str, axis: pd.DatetimeIndex) -> pd.DataFrame:
        """Return table ``key`` on ``axis``.

        A table with timestamps is aligned by instant and must cover every
        instant of ``axis``. A table without timestamps must have one row per
        instant. It is then aligned by position, with one warning per table.

        Raises:
            ValueError: If the table does not cover the axis, or has another length.
        """
        if axis.tz is None:
            raise ValueError(f"The time axis for table '{key}' must be tz-aware.")
        if key in self._tables or key in self._errors:
            table = self.table(key)
            missing = ~axis.isin(table.index)
            if missing.any():
                raise ValueError(
                    f"Table '{key}' does not cover the time axis: {int(missing.sum())} of {len(axis)} instants are "
                    f"missing, the first at {axis[missing][0]:%Y-%m-%d %H:%M:%S%z}."
                )
            return table.reindex(axis)

        frame = self._frames.get(key)
        if frame is None:
            raise ValueError(f"Table '{key}' is not in the data.")
        if len(frame) != len(axis):
            raise ValueError(
                f"Table '{key}' has no timestamps and {len(frame)} rows, but the time axis has {len(axis)}. "
                f"Add a '{C.DATETIME}' column, or give it one row per time step."
            )
        if key not in self._positional_warned:
            self._positional_warned.add(key)
            logger.warning(
                "Table '%s' has no timestamps; EnTiSe uses its rows in order, one per time step. "
                "Add a '%s' column to align it by time.",
                key,
                C.DATETIME,
            )
        aligned = frame.copy()
        aligned.index = axis
        return aligned


def output_index(axis: pd.DatetimeIndex, obj: Mapping[str, Any], data: Mapping[str, Any]) -> pd.DatetimeIndex:
    """Return the index of a method's output: ``axis`` in the site zone, or in UTC if the run asks for it.

    Every method of an object uses this, so all its outputs share one index.
    """
    zone = UTC if run_time(data).utc else site_zone(obj)
    return pd.DatetimeIndex(axis, name=C.DATETIME).tz_convert(zone)


def run_time(data: Mapping[str, Any]) -> RunTime:
    """Return the run's :class:`RunTime`, or build one for a direct method call."""
    current = data.get(RUN_TIME_KEY)
    return current if isinstance(current, RunTime) else RunTime(data)


# -- irradiance labels ------------------------------------------------------------------------------------------------


def _equation_of_time_hours(day_of_year: np.ndarray) -> np.ndarray:
    """Equation of time in hours (NOAA approximation)."""
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1)
    minutes = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    return minutes / 60.0


def check_irradiance_labels(frame: pd.DataFrame, longitude: float, *, table: str) -> float | None:
    """Warn if irradiance looks labelled at the end of its interval.

    For each day, the irradiance-weighted centre of the intervals is compared
    with solar noon. Start labels put it close to noon; end labels, as used by
    Open-Meteo, put it one step later. The check needs a step of at least one
    hour and 14 days with irradiance; otherwise it returns ``None``.

    Args:
        frame: A normalized table with ``global_horizontal_irradiance[W m-2]``.
        longitude: The site's longitude in degrees east.
        table: The table's name for the warning.

    Returns:
        The median offset from solar noon in hours, or ``None`` if not judged.
    """
    if C.SOLAR_GHI not in frame.columns or len(frame) < 2:
        return None
    # Hours since the epoch, as floats, keep the arithmetic independent of the timestamp unit.
    hours = frame.index.as_unit(UNIT).asi8 / 3.6e9
    step_hours = float(hours[1] - hours[0])
    if step_hours < 1.0:
        return None

    ghi = frame[C.SOLAR_GHI].astype(float).clip(lower=0.0).fillna(0.0).to_numpy()
    # Mean solar time of each interval's midpoint, split into day and hour of day.
    solar = hours + step_hours / 2.0 + longitude / 15.0
    day = np.floor(solar / 24.0)
    hour = solar - 24.0 * day
    used = ghi > 0
    if not used.any():
        return None
    totals = (
        pd.DataFrame({"day": day[used], "weighted": hour[used] * ghi[used], "ghi": ghi[used]})
        .groupby("day")[["weighted", "ghi"]]
        .sum()
    )
    if len(totals) < _MIN_DAYS_FOR_LABEL_CHECK:
        return None
    centre = (totals["weighted"] / totals["ghi"]).to_numpy()
    day_of_year = pd.DatetimeIndex(totals.index.to_numpy(dtype="int64").astype("datetime64[D]")).day_of_year.to_numpy()
    noon = 12.0 - _equation_of_time_hours(day_of_year)
    offset = float(np.median(centre - noon))

    if offset > 0.5 * step_hours:
        logger.warning(
            "Irradiance in table '%s' peaks about %.1f h after solar noon. EnTiSe reads each row as the mean over "
            "the interval that starts at its label; these values look labelled at the interval end, as Open-Meteo "
            "labels them. Shift the irradiance columns one row earlier, for example "
            "`df[columns] = df[columns].shift(-1)`.",
            table,
            offset,
        )
    return offset
