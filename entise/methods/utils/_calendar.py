"""Read calendar engines on the site's local wall clock.

Standard load profiles (BDEW), the heat model of DistrictHeatingSim and the
Load Profile Generator are defined on a plain local calendar with 24 hours on
every day. EnTiSe runs such an engine on a regular naive grid of local wall
clock time and reads each instant of the run at its legal wall-clock label:

- On the 23-hour day of the change to summer time, the skipped hour is never read.
- On the 25-hour day of the change back, the repeated hour is read twice.

Every row of the run covers ``[t, t + step)``. Its value is the mean of the
engine's profile over the same wall-clock span, so steps that are finer or
coarser than the engine's, or that start between two engine labels, read the
right share of the profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from entise.core._time import regular_step, run_time


def time_table(data: Mapping[str, Any], key: str) -> tuple[pd.DataFrame, pd.Timedelta]:
    """Return the run's normalized table ``key`` and its regular time step."""
    table = run_time(data).table(key)
    return table, regular_step(table.index, table=key)


def wall_clock(axis: pd.DatetimeIndex, zone: ZoneInfo) -> pd.DatetimeIndex:
    """Return the naive local wall-clock label of each instant."""
    return axis.tz_convert(zone).tz_localize(None)


def wall_window(axis: pd.DatetimeIndex, zone: ZoneInfo, step: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the wall-clock span ``[start, end)`` that the run reads."""
    wall = wall_clock(axis, zone)
    return wall.min(), wall.max() + step


def wall_years(axis: pd.DatetimeIndex, zone: ZoneInfo, step: pd.Timedelta) -> list[int]:
    """Return the calendar years that the wall-clock span of the run touches."""
    start, end = wall_window(axis, zone, step)
    last = end - pd.Timedelta(1, unit="us")
    return list(range(start.year, last.year + 1))


def _seconds(index: pd.DatetimeIndex | pd.Timestamp, origin: pd.Timestamp) -> np.ndarray:
    return np.asarray((index - origin) / pd.Timedelta(1, unit="s"), dtype=float)


def read_profile(
    profile: pd.Series, axis: pd.DatetimeIndex, zone: ZoneInfo, step: pd.Timedelta, *, name: str
) -> np.ndarray:
    """Return the mean of ``profile`` over each row's wall-clock span ``[t, t + step)``.

    Args:
        profile: Engine output on a regular naive local grid. The value at a
            label is the mean over the interval that starts there.
        axis: The run's UTC instants.
        zone: The site's time zone.
        step: The run's time step.
        name: The engine's name for error messages.

    Raises:
        ValueError: If the profile is not regular or does not cover the run.
    """
    profile = profile.sort_index()
    grid = pd.DatetimeIndex(profile.index)
    if len(grid) < 2:
        raise ValueError(f"[{name}] The engine returned fewer than two values.")
    resolution = grid[1] - grid[0]
    if not ((grid[1:] - grid[:-1]) == resolution).all():
        raise ValueError(f"[{name}] The engine returned an irregular time grid.")

    origin = grid[0]
    edges = np.append(_seconds(grid, origin), _seconds(grid[-1] + resolution, origin))
    width = resolution / pd.Timedelta(1, unit="s")
    integral = np.concatenate(([0.0], np.cumsum(profile.to_numpy(dtype=float) * width)))

    start = _seconds(wall_clock(axis, zone), origin)
    span = step / pd.Timedelta(1, unit="s")
    end = start + span
    if start.min() < edges[0] or end.max() > edges[-1]:
        raise ValueError(f"[{name}] The engine's profile does not cover the requested time span.")
    return (np.interp(end, edges, integral) - np.interp(start, edges, integral)) / span


def hourly_wall_series(values: pd.Series, zone: ZoneInfo, step: pd.Timedelta, *, whole_days: bool = False) -> pd.Series:
    """Return ``values`` (indexed by UTC instants) on a complete hourly local wall-clock grid.

    The grid covers the run's wall-clock span, extended to whole local days if
    ``whole_days`` is set. Values that fall into the same wall-clock hour are
    averaged, which includes the two instants of the repeated hour. Hours
    without a value, such as the skipped hour, are interpolated, and the edges
    are held.
    """
    wall = wall_clock(pd.DatetimeIndex(values.index), zone)
    hourly = pd.Series(values.to_numpy(dtype=float), index=wall).groupby(wall.floor("h")).mean()
    start, end = wall_window(pd.DatetimeIndex(values.index), zone, step)
    first, last = start.floor("h"), (end - pd.Timedelta(1, unit="us")).floor("h")
    if whole_days:
        first, last = first.normalize(), last.normalize() + pd.Timedelta(23, unit="h")
    grid = pd.date_range(first, last, freq="h")
    return hourly.reindex(grid).interpolate(limit_direction="both")


def scale_to_energy(power_w: np.ndarray, step: pd.Timedelta, energy_wh: float) -> np.ndarray:
    """Scale mean powers so that the run's energy ``Σ W · step`` equals ``energy_wh``."""
    hours = step / pd.Timedelta(1, unit="h")
    total = float(np.sum(power_w) * hours)
    if total <= 0:
        raise ValueError("The profile has no energy to scale.")
    return power_w * (energy_wh / total)
