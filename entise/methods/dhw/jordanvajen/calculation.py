"""
Core calculation functions for the Jordan & Vajen DHW method.

Water is drawn per local day of the site.

- **Shares and weekday factors.** Each event type has a share of the mean
  daily volume, from the weekly mean of its ``probability_day``. Within the
  type, ``probability_day`` also gives a factor per weekday, normalised so
  that its weekly mean is 1; holidays use Sunday's factor.
- **Daily volume.** A day's volume is a draw from the dwelling's daily demand,
  scaled by the seasonal factor and by Σ share × factor of the day's weekday.
  Weekday totals vary, and the annual event mix stays at the shares.
- **Events.** For each day and event type, the number of events is a Poisson
  draw around the day's expected volume of that type divided by the type's
  mean event volume. Each event's size comes from the type's own
  distribution, so a bath stays a bath.
- **Placement.** Events are placed at the activity times of the day's weekday
  (``probability``). A time is placed on the instant it names in the site's
  time zone: a time that does not exist when clocks go forward moves forward,
  and a time that happens twice when clocks go back takes the first
  occurrence. The row that contains the instant receives the volume.
- **Run total.** One factor over the whole run makes the total equal the sum
  of the daily volumes.
"""

import logging
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from entise._optional import requires_extra
from entise.constants import Columns as C
from entise.methods.dhw.jordanvajen.utils import _convert_time_to_seconds_of_day, _sample_event_volumes

logger = logging.getLogger(__name__)

SUNDAY = 6


def days_in_year(days: pd.DatetimeIndex) -> np.ndarray:
    """Return 366 for days in a leap year and 365 otherwise."""
    return np.where(days.is_leap_year, 366, 365)


def local_holidays(location, years) -> frozenset[date]:
    """Return the holidays of ``location`` ("country" or "subdivision,country") in ``years``."""
    if not isinstance(location, str) or not location.strip():
        return frozenset()
    parts = location.split(",")
    country = parts[-1].strip()
    subdiv = parts[0].strip() if len(parts) > 1 else None
    with requires_extra("dhw-jordanvajen", "holidays"):
        import holidays

    try:
        return frozenset(holidays.country_holidays(country=country, subdiv=subdiv, years=list(years)).keys())
    except (KeyError, ValueError, NotImplementedError) as exc:
        logger.warning(f"Invalid holiday location: {location}. Error: {exc}")
        return frozenset()


def _weekdays(days: pd.DatetimeIndex, holidays: frozenset[date]) -> np.ndarray:
    weekdays = days.day_of_week.to_numpy().copy()
    if holidays:
        weekdays[np.array([d in holidays for d in days.date], dtype=bool)] = SUNDAY
    return weekdays


def _day_weights(activity: pd.DataFrame) -> pd.DataFrame:
    """``probability_day`` of each event type (columns) on each weekday (rows), 1 if absent."""
    if C.PROBABILITY_DAY in activity.columns:
        weights = activity.groupby([C.DAY_OF_WEEK, C.EVENT])[C.PROBABILITY_DAY].first().unstack(C.EVENT)
    else:
        weights = activity.groupby([C.DAY_OF_WEEK, C.EVENT]).size().unstack(C.EVENT).notna().astype(float)
    return weights.reindex(range(7)).fillna(0.0).sort_index(axis=1)


def event_shares(activity: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return each event type's share of the mean daily volume and its factor per weekday.

    The share comes from the weekly mean of ``probability_day``. The factor is
    ``probability_day`` on that weekday divided by its weekly mean, so the
    factors of an event type average to 1 over a week.
    """
    weights = _day_weights(activity)
    weekly = weights.mean(axis=0)
    shares = weekly / weekly.sum() if weekly.sum() > 0 else weekly
    factors = weights.div(weekly.where(weekly > 0), axis=1).fillna(0.0)
    return shares, factors


def daily_volume(
    days: pd.DatetimeIndex,
    mean_annual_l: float,
    sigma: float,
    activity: pd.DataFrame,
    seasonal_variation: float,
    seasonal_peak_day: int,
    holidays: frozenset[date],
    rng: np.random.Generator,
) -> pd.Series:
    """Return the expected volume [l] of each local day.

    The base draw is a truncated normal around the mean daily volume of the
    day's year (``mean_annual_l`` divided by the actual number of days in that
    year). It is scaled by the seasonal factor
    ``1 + seasonal_variation * cos(2 pi (day_of_year - peak) / days_in_year)``
    and by Σ share × weekday factor over the event types (see :func:`event_shares`).
    """
    with requires_extra("dhw-jordanvajen", "scipy"):
        from scipy.stats import truncnorm

    length = days_in_year(days)
    mean = mean_annual_l / length
    draws = truncnorm(-1.0 / sigma, np.inf, loc=mean, scale=mean * sigma).rvs(size=len(days), random_state=rng)

    seasonal = 1.0 + seasonal_variation * np.cos(
        2.0 * np.pi * (days.day_of_year.to_numpy() - seasonal_peak_day) / length
    )

    shares, factors = event_shares(activity)
    weekday_total = (factors * shares).sum(axis=1).to_numpy()[_weekdays(days, holidays)]
    return pd.Series(draws * seasonal * weekday_total, index=days)


def _mean_event_volume(mean_flow: float, sigma_rel: float) -> float:
    """Mean of the truncated normal that event volumes are drawn from (see ``_sample_event_volumes``)."""
    with requires_extra("dhw-jordanvajen", "scipy"):
        from scipy.stats import truncnorm

    sigma = mean_flow * sigma_rel
    if sigma <= 0:
        return mean_flow
    return float(truncnorm(-mean_flow / sigma, np.inf, loc=mean_flow, scale=sigma).mean())


def sample_volume(
    axis: pd.DatetimeIndex,
    zone: ZoneInfo,
    step: pd.Timedelta,
    activity: pd.DataFrame,
    daily_volume: pd.Series,
    rng: np.random.Generator,
    holidays: frozenset[date],
    *,
    hours: np.ndarray | None = None,
    by_event: bool = False,
) -> pd.Series | pd.DataFrame:
    """Draw each local day's volume as events and place them on the run's rows.

    Args:
        axis: The run's UTC instants; row ``i`` covers ``[axis[i], axis[i + 1])``.
        zone: The site's time zone.
        step: The run's time step.
        activity: Activity table with day of week, time, event, probability,
            duration and flow rate (and optionally probability_day).
        daily_volume: Expected volume [l] of each local day, indexed by the day at midnight.
        rng: Random number generator.
        holidays: Local dates that count as Sundays.
        hours: Length of each row in hours; defaults to ``step`` for every row.
        by_event: Return one column per event type instead of the total.

    Returns:
        Volume [l] per row of the run. Over the run it equals the expected
        volume of the covered days; a partly covered day contributes the share
        of its activity that the run covers.
    """
    n = len(axis)
    hours = np.full(n, step / pd.Timedelta(1, unit="h")) if hours is None else np.asarray(hours, dtype=float)
    end = axis[-1] + pd.Timedelta(hours[-1] * 3600.0, unit="s")
    days = pd.DatetimeIndex(daily_volume.index)
    weekdays = _weekdays(days, holidays)
    shares, factors = event_shares(activity)
    expected_by_type = (factors * shares).to_numpy()[weekdays]
    totals = expected_by_type.sum(axis=1, keepdims=True)
    split = np.divide(expected_by_type, totals, out=np.zeros_like(expected_by_type), where=totals > 0)
    expected_by_type = daily_volume.to_numpy(dtype=float)[:, None] * split

    seconds = _convert_time_to_seconds_of_day(activity[C.TIME].astype(str))
    activity = activity.assign(_seconds=seconds)
    volumes = {}
    target_total = 0.0
    largest = None  # (expected volume, event, day of slot, row, scale) of the event type with the most water

    for j, event in enumerate(factors.columns):
        rows = activity[activity[C.EVENT] == event]
        first = rows.iloc[0]
        mean_flow = float(first[C.FLOW_RATE] * first[C.DURATION])
        sigma_rel = float(first[C.FLOW_RATE_SIGMA] / first[C.FLOW_RATE])

        day_parts, pos_parts, prob_parts, full_mass = [], [], [], np.zeros(len(days))
        for weekday, slots in rows.groupby(C.DAY_OF_WEEK):
            day_index = np.flatnonzero(weekdays == weekday)
            if not len(day_index):
                continue
            slot_seconds = slots["_seconds"].to_numpy(dtype="int64")
            probs = slots[C.PROBABILITY].to_numpy(dtype=float)
            full_mass[day_index] = probs.sum()

            offsets = (slot_seconds * 1_000_000_000).astype("timedelta64[ns]")
            wall = pd.DatetimeIndex((days[day_index].to_numpy()[:, None] + offsets[None, :]).ravel())
            instants = wall.tz_localize(zone, nonexistent="shift_forward", ambiguous=np.ones(len(wall), dtype=bool))
            instants = instants.tz_convert("UTC")
            covered = np.asarray((instants >= axis[0]) & (instants < end))
            pos = axis.searchsorted(instants, side="right") - 1

            day_parts.append(np.repeat(day_index, len(slot_seconds))[covered])
            pos_parts.append(pos[covered])
            prob_parts.append(np.tile(probs, len(day_index))[covered])

        volume = np.zeros(n)
        volumes[event] = volume
        if not day_parts:
            continue
        day_of_slot = np.concatenate(day_parts)
        pos = np.concatenate(pos_parts)
        prob = np.concatenate(prob_parts)
        keep = prob > 0
        day_of_slot, pos, prob = day_of_slot[keep], pos[keep], prob[keep]
        order = np.lexsort((pos, day_of_slot))
        day_of_slot, pos, prob = day_of_slot[order], pos[order], prob[order]

        covered_mass = np.bincount(day_of_slot, weights=prob, minlength=len(days))
        fraction = np.divide(covered_mass, full_mass, out=np.zeros(len(days)), where=full_mass > 0)
        expected = expected_by_type[:, j] * fraction
        target_total += float(expected.sum())
        if mean_flow <= 0 or not len(prob):
            continue

        # Inverse sampling within each day: day k occupies [k, k + 1) of one cumulative scale.
        starts = np.repeat(np.cumsum(covered_mass) - covered_mass, np.bincount(day_of_slot, minlength=len(days)))
        scale = day_of_slot + (np.cumsum(prob) - starts) / covered_mass[day_of_slot]
        last = np.r_[day_of_slot[1:] != day_of_slot[:-1], True]
        scale[last] = day_of_slot[last] + 1.0
        if largest is None or expected.sum() > largest[0]:
            largest = (float(expected.sum()), event, expected, pos, scale)

        # The number of events has the day's expected volume over the mean size of the drawn events.
        counts = rng.poisson(np.where(covered_mass > 0, expected / _mean_event_volume(mean_flow, sigma_rel), 0.0))
        if counts.sum() == 0:
            continue

        event_days = np.repeat(np.arange(len(days)), counts)
        picks = np.searchsorted(scale, event_days + rng.random(len(event_days)), side="right")

        sizes = _sample_event_volumes(mean_flow, sigma_rel, len(event_days), rng)
        volume += np.bincount(pos[picks], weights=sizes, minlength=n)

    # One factor over the whole run keeps the event sizes and makes the total exact.
    result = pd.DataFrame(volumes, index=axis)
    drawn_total = float(result.to_numpy().sum())
    if drawn_total > 0:
        result *= target_total / drawn_total
    elif target_total > 0 and largest is not None:
        # A run too small to draw any event still receives its volume, as one event of the largest type.
        _, event, expected, pos, scale = largest
        day = rng.choice(len(days), p=expected / expected.sum())
        pick = np.searchsorted(scale, day + rng.random(), side="right")
        result.loc[result.index[pos[pick]], event] = target_total

    return result if by_event else result.sum(axis=1)
