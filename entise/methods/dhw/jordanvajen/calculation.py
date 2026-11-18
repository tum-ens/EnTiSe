"""
Core calculation functions for the Jordan & Vajen DHW method.

Water is drawn per local day of the site:

1. Each local day gets a volume: a draw from the dwelling's daily demand,
   scaled by the day's seasonal factor and by its weekday weight (holidays
   count as Sundays).
2. The day's volume is split over the event types by their weight for that
   weekday (``probability_day``).
3. Each event type's volume is drawn as discrete events at that weekday's
   activity times (``probability``). A time is placed on the instant it names
   in the site's time zone: a time that does not exist when clocks go forward
   moves forward, and a time that happens twice when clocks go back takes the
   first occurrence. The row that contains the instant receives the volume.
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
    """Weight of each event type on each weekday, from ``probability_day`` (1 if absent)."""
    if C.PROBABILITY_DAY in activity.columns:
        weights = activity.groupby([C.DAY_OF_WEEK, C.EVENT])[C.PROBABILITY_DAY].first().unstack(C.EVENT)
    else:
        weights = activity.groupby([C.DAY_OF_WEEK, C.EVENT]).size().unstack(C.EVENT).notna().astype(float)
    return weights.reindex(range(7)).fillna(0.0).sort_index(axis=1)


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
    """Return the volume [l] of each local day.

    The base draw is a truncated normal around the mean daily volume of the
    day's year (``mean_annual_l`` divided by the actual number of days in that
    year). It is scaled by the seasonal factor
    ``1 + seasonal_variation * cos(2 pi (day_of_year - peak) / days_in_year)``
    and by the weekday weight, normalised so that an average week keeps the
    mean.
    """
    with requires_extra("dhw-jordanvajen", "scipy"):
        from scipy.stats import truncnorm

    length = days_in_year(days)
    mean = mean_annual_l / length
    draws = truncnorm(-1.0 / sigma, np.inf, loc=mean, scale=mean * sigma).rvs(size=len(days), random_state=rng)

    seasonal = 1.0 + seasonal_variation * np.cos(
        2.0 * np.pi * (days.day_of_year.to_numpy() - seasonal_peak_day) / length
    )

    per_weekday = _day_weights(activity).sum(axis=1).to_numpy()
    mean_weight = per_weekday.mean()
    weekday = per_weekday[_weekdays(days, holidays)] / mean_weight if mean_weight > 0 else np.ones(len(days))

    return pd.Series(draws * seasonal * weekday, index=days)


def sample_volume(
    axis: pd.DatetimeIndex,
    zone: ZoneInfo,
    step: pd.Timedelta,
    activity: pd.DataFrame,
    daily_volume: pd.Series,
    rng: np.random.Generator,
    holidays: frozenset[date],
) -> pd.Series:
    """Draw each local day's volume as events and place them on the run's rows.

    Args:
        axis: The run's UTC instants; row ``i`` covers ``[axis[i], axis[i] + step)``.
        zone: The site's time zone.
        step: The run's time step.
        activity: Activity table with day of week, time, event, probability,
            duration and flow rate (and optionally probability_day).
        daily_volume: Volume [l] of each local day, indexed by the day at midnight.
        rng: Random number generator.
        holidays: Local dates that count as Sundays.

    Returns:
        Volume [l] per row of the run. A day that the run covers completely
        receives exactly its daily volume; a partly covered day receives the
        share of its activity that the run covers.
    """
    n = len(axis)
    end = axis[-1] + step
    days = pd.DatetimeIndex(daily_volume.index)
    weekdays = _weekdays(days, holidays)
    weights = _day_weights(activity)
    shares = weights.to_numpy()[weekdays]
    totals = shares.sum(axis=1, keepdims=True)
    shares = np.divide(shares, totals, out=np.zeros_like(shares), where=totals > 0)

    seconds = _convert_time_to_seconds_of_day(activity[C.TIME].astype(str))
    activity = activity.assign(_seconds=seconds)
    volume = np.zeros(n)

    for j, event in enumerate(weights.columns):
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

            wall = (
                days[day_index].to_numpy()[:, None] + (slot_seconds * 1_000_000_000).astype("timedelta64[ns]")[None, :]
            )
            wall = pd.DatetimeIndex(wall.ravel())
            instants = wall.tz_localize(zone, nonexistent="shift_forward", ambiguous=np.ones(len(wall), dtype=bool))
            instants = instants.tz_convert("UTC")
            covered = np.asarray((instants >= axis[0]) & (instants < end))
            pos = axis.searchsorted(instants, side="right") - 1

            day_parts.append(np.repeat(day_index, len(slot_seconds))[covered])
            pos_parts.append(pos[covered])
            prob_parts.append(np.tile(probs, len(day_index))[covered])

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
        target = daily_volume.to_numpy(dtype=float) * shares[:, j] * fraction
        counts = np.where((target > 0) & (covered_mass > 0), np.maximum(1, np.rint(target / mean_flow)), 0).astype(int)
        if counts.sum() == 0:
            continue

        # Inverse sampling within each day: day k occupies [k, k + 1) of one cumulative scale.
        within = np.cumsum(prob) - np.repeat(
            np.cumsum(covered_mass) - covered_mass, np.bincount(day_of_slot, minlength=len(days))
        )
        scale = day_of_slot + within / covered_mass[day_of_slot]
        last = np.r_[day_of_slot[1:] != day_of_slot[:-1], True]
        scale[last] = day_of_slot[last] + 1.0
        event_days = np.repeat(np.arange(len(days)), counts)
        picks = np.searchsorted(scale, event_days + rng.random(len(event_days)), side="right")

        volumes = _sample_event_volumes(mean_flow, sigma_rel, len(event_days), rng)
        drawn = np.bincount(event_days, weights=volumes, minlength=len(days))
        volumes = volumes * np.divide(target, drawn, out=np.zeros(len(days)), where=drawn > 0)[event_days]
        volume += np.bincount(pos[picks], weights=volumes, minlength=n)

    return pd.Series(volume, index=axis)
