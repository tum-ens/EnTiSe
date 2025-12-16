"""
Core calculation functions for the Jordan & Vajen DHW method.

This module contains the core calculation functions used by the Jordan & Vajen DHW method
to generate DHW demand time series.
"""

import logging
from typing import Any, Dict, Tuple, Union

import holidays
import numpy as np
import pandas as pd

import entise.methods.dhw.defaults as defaults
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.dhw.jordanvajen.utils import (
    DAYS_PER_YEAR,
    SEASONAL_FACTOR,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    _convert_time_to_seconds_of_day,
    _find_nearest_activity_times,
    _sample_event_volumes,
)

logger = logging.getLogger(__name__)


def _calculate_volume_timeseries(
    activity_data: pd.DataFrame,
    annual_volumes: Dict[str, float],
    mean_flows: Dict[str, float],
    sigma_rel: Dict[str, float],
    daily_demand: Union[float, pd.Series],
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Generate a DHW volume time series using vectorized inverse sampling.

    This function generates a DHW volume time series by:
    1. Converting activity times to seconds of day
    2. Finding the nearest activity time for each timestamp
    3. Calculating probabilities for each timestamp
    4. Sampling event occurrences using inverse transform sampling
    5. Sampling event volumes from truncated normal distributions
    6. Aggregating the volumes into a time series

    Parameters:
    -----------
    activity_data : pd.DataFrame
        Activity data with columns:
        - event: Event type (e.g., 'Shower', 'Bath', 'Medium', 'Small')
        - time: Time of day (HH:MM:SS)
        - probability: Probability of event occurring at this time
        - probability_day: Probability of event occurring on this day
        - seasonal_factor: Seasonal variation factor
    annual_volumes : Dict[str, float]
        Annual volume for each event type in liters
    mean_flows : Dict[str, float]
        Mean flow per event for each event type in liters
    sigma_rel : Dict[str, float]
        Relative standard deviation for each event type
    daily_demand : Union[float, pd.Series]
        Daily demand in liters. If a Series, it should be indexed by date.
    index : pd.DatetimeIndex
        Timestamp index for the output time series
    rng : np.random.Generator
        Random number generator

    Returns:
    --------
    pd.Series
        DHW volume time series indexed by timestamp

    Raises:
    -------
    ValueError
        If activity_data is empty or missing required columns
    KeyError
        If an event type in activity_data is not in annual_volumes, mean_flows, or sigma_rel
    """
    # Validate inputs
    if activity_data.empty:
        raise ValueError("Activity data is empty")

    required_columns = [C.EVENT, C.TIME, C.PROBABILITY, C.PROBABILITY_DAY, SEASONAL_FACTOR]
    missing_columns = [col for col in required_columns if col not in activity_data.columns]
    if missing_columns:
        raise ValueError(f"Activity data missing required columns: {missing_columns}")

    # Pre-compute seconds‐of‐day for each timestamp
    logger.debug(f"Converting {len(index)} timestamps to seconds of day")
    seconds_of_day = index.dt.hour * SECONDS_PER_HOUR + index.dt.minute * SECONDS_PER_MINUTE + index.dt.second

    # Pre-compute per‐timestamp daily weight by matching each ts to its date
    if isinstance(daily_demand, pd.Series):
        logger.debug("Mapping daily demand to timestamps")
        daily_weights = daily_demand.reindex(index.dt.normalize()).values
    else:
        # If daily_demand is a scalar, use it for all timestamps
        logger.debug(f"Using constant daily demand: {daily_demand}")
        daily_weights = np.full(len(index), daily_demand)

    # Containers for event positions and volumes
    all_positions = []
    all_volumes = []

    # Process each event type
    event_types = activity_data[C.EVENT].unique()
    logger.debug(f"Processing {len(event_types)} event types: {', '.join(event_types)}")

    for event, df_ev in activity_data.groupby(C.EVENT):
        # Validate event type
        if event not in annual_volumes:
            raise KeyError(f"Event type '{event}' not in annual_volumes")
        if event not in mean_flows:
            raise KeyError(f"Event type '{event}' not in mean_flows")
        if event not in sigma_rel:
            raise KeyError(f"Event type '{event}' not in sigma_rel")

        # Compute number of events N
        N = int(round(annual_volumes[event] / mean_flows[event]))
        logger.debug(
            f"Event type '{event}': {N} events, annual volume={annual_volumes[event]:.1f} L, "
            f"mean flow={mean_flows[event]:.1f} L/event"
        )

        # Skip if no events
        if N <= 0:
            logger.warning(f"No events for event type '{event}' (N={N})")
            continue

        # Convert activity times to seconds of day
        activity_times = _convert_time_to_seconds_of_day(df_ev[C.TIME])

        # Apply probability_day × seasonal × holiday weighting before normalization
        weighted_probs = df_ev[C.PROBABILITY].values * df_ev[C.PROBABILITY_DAY].values * df_ev[SEASONAL_FACTOR].values

        # Find nearest activity time for each timestamp
        idxs, order = _find_nearest_activity_times(activity_times, seconds_of_day)

        # Reorder probabilities to match sorted activity times
        probs_sorted = weighted_probs[order]

        # Build probability distribution for inverse sampling
        base_probs = probs_sorted[idxs]

        # Weight by daily demand
        probs = base_probs * daily_weights

        # Normalize probabilities
        probs_sum = probs.sum()
        if probs_sum <= 0:
            logger.warning(f"Zero probability sum for event type '{event}'. Skipping.")
            continue

        probs /= probs_sum

        # Inverse‐sample event positions using CDF
        cdf = np.cumsum(probs)
        r = rng.random(N)
        positions = np.searchsorted(cdf, r, side="right")

        # Sample per-event volumes from truncated normal distribution
        volumes = _sample_event_volumes(mean_flows[event], sigma_rel[event], N, rng)

        # Add to containers
        all_positions.append(positions)
        all_volumes.append(volumes)

        logger.debug(f"Event type '{event}': sampled {len(positions)} positions and volumes")

    # Check if any events were sampled
    if not all_positions:
        logger.warning("No events were sampled. Returning zero time series.")
        return pd.Series(0.0, index=index)

    # Aggregate into time series
    logger.debug("Aggregating events into time series")
    positions = np.concatenate(all_positions)
    volumes = np.concatenate(all_volumes)

    # Use bincount for efficient aggregation
    flat = np.bincount(positions, weights=volumes, minlength=len(index))

    # Create Series with timestamp index
    result = pd.Series(flat, index=index)
    logger.debug(
        f"Created time series with {len(result)} points, "
        f"total volume={result.sum():.1f} L, "
        f"non-zero points={(result > 0).sum()} ({(result > 0).sum()/len(result)*100:.1f}%)"
    )

    return result


def _calculate_timeseries(
    weather: pd.DataFrame,
    activity_data: pd.DataFrame,
    daily_demand: Union[float, pd.Series],
    water_temp: pd.DataFrame,
    obj: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate DHW demand time series using stochastic inverse sampling.

    This function uses a vectorized inverse sampling approach to generate DHW demand time series.
    The approach is based on the Jordan & Vajen (2005) methodology and provides a more realistic
    representation of DHW usage patterns by:

    1. Sampling event occurrences based on probability distributions over time
    2. Sampling individual event volumes from truncated normal distributions
    3. Aggregating the volumes into a time series

    This approach is more efficient and accurate than the previous hybrid approach, with a
    computational complexity of O(E + T), where E is the number of events and T is the number
    of timestamps.

    Parameters:
    -----------
    weather : pd.DataFrame
        Weather data with datetime column (C.DATETIME)
    activity_data : pd.DataFrame
        Activity data with columns:
        - day: Day of week (0=Monday, 6=Sunday)
        - time: Time of day (HH:MM:SS)
        - event: Event type (e.g., 'Shower', 'Bath', 'Medium', 'Small')
        - probability: Probability of event occurring at this time
        - duration: Duration of event in seconds
        - flow_rate: Flow rate in liters/second
        - duration_sigma: Standard deviation of duration in seconds
        - flow_rate_sigma: Standard deviation of flow rate in liters/second
        - probability_day: Probability of event occurring on this day
    daily_demand : float or pd.Series
        Daily demand in liters. If a Series, it should be indexed by date.
        Used to calculate annual volumes for each event type.
    water_temp : pd.DataFrame
        Cold and hot water temperature in °C for every time step
        Columns: [O.TEMP_WATER_COLD, O.TEMP_WATER_HOT]
    obj : Dict[str, Any]
        Object parameters including:
        - O.SEED: Random seed for reproducibility (optional)
        - O.SEASONAL_VARIATION: Seasonal variation factor (optional)
        - O.SEASONAL_PEAK_DAY: Day of year with peak demand (optional)
        - O.HOLIDAYS_LOCATION: Holiday location(s) (optional)

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        Tuple containing:
        - volume_timeseries: DHW volume in liters for each timestamp
        - energy_timeseries: DHW energy demand in Wh for each timestamp

    Raises:
    -------
    ValueError
        If required columns are missing from activity_data or weather
    KeyError
        If required keys are missing from water_temp
    """
    # Validate inputs
    if C.DATETIME not in weather.columns:
        raise ValueError(f"Weather data missing required column: {C.DATETIME}")

    required_columns = [C.DAY_OF_WEEK, C.TIME, C.EVENT, C.PROBABILITY, C.DURATION, C.FLOW_RATE, C.FLOW_RATE_SIGMA]
    missing_columns = [col for col in required_columns if col not in activity_data.columns]
    if missing_columns:
        raise ValueError(f"Activity data missing required columns: {missing_columns}")

    required_keys = [O.TEMP_WATER_COLD, O.TEMP_WATER_HOT]
    missing_keys = [key for key in required_keys if key not in water_temp.columns]
    if missing_keys:
        raise KeyError(f"Water temperature data missing required columns: {missing_keys}")

    # Get parameters
    temp_hot = water_temp[O.TEMP_WATER_HOT]
    temp_cold = water_temp[O.TEMP_WATER_COLD]
    seed = obj.get(O.SEED, None)
    rng = np.random.default_rng(seed)
    logger.debug(f"Using random seed: {seed}")

    # Create index
    index = pd.to_datetime(weather[C.DATETIME], utc=True).dt.tz_convert(weather[C.DATETIME].iloc[0].tz)
    logger.debug(f"Created time index with {len(index)} timestamps")

    # Calculate simulation duration in years
    start_date = index.iat[0].date()
    end_date = index.iat[-1].date()
    days_in_simulation = (end_date - start_date).days + 1
    years_in_simulation = days_in_simulation / DAYS_PER_YEAR
    logger.info(
        f"Simulation period: {start_date} to {end_date} ({days_in_simulation} days, {years_in_simulation:.2f} years)"
    )

    # Determine weekdays
    days = pd.DataFrame({C.DATE: pd.date_range(start_date, end_date, freq="D")})
    days[C.DAY_OF_WEEK] = days[C.DATE].dt.weekday
    logger.debug(f"Created calendar with {len(days)} days")

    # Cross-join only matching weekday rows
    # `activity_data.day` is 0=Mon…6=Sun
    events = days.merge(activity_data, left_on=C.DAY_OF_WEEK, right_on=C.DAY_OF_WEEK, how="inner")
    logger.debug(f"Created events DataFrame with {len(events)} rows after day-of-week matching")

    # Include seasonal factor (sine wave)
    seasonal_var = obj.get(O.SEASONAL_VARIATION, defaults.DEFAULT_SEASONAL_VARIATION)
    seasonal_peak = obj.get(O.SEASONAL_PEAK_DAY, defaults.DEFAULT_SEASONAL_PEAK_DAY)
    events["dayofyear"] = events[C.DATE].dt.dayofyear
    events[SEASONAL_FACTOR] = 1 + seasonal_var * np.cos(
        2 * np.pi * (events["dayofyear"] - seasonal_peak) / DAYS_PER_YEAR
    )
    logger.info(f"Applied seasonal variation: amplitude={seasonal_var:.2f}, peak day={seasonal_peak}")

    # Holiday suppression (treat holidays as Sundays)
    location = obj.get(O.HOLIDAYS_LOCATION, [])
    if isinstance(location, str) and location.strip():
        location = location.split(",")
        country = location[-1].strip()
        subdiv = location[0].strip() if len(location) > 1 else None

        try:
            holiday_dates = set(
                holidays.country_holidays(country=country, subdiv=subdiv, years=index.dt.year.unique()).keys()
            )
            logger.info(
                f"Applied holiday suppression for {country}{f', {subdiv}' if subdiv else ''}: {len(holiday_dates)} holidays"
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid holiday location: {location}. Error: {str(e)}")
            holiday_dates = set()
    else:
        holiday_dates = set()
        logger.debug("No holiday suppression applied")

    # Flag holiday rows and override their 'day' to Sunday (6)
    is_hol = events[C.DATE].dt.date.isin(holiday_dates)
    events.loc[is_hol, C.DAY_OF_WEEK] = 6
    if is_hol.sum() > 0:
        logger.debug(f"Treated {is_hol.sum()} event days as holidays (Sunday)")

    # Re‐assign probability_day from the original activity_data
    if C.PROBABILITY_DAY in activity_data.columns:
        pd_map = activity_data.set_index([C.EVENT, C.DAY_OF_WEEK, C.TIME])[C.PROBABILITY_DAY]
        mi = pd.MultiIndex.from_frame(events[[C.EVENT, C.DAY_OF_WEEK, C.TIME]])
        events[C.PROBABILITY_DAY] = pd_map.reindex(mi).values
        logger.debug("Re-assigned probability_day values from activity data")

    # Calculate annual volumes for each event type
    # If daily_demand is a Series, calculate the total annual demand
    if isinstance(daily_demand, pd.Series):
        # Filter to dates within the simulation period
        daily_demand_filtered = daily_demand[
            (daily_demand.index.date >= start_date) & (daily_demand.index.date <= end_date)
        ]
        total_annual_demand = daily_demand_filtered.sum() * DAYS_PER_YEAR / days_in_simulation
        logger.info(f"Calculated total annual demand from daily demand series: {total_annual_demand:.1f} L/year")
    else:
        # If daily_demand is a scalar, multiply by days in a year
        total_annual_demand = daily_demand * DAYS_PER_YEAR
        logger.info(f"Calculated total annual demand from constant daily demand: {total_annual_demand:.1f} L/year")

    # Calculate annual volumes for each event type based on probability_day
    # If probability_day is not available, distribute evenly
    annual_volumes = {}
    mean_flows = {}
    sigma_rel = {}

    # Group activity data by event type
    event_groups = events.groupby(C.EVENT)
    event_types = list(event_groups.groups.keys())
    logger.info(f"Processing {len(event_types)} event types: {', '.join(event_types)}")

    # Check if probability_day column exists
    if C.PROBABILITY_DAY in activity_data.columns:
        # Calculate total probability_day across all event types
        total_prob_day = sum(group[C.PROBABILITY_DAY].iloc[0] for _, group in event_groups)
        logger.debug(f"Total probability_day across all event types: {total_prob_day:.4f}")

        # Distribute annual demand based on probability_day
        for event, group in event_groups:
            prob_day = group[C.PROBABILITY_DAY].iloc[0]
            annual_volumes[event] = total_annual_demand * prob_day / total_prob_day

            # Calculate mean flow and sigma for each event type
            mean_flows[event] = group[C.FLOW_RATE].iloc[0] * group[C.DURATION].iloc[0]  # liters per event
            sigma_rel[event] = group[C.FLOW_RATE_SIGMA].iloc[0] / group[C.FLOW_RATE].iloc[0]  # relative sigma

            logger.info(
                f"Event type '{event}': probability_day={prob_day:.4f}, annual volume={annual_volumes[event]:.1f} L, "
                f"mean flow={mean_flows[event]:.1f} L/event, sigma_rel={sigma_rel[event]:.4f}"
            )
    else:
        # If probability_day is not available, distribute evenly
        event_count = len(event_groups)
        logger.warning(
            f"probability_day column not found in activity data. Distributing demand evenly across {event_count} event types."
        )

        for event, group in event_groups:
            annual_volumes[event] = total_annual_demand / event_count

            # Calculate mean flow and sigma for each event type
            mean_flows[event] = group[C.FLOW_RATE].iloc[0] * group[C.DURATION].iloc[0]  # liters per event
            sigma_rel[event] = group[C.FLOW_RATE_SIGMA].iloc[0] / group[C.FLOW_RATE].iloc[0]  # relative sigma

            logger.info(
                f"Event type '{event}': annual volume={annual_volumes[event]:.1f} L, "
                f"mean flow={mean_flows[event]:.1f} L/event, sigma_rel={sigma_rel[event]:.4f}"
            )

    # Generate volume time series using vectorized inverse sampling
    logger.info("Generating volume time series using vectorized inverse sampling")
    ts_volume = _calculate_volume_timeseries(events, annual_volumes, mean_flows, sigma_rel, daily_demand, index, rng)

    # Scale to match total annual demand
    ts_volume *= total_annual_demand / ts_volume.sum()
    logger.debug(f"Scaled volume time series to match total annual demand: {total_annual_demand:.1f} L")

    # Log volume statistics
    volume_stats = {
        "total": ts_volume.sum(),
        "mean": ts_volume.mean(),
        "max": ts_volume.max(),
        "min": ts_volume.min(),
        "non_zero": (ts_volume > 0).sum(),
        "zero": (ts_volume == 0).sum(),
    }
    logger.info(
        f"Volume time series statistics: total={volume_stats['total']:.1f} L, "
        f"mean={volume_stats['mean']:.3f} L, max={volume_stats['max']:.3f} L, "
        f"non-zero points: {volume_stats['non_zero']} ({volume_stats['non_zero']/len(ts_volume)*100:.1f}%)"
    )

    # Calculate energy demand
    delta_t = temp_hot - temp_cold
    energy_factor = (
        defaults.DEFAULT_DENSITY_WATER / 1000 * defaults.DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600
    )  # Convert to Wh
    ts_energy = ts_volume * energy_factor

    # Log energy statistics
    energy_stats = {"total": ts_energy.sum(), "mean": ts_energy.mean(), "max": ts_energy.max(), "min": ts_energy.min()}
    logger.info(
        f"Energy time series statistics: total={energy_stats['total']:.1f} Wh, "
        f"mean={energy_stats['mean']:.3f} Wh, max={energy_stats['max']:.3f} Wh"
    )

    return ts_volume.round(3), ts_energy.round().astype(int)
