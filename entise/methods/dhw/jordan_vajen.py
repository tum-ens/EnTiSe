"""
Jordan & Vajen DHW (Domestic Hot Water) method.

This module implements a DHW method based on Jordan & Vajen (2005):
"DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS"

Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS.
Universität Marburg.
URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
"""

import os
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import truncnorm
import holidays

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types
import entise.methods.dhw.defaults as defaults

logger = logging.getLogger(__name__)


class JordanVajen(Method):
    """
    Jordan & Vajen DHW method based on dwelling size.

    This method calculates domestic hot water demand time series based on the size of the dwelling
    using the Jordan & Vajen (2005) methodology. It distributes daily demand according to activity
    profiles provided in the activity data filename.

    Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES 
    WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS. Universität Marburg.
    URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
    """
    types = [Types.DHW]
    name = "jordanvajen"
    required_keys = [O.WEATHER, O.DWELLING_SIZE]
    optional_keys = [
        O.DHW_ACTIVITY,
        O.DHW_DEMAND_PER_SIZE,
        O.HOLIDAYS_LOCATION,
        O.TEMP_WATER_COLD,
        O.TEMP_WATER_HOT,
        O.SEASONAL_VARIATION, 
        O.SEASONAL_PEAK_DAY,
        O.SEED,
    ]
    required_timeseries = [O.WEATHER]
    optional_timeseries = [
        O.DHW_ACTIVITY,
        O.DHW_DEMAND_PER_SIZE,
        O.TEMP_WATER_COLD,
        O.TEMP_WATER_HOT,
    ]
    output_summary = {
        f'{C.DEMAND}_{Types.DHW}_volume_total': 'total hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_volume_avg': 'average hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_volume_peak': 'peak hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_energy_total': 'total energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_energy_avg': 'average energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_energy_peak': 'peak energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_power_avg': 'average power for hot water in W',
        f'{C.DEMAND}_{Types.DHW}_power_max': 'maximum power for hot water in W',
        f'{C.DEMAND}_{Types.DHW}_power_min': 'minimum power for hot water in W',
    }
    output_timeseries = {
        f'{C.LOAD}_{Types.DHW}_volume': 'hot water demand in liters',
        f'{C.LOAD}_{Types.DHW}_energy': 'energy demand for hot water in Wh',
        f'{C.LOAD}_{Types.DHW}_power': 'power demand for hot water in W',
        f'{Types.DHW}_{O.TEMP_WATER_COLD}': 'cold water temperature in degrees Celsius',
        f'{Types.DHW}_{O.TEMP_WATER_HOT}': 'hot water temperature in degrees Celsius',
    }

    def generate(self, obj, data, ts_type):
        """
        Generate DHW demand time series.

        Parameters:
        -----------
        obj : dict
            Object parameters
        data : dict
            Input data
        ts_type : str
            Time series type

        Returns:
        --------
        dict
            Dictionary with summary and time series data
        """
        # Reproducible RNG
        seed = obj.get(O.SEED, None)
        rng = np.random.default_rng(seed)

        # Get parameters
        weather = data[O.WEATHER]
        weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME])
        dwelling_size = obj[O.DWELLING_SIZE]

        # Get activity data
        activity_data = data.get(O.DHW_ACTIVITY, None)
        if activity_data is None:
            activity_data = _get_activity_data('jordan_vajen')

        # Obtain statistical data for yearly demand
        demand_data = _get_demand_data('jordan_vajen')
        sizes = demand_data['dwelling_size'].values
        idx = np.abs(sizes - dwelling_size).argmin()
        m3_per_m2_a = demand_data.iloc[idx]['m3_per_m2_a']
        sigma = demand_data.iloc[idx]['sigma']

        # Compute mean & std in litres/day
        mean_daily_l = m3_per_m2_a * dwelling_size * 1e3 / 365
        sd_daily_l = mean_daily_l * sigma

        # Define truncation bounds (no negatives)
        a, b = (0 - mean_daily_l) / sd_daily_l, np.inf

        # Build a date index for each simulation day
        start = weather[C.DATETIME].iloc[0].normalize()
        end = weather[C.DATETIME].iloc[-1].normalize()
        days = pd.date_range(start, end, freq='D')

        # 5) sample once **per day** from the truncated normal
        dist = stats.truncnorm(a, b, loc=mean_daily_l, scale=sd_daily_l)
        daily_demand_l = pd.Series(dist.rvs(size=len(days), random_state=rng), index=days)

        # Get cold water temperature
        water_temp = _get_water_temperatures(obj, data, weather)

        # Generate time series
        ts_volume, ts_energy = _calculate_timeseries(
            weather, activity_data, daily_demand_l, water_temp, obj
        )

        # Convert energy into power
        # Calculate time interval in hours
        time_diff = pd.Series(weather[C.DATETIME]).diff().dt.total_seconds() / 3600
        # Use the median time difference for the first element
        time_diff.iloc[0] = time_diff.median()
        # Calculate power as energy divided by time
        ts_power = ts_energy / time_diff.values

        # Create output
        summary = {
            f'{C.DEMAND}_{Types.DHW}_volume_total': int(ts_volume.sum().round(0)),
            f'{C.DEMAND}_{Types.DHW}_volume_avg': float(ts_volume.mean().round(3)),
            f'{C.DEMAND}_{Types.DHW}_volume_peak': float(ts_volume.max().round(3)),
            f'{C.DEMAND}_{Types.DHW}_energy_total': int(ts_energy.sum()),
            f'{C.DEMAND}_{Types.DHW}_energy_avg': int(ts_energy.mean().round(0)),
            f'{C.DEMAND}_{Types.DHW}_energy_peak': int(ts_energy.max()),
            f'{C.DEMAND}_{Types.DHW}_power_avg': int(ts_power.mean().round(0)),
            f'{C.DEMAND}_{Types.DHW}_power_max': int(ts_power.max()),
            f'{C.DEMAND}_{Types.DHW}_power_min': int(ts_power.min()),
        }

        timeseries = pd.DataFrame({
            f'{C.LOAD}_{Types.DHW}_volume': ts_volume,
            f'{C.LOAD}_{Types.DHW}_energy': ts_energy,
            f'{C.LOAD}_{Types.DHW}_power': ts_power,
        }, index=weather[C.DATETIME])

        return {
            "summary": summary,
            "timeseries": timeseries
        }


def _vectorized_inverse_sampling(
    activity_data: pd.DataFrame,
    annual_volumes: dict,
    mean_flows: dict,
    sigma_rel: dict,
    daily_demand: pd.Series,
    index: pd.DatetimeIndex,
    rng: np.random.Generator
) -> pd.Series:
    """
    Returns a DHW volume time series by inverse‐sampling events vectorized,
    with optimized nearest‐neighbor lookup via searchsorted.
    """
    # Pre-compute seconds‐of‐day for each timestamp
    seconds_of_day = (
        index.dt.hour.values * 3600 +
        index.dt.minute.values * 60 +
        index.dt.second.values
    )
    # Pre-compute per‐timestamp daily weight by matching each ts to its date
    daily_weights = daily_demand.reindex(index.dt.normalize()).values

    all_positions = []
    all_volumes   = []

    for event, df_ev in activity_data.groupby('event'):
        # Compute number of events N
        N = int(round(annual_volumes[event] / mean_flows[event]))

        # Vectorized convert 'time' strings to seconds of day
        t_parts = df_ev['time'].str.split(':', expand=True).astype(int)
        activity_times = (
            t_parts[0] * 3600 +
            t_parts[1] * 60 +
            t_parts.get(2, 0)
        ).values

        # Apply probability_day × seasonal × holiday weighting before normalization
        weighted_probs = (
                df_ev['probability'].values
                * df_ev['probability_day'].values
                * df_ev['seasonal_factor'].values
        )

        # Sort activity times for searchsorted
        order = np.argsort(activity_times)
        times_sorted = activity_times[order]
        probs_sorted = weighted_probs[order]

        # Find nearest activity slot for every timestamp
        pos = np.searchsorted(times_sorted, seconds_of_day, side='left')
        left  = np.clip(pos - 1, 0, len(times_sorted) - 1)
        right = np.clip(pos,     0, len(times_sorted) - 1)

        left_diff  = np.abs(seconds_of_day - times_sorted[left])
        right_diff = np.abs(seconds_of_day - times_sorted[right])
        choose_right = right_diff < left_diff
        idxs = np.where(choose_right, right, left)

        # Build, weight by daily demand, then normalize
        base_probs = probs_sorted[idxs]
        probs = base_probs * daily_weights
        probs /= probs.sum()

        # Inverse‐sample event positions
        cdf = np.cumsum(probs)
        r   = rng.random(N)
        positions = np.searchsorted(cdf, r, side='right')

        # Sample per-event volumes (truncated normal)
        mu    = mean_flows[event]
        sigma = mu * sigma_rel[event]
        a, b  = (0 - mu)/sigma, np.inf
        dist  = truncnorm(a, b, loc=mu, scale=sigma)
        volumes = dist.rvs(size=N, random_state=rng)

        all_positions.append(positions)
        all_volumes.append(volumes)

    # Aggregate into time series
    positions = np.concatenate(all_positions)
    volumes   = np.concatenate(all_volumes)
    flat      = np.bincount(positions, weights=volumes, minlength=len(index))

    return pd.Series(flat, index=index)


def _get_activity_data(source: str, filename: str = 'dhw_activity.csv') -> pd.DataFrame:
    """
    Get activity data from filename.

    Parameters:
    -----------
    source : str
        Source directory (e.g., 'jordan_vajen')
    filename : str
        Filename (e.g., 'dhw_activity.csv')

    Returns:
    --------
    pd.DataFrame
        Activity data
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    path = os.path.join(root_dir, 'entise', 'data', Types.DHW, source, filename)

    return pd.read_csv(path)


def _get_demand_data(source: str, filename: str = 'dhw_demand_by_dwelling.csv'):
    """
    Get demand data from filename with fallback mechanism.

    Parameters:
    -----------
    source : str
        Source directory (e.g., 'jordan_vajen')
    filename : str
        Filename (e.g., 'dhw_demand_by_dwelling.csv')

    Returns:
    --------
    pd.DataFrame
        Demand data
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    path = os.path.join(root_dir, 'entise', 'data', Types.DHW, source, filename)

    return pd.read_csv(path)


def _get_water_temp(obj: dict, data: dict, key: str):
    water_temp = obj.get(key, None)
    if isinstance(water_temp, str):
        water_temp = data.get(water_temp, None)
    return water_temp


def _get_water_temp_cold(obj: dict, data: dict, df: pd.DataFrame, key: str):
    water_temp = _get_water_temp(obj, data, key)
    if water_temp is None:
        logger.info(f'{key} not defined. Using back-up method "VDI4655".')

        # Load data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        path = os.path.join(root_dir, 'entise', 'data', Types.DHW, 'shared', 'cold_water_temp_vdi4655.csv')
        water_temp_data = pd.read_csv(path)

        # Extract months from datetime index and match with water temperature data
        months = pd.Series(df.index.month, index=df.index)
        water_temp = months.map(water_temp_data.set_index(C.MONTH)[O.TEMP_WATER_COLD])

    return water_temp


def _get_water_temp_hot(obj: dict, data: dict, key: str):
    water_temp = _get_water_temp(obj, data, key)
    if water_temp is None:
        logger.info(f'{key} not defined. Using back-up method "constant value".')
        water_temp = defaults.DEFAULT_TEMP_HOT

    return water_temp


def _get_water_temperatures(obj, data, weather):
    """
    Get cold water temperature.

    Parameters:
    -----------
    obj : dict
        Object parameters
    weather : pd.DataFrame
        Weather data

    Returns:
    --------
    float
        Cold water temperature
    """
    df = pd.DataFrame(index=weather[C.DATETIME], columns=[O.TEMP_WATER_COLD, O.TEMP_WATER_HOT])
    df.index = pd.to_datetime(df.index)
    df[O.TEMP_WATER_COLD] = _get_water_temp_cold(obj, data, df, O.TEMP_WATER_COLD)
    df[O.TEMP_WATER_HOT] = _get_water_temp_hot(obj, data, O.TEMP_WATER_HOT)
    return df


def _calculate_timeseries(weather, activity_data, daily_demand, water_temp, obj):
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
        Weather data with datetime index
    activity_data : pd.DataFrame
        Activity data with columns ['day', 'time', 'event', 'probability', 'duration', 'flow_rate',
        'sigma_duration', 'sigma_flow_rate']
    daily_demand : float or pd.Series
        Daily demand in liters. Used to calculate annual volumes for each event type.
    water_temp : pd.DataFrame
        Cold and hot water temperature in °C for every time step
    obj : dict
        Object parameters
    chunk_size : int, optional
        Not used in this implementation, kept for backward compatibility.

    Returns:
    --------
    tuple
        (volume_timeseries, energy_timeseries)
    """
    # Get parameters
    temp_hot = water_temp[O.TEMP_WATER_HOT]
    temp_cold = water_temp[O.TEMP_WATER_COLD]
    seed = obj.get(O.SEED, None)
    rng = np.random.default_rng(seed)

    # Create index
    index = pd.to_datetime(weather[C.DATETIME])

    # Calculate simulation duration in years
    start_date = index.iat[0].date()
    end_date = index.iat[-1].date()
    days_in_simulation = (end_date - start_date).days + 1
    years_in_simulation = days_in_simulation / 365.0

    # Determine weekdays
    days = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')})
    days['weekday'] = days['date'].dt.weekday

    # Cross-join only matching weekday rows
    # `activity_data.day` is 0=Mon…6=Sun
    events = days.merge(
        activity_data,
        left_on='weekday',
        right_on='day',
        how='inner'
    )

    # Include seasonal factor (sine wave)
    seasonal_var = obj.get(O.SEASONAL_VARIATION, defaults.DEFAULT_SEASONAL_VARIATION)
    seasonal_peak = obj.get(O.SEASONAL_PEAK_DAY, defaults.DEFAULT_SEASONAL_PEAK_DAY)
    events['dayofyear'] = events['date'].dt.dayofyear
    events['seasonal_factor'] = (1 + seasonal_var * np.cos(2 * np.pi * (events['dayofyear'] - seasonal_peak) / 365))

    # Holiday suppression (0 on holiday, 1 otherwise)
    location = obj.get(O.HOLIDAYS_LOCATION, [])
    if isinstance(location, str):
        location = location.split(',')
        country = location[-1]
        subdiv = location[0] if len(location) > 1 else None
        holiday_dates = set(holidays.country_holidays(country=country, subdiv=subdiv, years=index.dt.year.unique()).keys())
    else:
        holiday_dates = set()

    # Flag holiday rows and override their 'day' to Sunday (6)
    is_hol = events['date'].dt.date.isin(holiday_dates)
    events.loc[is_hol, 'day'] = 6

    # Re‐assign probability_day from the original activity_data
    pd_map = activity_data.set_index(['event', 'day', 'time'])['probability_day']
    mi = pd.MultiIndex.from_frame(events[['event', 'day', 'time']])
    events['probability_day'] = pd_map.reindex(mi).values

    # Calculate annual volumes for each event type
    # If daily_demand is a Series, calculate the total annual demand
    if isinstance(daily_demand, pd.Series):
        # Filter to dates within the simulation period
        daily_demand_filtered = daily_demand[
            (daily_demand.index.date >= start_date) &
            (daily_demand.index.date <= end_date)
        ]
        total_annual_demand = daily_demand_filtered.sum() * 365.0 / days_in_simulation
    else:
        # If daily_demand is a scalar, multiply by days in a year
        total_annual_demand = daily_demand * 365.0

    # Calculate annual volumes for each event type based on probability_day
    # If probability_day is not available, distribute evenly
    annual_volumes = {}
    mean_flows = {}
    sigma_rel = {}

    # Group activity data by event type
    event_groups = events.groupby('event')

    # Check if probability_day column exists
    if 'probability_day' in activity_data.columns:
        # Calculate total probability_day across all event types
        total_prob_day = sum(group['probability_day'].iloc[0] for _, group in event_groups)

        # Distribute annual demand based on probability_day
        for event, group in event_groups:
            prob_day = group['probability_day'].iloc[0]
            annual_volumes[event] = total_annual_demand * prob_day / total_prob_day

            # Calculate mean flow and sigma for each event type
            mean_flows[event] = group['flow_rate'].iloc[0] * group['duration'].iloc[0]  # liters per event
            sigma_rel[event] = group['sigma_flow_rate'].iloc[0] / group['flow_rate'].iloc[0]  # relative sigma
    else:
        # If probability_day is not available, distribute evenly
        event_count = len(event_groups)
        for event, group in event_groups:
            annual_volumes[event] = total_annual_demand / event_count

            # Calculate mean flow and sigma for each event type
            mean_flows[event] = group['flow_rate'].iloc[0] * group['duration'].iloc[0]  # liters per event
            sigma_rel[event] = group['sigma_flow_rate'].iloc[0] / group['flow_rate'].iloc[0]  # relative sigma

    # Generate volume time series using vectorized inverse sampling
    ts_volume = _vectorized_inverse_sampling(
        events,
        annual_volumes,
        mean_flows,
        sigma_rel,
        daily_demand,
        index,
        rng
    )

    ts_volume *= total_annual_demand/ts_volume.sum()

    # Calculate energy demand
    delta_t = temp_hot - temp_cold
    energy_factor = defaults.DEFAULT_DENSITY_WATER / 1000 * defaults.DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600  # Convert to Wh
    ts_energy = ts_volume * energy_factor

    return ts_volume.round(3), ts_energy.round().astype(int)
