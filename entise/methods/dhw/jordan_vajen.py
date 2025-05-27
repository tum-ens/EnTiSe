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
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import truncnorm
import holidays

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types
import entise.methods.dhw.defaults as defaults

logger = logging.getLogger(__name__)

# Constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
LITERS_PER_M3 = 1000.0
JOULES_TO_WATT_HOURS = 1/3600.0
MAX_TIME_DIFF_MINUTES = 360  # Maximum time difference for timestamp-centric approach (6 hours)
DEFAULT_CHUNK_SIZE_DAYS = 365  # Default chunk size for processing large datasets (days)
DEFAULT_CHUNK_SIZE_HOURS = DEFAULT_CHUNK_SIZE_DAYS * 24  # Default chunk size for timestamp processing

# Internal strings for this method
SEASONAL_FACTOR = "seasonal_factor"


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
    required_keys = [O.DATETIMES, O.DWELLING_SIZE]
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
    required_timeseries = [O.DATETIMES]
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

    def generate(self, obj, data, ts_type: str = Types.DHW):
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
        datetimes = obj[O.DATETIMES]
        datetimes = data[datetimes]
        datetimes[C.DATETIME] = (pd.to_datetime(datetimes[C.DATETIME], utc=True).dt
                                 .tz_convert(pd.to_datetime(datetimes[C.DATETIME].iloc[0]).tz))
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
        start = datetimes[C.DATETIME].iloc[0].normalize()
        end = datetimes[C.DATETIME].iloc[-1].normalize()
        days = pd.date_range(start, end, freq='D')

        # 5) sample once **per day** from the truncated normal
        dist = stats.truncnorm(a, b, loc=mean_daily_l, scale=sd_daily_l)
        daily_demand_l = pd.Series(dist.rvs(size=len(days), random_state=rng), index=days)

        # Get cold water temperature
        water_temp = _get_water_temperatures(obj, data, datetimes)

        # Generate time series
        ts_volume, ts_energy = _calculate_timeseries(
            datetimes, activity_data, daily_demand_l, water_temp, obj
        )

        # Convert energy into power
        # Calculate time interval in hours
        time_diff = pd.Series(datetimes[C.DATETIME]).diff().dt.total_seconds() / 3600
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
        }, index=datetimes[C.DATETIME])

        return {
            "summary": summary,
            "timeseries": timeseries
        }


def _convert_time_to_seconds_of_day(time_strings: pd.Series) -> np.ndarray:
    """
    Convert time strings to seconds of day.

    This function converts time strings in the format "HH:MM:SS" or "HH:MM" to seconds of day.

    Parameters:
    -----------
    time_strings : pd.Series
        Series of time strings in the format "HH:MM:SS" or "HH:MM"

    Returns:
    --------
    np.ndarray
        Array of seconds of day
    """
    t_parts = time_strings.str.split(':', expand=True).astype(int)
    return (
        t_parts[0] * SECONDS_PER_HOUR +
        t_parts[1] * SECONDS_PER_MINUTE +
        t_parts.get(2, 0)  # Handle case where seconds are not provided
    ).values


def _find_nearest_activity_times(activity_times: np.ndarray, seconds_of_day: np.ndarray) -> np.ndarray:
    """
    Find the nearest activity time for each timestamp.

    This function finds the nearest activity time for each timestamp using binary search.

    Parameters:
    -----------
    activity_times : np.ndarray
        Array of activity times in seconds of day
    seconds_of_day : np.ndarray
        Array of timestamp times in seconds of day

    Returns:
    --------
    np.ndarray
        Array of indices into activity_times for each timestamp
    """
    # Sort activity times for searchsorted
    order = np.argsort(activity_times)
    times_sorted = activity_times[order]

    # Find nearest activity slot for every timestamp
    pos = np.searchsorted(times_sorted, seconds_of_day, side='left')
    left = np.clip(pos - 1, 0, len(times_sorted) - 1)
    right = np.clip(pos, 0, len(times_sorted) - 1)

    # Determine whether left or right index is closer
    left_diff = np.abs(seconds_of_day - times_sorted[left])
    right_diff = np.abs(seconds_of_day - times_sorted[right])
    choose_right = right_diff < left_diff

    # Return the index of the nearest activity time
    return np.where(choose_right, right, left), order


def _sample_event_volumes(
    mean_flow: float,
    sigma_relative: float,
    num_events: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample event volumes from a truncated normal distribution.

    This function samples event volumes from a truncated normal distribution
    with mean = mean_flow and standard deviation = mean_flow * sigma_relative.

    Parameters:
    -----------
    mean_flow : float
        Mean flow per event in liters
    sigma_relative : float
        Relative standard deviation (sigma/mean)
    num_events : int
        Number of events to sample
    rng : np.random.Generator
        Random number generator

    Returns:
    --------
    np.ndarray
        Array of sampled volumes
    """
    mu = mean_flow
    sigma = mu * sigma_relative

    # Ensure non-negative volumes
    a, b = (0 - mu) / sigma, np.inf

    # Create truncated normal distribution
    dist = truncnorm(a, b, loc=mu, scale=sigma)

    # Sample volumes
    return dist.rvs(size=num_events, random_state=rng)


def _calculate_volume_timeseries(
    activity_data: pd.DataFrame,
    annual_volumes: Dict[str, float],
    mean_flows: Dict[str, float],
    sigma_rel: Dict[str, float],
    daily_demand: Union[float, pd.Series],
    index: pd.DatetimeIndex,
    rng: np.random.Generator
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
    seconds_of_day = (
        index.dt.hour * SECONDS_PER_HOUR +
        index.dt.minute * SECONDS_PER_MINUTE +
        index.dt.second
    )

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
        logger.debug(f"Event type '{event}': {N} events, annual volume={annual_volumes[event]:.1f} L, "
                    f"mean flow={mean_flows[event]:.1f} L/event")

        # Skip if no events
        if N <= 0:
            logger.warning(f"No events for event type '{event}' (N={N})")
            continue

        # Convert activity times to seconds of day
        activity_times = _convert_time_to_seconds_of_day(df_ev[C.TIME])

        # Apply probability_day × seasonal × holiday weighting before normalization
        weighted_probs = (
            df_ev[C.PROBABILITY].values *
            df_ev[C.PROBABILITY_DAY].values *
            df_ev[SEASONAL_FACTOR].values
        )

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
        positions = np.searchsorted(cdf, r, side='right')

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
    logger.debug(f"Created time series with {len(result)} points, "
                f"total volume={result.sum():.1f} L, "
                f"non-zero points={(result > 0).sum()} ({(result > 0).sum()/len(result)*100:.1f}%)")

    return result


def _get_data_path(source: str, filename: str, subdir: str = Types.DHW) -> str:
    """
    Get the full path to a data file.

    This utility function constructs the full path to a data file based on the source directory,
    filename, and optional subdirectory.

    Parameters:
    -----------
    source : str
        Source directory (e.g., 'jordan_vajen')
    filename : str
        Filename (e.g., 'dhw_activity.csv')
    subdir : str, optional
        Subdirectory within the data directory, defaults to Types.DHW

    Returns:
    --------
    str
        Full path to the data file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    return os.path.join(root_dir, 'entise', 'data', subdir, source, filename)


def _get_activity_data(source: str, filename: str = 'dhw_activity.csv') -> pd.DataFrame:
    """
    Get activity data from filename.

    This function loads DHW activity data from a CSV file. The activity data contains information
    about different DHW events (shower, bath, etc.) and their characteristics (probability, duration,
    flow rate, etc.).

    Parameters:
    -----------
    source : str
        Source directory (e.g., 'jordan_vajen')
    filename : str
        Filename (e.g., 'dhw_activity.csv')

    Returns:
    --------
    pd.DataFrame
        Activity data with columns:
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - time: Time of day (HH:MM:SS)
        - event: Event type (e.g., 'Shower', 'Bath', 'Medium', 'Small')
        - probability: Probability of event occurring at this time
        - duration: Duration of event in seconds
        - flow_rate: Flow rate in liters/second
        - duration_sigma: Standard deviation of duration in seconds
        - flow_rate_sigma: Standard deviation of flow rate in liters/second
        - probability_day: Probability of event occurring on this day

    Raises:
    -------
    FileNotFoundError
        If the activity data file is not found
    ValueError
        If the activity data file has an invalid format or is missing required columns
    """
    try:
        path = _get_data_path(source, filename)
        logger.debug(f"Loading activity data from {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Activity data file not found: {path}")

        data = pd.read_csv(path)

        # Validate required columns
        required_columns = ['day_of_week', 'time', 'event', 'probability', 'duration', 'flow_rate']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Activity data missing required columns: {missing_columns}")

        logger.info(f"Loaded activity data: {len(data)} rows, {len(data['event'].unique())} event types")
        return data

    except pd.errors.EmptyDataError:
        logger.error(f"Activity data file is empty: {path}")
        raise ValueError(f"Activity data file is empty: {path}")

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing activity data file: {str(e)}")
        raise ValueError(f"Error parsing activity data file: {str(e)}")

    except Exception as e:
        logger.error(f"Error loading activity data: {str(e)}")
        raise


def _get_demand_data(source: str, filename: str = 'dhw_demand_by_dwelling.csv') -> pd.DataFrame:
    """
    Get demand data from filename.

    This function loads DHW demand data from a CSV file. The demand data contains information
    about DHW demand for different dwelling sizes.

    Parameters:
    -----------
    source : str
        Source directory (e.g., 'jordan_vajen')
    filename : str
        Filename (e.g., 'dhw_demand_by_dwelling.csv')

    Returns:
    --------
    pd.DataFrame
        Demand data with columns:
        - dwelling_size: Dwelling size in m²
        - m3_per_m2_a: Annual DHW demand in m³ per m² per year
        - sigma: Standard deviation of daily demand as a fraction of mean daily demand

    Raises:
    -------
    FileNotFoundError
        If the demand data file is not found
    ValueError
        If the demand data file has an invalid format or is missing required columns
    """
    try:
        path = _get_data_path(source, filename)
        logger.debug(f"Loading demand data from {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Demand data file not found: {path}")

        data = pd.read_csv(path)

        # Validate required columns
        required_columns = ['dwelling_size', 'm3_per_m2_a', 'sigma']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Demand data missing required columns: {missing_columns}")

        logger.info(f"Loaded demand data: {len(data)} dwelling sizes")
        return data

    except pd.errors.EmptyDataError:
        logger.error(f"Demand data file is empty: {path}")
        raise ValueError(f"Demand data file is empty: {path}")

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing demand data file: {str(e)}")
        raise ValueError(f"Error parsing demand data file: {str(e)}")

    except Exception as e:
        logger.error(f"Error loading demand data: {str(e)}")
        raise


def _get_water_temp(obj: Dict[str, Any], data: Dict[str, Any], key: str) -> Union[float, pd.Series, None]:
    """
    Get water temperature from object parameters or data.

    This function retrieves water temperature from either object parameters or data.
    If the temperature is specified as a string in the object parameters, it is used
    as a key to retrieve the temperature from the data.

    Parameters:
    -----------
    obj : Dict[str, Any]
        Object parameters
    data : Dict[str, Any]
        Input data
    key : str
        Key for water temperature (e.g., O.TEMP_WATER_COLD, O.TEMP_WATER_HOT)

    Returns:
    --------
    Union[float, pd.Series, None]
        Water temperature as a float or Series, or None if not found
    """
    water_temp = obj.get(key, None)
    logger.debug(f"Getting water temperature for key '{key}': {water_temp}")

    if isinstance(water_temp, str):
        if water_temp not in data:
            logger.warning(f"Water temperature key '{water_temp}' not found in data")
            return None
        water_temp = data.get(water_temp, None)
        logger.debug(f"Retrieved water temperature from data: {type(water_temp)}")

    return water_temp


def _get_water_temp_cold(obj: Dict[str, Any], data: Dict[str, Any], df: pd.DataFrame, key: str) -> pd.Series:
    """
    Get cold water temperature.

    This function retrieves cold water temperature from object parameters or data.
    If not found, it uses a fallback method based on the VDI4655 standard.

    Parameters:
    -----------
    obj : Dict[str, Any]
        Object parameters
    data : Dict[str, Any]
        Input data
    df : pd.DataFrame
        DataFrame with datetime index
    key : str
        Key for cold water temperature (e.g., O.TEMP_WATER_COLD)

    Returns:
    --------
    pd.Series
        Cold water temperature for each timestamp in df

    Raises:
    -------
    FileNotFoundError
        If the cold water temperature data file is not found
    ValueError
        If the cold water temperature data file has an invalid format
    """
    water_temp = _get_water_temp(obj, data, key)

    if water_temp is None:
        logger.info(f"'{key}' not defined. Using fallback method 'VDI4655'.")

        try:
            # Load VDI4655 cold water temperature data
            path = _get_data_path('shared', 'cold_water_temp_vdi4655.csv')
            logger.debug(f"Loading cold water temperature data from {path}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Cold water temperature data file not found: {path}")

            water_temp_data = pd.read_csv(path)

            # Validate required columns
            if C.MONTH not in water_temp_data.columns or O.TEMP_WATER_COLD not in water_temp_data.columns:
                raise ValueError(f"Cold water temperature data missing required columns: {C.MONTH} or {O.TEMP_WATER_COLD}")

            # Extract months from datetime index and match with water temperature data
            months = pd.Series(df.index.month, index=df.index)
            water_temp = months.map(water_temp_data.set_index(C.MONTH)[O.TEMP_WATER_COLD])

            # Check if any NaN values in the result
            if water_temp.isna().any():
                logger.warning(f"Some cold water temperatures could not be determined. Using default value {defaults.DEFAULT_TEMP_COLD}°C.")
                water_temp = water_temp.fillna(defaults.DEFAULT_TEMP_COLD)

            logger.info(f"Using VDI4655 cold water temperatures: min={water_temp.min():.1f}°C, max={water_temp.max():.1f}°C")

        except Exception as e:
            logger.error(f"Error getting cold water temperature: {str(e)}. Using default value {defaults.DEFAULT_TEMP_COLD}°C.")
            water_temp = pd.Series(defaults.DEFAULT_TEMP_COLD, index=df.index)

    # Convert to Series if it's a scalar
    if not isinstance(water_temp, pd.Series):
        logger.debug(f"Converting cold water temperature from {type(water_temp)} to Series")
        water_temp = pd.Series(water_temp, index=df.index)

    return water_temp


def _get_water_temp_hot(obj: Dict[str, Any], data: Dict[str, Any], df: pd.DataFrame, key: str) -> pd.Series:
    """
    Get hot water temperature.

    This function retrieves hot water temperature from object parameters or data.
    If not found, it uses a default constant value.

    Parameters:
    -----------
    obj : Dict[str, Any]
        Object parameters
    data : Dict[str, Any]
        Input data
    df : pd.DataFrame
        DataFrame with datetime index
    key : str
        Key for hot water temperature (e.g., O.TEMP_WATER_HOT)

    Returns:
    --------
    pd.Series
        Hot water temperature for each timestamp in df
    """
    water_temp = _get_water_temp(obj, data, key)

    if water_temp is None:
        logger.info(f"'{key}' not defined. Using default constant value {defaults.DEFAULT_TEMP_HOT}°C.")
        water_temp = defaults.DEFAULT_TEMP_HOT

    # Convert to Series if it's a scalar
    if not isinstance(water_temp, pd.Series):
        logger.debug(f"Converting hot water temperature from {type(water_temp)} to Series")
        water_temp = pd.Series(water_temp, index=df.index)

    return water_temp


def _get_water_temperatures(obj: Dict[str, Any], data: Dict[str, Any], weather: pd.DataFrame) -> pd.DataFrame:
    """
    Get water temperatures (cold and hot).

    This function retrieves cold and hot water temperatures for each timestamp in the weather data.

    Parameters:
    -----------
    obj : Dict[str, Any]
        Object parameters
    data : Dict[str, Any]
        Input data
    weather : pd.DataFrame
        Weather data with datetime column

    Returns:
    --------
    pd.DataFrame
        DataFrame with cold and hot water temperatures for each timestamp
        Columns: [O.TEMP_WATER_COLD, O.TEMP_WATER_HOT]
        Index: Same as weather[C.DATETIME]

    Raises:
    -------
    ValueError
        If the weather data does not have a datetime column
    """
    if C.DATETIME not in weather.columns:
        raise ValueError(f"Weather data missing required column: {C.DATETIME}")

    logger.debug(f"Getting water temperatures for {len(weather)} timestamps")

    # Create DataFrame with same index as weather
    df = pd.DataFrame(index=weather[C.DATETIME], columns=[O.TEMP_WATER_COLD, O.TEMP_WATER_HOT])

    # Convert index to datetime with consistent timezone
    df.index = pd.to_datetime(df.index, utc=True)
    if hasattr(weather[C.DATETIME].iloc[0], 'tz') and weather[C.DATETIME].iloc[0].tz is not None:
        df.index = df.index.tz_convert(weather[C.DATETIME].iloc[0].tz)

    # Get cold and hot water temperatures
    df[O.TEMP_WATER_COLD] = _get_water_temp_cold(obj, data, df, O.TEMP_WATER_COLD)
    df[O.TEMP_WATER_HOT] = _get_water_temp_hot(obj, data, df, O.TEMP_WATER_HOT)

    # Validate temperatures
    if (df[O.TEMP_WATER_HOT] <= df[O.TEMP_WATER_COLD]).any():
        logger.warning("Hot water temperature is not greater than cold water temperature for some timestamps")

    logger.info(f"Water temperatures: cold={df[O.TEMP_WATER_COLD].mean():.1f}°C, hot={df[O.TEMP_WATER_HOT].mean():.1f}°C")

    return df


def _calculate_timeseries(
    weather: pd.DataFrame,
    activity_data: pd.DataFrame,
    daily_demand: Union[float, pd.Series],
    water_temp: pd.DataFrame,
    obj: Dict[str, Any]
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
    logger.info(f"Simulation period: {start_date} to {end_date} ({days_in_simulation} days, {years_in_simulation:.2f} years)")

    # Determine weekdays
    days = pd.DataFrame({C.DATE: pd.date_range(start_date, end_date, freq='D')})
    days[C.DAY_OF_WEEK] = days[C.DATE].dt.weekday
    logger.debug(f"Created calendar with {len(days)} days")

    # Cross-join only matching weekday rows
    # `activity_data.day` is 0=Mon…6=Sun
    events = days.merge(
        activity_data,
        left_on=C.DAY_OF_WEEK,
        right_on=C.DAY_OF_WEEK,
        how='inner'
    )
    logger.debug(f"Created events DataFrame with {len(events)} rows after day-of-week matching")

    # Include seasonal factor (sine wave)
    seasonal_var = obj.get(O.SEASONAL_VARIATION, defaults.DEFAULT_SEASONAL_VARIATION)
    seasonal_peak = obj.get(O.SEASONAL_PEAK_DAY, defaults.DEFAULT_SEASONAL_PEAK_DAY)
    events['dayofyear'] = events[C.DATE].dt.dayofyear
    events[SEASONAL_FACTOR] = (1 + seasonal_var * np.cos(2 * np.pi * (events['dayofyear'] - seasonal_peak) / DAYS_PER_YEAR))
    logger.info(f"Applied seasonal variation: amplitude={seasonal_var:.2f}, peak day={seasonal_peak}")

    # Holiday suppression (treat holidays as Sundays)
    location = obj.get(O.HOLIDAYS_LOCATION, [])
    if isinstance(location, str) and location.strip():
        location = location.split(',')
        country = location[-1].strip()
        subdiv = location[0].strip() if len(location) > 1 else None

        try:
            holiday_dates = set(holidays.country_holidays(
                country=country,
                subdiv=subdiv,
                years=index.dt.year.unique()
            ).keys())
            logger.info(f"Applied holiday suppression for {country}{f', {subdiv}' if subdiv else ''}: {len(holiday_dates)} holidays")
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
            (daily_demand.index.date >= start_date) &
            (daily_demand.index.date <= end_date)
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

            logger.info(f"Event type '{event}': probability_day={prob_day:.4f}, annual volume={annual_volumes[event]:.1f} L, "
                       f"mean flow={mean_flows[event]:.1f} L/event, sigma_rel={sigma_rel[event]:.4f}")
    else:
        # If probability_day is not available, distribute evenly
        event_count = len(event_groups)
        logger.warning(f"probability_day column not found in activity data. Distributing demand evenly across {event_count} event types.")

        for event, group in event_groups:
            annual_volumes[event] = total_annual_demand / event_count

            # Calculate mean flow and sigma for each event type
            mean_flows[event] = group[C.FLOW_RATE].iloc[0] * group[C.DURATION].iloc[0]  # liters per event
            sigma_rel[event] = group[C.FLOW_RATE_SIGMA].iloc[0] / group[C.FLOW_RATE].iloc[0]  # relative sigma

            logger.info(f"Event type '{event}': annual volume={annual_volumes[event]:.1f} L, "
                       f"mean flow={mean_flows[event]:.1f} L/event, sigma_rel={sigma_rel[event]:.4f}")

    # Generate volume time series using vectorized inverse sampling
    logger.info("Generating volume time series using vectorized inverse sampling")
    ts_volume = _calculate_volume_timeseries(
        events,
        annual_volumes,
        mean_flows,
        sigma_rel,
        daily_demand,
        index,
        rng
    )

    # Scale to match total annual demand
    ts_volume *= total_annual_demand/ts_volume.sum()
    logger.debug(f"Scaled volume time series to match total annual demand: {total_annual_demand:.1f} L")

    # Log volume statistics
    volume_stats = {
        'total': ts_volume.sum(),
        'mean': ts_volume.mean(),
        'max': ts_volume.max(),
        'min': ts_volume.min(),
        'non_zero': (ts_volume > 0).sum(),
        'zero': (ts_volume == 0).sum()
    }
    logger.info(f"Volume time series statistics: total={volume_stats['total']:.1f} L, "
               f"mean={volume_stats['mean']:.3f} L, max={volume_stats['max']:.3f} L, "
               f"non-zero points: {volume_stats['non_zero']} ({volume_stats['non_zero']/len(ts_volume)*100:.1f}%)")

    # Calculate energy demand
    delta_t = temp_hot - temp_cold
    energy_factor = defaults.DEFAULT_DENSITY_WATER / 1000 * defaults.DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600  # Convert to Wh
    ts_energy = ts_volume * energy_factor

    # Log energy statistics
    energy_stats = {
        'total': ts_energy.sum(),
        'mean': ts_energy.mean(),
        'max': ts_energy.max(),
        'min': ts_energy.min()
    }
    logger.info(f"Energy time series statistics: total={energy_stats['total']:.1f} Wh, "
               f"mean={energy_stats['mean']:.3f} Wh, max={energy_stats['max']:.3f} Wh")

    return ts_volume.round(3), ts_energy.round().astype(int)
