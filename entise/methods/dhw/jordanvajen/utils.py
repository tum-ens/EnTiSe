"""
Utility functions for the Jordan & Vajen DHW method.

This module contains general utility functions used by the Jordan & Vajen DHW method.
"""

import os

import numpy as np
import pandas as pd

# Constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
LITERS_PER_M3 = 1000.0
JOULES_TO_WATT_HOURS = 1 / 3600.0
MAX_TIME_DIFF_MINUTES = 360  # Maximum time difference for timestamp-centric approach (6 hours)
DEFAULT_CHUNK_SIZE_DAYS = 365  # Default chunk size for processing large datasets (days)
DEFAULT_CHUNK_SIZE_HOURS = DEFAULT_CHUNK_SIZE_DAYS * 24  # Default chunk size for timestamp processing

# Internal strings for this method
SEASONAL_FACTOR = "seasonal_factor"


def _get_data_path(source: str, filename: str, subdir: str = "dhw") -> str:
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
        Subdirectory within the data directory, defaults to "dhw"

    Returns:
    --------
    str
        Full path to the data file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
    return os.path.join(root_dir, "entise", "data", subdir, source, filename)


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
    t_parts = time_strings.str.split(":", expand=True).astype(int)
    return (
        t_parts[0] * SECONDS_PER_HOUR
        + t_parts[1] * SECONDS_PER_MINUTE
        + t_parts.get(2, 0)  # Handle case where seconds are not provided
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
    pos = np.searchsorted(times_sorted, seconds_of_day, side="left")
    left = np.clip(pos - 1, 0, len(times_sorted) - 1)
    right = np.clip(pos, 0, len(times_sorted) - 1)

    # Determine whether left or right index is closer
    left_diff = np.abs(seconds_of_day - times_sorted[left])
    right_diff = np.abs(seconds_of_day - times_sorted[right])
    choose_right = right_diff < left_diff

    # Return the index of the nearest activity time
    return np.where(choose_right, right, left), order


def _sample_event_volumes(
    mean_flow: float, sigma_relative: float, num_events: int, rng: np.random.Generator
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
    from scipy.stats import truncnorm

    mu = mean_flow
    sigma = mu * sigma_relative

    # Ensure non-negative volumes
    a, b = (0 - mu) / sigma, np.inf

    # Create truncated normal distribution
    dist = truncnorm(a, b, loc=mu, scale=sigma)

    # Sample volumes
    return dist.rvs(size=num_events, random_state=rng)
