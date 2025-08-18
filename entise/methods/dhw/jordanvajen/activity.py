"""
Activity data handling functions for the Jordan & Vajen DHW method.

This module contains functions for loading and processing activity data
used by the Jordan & Vajen DHW method.
"""

import logging
import os

import pandas as pd

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.dhw.jordanvajen.utils import _get_data_path

logger = logging.getLogger(__name__)


def _get_activity_data(source: str, filename: str = "dhw_activity.csv") -> pd.DataFrame:
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
        required_columns = [C.DAY_OF_WEEK, C.TIME, C.EVENT, C.PROBABILITY, C.DURATION, C.FLOW_RATE]
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


def _get_demand_data(source: str, filename: str = "dhw_demand_by_dwelling.csv") -> pd.DataFrame:
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
        required_columns = [O.DWELLING_SIZE, O.YEARLY_DHW_DEMAND, O.SIGMA]
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
