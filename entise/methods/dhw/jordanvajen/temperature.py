"""
Water temperature handling functions for the Jordan & Vajen DHW method.

This module contains functions for handling water temperature data
used by the Jordan & Vajen DHW method.
"""

import logging
from typing import Any, Dict, Union

import pandas as pd

import entise.methods.dhw.defaults as defaults
from entise.constants import Columns as C
from entise.constants import Objects as O

logger = logging.getLogger(__name__)


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
    """Return the supplied cold-water temperature or the constant default."""
    water_temp = _get_water_temp(obj, data, key)
    if water_temp is None:
        water_temp = defaults.DEFAULT_TEMP_COLD

    if not isinstance(water_temp, pd.Series):
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
    if hasattr(weather[C.DATETIME].iloc[0], "tz") and weather[C.DATETIME].iloc[0].tz is not None:
        df.index = df.index.tz_convert(weather[C.DATETIME].iloc[0].tz)

    # Get cold and hot water temperatures
    df[O.TEMP_WATER_COLD] = _get_water_temp_cold(obj, data, df, O.TEMP_WATER_COLD)
    df[O.TEMP_WATER_HOT] = _get_water_temp_hot(obj, data, df, O.TEMP_WATER_HOT)

    # Validate temperatures
    if (df[O.TEMP_WATER_HOT] <= df[O.TEMP_WATER_COLD]).any():
        logger.warning("Hot water temperature is not greater than cold water temperature for some timestamps")

    logger.info(
        f"Water temperatures: cold={df[O.TEMP_WATER_COLD].mean():.1f}°C, hot={df[O.TEMP_WATER_HOT].mean():.1f}°C"
    )

    return df
