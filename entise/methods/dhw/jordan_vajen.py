"""DHW generation module based on the Jordan & Vajen methodology.

This module implements a domestic hot water (DHW) demand generation method based on
Jordan & Vajen (2005): "DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES
WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS". The implementation follows the
Method pattern established in the project architecture.

The module provides functionality to:
- Process input parameters for DHW demand calculation
- Generate daily demand values based on dwelling size
- Calculate DHW demand time series based on activity profiles
- Compute summary statistics for the generated time series

The main class, JordanVajen, inherits from the Method base class and implements the
required interface for integration with the EnTiSe framework.

Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER
PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS. Universität Marburg.
URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats as stats

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method
from entise.methods.dhw.defaults import (
    DEFAULT_SEASONAL_PEAK_DAY,
    DEFAULT_SEASONAL_VARIATION,
    DEFAULT_TEMP_COLD,
    DEFAULT_TEMP_HOT,
)
from entise.methods.dhw.jordanvajen.activity import _get_activity_data, _get_demand_data
from entise.methods.dhw.jordanvajen.calculation import _calculate_timeseries
from entise.methods.dhw.jordanvajen.temperature import _get_water_temperatures

logger = logging.getLogger(__name__)


class JordanVajen(Method):
    """Implements a DHW demand generation method based on Jordan & Vajen methodology.

    This class provides functionality to generate domestic hot water (DHW) demand
    time series based on dwelling size and activity profiles. It uses the Jordan & Vajen
    (2005) methodology to model DHW demand, taking into account factors such as
    seasonal variations, daily activity patterns, and water temperatures.

    The class follows the Method pattern defined in the EnTiSe framework, implementing
    the required interface for time series generation methods.

    Attributes:
        types (list): List of time series types this method can generate (DHW only).
        name (str): Name identifier for the method.
        required_keys (list): Required input parameters (datetimes, dwelling_size).
        optional_keys (list): Optional input parameters (activity data, temperatures, etc.).
        required_timeseries (list): Required time series inputs (datetimes).
        optional_timeseries (list): Optional time series inputs (activity data, temperatures).
        output_summary (dict): Mapping of output summary keys to descriptions.
        output_timeseries (dict): Mapping of output time series keys to descriptions.

    Example:
        >>> from entise.methods.dhw.jordan_vajen import JordanVajen
        >>> from entise.core.generator import TimeSeriesGenerator
        >>>
        >>> # Create a generator and add objects
        >>> gen = TimeSeriesGenerator()
        >>> gen.add_objects(objects_df)  # DataFrame with dwelling parameters
        >>>
        >>> # Generate time series
        >>> summary, timeseries = gen.generate(data)  # data contains datetime information
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
        f"{C.DEMAND}_{Types.DHW}_volume_total": "total hot water demand in liters",
        f"{C.DEMAND}_{Types.DHW}_volume_avg": "average hot water demand in liters",
        f"{C.DEMAND}_{Types.DHW}_volume_peak": "peak hot water demand in liters",
        f"{C.DEMAND}_{Types.DHW}_energy_total": "total energy demand for hot water in Wh",
        f"{C.DEMAND}_{Types.DHW}_energy_avg": "average energy demand for hot water in Wh",
        f"{C.DEMAND}_{Types.DHW}_energy_peak": "peak energy demand for hot water in Wh",
        f"{C.DEMAND}_{Types.DHW}_power_avg": "average power for hot water in W",
        f"{C.DEMAND}_{Types.DHW}_power_max": "maximum power for hot water in W",
        f"{C.DEMAND}_{Types.DHW}_power_min": "minimum power for hot water in W",
    }
    output_timeseries = {
        f"{C.LOAD}_{Types.DHW}_volume": "hot water demand in liters",
        f"{C.LOAD}_{Types.DHW}_energy": "energy demand for hot water in Wh",
        f"{C.LOAD}_{Types.DHW}_power": "power demand for hot water in W",
        f"{Types.DHW}_{O.TEMP_WATER_COLD}": "cold water temperature in degrees Celsius",
        f"{Types.DHW}_{O.TEMP_WATER_HOT}": "hot water temperature in degrees Celsius",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        ts_type: str = Types.DHW,
        *,
        datetimes: pd.DataFrame = None,
        dwelling_size: float = None,
        dhw_activity: pd.DataFrame = None,
        dhw_demand_per_size: pd.DataFrame = None,
        holidays_location: str = None,
        temp_water_cold: float = None,
        temp_water_hot: float = None,
        seasonal_variation: float = None,
        seasonal_peak_day: int = None,
        seed: int = None,
    ):
        """Generate DHW demand time series based on input parameters.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the DHW demand time series,
        and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing DHW system parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            ts_type (str, optional): Time series type to generate. Defaults to Types.DHW.
            datetimes (pd.DataFrame, required): DataFrame with datetime information. Defaults to None.
            dwelling_size (float, required): Size of the dwelling in square meters. Defaults to None.
            dhw_activity (pd.DataFrame, optional): Activity profiles for DHW demand. Defaults to None.
            dhw_demand_per_size (pd.DataFrame, optional): Demand data per dwelling size. Defaults to None.
            holidays_location (str, optional): Location for holiday calendar. Defaults to None.
            temp_water_cold (float, optional): Cold water temperature in degrees Celsius. Defaults to None.
            temp_water_hot (float, optional): Hot water temperature in degrees Celsius. Defaults to None.
            seasonal_variation (float, optional): Seasonal variation factor. Defaults to None.
            seasonal_peak_day (int, optional): Day of year with peak demand. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            dict: Dictionary containing:
                - "summary" (dict): Summary statistics including total demand,
                  average demand, and peak demand.
                - "timeseries" (pd.DataFrame): Time series of DHW demand
                  with timestamps as index.

        Raises:
            Exception: If required data is missing or invalid.

        Example:
            >>> jordanvajen = JordanVajen()
            >>> # Using explicit parameters
            >>> result = jordanvajen.generate(datetimes=datetimes_df, dwelling_size=100)
            >>> # Or using dictionaries
            >>> obj = {"datetimes": "datetimes", "dwelling_size": 100}
            >>> data = {"datetimes": datetimes_df}
            >>> result = jordanvajen.generate(obj=obj, data=data)
            >>> summary = result["summary"]
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            datetimes=datetimes,
            dwelling_size=dwelling_size,
            dhw_activity=dhw_activity,
            dhw_demand_per_size=dhw_demand_per_size,
            holidays_location=holidays_location,
            temp_water_cold=temp_water_cold,
            temp_water_hot=temp_water_hot,
            seasonal_variation=seasonal_variation,
            seasonal_peak_day=seasonal_peak_day,
            seed=seed,
        )

        processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

        ts_volume, ts_energy, ts_power = calculate_timeseries(processed_obj, processed_data)

        logger.debug(f"[DHW jordanvajen]: Generating {ts_type} data")

        return format_output(processed_data, ts_volume, ts_energy, ts_power)


def get_input_data(obj, data, method_type=Types.DHW):
    """Process and validate input data for DHW demand calculation.

    This function extracts required and optional parameters from the input dictionaries,
    applies default values where needed, performs data validation, and prepares the
    data for DHW demand calculation.

    Args:
        obj (dict): Dictionary containing DHW parameters such as dwelling size
            and temperature settings.
        data (dict): Dictionary containing input data such as datetime information
            and activity profiles.
        method_type (str, optional): Method type to use for prefixing. Defaults to Types.DHW.

    Returns:
        tuple: A tuple containing:
            - obj_out (dict): Processed object parameters with defaults applied.
            - data_out (dict): Processed data with required format for calculation.

    Raises:
        KeyError: If required parameters are missing.
        Exception: If required datetime data is missing.

    Notes:
        - Parameters can be specified with method-specific prefixes (e.g., "dhw:temp_water_cold")
          which will take precedence over generic parameters (e.g., "temp_water_cold").
        - Missing temperature values are automatically set to defaults.
        - Activity data is loaded from default sources if not provided.
    """
    # Check for required parameters
    dwelling_size_key = f"{method_type}:{O.DWELLING_SIZE}" if method_type else O.DWELLING_SIZE
    if O.DWELLING_SIZE not in obj and dwelling_size_key not in obj:
        logger.error(f"[DHW jordanvajen]: Missing required parameter {O.DWELLING_SIZE}")
        raise KeyError(f"Required parameter '{O.DWELLING_SIZE}' not found in object parameters")

    datetimes_key = f"{method_type}:{O.DATETIMES}" if method_type else O.DATETIMES
    if O.DATETIMES not in obj and datetimes_key not in obj:
        logger.error(f"[DHW jordanvajen]: Missing required parameter {O.DATETIMES}")
        raise KeyError(f"Required parameter '{O.DATETIMES}' not found in object parameters")

    # Process object parameters
    obj_out = {
        O.DWELLING_SIZE: Method.get_with_method_backup(obj, O.DWELLING_SIZE, method_type),
        O.DATETIMES: Method.get_with_method_backup(obj, O.DATETIMES, method_type),
        O.SEED: Method.get_with_method_backup(obj, O.SEED, method_type),
        O.TEMP_WATER_COLD: Method.get_with_method_backup(obj, O.TEMP_WATER_COLD, method_type, DEFAULT_TEMP_COLD),
        O.TEMP_WATER_HOT: Method.get_with_method_backup(obj, O.TEMP_WATER_HOT, method_type, DEFAULT_TEMP_HOT),
        O.SEASONAL_VARIATION: Method.get_with_method_backup(
            obj, O.SEASONAL_VARIATION, method_type, DEFAULT_SEASONAL_VARIATION
        ),
        O.SEASONAL_PEAK_DAY: Method.get_with_method_backup(
            obj, O.SEASONAL_PEAK_DAY, method_type, DEFAULT_SEASONAL_PEAK_DAY
        ),
        O.HOLIDAYS_LOCATION: Method.get_with_method_backup(obj, O.HOLIDAYS_LOCATION, method_type),
    }

    # Process data
    data_out = {
        O.DATETIMES: Method.get_with_backup(data, obj_out[O.DATETIMES]),
        O.DHW_ACTIVITY: Method.get_with_backup(data, O.DHW_ACTIVITY),
        O.DHW_DEMAND_PER_SIZE: Method.get_with_backup(data, O.DHW_DEMAND_PER_SIZE),
        O.TEMP_WATER_COLD: Method.get_with_backup(data, O.TEMP_WATER_COLD),
        O.TEMP_WATER_HOT: Method.get_with_backup(data, O.TEMP_WATER_HOT),
    }

    # Validate required data
    if data_out[O.DATETIMES] is None:
        logger.error("[DHW jordanvajen]: No datetime data")
        raise KeyError(f"Required data '{obj_out[O.DATETIMES]}' not found in input data")

    # Process datetime data
    datetimes = data_out[O.DATETIMES].copy()
    datetimes[C.DATETIME] = pd.to_datetime(datetimes[C.DATETIME], utc=True).dt.tz_convert(
        pd.to_datetime(datetimes[C.DATETIME].iloc[0]).tz
    )
    data_out[O.DATETIMES] = datetimes
    data_out["datetimes_index"] = datetimes[C.DATETIME]

    # Get activity data if not provided
    if data_out[O.DHW_ACTIVITY] is None:
        data_out[O.DHW_ACTIVITY] = _get_activity_data("jordan_vajen")

    # Create random number generator
    data_out["rng"] = np.random.default_rng(obj_out[O.SEED])

    return obj_out, data_out


def calculate_timeseries(obj, data):
    """Calculate DHW demand time series using the Jordan & Vajen methodology.

    This function generates daily demand values based on dwelling size and
    distributes them according to activity profiles to create a time series
    of DHW demand.

    Args:
        obj (dict): Dictionary containing processed DHW parameters such as:
            - dwelling_size: Size of the dwelling in square meters
            - temp_water_cold: Cold water temperature in degrees Celsius
            - temp_water_hot: Hot water temperature in degrees Celsius
            - seasonal_variation: Seasonal variation factor
            - seasonal_peak_day: Day of year with peak demand
        data (dict): Dictionary containing processed data such as:
            - datetimes: DataFrame with datetime information
            - dhw_activity: Activity profiles for DHW demand
            - rng: Random number generator

    Returns:
        tuple: A tuple containing:
            - ts_volume (pd.Series): Time series of DHW volume demand in liters
            - ts_energy (pd.Series): Time series of DHW energy demand in Wh
            - ts_power (pd.Series): Time series of DHW power demand in W

    Notes:
        - The function first calculates daily demand values based on dwelling size
        - It then distributes these values according to activity profiles
        - Finally, it calculates energy and power demand based on water temperatures
    """
    # Get parameters
    dwelling_size = obj[O.DWELLING_SIZE]
    datetimes = data[O.DATETIMES]
    activity_data = data[O.DHW_ACTIVITY]
    rng = data["rng"]

    # Obtain statistical data for yearly demand
    demand_data = _get_demand_data("jordan_vajen")
    sizes = demand_data["dwelling_size"].values
    idx = np.abs(sizes - dwelling_size).argmin()
    m3_per_m2_a = demand_data.iloc[idx]["m3_per_m2_a"]
    sigma = demand_data.iloc[idx]["sigma"]

    # Compute mean & std in litres/day
    mean_daily_l = m3_per_m2_a * dwelling_size * 1e3 / 365
    sd_daily_l = mean_daily_l * sigma

    # Define truncation bounds (no negatives)
    a, b = (0 - mean_daily_l) / sd_daily_l, np.inf

    # Build a date index for each simulation day
    start = datetimes[C.DATETIME].iloc[0].normalize()
    end = datetimes[C.DATETIME].iloc[-1].normalize()
    days = pd.date_range(start, end, freq="D")

    # Sample once per day from the truncated normal
    dist = stats.truncnorm(a, b, loc=mean_daily_l, scale=sd_daily_l)
    daily_demand_l = pd.Series(dist.rvs(size=len(days), random_state=rng), index=days)

    # Get water temperatures
    water_temp = _get_water_temperatures(obj, data, datetimes)

    # Generate volume and energy time series
    ts_volume, ts_energy = _calculate_timeseries(datetimes, activity_data, daily_demand_l, water_temp, obj)

    # Convert energy into power
    # Calculate time interval in hours
    time_diff = pd.Series(datetimes[C.DATETIME]).diff().dt.total_seconds() / 3600
    # Use the median time difference for the first element
    time_diff.iloc[0] = time_diff.median()
    # Calculate power as energy divided by time
    ts_power = ts_energy / time_diff.values

    return ts_volume, ts_energy, ts_power


def format_output(data, ts_volume, ts_energy, ts_power):
    """
    Format and postprocess the DHW demand time series into a structured output
    for reporting and downstream use.

    This function prepares both:
    1. A summary of key performance indicators (volume, energy, and power statistics),
    2. A time series DataFrame with raw and smoothed power demand values using
       three different smoothing techniques:
       - SMA (Simple Moving Average)
       - EWMA (Exponentially Weighted Moving Average)
       - Gaussian kernel smoothing

    These smoothed versions of the raw DHW power profile can be used to approximate
    the impact of thermal buffering in hot water storage tanks.

    Args:
        data (dict): A dictionary containing processed inputs, including:
            - 'datetimes_index' (pd.DatetimeIndex): The index to be used for the time series output.
        ts_volume (pd.Series): Time series of hot water demand in liters. Typically shaped using activity profiles.
        ts_energy (pd.Series): Time series of hot water energy demand in watt-hours (Wh), adjusted by water temperature.
        ts_power (pd.Series): Time series of instantaneous power demand in watts (W), derived from energy and timestep.

    Returns:
        dict:
            {
                "summary": dict with total, average, and peak demand metrics,
                "timeseries": pd.DataFrame with raw and smoothed DHW outputs
            }

    Internal Smoothing Methods:
        - create_sma_ts(ts, window=8):
            Applies a centered simple moving average over 8 samples (~2 hours).
            Fills edge NaNs with original raw series values.

        - create_ewma_ts(ts, alpha=0.1):
            Applies causal exponential smoothing with a decay factor α = 0.1,
            corresponding to a time constant τ ≈ 2 hours. Mimics reactive tank behavior.

        - create_gaussian_ts(ts, window=8, std=2):
            Applies symmetric Gaussian smoothing with std=2 over an 8-sample window (~2 hours).
            Mimics anticipatory or postprocessed smoothing as with idealized tanks.

    Notes:
        - All power time series (raw and smoothed) are rounded and converted to integer watts (W).
        - The smoothing methods are tuned based on typical tank behavior and timestep resolution (15 min).
        - Missing values from smoothing (at edges) are filled using the original raw signal for physical plausibility.
        - The volume and energy series remain unaltered.

    Example:
        >>> result = format_output(data, ts_volume, ts_energy, ts_power)
        >>> result['summary']
        {
            'dhw_volume_total': 1350,
            'dhw_volume_avg': 187.5,
            ...
        }
        >>> result['timeseries'].columns
        Index(['dhw_volume', 'dhw_energy', 'dhw_power',
               'dhw_power_sma', 'dhw_power_ewma', 'dhw_power_gaussian'],
              dtype='object')

    """

    def create_sma_ts(ts: pd.Series, window: int = 8) -> pd.Series:
        return ts.rolling(window=window, center=True).mean().fillna(ts).round()

    def create_ewma_ts(ts: pd.Series, alpha: float = 0.1) -> pd.Series:
        return ts.ewm(alpha=alpha, adjust=False).mean().fillna(ts).round()

    def create_gaussian_ts(ts: pd.Series, window: int = 8, std: float = 2) -> pd.Series:
        return ts.rolling(window=window, win_type="gaussian", center=True).mean(std=std).fillna(ts).round()

    summary = {
        f"{Types.DHW}_volume_total": int(ts_volume.sum().round(0)),
        f"{Types.DHW}_volume_avg": float(ts_volume.mean().round(3)),
        f"{Types.DHW}_volume_peak": float(ts_volume.max().round(3)),
        f"{Types.DHW}_energy_total": int(ts_energy.sum()),
        f"{Types.DHW}_energy_avg": int(ts_energy.mean().round(0)),
        f"{Types.DHW}_energy_peak": int(ts_energy.max()),
        f"{Types.DHW}_power_avg": int(ts_power.mean().round(0)),
        f"{Types.DHW}_power_max": int(ts_power.max()),
        f"{Types.DHW}_power_min": int(ts_power.min()),
    }

    # Create output timeseries
    sma = create_sma_ts(ts_power)
    ewma = create_ewma_ts(ts_power)
    gaussian = create_gaussian_ts(ts_power)

    timeseries = pd.DataFrame(
        {
            f"{Types.DHW}_volume": ts_volume,
            f"{Types.DHW}_energy": ts_energy.astype(int),
            f"{Types.DHW}_power": ts_power.astype(int),
            f"{Types.DHW}_power_sma": sma.astype(int),
            f"{Types.DHW}_power_ewma": ewma.astype(int),
            f"{Types.DHW}_power_gaussian": gaussian.astype(int),
        },
        index=data["datetimes_index"],
    )

    return {"summary": summary, "timeseries": timeseries}
