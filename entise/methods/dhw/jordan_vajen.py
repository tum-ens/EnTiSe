import logging

import numpy as np
import pandas as pd

from entise._optional import require_module
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._random import method_rng
from entise.core._time import output_index, site_zone
from entise.core.base import Method
from entise.methods.dhw.defaults import (
    DEFAULT_DENSITY_WATER,
    DEFAULT_SEASONAL_PEAK_DAY,
    DEFAULT_SEASONAL_VARIATION,
    DEFAULT_SPECIFIC_HEAT_WATER,
    DEFAULT_TEMP_COLD,
    DEFAULT_TEMP_HOT,
)
from entise.methods.dhw.jordanvajen.activity import _get_activity_data, _get_demand_data
from entise.methods.dhw.jordanvajen.calculation import daily_volume, local_holidays, sample_volume
from entise.methods.dhw.jordanvajen.temperature import _get_water_temperatures
from entise.methods.utils._calendar import time_table, wall_window

logger = logging.getLogger(__name__)


class JordanVajen(Method):
    """This module implements a domestic hot water (DHW) demand generation method based on
    Jordan & Vajen (2005): "DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES
    WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS". The implementation follows the
    Method pattern established in the project architecture.

    Water is drawn per local day of the object's ``timezone`` (UTC if unset):

    - Each event type (bath, shower, ...) has a share of the mean daily volume,
      from the weekly mean of its ``probability_day``. Within the type,
      ``probability_day`` gives a factor per weekday, normalised so that its weekly
      mean is 1; holidays use Sunday's factor. A day's expected volume is a draw
      from the dwelling's daily demand, scaled by the seasonal factor (using the
      actual length of the year) and by Σ share × factor of its weekday. Weekday
      totals therefore vary, and the annual event mix stays at the shares.
    - For each day and event type, the number of events is a Poisson draw around
      the expected volume over the type's mean event size, and each event keeps
      its own size. One factor over the whole run makes the total equal the sum
      of the days' expected volumes.
    - Events are placed at the activity times of the day's weekday. A time that
      does not exist on the day clocks go forward moves forward, and a time that
      happens twice on the day clocks go back takes its first occurrence.
    - ``power[W]`` is the energy of the interval that starts at the label divided
      by the interval's length.
    - Random draws come from the object's ``seed`` if it has one, otherwise from
      the run seed of ``Generator.generate`` combined with the object's ``id`` and
      the type. Either makes the time series reproducible; without a seed every
      run differs.

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

    types = [Types.DHW]
    name = "jordanvajen"
    required_keys = [O.DATETIMES, O.DWELLING_SIZE]
    optional_keys = {
        O.DHW_ACTIVITY: None,
        O.DHW_DEMAND_PER_SIZE: None,
        O.HOLIDAYS_LOCATION: None,
        O.TEMP_WATER_COLD: DEFAULT_TEMP_COLD,
        O.TEMP_WATER_HOT: DEFAULT_TEMP_HOT,
        O.SEASONAL_VARIATION: DEFAULT_SEASONAL_VARIATION,
        O.SEASONAL_PEAK_DAY: DEFAULT_SEASONAL_PEAK_DAY,
        O.SEED: None,
        O.TIMEZONE: None,
    }
    required_data = [O.DATETIMES]
    optional_data = [
        O.DHW_ACTIVITY,
        O.DHW_DEMAND_PER_SIZE,
        O.TEMP_WATER_COLD,
        O.TEMP_WATER_HOT,
    ]
    output_summary = {
        f"{Types.DHW}{SEP}volume_total[l]": "total hot water demand in liters",
        f"{Types.DHW}{SEP}volume_avg[l]": "average hot water demand in liters",
        f"{Types.DHW}{SEP}volume_peak[l]": "peak hot water demand in liters",
        f"{Types.DHW}{SEP}energy_total[Wh]": "total energy demand for hot water in Wh",
        f"{Types.DHW}{SEP}energy_avg[Wh]": "average energy demand for hot water in Wh",
        f"{Types.DHW}{SEP}energy_peak[Wh]": "peak energy demand for hot water in Wh",
        f"{Types.DHW}{SEP}power_avg[W]": "average power for hot water in W",
        f"{Types.DHW}{SEP}power_max[W]": "maximum power for hot water in W",
        f"{Types.DHW}{SEP}power_min[W]": "minimum power for hot water in W",
    }
    output_timeseries = {
        f"{Types.DHW}_volume": "hot water demand in liters",
        f"{Types.DHW}_energy": "energy demand for hot water in Wh",
        f"{Types.DHW}_power": "power demand for hot water in W",
        f"{Types.DHW}_power_sma": "smoothed power demand using simple moving average",
        f"{Types.DHW}_power_ewma": "smoothed power demand using exponential weighted moving average",
        f"{Types.DHW}_power_gaussian": "smoothed power demand using gaussian smoothing",
        f"{Types.DHW}_{O.TEMP_WATER_COLD}": "cold water temperature in degrees Celsius",
        f"{Types.DHW}_{O.TEMP_WATER_HOT}": "hot water temperature in degrees Celsius",
    }

    input_aliases = {"temp_water_cold": O.TEMP_WATER_COLD, "temp_water_hot": O.TEMP_WATER_HOT}

    def generate(
        self,
        obj: dict | None = None,
        data: dict | None = None,
        results: dict | None = None,
        ts_type: str = Types.DHW,
        **inputs,
    ) -> dict:
        """Generate DHW demand time series based on input parameters.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the DHW demand time series,
        and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing DHW system parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            results (dict, optional): Dictionary with results from previously generated time series
            ts_type (str, optional): Time series type to generate. Defaults to Types.DHW.
            **inputs: Any input of the method, by its key or by its Python name (the key
                without its unit, such as ``area`` for ``area[m2]``). Keywords override the
                same keys in ``obj``. A table input, such as ``datetimes``, can be a
                DataFrame. For example:

                - ``datetimes`` (DataFrame or str, required): The run's datetimes.
                - ``dwelling_size`` (``dwelling_size[m2]``, required): Size of the dwelling in m².
                - ``dhw_activity`` (DataFrame): Activity profiles for DHW demand.
                - ``dhw_demand_per_size`` (``dhw_demand_per_size[m2]``): Demand data per dwelling size.
                - ``holidays_location`` (str): Location for the holiday calendar.
                - ``temp_water_cold`` and ``temp_water_hot`` (``cold_water_temperature[C]``,
                  ``hot_water_temperature[C]``): Cold and hot water temperature in °C.
                - ``seasonal_variation`` (float): Seasonal variation factor.
                - ``seasonal_peak_day`` (int): Day of year with peak demand.
                - ``seed`` (int): The object's random seed. It takes precedence over the
                  run seed of ``Generator.generate``. Without either, each call draws
                  different random numbers.

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
            >>> obj = {"datetimes": "datetimes", "dwelling_size[m2]": 100}
            >>> data = {"datetimes": datetimes_df}
            >>> result = jordanvajen.generate(obj=obj, data=data)
            >>> summary = result["summary"]
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(obj, data, ts_type, **inputs)

        processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

        ts_volume, ts_energy, ts_power, water_temp = calculate_timeseries(processed_obj, processed_data)

        logger.debug(f"[DHW jordanvajen]: Generating {ts_type} data")

        return _format_output(processed_data, ts_volume, ts_energy, ts_power, water_temp)


def _named_table(obj: dict, data: dict, key: str):
    """Return the table the object names for ``key``, or the table under ``key`` in ``data``.

    Raises:
        ValueError: If the object names a table that is not in ``data``.
    """
    name = obj.get(key)
    if isinstance(name, str) and name.strip():
        table = Method.get_with_backup(data, name)
        if table is None:
            raise ValueError(f"Object '{obj.get(O.ID)}' names the {key} table '{name}', which is not in the data.")
        return table
    return Method.get_with_backup(data, key)


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
    obj_out = JordanVajen.get_relevant_objects(obj, method_type)

    # Process data
    data_out = {
        O.DATETIMES: Method.get_with_backup(data, obj_out.get(O.DATETIMES)),
        O.DHW_ACTIVITY: _named_table(obj_out, data, O.DHW_ACTIVITY),
        O.DHW_DEMAND_PER_SIZE: _named_table(obj_out, data, O.DHW_DEMAND_PER_SIZE),
        O.TEMP_WATER_COLD: Method.get_with_backup(data, O.TEMP_WATER_COLD),
        O.TEMP_WATER_HOT: Method.get_with_backup(data, O.TEMP_WATER_HOT),
    }

    # Validate required data
    if data_out[O.DATETIMES] is None:
        logger.error("[DHW jordanvajen]: No datetime data")
        raise KeyError(f"Required data '{obj_out.get(O.DATETIMES)}' not found in input data")

    # The run's instants and the site's local calendar
    table, step, hours = time_table(data, obj_out.get(O.DATETIMES), site_zone(obj_out))
    data_out[O.DATETIMES] = table
    data_out["step"] = step
    data_out["hours"] = hours
    data_out["datetimes_index"] = output_index(table.index, obj_out, data).to_series()

    # Use the bundled tables if none is given
    if data_out[O.DHW_ACTIVITY] is None:
        data_out[O.DHW_ACTIVITY] = _get_activity_data("jordan_vajen")
    if data_out[O.DHW_DEMAND_PER_SIZE] is None:
        data_out[O.DHW_DEMAND_PER_SIZE] = _get_demand_data("jordan_vajen")

    # The object's seed wins; otherwise the run seed, the object's id and the type place its random stream.
    data_out["rng"] = method_rng(data, obj.get(O.ID), method_type, seed=obj_out.get(O.SEED))

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
    dwelling_size = obj[O.DWELLING_SIZE]
    activity_data = data[O.DHW_ACTIVITY]
    rng = data["rng"]
    zone = site_zone(obj)
    axis = data[O.DATETIMES].index
    step = data["step"]

    # Statistical demand of the dwelling size
    demand_data = data[O.DHW_DEMAND_PER_SIZE]
    sizes = demand_data[O.DWELLING_SIZE].values
    idx = np.abs(sizes - dwelling_size).argmin()
    m3_per_m2_a = demand_data.iloc[idx][O.YEARLY_DHW_DEMAND]
    sigma = demand_data.iloc[idx][O.SIGMA]

    require_module("dhw-jordanvajen", "holidays")

    # Local days of the run and their volumes
    start, end = wall_window(axis, zone, step)
    days = pd.date_range(start.normalize(), (end - pd.Timedelta(1, unit="us")).normalize(), freq="D")
    holidays = local_holidays(obj.get(O.HOLIDAYS_LOCATION), sorted(set(days.year)))
    volume_per_day = daily_volume(
        days,
        m3_per_m2_a * dwelling_size * 1e3,
        sigma,
        activity_data,
        obj[O.SEASONAL_VARIATION],
        obj[O.SEASONAL_PEAK_DAY],
        holidays,
        rng,
    )
    ts_volume = sample_volume(axis, zone, step, activity_data, volume_per_day, rng, holidays, hours=data["hours"])

    # Water temperatures by local month, and the energy to heat the water
    local_times = pd.DataFrame({C.DATETIME: axis.tz_convert(zone)})
    water_temp = _get_water_temperatures(obj, data, local_times)
    delta_t = water_temp[O.TEMP_WATER_HOT].to_numpy(dtype=float) - water_temp[O.TEMP_WATER_COLD].to_numpy(dtype=float)
    energy_factor = DEFAULT_DENSITY_WATER / 1000 * DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600  # Wh per litre

    out_index = data["datetimes_index"].index
    ts_energy = pd.Series(np.round(ts_volume.to_numpy() * energy_factor).astype(int), index=out_index)
    ts_volume = pd.Series(ts_volume.to_numpy(), index=out_index).round(3)
    # Power is the energy of the interval that starts at the label, divided by its length.
    ts_power = ts_energy / data["hours"]
    water_temp.index = out_index

    return ts_volume, ts_energy, ts_power, water_temp


def _format_output(data, ts_volume, ts_energy, ts_power, water_temp, tank_duration_h: float = 2):
    """
    Format and postprocess the DHW demand time series into a structured output
    for reporting and downstream use.

    This function performs the final formatting step of the Jordan & Vajen DHW
    demand generator. It computes:
    1. Summary statistics for DHW volume, energy, and power.
    2. A time series DataFrame containing:
        - Raw DHW power, energy, and volume
        - Smoothed DHW power using three realistic smoothing methods:
            - SMA (Simple Moving Average)
            - EWMA (Exponentially Weighted Moving Average)
            - Gaussian smoothing

    The smoothing is parameterized by a single physically meaningful value:
    the storage tank's effective thermal buffering duration in **hours**. Based
    on this value and the time resolution of the input time series, all internal
    smoothing parameters (window size, decay rate, kernel width) are derived
    automatically.

    Smoothing methods applied:
    1. **SMA (Simple Moving Average)**
       - Window size = tank_duration / timestep
       - Centered moving average, symmetric
       - Approximates blunt buffering effect with fixed delay

    2. **EWMA (Exponential Weighted Moving Average)**
       - α = Δt / (τ + Δt), where τ = tank_duration
       - Causal, lagging response
       - Models reactive systems (e.g., thermostatically controlled tanks)

    3. **Gaussian smoothing**
       - Symmetric, bell-shaped weight curve
       - σ = window / 4, window = tank_duration / timestep
       - Best for anticipatory smoothing or postprocessed ideal tank behavior

    All smoothing methods:
    - Fill edge NaNs with the original raw signal
    - Round to integer W to match common DHW device behavior
    - Are computed only for the power profile (volume and energy remain unchanged)

    Args:
        data (dict): Input dictionary containing at least:
            - 'datetimes_index': pd.DatetimeIndex
        ts_volume (pd.Series): Volume demand time series [liters].
        ts_energy (pd.Series): Energy demand time series [Wh].
        ts_power (pd.Series): Power demand time series [W].
        tank_duration_h (float): Desired smoothing duration in hours (e.g. 2.0 for 2 hours).

    Returns:
        dict: {
            "summary": demand statistics,
            "timeseries": DataFrame with raw and smoothed DHW demand
        }

    Notes:
    - This function assumes the input time series are at constant resolution (e.g. 15 min).
    - It automatically infers time step size and scales smoothing accordingly.
    - Internally rounds and casts all power signals to `int` for consistency with discrete appliance modeling.
    - This function is tailored to hot water tanks but may generalize to other thermal buffers.

    Examples:
    >>> result = _format_output(data, ts_volume, ts_energy, ts_power, water_temp, tank_duration_h=2.0)
    >>> summary = result["summary"]
    >>> result["timeseries"].columns
    Index(['dhw_volume', 'dhw_energy', 'dhw_power',
           'dhw_power_sma', 'dhw_power_ewma', 'dhw_power_gaussian'],
          dtype='object')
    """

    # Infer time step resolution [hours]
    time_index = data["datetimes_index"]
    timestep_h = time_index.diff().dt.total_seconds().median() / 3600

    # Convert smoothing duration to number of steps
    n_steps = max(3, int(round(tank_duration_h / timestep_h)))

    def create_sma_ts(ts: pd.Series) -> pd.Series:
        return ts.rolling(window=n_steps, center=True).mean().fillna(ts).round()

    def create_ewma_ts(ts: pd.Series) -> pd.Series:
        # α = Δt / (τ + Δt)  with τ = tank_duration_h
        alpha = timestep_h / (tank_duration_h + timestep_h)
        return ts.ewm(alpha=alpha, adjust=False).mean().fillna(ts).round()

    def create_gaussian_ts(ts: pd.Series) -> pd.Series:
        sigma = n_steps / 4  # Rule of thumb: window ≈ 4σ → use σ = window / 4
        return ts.rolling(window=n_steps, win_type="gaussian", center=True).mean(std=sigma).fillna(ts).round()

    # Summary statistics
    summary = {
        f"{Types.DHW}{SEP}volume_total[l]": int(ts_volume.sum().round(0)),
        f"{Types.DHW}{SEP}volume_avg[l]": float(ts_volume.mean().round(3)),
        f"{Types.DHW}{SEP}volume_peak[l]": float(ts_volume.max().round(3)),
        f"{Types.DHW}{SEP}energy_total[Wh]": int(ts_energy.sum()),
        f"{Types.DHW}{SEP}energy_avg[Wh]": int(ts_energy.mean().round(0)),
        f"{Types.DHW}{SEP}energy_peak[Wh]": int(ts_energy.max()),
        f"{Types.DHW}{SEP}power_avg[W]": int(ts_power.mean().round(0)),
        f"{Types.DHW}{SEP}power_max[W]": int(ts_power.max()),
        f"{Types.DHW}{SEP}power_min[W]": int(ts_power.min()),
    }

    # Apply smoothing
    sma = create_sma_ts(ts_power)
    ewma = create_ewma_ts(ts_power)
    gaussian = create_gaussian_ts(ts_power)

    # Combine all time series into output
    timeseries = pd.DataFrame(
        {
            f"{Types.DHW}{SEP}volume[l]": ts_volume,
            f"{Types.DHW}{SEP}energy[Wh]": ts_energy.astype(int),
            f"{Types.DHW}{SEP}power[W]": ts_power.astype(int),
            f"{Types.DHW}{SEP}power_sma[W]": sma.astype(int),
            f"{Types.DHW}{SEP}power_ewma[W]": ewma.astype(int),
            f"{Types.DHW}{SEP}power_gaussian[W]": gaussian.astype(int),
            f"{Types.DHW}{SEP}{O.TEMP_WATER_COLD}": water_temp.loc[:, O.TEMP_WATER_COLD],
            f"{Types.DHW}{SEP}{O.TEMP_WATER_HOT}": water_temp.loc[:, O.TEMP_WATER_HOT],
        },
        index=time_index,
    )

    return {"summary": summary, "timeseries": timeseries}
