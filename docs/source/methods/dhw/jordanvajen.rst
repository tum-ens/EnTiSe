jordanvajen
===========

Overview
--------

This module implements a domestic hot water (DHW) demand generation method based on
Jordan & Vajen (2005): "DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES
WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS". The implementation follows the
Method pattern established in the project architecture.

Water is drawn per local day of the object's ``timezone`` (UTC if unset):

- Each event type (bath, shower, ...) has a share of the mean daily volume, from the weekly mean of its ``probability_day``. Within the type, ``probability_day`` gives a factor per weekday, normalised so that its weekly mean is 1; holidays use Sunday's factor. A day's expected volume is a draw from the dwelling's daily demand, scaled by the seasonal factor (using the actual length of the year) and by Σ share × factor of its weekday. Weekday totals therefore vary, and the annual event mix stays at the shares.

- For each day and event type, the number of events is a Poisson draw around the expected volume over the type's mean event size, and each event keeps its own size. One factor over the whole run makes the total equal the sum of the days' expected volumes.

- Events are placed at the activity times of the day's weekday. A time that does not exist on the day clocks go forward moves forward, and a time that happens twice on the day clocks go back takes its first occurrence.

- ``power[W]`` is the energy of the interval that starts at the label divided by the interval's length.

- Random draws come from the object's ``seed`` if it has one, otherwise from the run seed of ``Generator.generate`` combined with the object's ``id`` and the type. Either makes the time series reproducible; without a seed every run differs.

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


Key facts
---------

- Method key: ``jordanvajen``

- Supported types:

  
  - ``dhw``
  


Requirements
------------

Install this method with ``pip install "entise[dhw-jordanvajen]"``.

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``datetimes``
- ``dwelling_size[m2]``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``dhw_activity``
- ``dhw_demand_per_size[m2]``
- ``holidays_location``
- ``cold_water_temperature[C]``
- ``hot_water_temperature[C]``
- ``seasonal_variation``
- ``seasonal_peak_day``
- ``seed``
- ``timezone``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``datetimes``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``dhw_activity``
- ``dhw_demand_per_size[m2]``
- ``cold_water_temperature[C]``
- ``hot_water_temperature[C]``



Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Description
   
   * - ``dhw_volume_total``
     - total hot water demand in liters
   
   * - ``dhw_volume_avg``
     - average hot water demand in liters
   
   * - ``dhw_volume_peak``
     - peak hot water demand in liters
   
   * - ``dhw_energy_total``
     - total energy demand for hot water in Wh
   
   * - ``dhw_energy_avg``
     - average energy demand for hot water in Wh
   
   * - ``dhw_energy_peak``
     - peak energy demand for hot water in Wh
   
   * - ``dhw_power_avg``
     - average power for hot water in W
   
   * - ``dhw_power_max``
     - maximum power for hot water in W
   
   * - ``dhw_power_min``
     - minimum power for hot water in W
   


Timeseries columns
~~~~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   
   * - ``dhw_volume``
     - hot water demand in liters
   
   * - ``dhw_energy``
     - energy demand for hot water in Wh
   
   * - ``dhw_power``
     - power demand for hot water in W
   
   * - ``dhw_power_sma``
     - smoothed power demand using simple moving average
   
   * - ``dhw_power_ewma``
     - smoothed power demand using exponential weighted moving average
   
   * - ``dhw_power_gaussian``
     - smoothed power demand using gaussian smoothing
   
   * - ``dhw_cold_water_temperature[C]``
     - cold water temperature in degrees Celsius
   
   * - ``dhw_hot_water_temperature[C]``
     - hot water temperature in degrees Celsius
   


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           results: dict | None = None,
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
               results (dict, optional): Dictionary with results from previously generated time series
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
               seed (int, optional): The object's random seed. With a seed the time series is
                   reproducible; it takes precedence over the run seed of ``Generator.generate``.
                   Without either, every call draws different numbers. Defaults to None.

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

           ts_volume, ts_energy, ts_power, water_temp = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[DHW jordanvajen]: Generating {ts_type} data")

           return _format_output(processed_data, ts_volume, ts_energy, ts_power, water_temp)

