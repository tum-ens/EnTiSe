jordanvajen
=========================


**Method Key:** ``jordanvajen``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------

Generate DHW demand time series based on input parameters.

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

Requirements
-------------

Required Keys
~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Type

   * - ``datetimes``
     - ``str``

   * - ``dwelling_size``
     - ``str``




Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``datetimes``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(self,
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
                   seed: int = None):
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
               obj, data,
               datetimes=datetimes, dwelling_size=dwelling_size,
               dhw_activity=dhw_activity, dhw_demand_per_size=dhw_demand_per_size,
               holidays_location=holidays_location,
               temp_water_cold=temp_water_cold, temp_water_hot=temp_water_hot,
               seasonal_variation=seasonal_variation, seasonal_peak_day=seasonal_peak_day,
               seed=seed
           )

           # Continue with existing implementation
           processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

           ts_volume, ts_energy, ts_power = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[DHW jordanvajen]: Generating {ts_type} data")

           # Create output summary
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

           # Create output timeseries
           timeseries = pd.DataFrame({
               f'{C.LOAD}_{Types.DHW}_volume': ts_volume,
               f'{C.LOAD}_{Types.DHW}_energy': ts_energy,
               f'{C.LOAD}_{Types.DHW}_power': ts_power,
           }, index=processed_data['datetimes_index'])

           return {
               "summary": summary,
               "timeseries": timeseries
           }
