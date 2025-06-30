pvlib
=========================


**Method Key:** ``pvlib``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------

Generate PV power time series based on input parameters and weather data.

This method implements the abstract generate method from the Method base class.
It processes the input parameters, calculates the PV generation time series,
and returns both the time series and summary statistics.

Args:
    obj (dict, optional): Dictionary containing PV system parameters. Defaults to None.
    data (dict, optional): Dictionary containing input data. Defaults to None.
    ts_type (str, optional): Time series type to generate. Defaults to Types.PV.
    latitude (float, optional): Geographic latitude in degrees. Defaults to None.
    longitude (float, optional): Geographic longitude in degrees. Defaults to None.
    weather (pd.DataFrame, optional): Weather data with solar radiation. Defaults to None.
    power (float, optional): System power rating in watts. Defaults to None.
    azimuth (float, optional): Panel azimuth angle in degrees (0=North, 90=East, 180=South, 270=West).
                                Defaults to None.
    tilt (float, optional): Panel tilt angle in degrees (0=horizontal, 90=vertical). Defaults to None.
    altitude (float, optional): Site altitude in meters. Defaults to None.
    pv_arrays (dict, optional): PV array configuration parameters. Defaults to None.
    pv_inverter (dict, optional): PV inverter configuration parameters. Defaults to None.

Returns:
    dict: Dictionary containing:
        - "summary" (dict): Summary statistics including total generation,
          maximum generation, and full load hours.
        - "timeseries" (pd.DataFrame): Time series of PV power generation
          with timestamps as index.

Raises:
    Exception: If required data is missing or invalid.

Example:
    >>> pvlib = PVLib()
    >>> # Using explicit parameters
    >>> result = pvlib.generate(latitude=48.1, longitude=11.6, power=5000, weather=weather_df)
    >>> # Or using dictionaries
    >>> obj = {"latitude": 48.1, "longitude": 11.6, "power": 5000}
    >>> data = {"weather": weather_df}  # DataFrame with solar radiation data
    >>> result = pvlib.generate(obj=obj, data=data)
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

   * - ``latitude``
     - ``str``

   * - ``longitude``
     - ``str``

   * - ``weather``
     - ``str``




Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``weather``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           ts_type: str = Types.PV,
           *,
           latitude: float = None,
           longitude: float = None,
           weather: pd.DataFrame = None,
           power: float = None,
           azimuth: float = None,
           tilt: float = None,
           altitude: float = None,
           pv_arrays: dict = None,
           pv_inverter: dict = None,
       ):
           """Generate PV power time series based on input parameters and weather data.

           This method implements the abstract generate method from the Method base class.
           It processes the input parameters, calculates the PV generation time series,
           and returns both the time series and summary statistics.

           Args:
               obj (dict, optional): Dictionary containing PV system parameters. Defaults to None.
               data (dict, optional): Dictionary containing input data. Defaults to None.
               ts_type (str, optional): Time series type to generate. Defaults to Types.PV.
               latitude (float, optional): Geographic latitude in degrees. Defaults to None.
               longitude (float, optional): Geographic longitude in degrees. Defaults to None.
               weather (pd.DataFrame, optional): Weather data with solar radiation. Defaults to None.
               power (float, optional): System power rating in watts. Defaults to None.
               azimuth (float, optional): Panel azimuth angle in degrees (0=North, 90=East, 180=South, 270=West).
                                           Defaults to None.
               tilt (float, optional): Panel tilt angle in degrees (0=horizontal, 90=vertical). Defaults to None.
               altitude (float, optional): Site altitude in meters. Defaults to None.
               pv_arrays (dict, optional): PV array configuration parameters. Defaults to None.
               pv_inverter (dict, optional): PV inverter configuration parameters. Defaults to None.

           Returns:
               dict: Dictionary containing:
                   - "summary" (dict): Summary statistics including total generation,
                     maximum generation, and full load hours.
                   - "timeseries" (pd.DataFrame): Time series of PV power generation
                     with timestamps as index.

           Raises:
               Exception: If required data is missing or invalid.

           Example:
               >>> pvlib = PVLib()
               >>> # Using explicit parameters
               >>> result = pvlib.generate(latitude=48.1, longitude=11.6, power=5000, weather=weather_df)
               >>> # Or using dictionaries
               >>> obj = {"latitude": 48.1, "longitude": 11.6, "power": 5000}
               >>> data = {"weather": weather_df}  # DataFrame with solar radiation data
               >>> result = pvlib.generate(obj=obj, data=data)
               >>> summary = result["summary"]
               >>> timeseries = result["timeseries"]
           """
           # Process keyword arguments
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               latitude=latitude,
               longitude=longitude,
               weather=weather,
               power=power,
               azimuth=azimuth,
               tilt=tilt,
               altitude=altitude,
               pv_arrays=pv_arrays,
               pv_inverter=pv_inverter,
           )

           # Continue with existing implementation
           processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

           ts = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[PV pvlib]: Generating {ts_type} data")

           timestep = processed_data[O.WEATHER][C.DATETIME].diff().dt.total_seconds().dropna().mode()[0]
           summary = {
               f"{C.GENERATION}_{Types.PV}": (ts.sum() * timestep / 3600).round().astype(int),
               f"{O.GEN_MAX}_{Types.PV}": ts.max().round().astype(int),
               f"{C.FLH}_{Types.PV}": (ts.sum() * timestep / 3600 / processed_obj[O.POWER]).round().astype(int),
           }

           ts = ts.rename(columns={"p_mp": f"{C.POWER}_{Types.PV}"})

           return {
               "summary": summary,
               "timeseries": ts,
           }
