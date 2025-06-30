wplib
=========================


**Method Key:** ``wplib``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------

Generate wind power time series based on input parameters and weather data.

This method implements the abstract generate method from the Method base class.
It processes the input parameters, calculates the wind power generation time series,
and returns both the time series and summary statistics.

Args:
    obj (dict, optional): Dictionary containing wind turbine parameters. Defaults to None.
    data (dict, optional): Dictionary containing input data. Defaults to None.
    ts_type (str, optional): Time series type to generate. Defaults to Types.WIND.
    latitude (float, optional): Geographic latitude in degrees. Defaults to None.
    longitude (float, optional): Geographic longitude in degrees. Defaults to None.
    weather (pd.DataFrame, optional): Weather data with wind speed and direction. Defaults to None.
    power (float, optional): System power rating in watts. Defaults to None.
    turbine_type (str, optional): Type of wind turbine. Defaults to None.
    hub_height (float, optional): Hub height of the turbine in meters. Defaults to None.
    altitude (float, optional): Site altitude in meters. Defaults to None.

Returns:
    dict: Dictionary containing:
        - "summary" (dict): Summary statistics including total generation,
          maximum generation, and full load hours.
        - "timeseries" (pd.DataFrame): Time series of wind power generation
          with timestamps as index.

Raises:
    Exception: If required data is missing or invalid.

Example:
    >>> windlib = WPLib()
    >>> # Using explicit parameters
    >>> result = windlib.generate(power=5000, weather=weather_df)
    >>> # Or using dictionaries
    >>> obj = {"power": 5000, "weather": "weather"}
    >>> data = {"weather": weather_df}  # DataFrame with wind data
    >>> result = windlib.generate(obj=obj, data=data)
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
           ts_type: str = Types.WIND,
           *,
           weather: pd.DataFrame = None,
           power: float = None,
           turbine_type: str = None,
           hub_height: float = None,
       ):
           """Generate wind power time series based on input parameters and weather data.

           This method implements the abstract generate method from the Method base class.
           It processes the input parameters, calculates the wind power generation time series,
           and returns both the time series and summary statistics.

           Args:
               obj (dict, optional): Dictionary containing wind turbine parameters. Defaults to None.
               data (dict, optional): Dictionary containing input data. Defaults to None.
               ts_type (str, optional): Time series type to generate. Defaults to Types.WIND.
               latitude (float, optional): Geographic latitude in degrees. Defaults to None.
               longitude (float, optional): Geographic longitude in degrees. Defaults to None.
               weather (pd.DataFrame, optional): Weather data with wind speed and direction. Defaults to None.
               power (float, optional): System power rating in watts. Defaults to None.
               turbine_type (str, optional): Type of wind turbine. Defaults to None.
               hub_height (float, optional): Hub height of the turbine in meters. Defaults to None.
               altitude (float, optional): Site altitude in meters. Defaults to None.

           Returns:
               dict: Dictionary containing:
                   - "summary" (dict): Summary statistics including total generation,
                     maximum generation, and full load hours.
                   - "timeseries" (pd.DataFrame): Time series of wind power generation
                     with timestamps as index.

           Raises:
               Exception: If required data is missing or invalid.

           Example:
               >>> windlib = WPLib()
               >>> # Using explicit parameters
               >>> result = windlib.generate(power=5000, weather=weather_df)
               >>> # Or using dictionaries
               >>> obj = {"power": 5000, "weather": "weather"}
               >>> data = {"weather": weather_df}  # DataFrame with wind data
               >>> result = windlib.generate(obj=obj, data=data)
               >>> summary = result["summary"]
               >>> timeseries = result["timeseries"]
           """
           # Process keyword arguments
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               weather=weather,
               power=power,
               turbine_type=turbine_type,
               hub_height=hub_height,
           )

           # Continue with existing implementation
           processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

           ts = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[WIND windlib]: Generating {ts_type} data")

           timestep = processed_data[O.WEATHER].index.diff().total_seconds().dropna()[0]
           summary = {
               f"{C.GENERATION}_{Types.WIND}": (ts.sum() * timestep / 3600).round().astype(int),
               f"{O.GEN_MAX}_{Types.WIND}": ts.max().round().astype(int),
               f"{C.FLH}_{Types.WIND}": (ts.sum() * timestep / 3600 / processed_obj[O.POWER]).round().astype(int),
           }

           ts = ts.rename(columns={O.POWER: f"{C.POWER}_{Types.WIND}"})

           return {
               "summary": summary,
               "timeseries": ts,
           }
