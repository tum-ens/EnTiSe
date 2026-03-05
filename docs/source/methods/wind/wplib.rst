wplib
=====

Overview
--------

Wind power generation using windpowerlib model chain (AC power output).

This module integrates the windpowerlib package to compute wind turbine power
from meteorological inputs and turbine metadata. It constructs a windpowerlib
ModelChain with a chosen turbine (by type/name), corrects wind speed to hub height
(if required), and converts met data (wind speed, temperature, pressure) into hub‑height
conditions and electrical output.

Key capabilities:

- Accept a site weather DataFrame and turbine parameters (type, hub height, rated power).

- Use windpowerlib's power curve–based ModelChain for robust, transparent calculations.

- Return AC power time series along with summary KPIs (max generation, full‑load hours).

Reference: windpowerlib — https://windpowerlib.readthedocs.io/


Key facts
---------

- Method key: ``wplib``

- Supported types:

  
  - ``wind``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``power[W]``
- ``turbine_type``
- ``hub_height[m]``
- ``wind_model``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``wind_model``



Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Description
   
   * - ``generation[Wh]_wind``
     - total wind power generation
   
   * - ``maximum_generation[W]_wind``
     - maximum wind power generation
   
   * - ``full_load_hours[h]_wind``
     - full load hours
   


Timeseries columns
~~~~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   
   * - ``generation[Wh]_wind``
     - wind power generation
   


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           results: dict | None = None,
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
               results (dict, optional): Dictionary with results from previously generated time series
               ts_type (str, optional): Time series type to generate. Defaults to Types.WIND.
               weather (pd.DataFrame, optional): Weather data with wind speed and direction. Defaults to None.
               power (float, optional): System power rating in watts. Defaults to None.
               turbine_type (str, optional): Type of wind turbine. Defaults to None.
               hub_height (float, optional): Hub height of the turbine in meters. Defaults to None.

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
           processed_obj, processed_data = self.get_input_data(processed_obj, processed_data, ts_type)

           ts = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[WIND windlib]: Generating {ts_type} data")

           return self._format_output(processed_obj, processed_data, ts)


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data, method_type=Types.WIND):
           """Process and validate input data for wind power generation calculation.

           This function extracts required and optional parameters from the input dictionaries,
           applies default values where needed, performs data validation, and prepares the
           data for wind power generation calculation.

           Args:
               obj (dict): Dictionary containing wind turbine parameters such as location,
                   turbine type, and power rating.
               data (dict): Dictionary containing input data such as weather information.
               method_type (str, optional): Method type to use for prefixing. Defaults to Types.WIND.

           Returns:
               tuple: A tuple containing:
                   - obj_out (dict): Processed object parameters with defaults applied.
                   - data_out (dict): Processed data with required format for calculation.

           Raises:
               Exception: If required weather data is missing.

           Notes:
               - Parameters can be specified with method-specific prefixes (e.g., "wind:altitude")
                 which will take precedence over generic parameters (e.g., "altitude").
               - Weather data is processed to match the format required by windpowerlib.
           """
           obj_out = {
               O.ID: Method.get_with_backup(obj, O.ID),
               O.POWER: Method.get_with_method_backup(obj, O.POWER, method_type, DEFAULT_POWER),
               O.TURBINE_TYPE: Method.get_with_method_backup(obj, O.TURBINE_TYPE, method_type, DEFAULT_TURBINE_TYPE),
               O.HUB_HEIGHT: Method.get_with_method_backup(obj, O.HUB_HEIGHT, method_type, DEFAULT_HUB_HEIGHT),
           }

           data_out = {
               O.WEATHER: Method.get_with_backup(data, O.WEATHER),
           }

           # Process weather data
           if data_out[O.WEATHER] is not None:
               weather = data_out[O.WEATHER].copy()
               weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME], utc=False)
               weather.index = pd.to_datetime(weather[C.DATETIME], utc=True)
               data_out[O.WEATHER] = self.process_weather_data(weather)
           else:
               logger.error("[WIND windlib]: No weather data")
               raise Exception(f"{O.WEATHER} not available")

           return obj_out, data_out


- process_weather_data

  .. code-block:: python

         def process_weather_data(self, weather_data):
           """Process weather data to match windpowerlib requirements.

           Args:
               weather_data (pd.DataFrame): Raw weather data.

           Returns:
               pd.DataFrame: Processed weather data with the format required by windpowerlib.
           """
           weather = weather_data.copy()

           # Add roughness length if not present
           if C.ROUGHNESS_LENGTH not in weather.columns:
               weather[C.ROUGHNESS_LENGTH] = ROUGHNESS_LENGTH

           # Select and rename required columns
           weather, info = self._obtain_weather_info(weather)

           weather = self._create_multiindex(weather, info)

           weather.rename(
               columns={
                   C.TEMP_AIR.split("[")[0]: "temperature",
                   C.SURFACE_AIR_PRESSURE.split("[")[0]: "pressure",
                   C.WIND_SPEED.split("[")[0]: "wind_speed",
                   C.WIND_DIRECTION.split("[")[0]: "wind_direction",
                   C.ROUGHNESS_LENGTH.split("[")[0]: "roughness_length",
               },
               inplace=True,
               level=0,
           )

           weather["temperature"] += UConv.CELSIUS2KELVIN.value

           return weather

