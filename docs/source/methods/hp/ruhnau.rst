ruhnau
======

Overview
--------

Heat pump COP model following Ruhnau et al. (2019).

Purpose and scope:

- Computes coefficient of performance (COP) time series for heat pumps by relating COP to the instantaneous temperature lift between the source (ambient/ground/water) and the sink (space heating or DHW). Source/sink types map to typical supply setpoints and optional temperature gradients.

Method outline:

- Parameterize COP as a quadratic function of the temperature lift ΔT based on coefficients for air/soil/water systems, and adjust via a correction factor to reflect realistic performance (defrosting, auxiliaries).

- Provide parallel COP streams for space heating and DHW if desired.

Typical use:

- Combine with HVAC load time series to estimate electrical demand of HP systems, or evaluate seasonal performance under different setpoints.

Reference:

- Ruhnau, O., Hirth, L., & Praktiknjo, A. (2019). Time series of heat demand and heat pump efficiency for energy system modeling. Energy Policy, 125, 704–715. doi:10.1016/j.enpol.2019.111179


Key facts
---------

- Method key: ``ruhnau``

- Supported types:

  
  - ``hp``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``hp_source``
- ``hp_sink``
- ``sink_temperature[C]``
- ``gradient_sink``
- ``water_temperature[C]``
- ``correction_factor``
- ``hp_system``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``hp_system``



Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Description
   
   * - ``hp:heating_avg[1]``
     - average heating COP value
   
   * - ``hp:heating_min[1]``
     - minimum heating COP value
   
   * - ``hp:heating_max[1]``
     - maximum heating COP value
   
   * - ``hp:dhw_avg[1]``
     - average DHW COP value
   
   * - ``hp:dhw_min[1]``
     - minimum DHW COP value
   
   * - ``hp:dhw_max[1]``
     - maximum DHW COP value
   


Timeseries columns
~~~~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   
   * - ``hp:heating[1]``
     - heating COP time series
   
   * - ``hp:dhw[1]``
     - DHW COP time series
   


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           results: dict | None = None,
           ts_type: str = Types.HP,
           *,
           weather: pd.DataFrame = None,
           hp_source: str = None,
           hp_sink: str = None,
           temp_sink: float = None,
           gradient_sink: float = None,
           temp_water: float = None,
           correction_factor: float = None,
           hp_system: dict = None,
           cop_coefficients: dict = None,
       ):
           """Generate heat pump COP time series for both heating and DHW.

           Args:
               obj (dict, optional): Dictionary with heat pump parameters
               data (dict, optional): Dictionary with input data
               results (dict, optional): Dictionary with results from previously generated time series
               ts_type (str, optional): Time series type to generate
               weather (pd.DataFrame, optional): Weather data with temperatures
               hp_source (str, optional): Heat pump source type ('ASHP', 'GSHP', 'WSHP')
               hp_sink (str, optional): Heat sink type ('floor', 'radiator')
               temp_sink (float, optional): Temperature setting for heating
               gradient_sink (float, optional): Gradient setting for heating
               temp_water (float, optional): Temperature setting for DHW
               correction_factor (float, optional): Efficiency correction factor
               hp_system (dict, optional): System configuration for heat pump
               cop_coefficients (dict, optional): Custom coefficients for COP calculation (a, b, c)

           Returns:
               dict: Dictionary with summary statistics and COP time series for both heating and DHW

           Raises:
               ValueError: If required parameters are missing or invalid
           """
           # Process inputs
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               weather=weather,
               hp_source=hp_source,
               hp_sink=hp_sink,
               hp_temp=temp_sink,
               gradient_sink=gradient_sink,
               temp_water=temp_water,
               correction_factor=correction_factor,
               cop_coefficients=cop_coefficients,
           )

           # Get input data with validation
           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data)

           # Calculate heating COP time series
           heating_cop_series = _calculate_heating_cop_series(processed_obj, processed_data)

           # Calculate DHW COP time series
           dhw_cop_series = _calculate_dhw_cop_series(processed_obj, processed_data)

           return self._format_output(heating_cop_series, dhw_cop_series, processed_data)

