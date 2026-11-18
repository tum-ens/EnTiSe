demandlib_heat
==============

Overview
--------

Space heating demand using demandlib’s temperature-driven BDEW model.

Purpose and scope:

- Wraps demandlib’s BDEW methodology to synthesize hourly heating energy profiles driven by outdoor air temperature and building archetype parameters. Profiles are scaled so that the energy over the run equals ``Objects.DEMAND_KWH``.

Notes:

- Provide weather with timestamps and air temperature. The engine runs on the hourly local wall clock of the object's ``timezone`` (UTC if unset). On the 23-hour day of the change to summer time the skipped hour is not read; on the 25-hour day the repeated hour is read twice. Each row is the mean power over the interval that starts at its label.

- Building type/class and wind class adjust the sensitivity and seasonal shape as per BDEW assumptions.

Reference:

- demandlib (BDEW heat): https://demandlib.readthedocs.io/

- BDEW guideline (German Association of Energy and Water Industries).


Key facts
---------

- Method key: ``demandlib_heat``

- Supported types:

  
  - ``heating``
  


Requirements
------------

Install this method with ``pip install "entise[heating-demandlib]"``.

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``demand[kWh]``
- ``building_type``
- ``building_class``
- ``wind_class``
- ``holidays_location``
- ``timezone``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Description
   
   * - ``heating:demand[kWh]``
     - total heating demand
   


Timeseries columns
~~~~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   
   * - ``heating:load[W]``
     - heating load
   


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           results: dict = None,
           ts_type: str = Types.HEATING,
           *,
           annual_demand_kwh: float = None,
           weather: pd.DataFrame = None,
           building_type: str = None,
           building_class: int = None,
           wind_class: int = None,
           holidays_location: Optional[str] = None,
       ):
           """Generate a heating load timeseries using demandlib's BDEW heat model.

           This method prepares inputs (including optional overrides), computes the
           heating demand profile based on outdoor air temperature and BDEW
           parameters, and returns a summary and timeseries.

           Args:
               obj: Object dictionary with inputs like demand, building params, weather key.
               data: Data dictionary containing the weather dataframe under O.WEATHER.
               results: Unused placeholder for interface compatibility.
               ts_type: Timeseries type, defaults to Types.HEATING.
               annual_demand_kwh: Optional heat demand in kWh over the simulated run (the
                   keyword keeps its historical name); defaults to 1.0.
               weather: Optional weather dataframe override with C.DATETIME and C.TEMP_AIR.
               building_type: Optional BDEW building type (e.g., "EFH").
               building_class: Optional BDEW building class (int).
               wind_class: Optional BDEW wind class (int).
               holidays_location: Optional country/region code for holidays (e.g., "DE").

           Returns:
               dict: {"summary": {...}, "timeseries": DataFrame with column
               "HEATING|load[W]" indexed like the input weather datetimes.
           """
           # Process keyword arguments
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               annual_demand_kwh=annual_demand_kwh,
               weather=weather,
               building_type=building_type,
               building_class=building_class,
               wind_class=wind_class,
               holidays_location=holidays_location,
           )
           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

           ts = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[demandlib heat]: Generating {ts_type} data")

           return self._format_output(ts, processed_obj, processed_data)

