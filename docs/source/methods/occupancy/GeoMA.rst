GeoMA
=====

Overview
--------

Binary occupancy inference from electricity demand via a Geometric Moving Average (GeoMA).

Purpose and scope:

- Compares the log‑transformed instantaneous power against its exponentially weighted moving average (EWM). If the current reading exceeds the EWM (geometric mean on the original scale), occupancy=1 else 0. An optional nightly schedule can force unoccupied states during specified hours.

Notes:

- Electricity demand supplies the timestamp index; no separate clock needed.

- Uses log10 with a small epsilon to avoid log(0) and stabilize low values.

- Smoothing parameter ``Objects.LAMBDA`` tunes responsiveness of the EWM.

Related methods:

- See also PHT (Page–Hinkley Test) for change‑point based detection on the same input.


Key facts
---------

- Method key: ``GeoMA``

- Supported types:

  
  - ``occupancy``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``lambda_occ``
- ``night_schedule``
- ``night_schedule_start``
- ``night_schedule_end``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``electricity``



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
   
   * - ``occupancy:average_occupancy``
     - average occupancy
   


Timeseries columns
~~~~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   
   * - ``occupancy:occupancy[1]``
     - binary occupancy schedule (0/1)
   


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           results: dict = None,
           ts_type: str = Types.OCCUPANCY,
           *,
           lambda_occ: float = None,
           night_schedule: bool = None,
           night_schedule_start: int = None,
           night_schedule_end: int = None,
       ):
           """
           Generate a binary occupancy schedule from electricity demand using GeoMA.

           This method is a thin orchestrator around calculate_timeseries. It prepares inputs,
           applies defaults for optional parameters (via the Method helpers), and formats the
           output as expected by the framework (summary and timeseries).

           Args:
               obj (dict, optional):
                   Object parameters. Relevant keys (under the current method type) include:
                   - O.LAMBDA (float): Exponential smoothing parameter in (0, 1].
                   - O.NIGHT_SCHEDULE (bool): Whether to enforce a nightly schedule.
                   - O.NIGHT_SCHEDULE_START (int): Start hour of nightly off period [0-23].
                   - O.NIGHT_SCHEDULE_END (int): End hour of nightly off period [0-23].
               data (dict, optional):
                   Not used directly for this method. The electricity time series is expected
                   to be present in results under Types.ELECTRICITY.
               results (dict, optional):
                   Dictionary with previously computed time series. Must contain an entry for
                   Types.ELECTRICITY with key Keys.TIMESERIES that provides a pandas DataFrame
                   with a datetime column Columns.DATETIME and at least one power column.
               ts_type (str, optional):
                   Target time series type. Defaults to Types.OCCUPANCY.
               lambda_occ (float, optional): Exponential smoothing factor for the EWM used
                   to compute the geometric moving average. Typical range (0.05–0.5).
               night_schedule (bool, optional): If True, apply nightly zeroing.
               night_schedule_start (int, optional): Hour of day marking the start of
                   the nightly off period (e.g., 18:00).
               night_schedule_end (int, optional):  Hour of day marking the end of the
                   nightly off period (e.g., 00:00).

           Returns:
               dict: A dictionary with two keys:
                   - "summary" (dict): Contains aggregated indicators, including
                     f"{Types.OCCUPANCY}{SEP}{Objects.OCCUPANCY_AVG}" with the average occupancy.
                   - "timeseries" (pd.DataFrame): A DataFrame indexed by datetime with one column
                     f"{Types.OCCUPANCY}{SEP}{Objects.OCCUPANCY}" containing 0/1 occupancy states.

           Raises:
               KeyError: If the required electricity time series is missing from results.
               ValueError: If the electricity data lacks a datetime column or any power column.

           Examples:
               >>> method = GeoMA()
               >>> out = method.generate(results={Types.ELECTRICITY: {K.TIMESERIES: elec_df}})
               >>> out["timeseries"].head()
           """

           # Process keyword arguments
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               lambda_occ=lambda_occ,
               night_schedule=night_schedule,
               night_schedule_start=night_schedule_start,
               night_schedule_end=night_schedule_end,
           )

           # Get input data
           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, results, ts_type)

           # Compute temperature and energy demand
           occ_schedule = calculate_timeseries(processed_obj, processed_data)

           return self._format_output(occ_schedule, processed_data)

