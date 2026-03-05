PHT
===

Overview
--------

Binary occupancy inference from electricity demand using the Page–Hinkley Test (PHT).

Purpose and scope:

- Applies a Page–Hinkley change detector to log‑power to identify sustained upward/downward drifts. Upward drifts set occupancy=1; downward drifts reset to 0. Optional nightly rules can force off‑hours to unoccupied.

Notes:

- Electricity load provides the timestamps; no separate clock is required.

- Power is log10‑transformed with an epsilon to stabilize low values.

- Sensitivity is tuned via lambda (running mean), baseline_offset, and detection_threshold.

References:

- Page, E. S. (1954). Continuous Inspection Schemes. Biometrika.

- Hinkley, D. V. (1971). Inference about the change‑point in a sequence of random variables.


Key facts
---------

- Method key: ``PHT``

- Supported types:

  
  - ``occupancy``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``detection_threshold``
- ``baseline_offset``
- ``night_schedule``
- ``night_schedule_start``
- ``night_schedule_end``
- ``lambda_occ``



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
           baseline_offset: float = None,
           detection_threshold: int = None,
           night_schedule: bool = None,
           night_schedule_start: int = None,
           night_schedule_end: int = None,
       ):
           """
           Generate a binary occupancy schedule from electricity demand using PHT.

           This method prepares inputs, applies defaults for optional parameters via the base
           Method utilities, delegates computation to calculate_timeseries, and formats the
           output (summary and timeseries) according to the framework conventions.

           Args:
               obj (dict, optional):
                   Object parameters. Relevant keys (under the current method type) include:
                   - O.LAMBDA (float): Exponential smoothing parameter used for the running average.
                   - O.BASELINE_OFFSET (float): Baseline offset subtracted from deviations to
                     tune sensitivity (positive values reduce false positives).
                   - O.DETECTION_THRESHOLD (float | int): Threshold for detecting drifts in the
                     Page–Hinkley statistic; larger values make detection less sensitive.
                   - O.NIGHT_SCHEDULE (bool): Whether to enforce a nightly schedule.
                   - O.NIGHT_SCHEDULE_START (int): Start hour of nightly off period [0-23].
                   - O.NIGHT_SCHEDULE_END (int): End hour of nightly off period [0-23].
               data (dict, optional):
                   Not used directly. The electricity time series is expected to be available in
                   results under Types.ELECTRICITY.
               results (dict, optional):
                   Dictionary with previously computed time series. Must contain an entry for
                   Types.ELECTRICITY with key Keys.TIMESERIES that provides a pandas DataFrame
                   with a datetime column Columns.DATETIME and at least one power column.
               ts_type (str, optional):
                   Target time series type. Defaults to Types.OCCUPANCY.
               lambda_occ (float, optional): Smoothing factor for the running average used by PHT.
               baseline_offset (float, optional): Adjusts the deviation baseline.
               detection_threshold (int, optional): Sets the change detection sensitivity.
               night_schedule (bool, optional): If True, apply nightly zeroing.
               night_schedule_start (int, optional): Hour marking the start of the nightly off period.
               night_schedule_end (int, optional): Hour marking the end of the nightly off period.

           Returns:
               dict: A dictionary with two keys:
                   - "summary" (dict): Contains aggregated indicators, including
                     f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}" with the average occupancy.
                   - "timeseries" (pd.DataFrame): A DataFrame indexed by datetime with one column
                     f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY}" containing 0/1 occupancy states.

           Raises:
               KeyError: If the required electricity time series is missing from results.
               ValueError: If the electricity data lacks a datetime column or any power column.

           Examples:
               >>> method = PHT()
               >>> out = method.generate(results={Types.ELECTRICITY: {K.TIMESERIES: elec_df}})
               >>> out["timeseries"].head()
           """

           # Process keyword arguments
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               lambda_occ=lambda_occ,
               baseline_offset=baseline_offset,
               detection_threshold=detection_threshold,
               night_schedule=night_schedule,
               night_schedule_start=night_schedule_start,
               night_schedule_end=night_schedule_end,
           )

           # Get input data
           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, results, ts_type)

           occ_schedule = calculate_timeseries(processed_obj, processed_data)

           return self._format_output(occ_schedule, processed_data)

