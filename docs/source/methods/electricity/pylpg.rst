pylpg
=====

Overview
--------

PyLPG electricity demand generation (EnTiSe standard interface).

The Load Profile Generator simulates a plain local calendar in minutes. EnTiSe
runs it over the local wall-clock span of the run in the object's ``timezone``
(UTC if unset). Each row is the mean power over the interval that starts at
its label. On the 23-hour day of the change to summer time the skipped hour is
not read; on the 25-hour day the repeated hour is read twice.

pylpg chooses the household members' ages and genders with Python's global
random numbers, which EnTiSe does not seed: the run seed of
``Generator.generate`` does not make PyLPG profiles reproducible.


Key facts
---------

- Method key: ``pylpg``

- Supported types:

  
  - ``electricity``
  


Requirements
------------

Install this method with ``pip install "entise[electricity-pylpg]"``.

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``households``
- ``occupants_per_household``
- ``datetimes``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``energy_intensity``
- ``timezone``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


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
   
   * - ``electricity:demand[Wh]``
     - total electricity demand
   
   * - ``electricity:load_max[W]``
     - maximum electricity load
   


Timeseries columns
~~~~~~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   
   * - ``electricity:load[W]``
     - electricity load
   


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self,
           obj: dict = None,
           data: dict = None,
           results: dict = None,
           ts_type: str = Types.ELECTRICITY,
           *,
           households: Optional[int] = None,
           occupants_per_household: Optional[int] = None,
           datetimes: Optional[pd.DataFrame] = None,
           energy_intensity: Optional[str] = None,
       ) -> dict:
           """Generate an electricity load timeseries using the PyLPG backend.

           This is the public entry point for the PyLPG electricity method. It accepts
           either an object/data mapping or keyword overrides, prepares inputs,
           executes PyLPG year-by-year to produce minute-resolution household
           electricity energies, aggregates them energy-conservingly to the requested
           timestep, converts to average power [W], and returns a summary and
           timeseries dataframe.

           Notes:
           - The instants come from the O.DATETIMES table, and the local calendar
             from the object's ``timezone`` (UTC if unset).
           - Timestep must be a multiple of 60 seconds (>= 60 s). Other steps are
             rejected because PyLPG natively produces minute energies.

           Args:
               obj: Optional object dictionary providing inputs. Recognized keys:
                   - O.ID: Optional identifier used in log messages.
                   - O.HOUSEHOLDS: Number of households to simulate (int, required).
                   - O.OCCUPANTS_PER_HOUSEHOLD: Occupants per household (int, required).
                   - O.ENERGY_INTENSITY: Optional PyLPG energy intensity profile name.
                   - O.DATETIMES: Optional override key for selecting the datetimes
                     timeseries from the data mapping.
               data: Data dictionary containing required timeseries. Must include an
                   entry for O.DATETIMES that is a DataFrame with a C.DATETIME column
                   of wall-clock timestamps (tz-naive or parseable strings).
               results: Unused placeholder for interface compatibility.
               ts_type: Timeseries type; defaults to Types.ELECTRICITY (ignored here).
               households: Keyword override for O.HOUSEHOLDS.
               occupants_per_household: Keyword override for O.OCCUPANTS_PER_HOUSEHOLD.
               datetimes: Keyword override providing the O.DATETIMES DataFrame.
               energy_intensity: Keyword override for O.ENERGY_INTENSITY (e.g., a
                   PyLPG intensity scenario name).

           Returns:
               dict: A dictionary with keys:
                   - "summary": Mapping with total electricity demand [Wh] and
                     maximum load [W] over the horizon.
                   - "timeseries": DataFrame with one column
                     "ELECTRICITY|load[W]" containing integer Watts indexed like
                     the input datetimes.
           """
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               households=households,
               occupants_per_household=occupants_per_household,
               datetimes=datetimes,
               energy_intensity=energy_intensity,
           )

           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

           ts = calculate_timeseries(processed_obj, processed_data)

           logger.debug("[pylpg]: Generated successfully.")
           return self._format_output(ts, processed_obj, processed_data)

