demandlib_electricity
=====================

Overview
--------

Electricity demand using demandlib's BDEW standard load profiles (SLPs).

This method wraps demandlib's implementation of the German BDEW standard load
profiles to generate high‑quality electricity demand time series. Given a
target time horizon (``Objects.DATETIMES``) and an annual energy demand in
kWh, it constructs the canonical 15‑minute SLP for the relevant calendar
year(s), scales it to the requested annual demand, and aligns it to the
target resolution using energy‑conserving resampling rules.

Key characteristics:

- Profile families such as household (H0) and commercial (Gx) are supported via ``Objects.PROFILE``; defaults to an H0 dynamic profile.

- Public holidays can be considered by providing ``Objects.HOLIDAYS_LOCATION``.

- Output power is provided in Watts and indexed like the provided datetimes.

Reference:

- demandlib documentation (BDEW SLPs): https://demandlib.readthedocs.io/

- BDEW guideline (German Association of Energy and Water Industries).


Key facts
---------

- Method key: ``demandlib_electricity``

- Supported types:

  
  - ``electricity``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``datetimes``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``demand[kWh]``
- ``profile``
- ``holidays_location``



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
           profile: str = None,
           demand_kwh: float = None,
           weather: pd.DataFrame = None,
           holidays_location: Optional[str] = None,
       ) -> dict:
           """Generate an electricity demand timeseries using demandlib BDEW profiles.

           This is the public entry point for the electricity method. It accepts either
           an object/data mapping or keyword overrides, prepares inputs, computes the
           load profile at the requested resolution, and returns a summary and
           timeseries dataframe.

           Args:
               obj: Object dictionary providing inputs (e.g., demand, profile key).
               data: Data dictionary containing required timeseries (O.DATETIMES).
               results: Unused placeholder for interface compatibility.
               ts_type: Timeseries type, defaults to Types.ELECTRICITY.
               profile: Optional BDEW profile key (e.g., "h0", "g0"); defaults to
                   method-level DEFAULT_PROFILE when not provided.
               demand_kwh: Optional annual demand in kWh; defaults to 1.0 if not provided.
               weather: Unused for this method; accepted for API symmetry.
               holidays_location: Optional holidays country/region code used by demandlib
                   to adjust profiles (e.g., "DE").

           Returns:
               dict: A dictionary with keys:
                   - "summary": Mapping with total demand [Wh] and max load [W].
                   - "timeseries": DataFrame indexed like the input datetimes with one
                     column "ELECTRICITY|load[W]" of integer W values.
           """
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               profile=profile,
               demand_kwh=demand_kwh,
               weather=weather,
               holidays_location=holidays_location,
           )
           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

           ts = calculate_timeseries(processed_obj, processed_data)

           logger.debug("[demandlib elec]: Generated succesfully.")

           return self._format_output(ts, processed_data)

