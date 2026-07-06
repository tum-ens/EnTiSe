districtheatingsim
==================

Overview
--------

Space heating demand using DistrictHeatingSim’s BDEW methodology.

Purpose and scope:

- Wraps the DistrictHeatingSim BDEW implementation to synthesize hourly heating energy profiles for a selected BDEW profile type (e.g., HEF03), scaled to an annual demand target and aligned to the requested timestep.

- Optionally splits the total load into space heating and DHW shares via the dhw_share parameter.

Notes:

- Provide weather with a datetime column and air temperature. The method derives wall‑clock timestamps and applies a power‑preserving alignment when your resolution differs from the hourly native resolution of the underlying model.

- The first three characters of profile_type define the building group, the last two its subtype, e.g., "HEF03".

Reference:

- DistrictHeatingSim (BDEW heat requirement): https://github.com/rl-institut/DistrictHeatingSim

- BDEW guideline (German Association of Energy and Water Industries).


Key facts
---------

- Method key: ``districtheatingsim``

- Supported types:

  
  - ``heating``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``demand[kWh]``
- ``profile_type``
- ``dhw_share``



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
     - total heating load per timestep
   
   * - ``heating:load_space[W]``
     - space heating load per timestep (derived via dhw_share)
   
   * - ``heating:load_dhw[W]``
     - DHW load per timestep (derived via dhw_share)
   


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
           profile_type: str = None,
           dhw_share: Optional[float] = None,
       ):
           """Generate a heating load timeseries using DistrictHeatingSim's BDEW heat model.

           This method prepares inputs (including optional overrides), computes the
           heating demand profile using DistrictHeatingSim’s BDEW implementation
           based on outdoor air temperature and profile parameters, and returns a
           summary and timeseries.

           Args:
               obj: Object dictionary with inputs like annual demand, profile type,
                    DHW share, and weather key.
               data: Data dictionary containing the weather dataframe under O.WEATHER.
               results: Unused placeholder for interface compatibility.
               ts_type: Timeseries type, defaults to Types.HEATING.
               annual_demand_kwh: Optional annual heat demand in kWh.
               weather: Optional weather dataframe override with C.DATETIME and C.TEMP_AIR.
               profile_type: Optional BDEW profile type (e.g., "HEF03", "GBH01").
                             First three characters define building type, last two subtype.
               dhw_share: Optional domestic hot water share (0–1), passed to
                          DistrictHeatingSim as real_ww_share.

           Returns:
               dict: {"summary": {...}, "timeseries": DataFrame with columns:
                 - "heating:load[W]" (total)
                 - "heating:load_space[W]" (derived)
                 - "heating:load_dhw[W]" (derived)
               indexed like the input weather datetimes.
           """
           processed_obj, processed_data = self._process_kwargs(
               obj,
               data,
               annual_demand_kwh=annual_demand_kwh,
               weather=weather,
               profile_type=profile_type,
               dhw_share=dhw_share,
           )
           processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

           ts_total_w = calculate_timeseries(processed_obj, processed_data)

           logger.debug(f"[districtheatsim heat]: Generating {ts_type} data")

           return self._format_output(ts_total_w, processed_obj)

