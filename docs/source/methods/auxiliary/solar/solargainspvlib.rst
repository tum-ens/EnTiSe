solargainspvlib
===============

Overview
--------

Perform calculations of solar gains for buildings using irradiance models.

This class provides methods to process input data and calculate solar gains by considering
weather conditions, window configurations, and solar irradiance models. It integrates
with `pvlib` to compute solar positions and irradiance values. The class supports different
irradiance models, such as "isotropic" and "haydavies", and handles missing input gracefully.


Key facts
---------

- Method key: ``SolarGainsPVLib``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``id``
- ``latitude[degree]``
- ``longitude[degree]``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``
- ``windows``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


- None


Timeseries columns
~~~~~~~~~~~~~~~~~~


- None


Public methods
--------------


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           object_id = obj[O.ID]
           windows = data.get(O.WINDOWS, None)
           if windows is not None:
               windows = windows.loc[windows[O.ID] == object_id]
               windows = windows if not windows.empty else None
           input_data = {
               "latitude": obj[O.LAT],
               "longitude": obj[O.LON],
               "weather": data[O.WEATHER],
               "windows": windows,
           }
           return input_data


- run

  .. code-block:: python

         def run(self, weather, windows, latitude, longitude):
           """Calculate solar gains for a building.

           Args:
               weather (pd.DataFrame): Weather data.
               windows (pd.DataFrame): Windows data.
               latitude (float): Latitude.
               longitude (float): Longitude.
               model (str, optional): Irradiance model to use. Default is "isotropic".

           Returns:
               pd.DataFrame: Solar gains for each timestep.

           Raises:
               ValueError: If the irradiance model is unknown.

           Caching rules:
           - Cache solpos and poa_global per weather frame; never across weather frames.
           - Do NOT cache final total solar gains.
           - The solar position is calculated at the centre of the object's 0.1 degree grid cell,
             so all objects in a cell share it and the result does not depend on calculation order.
           - Window fingerprint for POA cache uses only tilt and orientation.
           """
           if windows is None:
               return pd.DataFrame({O.GAINS_SOLAR: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)

           cache = _frame_cache(weather)
           timezone_info = weather.index[0].tzinfo
           tz_offset = 0 if timezone_info is None else timezone_info.utcoffset(None).total_seconds() / 3600
           cell = _cell_centre(latitude, longitude)

           # Cache solar position (solpos)
           sp_key = (cell, tz_offset)
           solpos = cache.solpos.get(sp_key)
           if solpos is None:
               location = pvlib.location.Location(cell[0], cell[1], tz=tz_offset)
               solpos = location.get_solarposition(pd.to_datetime(weather.index, utc=True), method="nrel_numba")
               cache.solpos[sp_key] = solpos

           total_solar_gains = np.zeros(len(weather), dtype=np.float32)
           zenith = solpos["zenith"]
           azimuth = solpos["azimuth"]
           ghi = weather[C.SOLAR_GHI]
           dhi = weather[C.SOLAR_DHI]
           dni = weather[C.SOLAR_DNI]

           # Loop windows; cache POA per (grid cell/tilt/azimuth) within this weather frame
           for _, window in windows.iterrows():
               tilt = float(window[C.TILT])
               orientation = float(window[C.ORIENTATION])
               poa_key = (cell, tz_offset, round(tilt, 3), round(orientation, 3))
               poa = cache.poa.get(poa_key)
               if poa is None:
                   irr = pvlib.irradiance.get_total_irradiance(
                       surface_tilt=tilt,
                       surface_azimuth=orientation,
                       solar_zenith=zenith,
                       solar_azimuth=azimuth,
                       dni=dni,
                       ghi=ghi,
                       dhi=dhi,
                       dni_extra=None,
                       model="isotropic",
                   )
                   poa = irr["poa_global"].to_numpy(dtype=np.float32, copy=False)
                   cache.poa[poa_key] = poa

               # Compute window gains from POA. Cast scalars to np.float32 so
               # `poa * area * g * sh` stays float32 (Python `float` would promote
               # the intermediate array to float64 and force a copy back).
               area = np.float32(window[C.AREA])
               g = np.float32(window[C.G_VALUE])
               sh = np.float32(window[C.SHADING])
               total_solar_gains += poa * area * g * sh

           return pd.DataFrame({O.GAINS_SOLAR: total_solar_gains}, index=weather.index)

