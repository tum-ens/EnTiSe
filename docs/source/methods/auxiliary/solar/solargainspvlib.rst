SolarGainsPVLib
=========================


**Method Key:** ``SolarGainsPVLib``

.. note::
   This is the class name. For auxiliary methods, the key is determined by the selector class.


Description
-----------

Calculate solar gains for a building.

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

Requirements
-------------

Required Keys
~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Type
   
   * - ``id``
     - ``str``
   
   * - ``latitude``
     - ``str``
   
   * - ``longitude``
     - ``str``
   



Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``weather``








**Timeseries Key:** ``windows``












Dependencies
-------------


- None


Methods
-------


**get_input_data**:


  .. code-block:: python

         def get_input_data(self, obj, data):
           object_id = obj[O.ID]
           windows = data.get(O.WINDOWS, None)
           if windows is not None:
               windows = windows.loc[windows[O.ID] == object_id]
               windows = windows if not windows.empty else None
           input_data = {
               O.LAT: obj[O.LAT],
               O.LON: obj[O.LON],
               "model": obj.get("model", "isotropic"),
               O.WEATHER: data[O.WEATHER],
               O.WINDOWS: windows,
           }
           return input_data



**run**:


  .. code-block:: python

         def run(self, weather, windows, latitude, longitude, model="isotropic"):
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
           """
           if windows is None:
               return pd.DataFrame({O.GAINS_SOLAR: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)

           # Obtain all relevant information upfront
           altitude = pvlib.location.lookup_altitude(latitude, longitude)
           timezone = weather.index.tzinfo or "UTC"
           location = pvlib.location.Location(latitude, longitude, altitude=altitude, tz=timezone)
           solpos = location.get_solarposition(weather.index)

           # Calculate values depending on model
           if model == 'haydavies':
               dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
               dni = pvlib.irradiance.dirint(ghi=weather[C.SOLAR_GHI],
                                           solar_zenith=solpos['apparent_zenith'],
                                           times=weather.index).fillna(0)
           elif model == 'isotropic':
               dni_extra = None
               dni = weather[C.SOLAR_DNI]
           else:
               raise ValueError('Unknown irradiance model.')

           total_solar_gains = np.zeros(len(weather), dtype=np.float32)
           for _, window in windows.iterrows():
               # Compute irradiance for this window
               irr = pvlib.irradiance.get_total_irradiance(
                   surface_tilt=window[C.TILT],
                   surface_azimuth=window[C.ORIENTATION],
                   solar_zenith=solpos["zenith"],
                   solar_azimuth=solpos["azimuth"],
                   dni=dni,
                   ghi=weather[C.SOLAR_GHI],
                   dhi=weather[C.SOLAR_DHI],
                   dni_extra=dni_extra,
                   model=model
               )
               poa_global = irr["poa_global"]
               window_gains = poa_global * window["area"] * window["transmittance"] * window["shading"]

               # Accumulate the gains
               total_solar_gains += window_gains.to_numpy(dtype=np.float32)
           return pd.DataFrame({O.GAINS_SOLAR: total_solar_gains}, index=weather.index)


