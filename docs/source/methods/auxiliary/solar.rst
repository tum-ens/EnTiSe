.. _auxiliary_solar:

Solar Gains
==========

Description
----------

Solar gains auxiliary methods calculate the heat gains from solar radiation through windows and other transparent surfaces. These gains are used by HVAC methods to calculate the heating and cooling demand.

Available Strategies
------------------

EnTiSe provides the following strategy for calculating solar gains:

SolarGainsPVLib
~~~~~~~~~~~~~

A strategy that uses the PVLib library to calculate solar gains through windows.

**Requirements**:

- ``windows``: Name of the time series containing window data.
- ``weather``: Name of the time series containing weather data.
- ``latitude``: Latitude of the location in degrees.
- ``longitude``: Longitude of the location in degrees.

**Window Data Format**:

The window data should be a DataFrame with the following columns:

- ``id``: Window identifier.
- ``area``: Window area in square meters.
- ``orientation``: Orientation in degrees (0 = North, 90 = East, 180 = South, 270 = West).
- ``tilt``: Tilt angle in degrees (0 = horizontal, 90 = vertical).
- ``g_value``: Solar heat gain coefficient (g-value) of the window.

**Weather Data Format**:

The weather data should be a DataFrame with the following columns:

- ``datetime``: Timestamp.
- ``solar_ghi``: Global horizontal irradiance in W/m².
- ``solar_dhi``: Diffuse horizontal irradiance in W/m².
- ``solar_dni``: Direct normal irradiance in W/m².

**Example**:

.. code-block:: python

    object_row = {
        "id": "building1",
        "latitude": 48.1,
        "longitude": 11.6,
    }
    
    # Window data
    windows_df = pd.DataFrame({
        "id": ["window1", "window2", "window3"],
        "area": [2.0, 3.0, 1.5],  # m²
        "orientation": [90, 180, 270],  # degrees
        "tilt": [90, 90, 90],  # degrees (vertical)
        "g_value": [0.6, 0.6, 0.6],
    })
    
    # Weather data
    weather_df = pd.DataFrame({
        "datetime": pd.date_range(start="2025-01-01", periods=24, freq="h"),
        "solar_ghi": [100.0, 200.0, 300.0, ...],  # W/m²
        "solar_dhi": [20.0, 40.0, 60.0, ...],  # W/m²
        "solar_dni": [80.0, 160.0, 240.0, ...],  # W/m²
    })
    
    data = {
        "windows": windows_df,
        "weather": weather_df
    }

**Model Parameter**:

The ``model`` parameter can be used to specify the irradiance model:

- ``isotropic``: Isotropic sky model (default).
- ``klucher``: Klucher model.
- ``hay_davies``: Hay-Davies model.
- ``reindl``: Reindl model.
- ``king``: King model.
- ``perez``: Perez model.

Example:

.. code-block:: python

    # Using the Perez model
    solar_gains = SolarGainsPVLib().run(weather_df, windows_df, latitude=48.1, longitude=11.6, model="perez")