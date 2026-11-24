===============
Weather Service
===============

The Weather Service module provides a modular approach to retrieving weather data from various sources. It includes a base ``WeatherProvider`` interface, specific provider implementations, and a ``WeatherService`` facade for easy access to weather data.

Install the service with ``pip install "entise[weather]"``.

.. _open-meteo-terms:

Open-Meteo data and API terms
-----------------------------

`Weather data by Open-Meteo <https://open-meteo.com/>`_ is licensed
under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
Credit Open-Meteo where you display its data, link the license and
describe any changes. See the `Open-Meteo data license
<https://open-meteo.com/en/licence#licence>`_.

The hosted API has separate `terms of use
<https://open-meteo.com/en/terms>`_. Open-Meteo limits its free hosted
endpoint to non-commercial use and applies request quotas. Check the
terms for your use of the hosted service.

.. _example-weather-attribution:

Sample weather
--------------

The examples use `weather data by Open-Meteo
<https://open-meteo.com/>`_ under
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
EnTiSe converted units, renamed columns, relabelled hourly irradiance
at the start of each interval and added mid-interval solar elevation.
The notice beside ``examples/common_data/weather.csv`` describes the
changes in detail.

Architecture
------------

The Weather Service is designed with a modular architecture:

1. **WeatherProvider Interface**: A base interface that defines the contract for all weather data providers.
2. **Provider Implementations**: Specific implementations of the ``WeatherProvider`` interface for different data sources.

   * Currently includes ``OpenMeteoProvider`` for retrieving data from the Open-Meteo API.

3. **WeatherService Facade**: A facade that provides a simple interface for accessing weather data from different providers.

Time Labels and Time Zones
--------------------------

The service returns data on EnTiSe's :ref:`time contract <time-contract>`:

* **Start labels.** Each row is labelled at the start of its hour. Open-Meteo
  labels its hourly aggregates (radiation means, precipitation, rain and
  snowfall sums, evapotranspiration, sunshine duration and gusts) at the end of
  the hour, so the service moves them one hour earlier. Instantaneous values,
  such as air temperature, wind speed or the ``*_instant`` radiation variables,
  keep their labels.
* **One extra day.** To fill the last requested hour, the service requests one
  more day than ``end_date`` and drops it again, so the result covers the local
  days from ``start_date`` to ``end_date``, both included.
* **Solar elevation.** ``solar_elevation_angle[degree]`` is the sun's elevation
  at the middle of each hour.
* **Time zones.** ``timezone`` sets both the local days of the request and the
  zone of the returned timestamps:

  * ``None`` (default): UTC days and UTC timestamps;
  * an IANA name, such as ``"Europe/Berlin"``: that zone's local days and
    timestamps. Case and surrounding spaces are ignored;
  * ``"auto"``: the zone Open-Meteo assigns to the location.

  Any other value raises a ``ValueError`` before the request.

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from entise.services.weather import WeatherService

   # Create a weather service instance
   weather_service = WeatherService()

   # Get weather data for a specific location and time period
   df = weather_service.get_weather_data(
       latitude=49.71754,
       longitude=11.05877,
       start_date="2022-01-01",
       end_date="2022-12-31",
       timezone="Europe/Berlin"
   )

   # Calculate derived variables
   df = weather_service.calculate_derived_variables(df, variables=["absolute_humidity"])

   # Save to CSV
   df.to_csv('weather_data.csv', index=False)

Specifying Features
~~~~~~~~~~~~~~~~~~~

You can specify which weather features to retrieve:

.. code-block:: python

   df = weather_service.get_weather_data(
       latitude=49.71754,
       longitude=11.05877,
       start_date="2022-01-01",
       end_date="2022-12-31",
       timezone="Europe/Berlin",
       features=[
           "temperature_2m",
           "relative_humidity_2m",
           "shortwave_radiation",
           "diffuse_radiation"
       ]
   )

Using a Different Provider
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have implemented additional providers, you can specify which one to use:

.. code-block:: python

   df = weather_service.get_weather_data(
       provider="custom_provider",
       latitude=49.71754,
       longitude=11.05877,
       start_date="2022-01-01",
       end_date="2022-12-31"
   )

Registering a Custom Provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can register your own custom provider:

.. code-block:: python

   from entise.services.weather import WeatherService, WeatherProvider

   class CustomProvider(WeatherProvider):
       def get_weather_data(self, latitude, longitude, start_date, end_date,
                          timezone=None, features=None, **kwargs):
           # Implementation for retrieving weather data
           # ...
           return df

   # Create a weather service instance
   weather_service = WeatherService()

   # Register the custom provider
   weather_service.register_provider("custom_provider", CustomProvider())

   # Use the custom provider
   df = weather_service.get_weather_data(
       provider="custom_provider",
       latitude=49.71754,
       longitude=11.05877,
       start_date="2022-01-01",
       end_date="2022-12-31"
   )

Available Features
------------------

For a complete list of available features, refer to the ``AVAILABLE_FEATURES`` set in the ``OpenMeteoProvider`` class.

Derived Variables
-----------------

The ``WeatherService`` can calculate the following derived variables:

* absolute_humidity: Calculated from relative humidity, temperature, and pressure

You can extend the ``calculate_derived_variables`` method to add more derived variables as needed.
