==============
Weather Service
==============

The Weather Service module provides a modular approach to retrieving weather data from various sources. It includes a base ``WeatherProvider`` interface, specific provider implementations, and a ``WeatherService`` facade for easy access to weather data.

Architecture
-----------

The Weather Service is designed with a modular architecture:

1. **WeatherProvider Interface**: A base interface that defines the contract for all weather data providers.
2. **Provider Implementations**: Specific implementations of the ``WeatherProvider`` interface for different data sources.

   * Currently includes ``OpenMeteoProvider`` for retrieving data from the Open-Meteo API.

3. **WeatherService Facade**: A facade that provides a simple interface for accessing weather data from different providers.

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
