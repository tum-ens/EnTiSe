"""
This module defines weather-related features supported by the system, with mapping
to standard climate and forecast (CF) names, Open-Meteo units, and CF-compliant
units. Additionally, it specifies the default inclusion state of each feature.

The features encompassed in this module represent a wide range of meteorological
parameters such as temperature, humidity, wind speed, precipitation, and radiation.
Each feature includes metadata for compatibility with Open-Meteo and CF conventions.
"""

import os
from typing import Any, List, Optional

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from entise.services.weather.weather import WeatherProvider, logger

FEATURES = {
    "temperature_2m": {
        "cf": "air_temperature",
        "unit": "C",
        "cf_unit": "C",  # Kelvin in CF, but due to usability we keep Celsius
        "default": True,
    },
    "relative_humidity_2m": {
        "cf": "relative_humidity",
        "unit": "%",
        "cf_unit": "1",
        "default": True,
    },
    "dew_point_2m": {
        "cf": "dew_point_temperature",
        "unit": "C",
        "cf_unit": "C",
        "default": False,
    },
    "apparent_temperature": {
        "cf": "apparent_temperature",
        "unit": "C",
        "cf_unit": "C",
        "default": False,
    },
    "precipitation": {
        "cf": "precipitation_amount",
        "unit": "mm",
        "cf_unit": "kg m-2",
        "default": False,
    },
    "rain": {
        "cf": "rainfall_amount",
        "unit": "mm",
        "cf_unit": "kg m-2",
        "default": False,
    },
    "snowfall": {
        "cf": "thickness_of_snowfall_amount",
        "unit": "cm",
        "cf_unit": "m",
        "default": False,
    },
    "snow_depth": {
        "cf": "surface_snow_thickness",
        "unit": "m",
        "cf_unit": "m",
        "default": False,
    },
    "weather_code": {
        "cf": "weather_code",
        "unit": "1",
        "cf_unit": "1",
        "default": False,
    },
    "pressure_msl": {
        "cf": "air_pressure_at_sea_level",
        "unit": "hPa",
        "cf_unit": "Pa",
        "default": False,
    },
    "surface_pressure": {
        "cf": "surface_air_pressure",
        "unit": "hPa",
        "cf_unit": "Pa",
        "default": True,
    },
    "cloud_cover": {
        "cf": "cloud_area_fraction",
        "unit": "%",
        "cf_unit": "1",
        "default": False,
    },
    "cloud_cover_low": {
        "cf": "cloud_area_fraction_in_atmosphere_layer_low",
        "unit": "%",
        "cf_unit": "1",
        "default": False,
    },
    "cloud_cover_mid": {
        "cf": "cloud_area_fraction_in_atmosphere_layer_mid",
        "unit": "%",
        "cf_unit": "1",
        "default": False,
    },
    "cloud_cover_high": {
        "cf": "cloud_area_fraction_in_atmosphere_layer_high",
        "unit": "%",
        "cf_unit": "1",
        "default": False,
    },
    "et0_fao_evapotranspiration": {
        "cf": "water_evapotranspiration_amount",
        "unit": "mm",
        "cf_unit": "kg m-2",
        "default": False,
    },
    "vapour_pressure_deficit": {
        "cf": "vapour_pressure_deficit_in_air",
        "unit": "kPa",
        "cf_unit": "Pa",
        "default": False,
    },
    "wind_speed_10m": {
        "cf": "wind_speed",
        "unit": "m s-1",
        "cf_unit": "m s-1",
        "default": True,
    },
    "wind_speed_100m": {
        "cf": "wind_speed",
        "unit": "m s-1",
        "cf_unit": "m s-1",
        "default": True,
    },
    "wind_direction_10m": {
        "cf": "wind_from_direction",
        "unit": "degree",
        "cf_unit": "degree",
        "default": True,
    },
    "wind_direction_100m": {
        "cf": "wind_from_direction",
        "unit": "degree",
        "cf_unit": "degree",
        "default": True,
    },
    "wind_gusts_10m": {
        "cf": "wind_speed_of_gust",
        "unit": "m s-1",
        "cf_unit": "m s-1",
        "default": False,
    },
    "soil_temperature_0_to_7cm": {
        "cf": "soil_temperature",
        "unit": "C",
        "cf_unit": "C",
        "default": False,
    },
    "soil_temperature_7_to_28cm": {
        "cf": "soil_temperature",
        "unit": "C",
        "cf_unit": "C",
        "default": False,
    },
    "soil_temperature_28_to_100cm": {
        "cf": "soil_temperature",
        "unit": "C",
        "cf_unit": "C",
        "default": False,
    },
    "soil_temperature_100_to_255cm": {
        "cf": "soil_temperature",
        "unit": "C",
        "cf_unit": "C",
        "default": True,
    },
    "soil_moisture_0_to_7cm": {
        "cf": "soil_moisture",
        "unit": "m3 m-3",
        "cf_unit": "m3 m-3",
        "default": False,
    },
    "soil_moisture_7_to_28cm": {
        "cf": "soil_moisture",
        "unit": "m3 m-3",
        "cf_unit": "m3 m-3",
        "default": False,
    },
    "soil_moisture_28_to_100cm": {
        "cf": "soil_moisture",
        "unit": "m3 m-3",
        "cf_unit": "m3 m-3",
        "default": False,
    },
    "soil_moisture_100_to_255cm": {
        "cf": "soil_moisture",
        "unit": "m3 m-3",
        "cf_unit": "m3 m-3",
        "default": True,
    },
    "shortwave_radiation": {
        "cf": "global_horizontal_irradiance",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": True,
    },
    "direct_radiation": {
        "cf": "direct_horizontal_irradiance",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "diffuse_radiation": {
        "cf": "diffuse_horizontal_irradiance",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": True,
    },
    "direct_normal_irradiance": {
        "cf": "direct_normal_irradiance",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": True,
    },
    "global_tilted_irradiance": {
        "cf": "global_tilted_irradiance",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "terrestrial_radiation": {
        "cf": "surface_upwelling_longwave_flux_in_air",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": True,
    },
    "shortwave_radiation_instant": {
        "cf": "global_horizontal_irradiance_instant",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "direct_radiation_instant": {
        "cf": "direct_horizontal_irradiance_instant",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "diffuse_radiation_instant": {
        "cf": "diffuse_horizontal_irradiance_instant",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "direct_normal_irradiance_instant": {
        "cf": "direct_normal_irradiance_instant",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "global_tilted_irradiance_instant": {
        "cf": "global_tilted_irradiance_instant",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "terrestrial_radiation_instant": {
        "cf": "surface_upwelling_longwave_flux_in_air",
        "unit": "W m-2",
        "cf_unit": "W m-2",
        "default": False,
    },
    "is_day": {
        "cf": "is_day",
        "unit": "1",
        "cf_unit": "1",
        "default": False,
    },
    "sunshine_duration": {
        "cf": "duration_of_sunshine",
        "unit": "s",
        "cf_unit": "s",
        "default": False,
    },
}


class OpenMeteoProvider(WeatherProvider):
    """Weather data provider using the Open-Meteo API.

    This provider retrieves weather data from the Open-Meteo API and can convert
    variable names to CF (Climate and Forecast) standard names. The conversion
    preserves height information from the original Open-Meteo names.

    Examples of conversion:
        - temperature_2m → air_temperature_2m
        - wind_speed_10m → wind_speed_10m
        - soil_temperature_0_to_7cm → soil_temperature_0.07m (converts cm to m)
        - shortwave_radiation → surface_downwelling_shortwave_flux_in_air
    """

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the Open-Meteo provider.

        Args:
            cache_dir: Directory to cache API responses
        """
        self.cache_dir = cache_dir
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self._setup_client()

    def _setup_client(self):
        """Set up the Open-Meteo API client with caching."""
        cache_session = requests_cache.CachedSession(self.cache_dir, expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    # Test convenience wrapper expected by unit tests
    def _validate_features(self, features: List[str]) -> List[str]:
        return super()._validate_features(features, FEATURES)

    def _remove_cache_file(self):
        """Remove the cache file after data retrieval."""
        cache_file = f"{self.cache_dir}.sqlite"
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                logger.info(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

    def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
        features: Optional[List[str]] = None,
        use_cf_names: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieve weather data from Open-Meteo API.

        This method retrieves weather data for a specific location and time period
        from the Open-Meteo API. It can return the data with either the original
        Open-Meteo variable names or with CF (Climate and Forecast) standard names.

        The CF standard names conversion preserves height information and follows
        these patterns:
        - temperature_2m → air_temperature_2m
        - soil_temperature_0_to_7cm → soil_temperature_0.07m
        - shortwave_radiation → surface_downwelling_shortwave_flux_in_air

        Args:
            latitude: Location latitude in degrees
            longitude: Location longitude in degrees
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            timezone: Timezone name (e.g., 'UTC', 'Europe/Berlin')
            features: List of weather parameters to retrieve
            use_cf_names: Whether to convert Open-Meteo names to CF standard names.
                          When True, column names in the returned DataFrame will use
                          CF standard names. When False (default), the original
                          Open-Meteo names will be used.
            **kwargs: Additional parameters for the API request

        Returns:
            DataFrame with weather data using either original Open-Meteo names
            or CF standard names based on the use_cf_names parameter
        """

        if features is None:
            features = self.use_default_features(FEATURES)

        # Validate features
        validated_features = self._validate_features(features)

        # Set up API parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone or "auto",
            "hourly": validated_features,
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
        }

        # Make API request
        try:
            response = self.client.weather_api(self.url, params=params)[0]
            df = self._response_to_dataframe(response, params["hourly"])
            if timezone and timezone.upper() != "UTC":
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(timezone)

            # Remove the cache file after successful data retrieval
            # self._remove_cache_file()

            return df
        except Exception as e:
            logger.error(f"Error retrieving data from Open-Meteo API: {e}")
            raise

    def _response_to_dataframe(self, response: Any, features: list) -> pd.DataFrame:
        """Convert API response to DataFrame.

        Args:
            response: Open-Meteo API response
            features: List of weather features to include

        Returns:
            DataFrame with weather data with enriched CF column names including units and height (when applicable)
        """
        hourly = response.Hourly()

        # Create DataFrame with datetime index
        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        # Add all variables to the DataFrame
        for idx, feature in enumerate(features):
            hourly_data[feature] = hourly.Variables(idx).ValuesAsNumpy()

        # Create DataFrame
        df = pd.DataFrame(data=hourly_data)

        # Add sun elevation angle if not already included
        if "solar_elevation_angle" not in df.columns:
            elevation = response.Elevation()
            hs = self._calculate_sun_elevation(
                pd.to_datetime(hourly.Time(), unit="s", utc=True),
                pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                pd.Timedelta(seconds=hourly.Interval()),
                response.Latitude(),
                response.Longitude(),
                elevation,
            )
            df["solar_elevation_angle[degree]"] = hs.values

        df = self.convert_units(df, FEATURES)

        # Rename columns to enriched CF names including units and optional height
        rename_map = {}
        for col in df.columns:
            if col == "datetime":
                continue
            rename_map[col] = self._convert_to_enriched_name(col, FEATURES)
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        return df
