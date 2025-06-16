"""Weather data service module.

This module provides a modular approach to retrieving weather data from various sources.
It includes a base WeatherProvider interface, specific provider implementations,
and a WeatherService facade for easy access to weather data.
"""

import logging
import os
from typing import Any, List, Optional

import numpy as np
import openmeteo_requests
import pandas as pd
import pvlib
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)


class WeatherProvider:
    """Base interface for all weather data providers."""

    def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
        features: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieve weather data for a specific location and time period.

        Args:
            latitude: Location latitude in degrees
            longitude: Location longitude in degrees
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            timezone: Timezone name (e.g., 'UTC', 'Europe/Berlin')
            features: List of weather parameters to retrieve
            **kwargs: Additional provider-specific parameters

        Returns:
            DataFrame with standardized weather data columns
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenMeteoProvider(WeatherProvider):
    """Weather data provider using the Open-Meteo API."""

    # Default features to retrieve if none specified
    DEFAULT_FEATURES = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "soil_temperature_100_to_255cm",
        "soil_moisture_100_to_255cm",
        "shortwave_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "terrestrial_radiation",
    ]

    # Set of all available features from the Open-Meteo API
    AVAILABLE_FEATURES = {
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "weather_code",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "wind_gusts_10m",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "terrestrial_radiation",
        "shortwave_radiation_instant",
        "direct_radiation_instant",
        "diffuse_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
        "terrestrial_radiation_instant",
        "is_day",
        "sunshine_duration",
    }

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

    def _validate_features(self, features: List[str]) -> List[str]:
        """Validate that requested features are available.

        Args:
            features: List of features to validate

        Returns:
            List of validated features

        Raises:
            ValueError: If any feature is not available
        """
        for feature in features:
            if feature not in self.AVAILABLE_FEATURES:
                raise ValueError(
                    f"Feature '{feature}' is not available. "
                    f"Please choose from the following list: {sorted(list(self.AVAILABLE_FEATURES))}"
                )
        return features

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
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieve weather data from Open-Meteo API.

        Args:
            latitude: Location latitude in degrees
            longitude: Location longitude in degrees
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            timezone: Timezone name (e.g., 'UTC', 'Europe/Berlin')
            features: List of weather parameters to retrieve
            **kwargs: Additional parameters for the API request

        Returns:
            DataFrame with weather data
        """
        # Use default features if none provided
        if features is None:
            features = self.DEFAULT_FEATURES

        # Validate features
        validated_features = self._validate_features(features)

        # Set up API parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto" if timezone is None else timezone,
            "hourly": validated_features,
            "wind_speed_unit": kwargs.get("wind_speed_unit", "ms"),
        }

        # Make API request
        try:
            response = self.client.weather_api(self.url, params=params)[0]
            df = self._response_to_dataframe(response, params["hourly"])

            # Remove the cache file after successful data retrieval
            self._remove_cache_file()

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
            DataFrame with weather data
        """
        hourly = response.Hourly()
        features = features

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
        for feature in features:
            hourly_data[feature] = hourly.Variables(features.index(feature)).ValuesAsNumpy()

        # Create DataFrame
        df = pd.DataFrame(data=hourly_data)

        # Add sun elevation angle if not already included
        if "sun_elevation" not in df.columns:
            elevation = response.Elevation()
            hs = self._calculate_sun_elevation(
                pd.to_datetime(hourly.Time(), unit="s", utc=True),
                pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                pd.Timedelta(seconds=hourly.Interval()),
                response.Latitude(),
                response.Longitude(),
                elevation,
            )
            df["sun_elevation"] = hs.values

        return df

    def _calculate_sun_elevation(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        freq: pd.Timedelta,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> pd.Series:
        """Calculate sun elevation angle.

        Args:
            start_date: Start datetime
            end_date: End datetime
            latitude: Location latitude in degrees
            longitude: Location longitude in degrees
            altitude: Location altitude in meters

        Returns:
            Series with sun elevation angles
        """
        # Generate time range
        times = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive="left")

        # Get solar position data
        solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)

        # Calculate sun elevation angle (degrees)
        sun_elevation_angle = 90 - solar_position["zenith"]
        sun_elevation_angle[sun_elevation_angle < 0] = 0

        return sun_elevation_angle


class WeatherService:
    """Facade for accessing weather data from different providers."""

    def __init__(self):
        """Initialize the weather service."""
        self.providers = {"openmeteo": OpenMeteoProvider()}

    def register_provider(self, name: str, provider: WeatherProvider):
        """Register a new weather data provider.

        Args:
            name: Name of the provider
            provider: Provider instance
        """
        if not isinstance(provider, WeatherProvider):
            raise TypeError("Provider must implement WeatherProvider interface")
        self.providers[name] = provider

    def get_weather_data(self, provider: str = "openmeteo", **kwargs) -> pd.DataFrame:
        """Get weather data using the specified provider.

        Args:
            provider: Name of the provider to use
            **kwargs: Provider-specific parameters

        Returns:
            DataFrame with weather data
        """
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not found")

        return self.providers[provider].get_weather_data(**kwargs)

    def calculate_derived_variables(self, df: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate derived weather variables.

        Args:
            df: Weather data DataFrame
            variables: List of variables to calculate

        Returns:
            DataFrame with additional variables
        """
        result = df.copy()

        if variables is None or "absolute_humidity" in variables:
            if all(col in result.columns for col in ["relative_humidity_2m", "temperature_2m", "surface_pressure"]):
                result["absolute_humidity"] = self._calculate_absolute_humidity(
                    result["relative_humidity_2m"], result["temperature_2m"], result["surface_pressure"]
                )

        return result

    @staticmethod
    def _calculate_absolute_humidity(
        relative_humidity: pd.Series, temperature: pd.Series, pressure: pd.Series
    ) -> pd.Series:
        """Calculate absolute humidity from relative humidity.

        Args:
            relative_humidity: Relative humidity in percent
            temperature: Temperature in Celsius
            pressure: Atmospheric pressure in hPa

        Returns:
            Absolute humidity
        """
        # Calculate saturation vapor pressure
        T = temperature  # Temperature in Celsius
        P_sat = 6.112 * np.exp((17.67 * T) / (T + 237.3))  # Magnus-Tetens formula

        # Calculate vapor pressure
        P_v = relative_humidity / 100 * P_sat

        # Calculate absolute humidity
        P_a = pressure  # Total atmospheric pressure
        absolute_humidity = (0.622 * P_v) / (P_a - P_v)

        return absolute_humidity


if __name__ == "__main__":
    # Example usage
    weather_service = WeatherService()

    # Get weather data for a specific location and time period
    df = weather_service.get_weather_data(
        latitude=49.71754, longitude=11.05877, start_date="2022-01-01", end_date="2022-12-31", timezone="Europe/Berlin"
    )

    # Calculate derived variables
    df = weather_service.calculate_derived_variables(df, variables=["absolute_humidity"])

    # Save to CSV
    df.to_csv("weather_data.csv", index=False)

    print(f"Retrieved weather data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
