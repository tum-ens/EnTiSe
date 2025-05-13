import time
import os
import sys
import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import pvlib
import matplotlib.pyplot as plt
from src.constants.columns import Columns as C

# Possible sources:
# https://open-meteo.com
# https://www.visualcrossing.com/resources/documentation/weather-data/where-can-you-find-high-quality-historical-weather-data-at-a-low-cost/
# https://www.visualcrossing.com/weather-api
# https://www.weatherbit.io/api/historical-weather-api
# https://www.weatherapi.com/api.aspx

NAMES = {
    'temperature_2m': C.TEMP_OUT,
    'shortwave_radiation': C.SOLAR_GHI,
}

class WeatherAPI:

    def __init__(self, start_date: str, end_date: str, timezone: str,
                 latitude: float, longitude: float, location: str = None,
                 features: list = ("temperature_2m", "relative_humidity_2m", "surface_pressure",
                                   "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance",
                                   "terrestrial_radiation"),
                 path: str = '.',
                 api: str = 'openmeteo'):
        self.path_save = path
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone,
            "latitude": latitude,
            "longitude": longitude,
            "hourly": self.check_if_features_exist(features),
        }
        self.location = location

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.api = openmeteo_requests.Client(session=retry_session)

        self.response = None
        self.data = None

    def get_weather_data(self):
        self.response = self.api.weather_api(self.url, params=self.params)[0]
        return self.response

    def convert_to_dataframe(self):
        hourly = self.response.Hourly()
        features = self.params["hourly"]

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        for feature in features:
            hourly_data[feature] = hourly.Variables(features.index(feature)).ValuesAsNumpy()

        self.data = pd.DataFrame(data=hourly_data)

        return self.data

    def create_weather_files(self, absolute_humidity: bool = False):
        # Extract data from response
        lat = self.response.Latitude()
        lon = self.response.Longitude()
        elev = self.response.Elevation()
        hourly = self.response.Hourly()

        if absolute_humidity:
            self.data['absolute_humidity_2m'] = self.absolute_humidity(self.data['relative_humidity_2m'],
                                                                       self.data['temperature_2m'],
                                                                       self.data['surface_pressure'])

        # Create solar radiation file
        self.data['sun_elevation_angle'] = self.calculate_sunelevationangle(
            pd.to_datetime(hourly.Time(), unit="s", utc=True),
            pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            lat, lon, elev)

        # Save entire data to file
        self.data.set_index('date', inplace=True)
        self.data.to_csv(os.path.join(self.path_save, f'{self.location}_weather.csv'))

        df_weather = self.data.loc[:, ['temperature_2m', 'shortwave_radiation']]
        df_weather.index.name = 'datetime'
        df_weather.rename(columns=NAMES, inplace=True, errors='ignore')

        df_weather.to_csv(os.path.join(self.path_save, f'{self.location}_weather_short.csv'))

        print(f"Request with Coordinates {self.response.Latitude()}°N {self.response.Longitude()}°E done.")

    @staticmethod
    def check_if_features_exist(features: list):
        available_features = {"temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
                              "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
                              "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
                              "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
                              "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
                              "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
                              "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
                              "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
                              "soil_moisture_100_to_255cm", "shortwave_radiation", "direct_radiation",
                              "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
                              "terrestrial_radiation", "shortwave_radiation_instant", "direct_radiation_instant",
                              "diffuse_radiation_instant", "direct_normal_irradiance_instant",
                              "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
                              "is_day", "sunshine_duration"}

        # Loop through features and check if they are available
        for feature in features:
            if feature not in available_features:
                raise ValueError(f"Feature '{feature}' is not available. "
                                 f"Please choose from the following list: {sorted(list(available_features))}")

        return features

    @staticmethod
    def absolute_humidity(relative_humidity, temperature, pressure):
        # Calculate saturation vapor pressure
        T = temperature  # Temperature in Celsius
        P_sat = 6.112 * np.exp((17.67 * T) / (T + 237.3))  # Magnus-Tetens formula

        # Calculate vapor pressure
        P_v = relative_humidity/100 * P_sat

        # Calculate absolute humidity
        P_a = pressure  # Total atmospheric pressure
        absolute_humidity = (0.622 * P_v) / (P_a - P_v)

        return absolute_humidity

    @ staticmethod
    def calculate_sunelevationangle(start_date, end_date, latitude, longitude, altitude, stepsize: str = 'h',
                                    timezone='UTC'):
        # Calculate sun elevation angle as data does not include it
        # Generate time range (e.g., one year)
        times = pd.date_range(start=start_date, end=end_date, freq=stepsize, tz=timezone, inclusive='right')

        # Get solar position data for the specified location and time range
        solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)

        # Calculate sun elevation angle (degrees)
        sun_elevation_angle = 90 - solar_position['zenith']
        sun_elevation_angle[sun_elevation_angle < 0] = 0
        hs = sun_elevation_angle.reset_index(drop=True)
        hs.name = None
        return hs


if __name__ == '__main__':

    # Input parameters
    dir_path = os.path.join('.')
    location = 'Forchheim'
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    tz = 'Europe/Berlin'
    lat = 49.71754
    lon = 11.05877

    # Set up the weather API object
    api = WeatherAPI(start_date=start_date, end_date=end_date, timezone=tz,
                     latitude=lat, longitude=lon, location=location, path=dir_path)

    # Get weather data
    weather = api.get_weather_data()

    # Convert to dataframe
    weather_df = api.convert_to_dataframe()

    # Create weather files
    api.create_weather_files()
