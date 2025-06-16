"""
Unit tests for the weather service module.

This module contains unit tests for the WeatherProvider interface,
OpenMeteoProvider implementation, and WeatherService facade.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from entise.services.weather import OpenMeteoProvider, WeatherProvider, WeatherService


class TestWeatherProvider:
    """Tests for the WeatherProvider interface."""

    def test_get_weather_data_not_implemented(self):
        """Test that the base WeatherProvider raises NotImplementedError."""
        provider = WeatherProvider()
        with pytest.raises(NotImplementedError):
            provider.get_weather_data(
                latitude=49.71754, longitude=11.05877, start_date="2022-01-01", end_date="2022-01-31"
            )


class TestOpenMeteoProvider:
    """Tests for the OpenMeteoProvider implementation."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("openmeteo_requests.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_cache_session(self):
        """Create a mock cache session for testing."""
        with patch("requests_cache.CachedSession") as mock_session:
            yield mock_session

    @pytest.fixture
    def mock_retry(self):
        """Create a mock retry session for testing."""
        with patch("retry_requests.retry") as mock_retry:
            yield mock_retry

    @pytest.fixture
    def mock_os_path_exists(self):
        """Mock os.path.exists to return True."""
        with patch("os.path.exists", return_value=True) as mock_exists:
            yield mock_exists

    @pytest.fixture
    def mock_os_remove(self):
        """Mock os.remove to do nothing."""
        with patch("os.remove") as mock_remove:
            yield mock_remove

    @pytest.fixture
    def mock_response(self):
        """Create a mock API response for testing."""
        mock_response = MagicMock()

        # Mock Hourly object
        mock_hourly = MagicMock()
        mock_hourly.Time.return_value = int(datetime(2022, 1, 1).timestamp())
        mock_hourly.TimeEnd.return_value = int(datetime(2022, 1, 2).timestamp())
        mock_hourly.Interval.return_value = 3600  # 1 hour
        mock_hourly.VariableCount.return_value = 3

        # Mock variable metadata
        mock_hourly.VariableMetadata.side_effect = lambda i: MagicMock(
            Name=lambda: ["temperature_2m", "relative_humidity_2m", "surface_pressure"][i]
        )

        # Mock variables
        mock_var1 = MagicMock()
        mock_var1.ValuesAsNumpy.return_value = np.array([10.0, 11.0, 12.0])
        mock_var2 = MagicMock()
        mock_var2.ValuesAsNumpy.return_value = np.array([80.0, 75.0, 70.0])
        mock_var3 = MagicMock()
        mock_var3.ValuesAsNumpy.return_value = np.array([1013.0, 1012.0, 1011.0])

        mock_hourly.Variables.side_effect = lambda i: [mock_var1, mock_var2, mock_var3][i]

        # Set up the response
        mock_response.Hourly.return_value = mock_hourly
        mock_response.Latitude.return_value = 49.71754
        mock_response.Longitude.return_value = 11.05877
        mock_response.Elevation.return_value = 100.0

        return mock_response

    def test_init(self):
        """Test initialization of OpenMeteoProvider."""
        provider = OpenMeteoProvider(cache_dir=".test_cache")
        assert provider.cache_dir == ".test_cache"
        assert provider.url == "https://archive-api.open-meteo.com/v1/archive"

    def test_validate_features_valid(self):
        """Test validation of valid features."""
        provider = OpenMeteoProvider()
        features = ["temperature_2m", "relative_humidity_2m"]
        validated = provider._validate_features(features)
        assert validated == features

    def test_validate_features_invalid(self):
        """Test validation of invalid features."""
        provider = OpenMeteoProvider()
        features = ["temperature_2m", "invalid_feature"]
        with pytest.raises(ValueError):
            provider._validate_features(features)

    def test_remove_cache_file(self, mock_os_path_exists, mock_os_remove):
        """Test removal of cache file."""
        provider = OpenMeteoProvider(cache_dir=".test_cache")
        provider._remove_cache_file()
        mock_os_path_exists.assert_called_once_with(".test_cache.sqlite")
        mock_os_remove.assert_called_once_with(".test_cache.sqlite")

    def test_remove_cache_file_not_exists(self):
        """Test removal of cache file when it doesn't exist."""
        with patch("os.path.exists", return_value=False) as mock_exists:
            with patch("os.remove") as mock_remove:
                provider = OpenMeteoProvider(cache_dir=".test_cache")
                provider._remove_cache_file()
                mock_exists.assert_called_once_with(".test_cache.sqlite")
                mock_remove.assert_not_called()

    def test_remove_cache_file_error(self, mock_os_path_exists):
        """Test handling of errors when removing cache file."""
        with patch("os.remove", side_effect=Exception("Test error")) as mock_remove:
            with patch("entise.services.weather.logger.warning") as mock_warning:
                provider = OpenMeteoProvider(cache_dir=".test_cache")
                provider._remove_cache_file()
                mock_os_path_exists.assert_called_once_with(".test_cache.sqlite")
                mock_remove.assert_called_once_with(".test_cache.sqlite")
                mock_warning.assert_called_once()

    @patch("pvlib.solarposition.get_solarposition")
    def test_calculate_sun_elevation(self, mock_get_solarposition):
        """Test calculation of sun elevation angles."""
        # Mock the solar position data
        mock_solar_position = pd.DataFrame({"zenith": [30.0, 40.0, 50.0]})
        mock_get_solarposition.return_value = mock_solar_position

        provider = OpenMeteoProvider()
        start_date = pd.Timestamp("2022-01-01")
        end_date = pd.Timestamp("2022-01-01 02:00:00")
        result = provider._calculate_sun_elevation(
            start_date, end_date, pd.Timedelta(hours=1), 49.71754, 11.05877, 100.0
        )

        assert len(result) == 3
        assert result.iloc[0] == 60.0  # 90 - 30
        assert result.iloc[1] == 50.0  # 90 - 40
        assert result.iloc[2] == 40.0  # 90 - 50

    def test_get_weather_data(self, mock_client, mock_response):
        """Test getting weather data from the API."""
        # Set up the mock client to return our mock response
        mock_client_instance = mock_client.return_value
        mock_client_instance.weather_api.return_value = [mock_response]

        # Create the provider with the mocked client
        provider = OpenMeteoProvider()
        provider.client = mock_client_instance

        # Mock the _response_to_dataframe method
        with patch.object(provider, "_response_to_dataframe") as mock_to_df:
            mock_df = pd.DataFrame(
                {
                    "datetime": pd.date_range("2022-01-01", periods=3, freq="h"),
                    "temperature_2m": [10.0, 11.0, 12.0],
                    "relative_humidity_2m": [80.0, 75.0, 70.0],
                    "surface_pressure": [1013.0, 1012.0, 1011.0],
                }
            )
            mock_to_df.return_value = mock_df

            # Mock the _remove_cache_file method
            with patch.object(provider, "_remove_cache_file") as mock_remove_cache:
                # Call the method
                result = provider.get_weather_data(
                    latitude=49.71754,
                    longitude=11.05877,
                    start_date="2022-01-01",
                    end_date="2022-01-31",
                    timezone="Europe/Berlin",
                    features=["temperature_2m", "relative_humidity_2m", "surface_pressure"],
                )

                # Check that the client was called with the correct parameters
                mock_client_instance.weather_api.assert_called_once()
                args, kwargs = mock_client_instance.weather_api.call_args
                assert args[0] == "https://archive-api.open-meteo.com/v1/archive"
                assert kwargs["params"]["latitude"] == 49.71754
                assert kwargs["params"]["longitude"] == 11.05877
                assert kwargs["params"]["start_date"] == "2022-01-01"
                assert kwargs["params"]["end_date"] == "2022-01-31"
                assert kwargs["params"]["timezone"] == "Europe/Berlin"
                assert set(kwargs["params"]["hourly"]) == {"temperature_2m", "relative_humidity_2m", "surface_pressure"}

                # Check that the response was processed correctly
                mock_to_df.assert_called_once_with(mock_response, kwargs["params"]["hourly"])

                # Check that the cache file was removed
                mock_remove_cache.assert_called_once()

                # Check the result
                assert result is mock_df

    def test_get_weather_data_error(self, mock_client):
        """Test handling of API errors."""
        # Set up the mock client to raise an exception
        mock_client_instance = mock_client.return_value
        mock_client_instance.weather_api.side_effect = Exception("API error")

        # Create the provider with the mocked client
        provider = OpenMeteoProvider()
        provider.client = mock_client_instance

        # Call the method and check that it raises the exception
        with pytest.raises(Exception, match="API error"):
            provider.get_weather_data(
                latitude=49.71754, longitude=11.05877, start_date="2022-01-01", end_date="2022-01-31"
            )


class TestWeatherService:
    """Tests for the WeatherService facade."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock weather provider for testing."""
        mock_provider = Mock(spec=WeatherProvider)
        mock_df = pd.DataFrame(
            {
                "datetime": pd.date_range("2022-01-01", periods=3, freq="h"),
                "temperature_2m": [10.0, 11.0, 12.0],
                "relative_humidity_2m": [80.0, 75.0, 70.0],
                "surface_pressure": [1013.0, 1012.0, 1011.0],
            }
        )
        mock_provider.get_weather_data.return_value = mock_df
        return mock_provider

    def test_init(self):
        """Test initialization of WeatherService."""
        service = WeatherService()
        assert "openmeteo" in service.providers
        assert isinstance(service.providers["openmeteo"], OpenMeteoProvider)

    def test_register_provider(self, mock_provider):
        """Test registering a new provider."""
        service = WeatherService()
        service.register_provider("test_provider", mock_provider)
        assert "test_provider" in service.providers
        assert service.providers["test_provider"] is mock_provider

    def test_register_provider_invalid(self):
        """Test registering an invalid provider."""
        service = WeatherService()
        with pytest.raises(TypeError):
            service.register_provider("invalid_provider", "not a provider")

    def test_get_weather_data(self, mock_provider):
        """Test getting weather data using a provider."""
        service = WeatherService()
        service.providers["test_provider"] = mock_provider

        result = service.get_weather_data(
            provider="test_provider",
            latitude=49.71754,
            longitude=11.05877,
            start_date="2022-01-01",
            end_date="2022-01-31",
        )

        mock_provider.get_weather_data.assert_called_once_with(
            latitude=49.71754, longitude=11.05877, start_date="2022-01-01", end_date="2022-01-31"
        )

        assert result is mock_provider.get_weather_data.return_value

    def test_get_weather_data_provider_not_found(self):
        """Test getting weather data with a non-existent provider."""
        service = WeatherService()
        with pytest.raises(ValueError, match="Provider 'non_existent' not found"):
            service.get_weather_data(provider="non_existent")

    def test_calculate_derived_variables(self):
        """Test calculation of derived variables."""
        service = WeatherService()

        # Create a test DataFrame
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2022-01-01", periods=3, freq="h"),
                "temperature_2m": [10.0, 11.0, 12.0],
                "relative_humidity_2m": [80.0, 75.0, 70.0],
                "surface_pressure": [1013.0, 1012.0, 1011.0],
            }
        )

        # Mock the _calculate_absolute_humidity method
        with patch.object(service, "_calculate_absolute_humidity") as mock_calc:
            mock_calc.return_value = pd.Series([0.01, 0.02, 0.03])

            # Call the method
            result = service.calculate_derived_variables(df, variables=["absolute_humidity"])

            # Check that the calculation method was called with the correct parameters
            mock_calc.assert_called_once()
            args, _ = mock_calc.call_args
            assert args[0].equals(df["relative_humidity_2m"])
            assert args[1].equals(df["temperature_2m"])
            assert args[2].equals(df["surface_pressure"])

            # Check the result
            assert "absolute_humidity" in result.columns
            assert result["absolute_humidity"].equals(mock_calc.return_value)

    def test_calculate_derived_variables_no_variables(self):
        """Test calculation of derived variables with no variables specified."""
        service = WeatherService()

        # Create a test DataFrame
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2022-01-01", periods=3, freq="h"),
                "temperature_2m": [10.0, 11.0, 12.0],
                "relative_humidity_2m": [80.0, 75.0, 70.0],
                "surface_pressure": [1013.0, 1012.0, 1011.0],
            }
        )

        # Mock the _calculate_absolute_humidity method
        with patch.object(service, "_calculate_absolute_humidity") as mock_calc:
            mock_calc.return_value = pd.Series([0.01, 0.02, 0.03])

            # Call the method
            result = service.calculate_derived_variables(df)

            # Check that the calculation method was called
            mock_calc.assert_called_once()

            # Check the result
            assert "absolute_humidity" in result.columns

    def test_calculate_derived_variables_missing_columns(self):
        """Test calculation of derived variables with missing columns."""
        service = WeatherService()

        # Create a test DataFrame with missing columns
        df = pd.DataFrame(
            {"datetime": pd.date_range("2022-01-01", periods=3, freq="h"), "temperature_2m": [10.0, 11.0, 12.0]}
        )

        # Mock the _calculate_absolute_humidity method
        with patch.object(service, "_calculate_absolute_humidity") as mock_calc:
            # Call the method
            result = service.calculate_derived_variables(df, variables=["absolute_humidity"])

            # Check that the calculation method was not called
            mock_calc.assert_not_called()

            # Check the result
            assert "absolute_humidity" not in result.columns

    def test_calculate_absolute_humidity(self):
        """Test calculation of absolute humidity."""
        service = WeatherService()

        # Create test data
        relative_humidity = pd.Series([80.0, 75.0, 70.0])
        temperature = pd.Series([10.0, 11.0, 12.0])
        pressure = pd.Series([1013.0, 1012.0, 1011.0])

        # Call the method
        result = service._calculate_absolute_humidity(relative_humidity, temperature, pressure)

        # Check the result
        assert len(result) == 3
        assert all(result > 0)  # Absolute humidity should be positive
