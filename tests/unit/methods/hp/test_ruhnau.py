"""
Tests for the Ruhnau heat pump COP method.
"""


import numpy as np
import pandas as pd
import pytest

from entise.constants import Types
from entise.methods.hp.ruhnau import HeatPumpSinks, HeatPumpSources, Ruhnau, TemperatureColumns


class TestRuhnau:
    @pytest.fixture
    def weather_data(self):
        """Create synthetic weather data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=24, freq="H")
        temperatures = np.sin(np.linspace(0, 2 * np.pi, 24)) * 10 + 10  # Temperatures between 0 and 20°C
        df = pd.DataFrame({"temperature_2m": temperatures}, index=dates)
        return df

    def test_initialization(self):
        """Test that the class initializes correctly."""
        ruhnau = Ruhnau()
        assert ruhnau.name == "Ruhnau"
        assert Types.HP in ruhnau.types

    def test_generate_basic(self, weather_data):
        """Test basic generation of COP time series."""
        ruhnau = Ruhnau()
        result = ruhnau.generate(weather=weather_data)

        # Check that the result has the expected structure
        assert "timeseries" in result
        assert "summary" in result

        # Check that the timeseries contains the COP column
        ts = result["timeseries"]
        assert "cop" in ts.columns

        # Check that the values are reasonable (COP typically between 1 and 8)
        assert ts["cop"].min() > 0
        assert ts["cop"].max() < 10

        # Check that the summary contains the expected information
        summary = result["summary"]
        assert "hp_source" in summary
        assert "hp_sink" in summary
        assert "hp_temp" in summary
        assert "correction_factor" in summary
        assert "cop_min" in summary
        assert "cop_max" in summary
        assert "cop_mean" in summary

        # Check that the default parameters are used
        assert summary["hp_source"] == ruhnau.DEFAULT_SOURCE
        assert summary["hp_sink"] == ruhnau.DEFAULT_SINK
        assert summary["hp_temp"] == ruhnau.DEFAULT_TEMP

    def test_correction_factor(self, weather_data):
        """Test that the correction factor is applied correctly."""
        ruhnau = Ruhnau()
        result1 = ruhnau.generate(weather=weather_data, correction_factor=1.0)
        result2 = ruhnau.generate(weather=weather_data, correction_factor=0.5)

        # The COP values with correction_factor=0.5 should be half of those with correction_factor=1.0
        ts1 = result1["timeseries"]
        ts2 = result2["timeseries"]

        # Check that the COP values are scaled by the correction factor
        np.testing.assert_allclose(ts2["cop"], ts1["cop"] * 0.5)

        # Check that the correction factor is included in the summary
        assert result1["summary"]["correction_factor"] == 1.0
        assert result2["summary"]["correction_factor"] == 0.5

    def test_temperature_calculations(self, weather_data):
        """Test that temperature calculations are performed correctly."""
        ruhnau = Ruhnau()

        # Access the private method for testing
        processed_weather = ruhnau._prepare_weather_data(weather_data)

        # Check that all required temperature columns exist
        assert TemperatureColumns.AIR_TEMP in processed_weather.columns
        assert TemperatureColumns.SOIL_TEMP in processed_weather.columns
        assert TemperatureColumns.GROUNDWATER_TEMP in processed_weather.columns

        # Check that groundwater temperature is constant
        assert (processed_weather[TemperatureColumns.GROUNDWATER_TEMP] == 10).all()

    def test_specific_parameters(self, weather_data):
        """Test that the specified parameters are used when provided."""
        ruhnau = Ruhnau()

        # Test with specific source, sink, and temperature
        result = ruhnau.generate(
            weather=weather_data, hp_source=HeatPumpSources.SOIL, hp_sink=HeatPumpSinks.RADIATOR, temp_sink=40
        )

        # Check that the specified parameters are used
        summary = result["summary"]
        assert summary["hp_source"] == HeatPumpSources.SOIL
        assert summary["hp_sink"] == HeatPumpSinks.RADIATOR
        assert summary["hp_temp"] == 40

    def test_invalid_parameters(self, weather_data):
        """Test that an error is raised when invalid parameters are provided."""
        ruhnau = Ruhnau()

        # Test with invalid source
        with pytest.raises(ValueError, match="Invalid heat pump source"):
            ruhnau.generate(weather=weather_data, hp_source="invalid_source")

        # Test with invalid sink
        with pytest.raises(ValueError, match="Invalid heat pump sink"):
            ruhnau.generate(weather=weather_data, hp_sink="invalid_sink")

    def test_closest_temperature(self, weather_data):
        """Test that the closest available temperature is used when the specified temperature is not available."""
        ruhnau = Ruhnau()

        # Test with a temperature that is not available for the specified sink
        # For floor heating, available temperatures are [25, 30, 35]
        result = ruhnau.generate(
            weather=weather_data,
            hp_sink=HeatPumpSinks.FLOOR,
            temp_sink=32,  # Not in [25, 30, 35], should use 30
        )

        # Check that the closest available temperature is used
        summary = result["summary"]
        assert summary["hp_temp"] == 30  # Closest to 32 in [25, 30, 35]
