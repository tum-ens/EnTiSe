import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.vdi4655 import VDI4655ColdWaterTemperatureDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

@pytest.fixture
def mock_cold_water_temp_data():
    """Create mock cold water temperature data for testing."""
    return pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'temperature_c': [8.0, 7.5, 8.0, 9.5, 11.0, 13.0, 14.5, 15.0, 14.0, 12.5, 10.5, 9.0]
    })

@pytest.fixture
def mock_activity_data():
    """Create mock activity data for testing."""
    return pd.DataFrame({
        'day': [0, 0, 0],
        'time': ['06:00:00', '12:00:00', '18:00:00'],
        'event': ['shower', 'sink', 'bath'],
        'probability': [0.5, 0.3, 0.2],
        'duration': [300, 60, 600],
        'flow_rate': [0.133, 0.067, 0.2],
        'sigma_duration': [120, 20, 180],
        'sigma_flow_rate': [0.033, 0.015, 0.05]
    })

@pytest.fixture
def mock_weather_data():
    """Create mock weather data for testing."""
    # Create a weather dataframe with data for multiple months
    index = pd.date_range('2025-01-01', periods=90, freq='D')
    return pd.DataFrame({
        C.DATETIME: index,
        C.TEMP_OUT: np.full(90, 10.0)
    }, index=index)

@pytest.fixture
def mock_obj():
    """Create mock object data for testing."""
    return {
        O.ID: 'test_obj',
        O.WEATHER: 'weather',
        O.TEMP_HOT: 60,
        O.SEASONAL_VARIATION: 0.1,
        O.SEASONAL_PEAK_DAY: 15
    }

@pytest.fixture
def mock_data(mock_weather_data):
    """Create mock data for testing."""
    return {
        O.WEATHER: mock_weather_data
    }

def test_vdi4655_cold_water_temperature_init():
    """Test initialization of VDI4655ColdWaterTemperatureDHW."""
    dhw = VDI4655ColdWaterTemperatureDHW()
    assert dhw.name == "VDI4655ColdWaterTemperatureDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.TEMP_COLD in dhw.optional_keys

def test_calculate_daily_demand():
    """Test _calculate_daily_demand method."""
    dhw = VDI4655ColdWaterTemperatureDHW()
    with pytest.raises(NotImplementedError):
        dhw._calculate_daily_demand({}, {})

# Create a concrete implementation of VDI4655ColdWaterTemperatureDHW for testing
class ConcreteVDI4655DHW(VDI4655ColdWaterTemperatureDHW):
    """Concrete implementation of VDI4655ColdWaterTemperatureDHW for testing."""
    
    def _calculate_daily_demand(self, obj, data):
        """Return a fixed daily demand for testing."""
        return 100.0  # 100 liters per day

@patch('pandas.read_csv')
def test_generate_with_concrete_implementation(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_cold_water_temp_data):
    """Test generate method with a concrete implementation."""
    # Mock the read_csv function to return mock activity data and cold water temperature data
    mock_read_csv.side_effect = [mock_activity_data, mock_cold_water_temp_data]
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(ConcreteVDI4655DHW, 'get_data_file', return_value='cold_water_temperature.csv'):
        dhw = ConcreteVDI4655DHW()
        result = dhw.generate(mock_obj, mock_data, Types.DHW)
    
    # Check that the result has the expected structure
    assert 'summary' in result
    assert 'timeseries' in result
    
    # Check summary values
    summary = result['summary']
    assert f'{C.DEMAND}_{Types.DHW}_volume' in summary
    assert f'{C.DEMAND}_{Types.DHW}_energy' in summary
    
    # Check timeseries values
    ts = result['timeseries']
    assert f'{C.LOAD}_{Types.DHW}_volume' in ts.columns
    assert f'{C.LOAD}_{Types.DHW}_energy' in ts.columns
    assert len(ts) == 90  # Should match the length of the weather data

@patch('pandas.read_csv')
def test_generate_timeseries_with_variable_temp(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_cold_water_temp_data):
    """Test _generate_timeseries_with_variable_temp method."""
    # Mock the read_csv function to return mock activity data and cold water temperature data
    mock_read_csv.side_effect = [mock_activity_data, mock_cold_water_temp_data]
    
    # Create a spy on the _generate_timeseries_with_variable_temp method to check its inputs
    with patch.object(ConcreteVDI4655DHW, '_generate_timeseries_with_variable_temp', wraps=ConcreteVDI4655DHW._generate_timeseries_with_variable_temp) as spy:
        # Mock the get_data_file method to return a fixed path
        with patch.object(ConcreteVDI4655DHW, 'get_data_file', return_value='cold_water_temperature.csv'):
            dhw = ConcreteVDI4655DHW()
            result = dhw.generate(mock_obj, mock_data, Types.DHW)
    
    # Check that the _generate_timeseries_with_variable_temp method was called with the correct arguments
    spy.assert_called_once()
    _, activity_data, daily_demand, cold_water_temp_data, temp_hot, seasonal_variation, seasonal_peak_day = spy.call_args[0]
    
    # Check that the arguments have the expected values
    assert daily_demand == 100.0
    assert temp_hot == 60
    assert seasonal_variation == 0.1
    assert seasonal_peak_day == 15
    assert len(cold_water_temp_data) == 12  # 12 months
    assert 'month' in cold_water_temp_data.columns
    assert 'temperature_c' in cold_water_temp_data.columns

@patch('pandas.read_csv')
def test_monthly_temperature_variation(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_cold_water_temp_data):
    """Test that the cold water temperature varies by month."""
    # Mock the read_csv function to return mock activity data and cold water temperature data
    mock_read_csv.side_effect = [mock_activity_data, mock_cold_water_temp_data]
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(ConcreteVDI4655DHW, 'get_data_file', return_value='cold_water_temperature.csv'):
        dhw = ConcreteVDI4655DHW()
        result = dhw.generate(mock_obj, mock_data, Types.DHW)
    
    # Get the timeseries
    ts = result['timeseries']
    
    # Check that the energy demand varies by month
    # January (month 1)
    january_energy = ts.loc['2025-01-01':, f'{C.LOAD}_{Types.DHW}_energy'].mean()
    
    # February (month 2)
    february_energy = ts.loc['2025-02-01':, f'{C.LOAD}_{Types.DHW}_energy'].mean()
    
    # March (month 3)
    march_energy = ts.loc['2025-03-01':, f'{C.LOAD}_{Types.DHW}_energy'].mean()
    
    # The energy demand should be different for different months due to different cold water temperatures
    assert january_energy != february_energy
    assert january_energy != march_energy
    assert february_energy != march_energy