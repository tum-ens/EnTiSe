import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.jordan_vajen import JordanVajenDwellingSizeDHW, JordanVajenWeekdayActivityDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

@pytest.fixture
def mock_demand_data():
    """Create mock demand data for testing."""
    return pd.DataFrame({
        'dwelling_size': [0, 40, 55, 70, 85, 100, 120, 140, 160, 180],
        'm3_per_m2_a': [0.28, 0.25, 0.26, 0.26, 0.23, 0.21, 0.18, 0.16, 0.13, 0.14],
        'sigma': [0.14, 0.23, 0.10, 0.14, 0.10, 0.11, 0.11, 0.12, 0.09, 0.11]
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
    index = pd.date_range('2025-01-01', periods=24, freq='H')
    return pd.DataFrame({
        C.DATETIME: index,
        C.TEMP_OUT: np.full(24, 10.0)
    }, index=index)

@pytest.fixture
def mock_obj():
    """Create mock object data for testing."""
    return {
        O.ID: 'test_obj',
        O.WEATHER: 'weather',
        O.DWELLING_SIZE: 100,
        O.TEMP_COLD: 10,
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

def test_jordan_vajen_dwelling_size_init():
    """Test initialization of JordanVajenDwellingSizeDHW."""
    dhw = JordanVajenDwellingSizeDHW()
    assert dhw.name == "JordanVajenDwellingSizeDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.DWELLING_SIZE in dhw.required_keys
    assert O.DHW_DEMAND_FILE in dhw.optional_keys

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand(mock_normal, mock_read_csv, mock_obj, mock_data, mock_demand_data):
    """Test _calculate_daily_demand method."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 120.0
    
    dhw = JordanVajenDwellingSizeDHW()
    daily_demand = dhw._calculate_daily_demand(mock_obj, mock_data)
    
    # Check that the daily demand is calculated correctly
    # For dwelling_size=100, m3_per_m2_a=0.21, the annual demand is 21 m³
    # The daily demand is 21 * 1000 / 365 = 57.53 liters
    # With random variation, it's 120.0 liters (mocked)
    assert daily_demand == 120.0
    
    # Check that the read_csv function was called with the correct arguments
    mock_read_csv.assert_called_once()
    
    # Check that the random.normal function was called with the correct arguments
    # For dwelling_size=100, m3_per_m2_a=0.21, sigma=0.11
    # The mean is 21 * 1000 / 365 = 57.53 liters
    # The standard deviation is 57.53 * 0.11 = 6.33 liters
    expected_mean = 0.21 * 100 * 1000 / 365
    expected_std = expected_mean * 0.11
    mock_normal.assert_called_once_with(expected_mean, expected_std)

def test_get_default_activity_file():
    """Test get_default_activity_file method."""
    dhw = JordanVajenDwellingSizeDHW()
    path = dhw.get_default_activity_file()
    assert path == os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

@patch('pandas.read_csv')
def test_generate(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_demand_data):
    """Test generate method."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_data]
    
    # Mock the _calculate_daily_demand method to return a fixed value
    with patch.object(JordanVajenDwellingSizeDHW, '_calculate_daily_demand', return_value=100.0):
        dhw = JordanVajenDwellingSizeDHW()
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
    assert len(ts) == 24  # Should match the length of the weather data

def test_jordan_vajen_weekday_activity_init():
    """Test initialization of JordanVajenWeekdayActivityDHW."""
    dhw = JordanVajenWeekdayActivityDHW()
    assert dhw.name == "JordanVajenWeekdayActivityDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys

def test_jordan_vajen_weekday_activity_get_default_activity_file():
    """Test get_default_activity_file method."""
    dhw = JordanVajenWeekdayActivityDHW()
    path = dhw.get_default_activity_file()
    assert path == os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

def test_jordan_vajen_weekday_activity_calculate_daily_demand():
    """Test _calculate_daily_demand method."""
    dhw = JordanVajenWeekdayActivityDHW()
    with pytest.raises(NotImplementedError):
        dhw._calculate_daily_demand({}, {})