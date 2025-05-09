import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.hendron_burch import HendronBurchOccupantsDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

@pytest.fixture
def mock_demand_data():
    """Create mock demand data for testing."""
    return pd.DataFrame({
        'occupants': [1, 2, 3, 4, 5, 6, 7, 8],
        'liters_per_day': [36, 57, 76, 95, 114, 132, 151, 170],
        'sigma': [7.2, 11.4, 15.2, 19.0, 22.8, 26.4, 30.2, 34.0]
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
        O.OCCUPANTS: 4,
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

def test_hendron_burch_occupants_init():
    """Test initialization of HendronBurchOccupantsDHW."""
    dhw = HendronBurchOccupantsDHW()
    assert dhw.name == "HendronBurchOccupantsDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.OCCUPANTS in dhw.required_keys
    assert O.DHW_DEMAND_FILE in dhw.optional_keys

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand(mock_normal, mock_read_csv, mock_obj, mock_data, mock_demand_data):
    """Test _calculate_daily_demand method."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 100.0
    
    dhw = HendronBurchOccupantsDHW()
    daily_demand = dhw._calculate_daily_demand(mock_obj, mock_data)
    
    # Check that the daily demand is calculated correctly
    # For occupants=4, liters_per_day=95, sigma=19.0
    # With random variation, it's 100.0 liters (mocked)
    assert daily_demand == 100.0
    
    # Check that the read_csv function was called with the correct arguments
    mock_read_csv.assert_called_once()
    
    # Check that the random.normal function was called with the correct arguments
    # For occupants=4, liters_per_day=95, sigma=19.0
    mock_normal.assert_called_once_with(95.0, 19.0)

def test_get_default_activity_file():
    """Test get_default_activity_file method."""
    dhw = HendronBurchOccupantsDHW()
    path = dhw.get_default_activity_file()
    assert path == os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

@patch('pandas.read_csv')
def test_generate(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_demand_data):
    """Test generate method."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_data]
    
    # Mock the _calculate_daily_demand method to return a fixed value
    with patch.object(HendronBurchOccupantsDHW, '_calculate_daily_demand', return_value=100.0):
        dhw = HendronBurchOccupantsDHW()
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

@patch('pandas.read_csv')
def test_with_custom_demand_file(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_demand_data):
    """Test with custom demand file."""
    # Add a custom demand file to the object
    obj_with_custom_file = mock_obj.copy()
    obj_with_custom_file[O.DHW_DEMAND_FILE] = 'custom_demand_file.csv'
    
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_data]
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(HendronBurchOccupantsDHW, 'get_data_file', return_value='custom_demand_file.csv'):
        # Mock the _calculate_daily_demand method to return a fixed value
        with patch.object(HendronBurchOccupantsDHW, '_calculate_daily_demand', return_value=100.0):
            dhw = HendronBurchOccupantsDHW()
            result = dhw.generate(obj_with_custom_file, mock_data, Types.DHW)
    
    # Check that the result has the expected structure
    assert 'summary' in result
    assert 'timeseries' in result