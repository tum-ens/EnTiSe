import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.iea_annex42 import IEAAnnex42HouseholdTypeDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

@pytest.fixture
def mock_demand_data():
    """Create mock demand data for testing."""
    return pd.DataFrame({
        'household_type': ['single_adult', 'couple_no_children', 'family_with_children', 'elderly_couple', 'shared_accommodation'],
        'liters_per_day': [50, 80, 120, 70, 100],
        'sigma': [10, 16, 24, 14, 20]
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
        O.HOUSEHOLD_TYPE: 'family_with_children',
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

def test_iea_annex42_household_type_init():
    """Test initialization of IEAAnnex42HouseholdTypeDHW."""
    dhw = IEAAnnex42HouseholdTypeDHW()
    assert dhw.name == "IEAAnnex42HouseholdTypeDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.HOUSEHOLD_TYPE in dhw.required_keys
    assert O.DHW_DEMAND_FILE in dhw.optional_keys

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand(mock_normal, mock_read_csv, mock_obj, mock_data, mock_demand_data):
    """Test _calculate_daily_demand method."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 130.0
    
    dhw = IEAAnnex42HouseholdTypeDHW()
    daily_demand = dhw._calculate_daily_demand(mock_obj, mock_data)
    
    # Check that the daily demand is calculated correctly
    # For household_type='family_with_children', liters_per_day=120, sigma=24
    # With random variation, it's 130.0 liters (mocked)
    assert daily_demand == 130.0
    
    # Check that the read_csv function was called with the correct arguments
    mock_read_csv.assert_called_once()
    
    # Check that the random.normal function was called with the correct arguments
    # For household_type='family_with_children', liters_per_day=120, sigma=24
    mock_normal.assert_called_once_with(120.0, 24.0)

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand_unknown_household_type(mock_normal, mock_read_csv, mock_obj, mock_data, mock_demand_data):
    """Test _calculate_daily_demand method with unknown household type."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 60.0
    
    # Create a copy of the object with an unknown household type
    obj_with_unknown_type = mock_obj.copy()
    obj_with_unknown_type[O.HOUSEHOLD_TYPE] = 'unknown_type'
    
    # Mock the logger to avoid warnings in the test output
    with patch('entise.methods.dhw.iea_annex42.logger'):
        dhw = IEAAnnex42HouseholdTypeDHW()
        daily_demand = dhw._calculate_daily_demand(obj_with_unknown_type, mock_data)
    
    # Check that the daily demand is calculated correctly
    # For unknown household type, it should use the first row in the data
    # For the first row, liters_per_day=50, sigma=10
    # With random variation, it's 60.0 liters (mocked)
    assert daily_demand == 60.0
    
    # Check that the read_csv function was called with the correct arguments
    mock_read_csv.assert_called_once()
    
    # Check that the random.normal function was called with the correct arguments
    # For the first row, liters_per_day=50, sigma=10
    mock_normal.assert_called_once_with(50.0, 10.0)

def test_get_default_activity_file():
    """Test get_default_activity_file method."""
    dhw = IEAAnnex42HouseholdTypeDHW()
    path = dhw.get_default_activity_file()
    assert path == os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

@patch('pandas.read_csv')
def test_generate(mock_read_csv, mock_obj, mock_data, mock_activity_data, mock_demand_data):
    """Test generate method."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_data]
    
    # Mock the _calculate_daily_demand method to return a fixed value
    with patch.object(IEAAnnex42HouseholdTypeDHW, '_calculate_daily_demand', return_value=100.0):
        dhw = IEAAnnex42HouseholdTypeDHW()
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
    with patch.object(IEAAnnex42HouseholdTypeDHW, 'get_data_file', return_value='custom_demand_file.csv'):
        # Mock the _calculate_daily_demand method to return a fixed value
        with patch.object(IEAAnnex42HouseholdTypeDHW, '_calculate_daily_demand', return_value=100.0):
            dhw = IEAAnnex42HouseholdTypeDHW()
            result = dhw.generate(obj_with_custom_file, mock_data, Types.DHW)
    
    # Check that the result has the expected structure
    assert 'summary' in result
    assert 'timeseries' in result