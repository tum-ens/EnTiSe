import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.user import UserDefinedDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

@pytest.fixture
def mock_demand_by_dwelling_data():
    """Create mock demand data by dwelling size for testing."""
    return pd.DataFrame({
        'dwelling_size': [0, 40, 55, 70, 85, 100, 120, 140, 160, 180],
        'm3_per_m2_a': [0.28, 0.25, 0.26, 0.26, 0.23, 0.21, 0.18, 0.16, 0.13, 0.14],
        'sigma': [0.14, 0.23, 0.10, 0.14, 0.10, 0.11, 0.11, 0.12, 0.09, 0.11]
    })

@pytest.fixture
def mock_demand_by_occupants_data():
    """Create mock demand data by occupants for testing."""
    return pd.DataFrame({
        'occupants': [1, 2, 3, 4, 5, 6, 7, 8],
        'liters_per_day': [36, 57, 76, 95, 114, 132, 151, 170],
        'sigma': [7.2, 11.4, 15.2, 19.0, 22.8, 26.4, 30.2, 34.0]
    })

@pytest.fixture
def mock_demand_by_household_type_data():
    """Create mock demand data by household type for testing."""
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
def mock_obj_dwelling():
    """Create mock object data with dwelling size for testing."""
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
def mock_obj_occupants():
    """Create mock object data with occupants for testing."""
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
def mock_obj_household_type():
    """Create mock object data with household type for testing."""
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
def mock_obj_no_params():
    """Create mock object data with no specific parameters for testing."""
    return {
        O.ID: 'test_obj',
        O.WEATHER: 'weather',
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

def test_user_defined_dhw_init():
    """Test initialization of UserDefinedDHW."""
    dhw = UserDefinedDHW()
    assert dhw.name == "UserDefinedDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.DWELLING_SIZE in dhw.optional_keys
    assert O.OCCUPANTS in dhw.optional_keys
    assert O.HOUSEHOLD_TYPE in dhw.optional_keys
    assert O.DHW_DEMAND_FILE in dhw.optional_keys
    assert O.DHW_ACTIVITY_FILE in dhw.optional_keys

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand_with_dwelling_size(mock_normal, mock_read_csv, mock_obj_dwelling, mock_data, mock_demand_by_dwelling_data):
    """Test _calculate_daily_demand method with dwelling size."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_by_dwelling_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 120.0
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(UserDefinedDHW, 'get_data_file', return_value='dhw_demand_by_dwelling.csv'):
        dhw = UserDefinedDHW()
        daily_demand = dhw._calculate_daily_demand(mock_obj_dwelling, mock_data)
    
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

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand_with_occupants(mock_normal, mock_read_csv, mock_obj_occupants, mock_data, mock_demand_by_occupants_data):
    """Test _calculate_daily_demand method with occupants."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_by_occupants_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 100.0
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(UserDefinedDHW, 'get_data_file', return_value='dhw_demand_by_occupants.csv'):
        dhw = UserDefinedDHW()
        daily_demand = dhw._calculate_daily_demand(mock_obj_occupants, mock_data)
    
    # Check that the daily demand is calculated correctly
    # For occupants=4, liters_per_day=95, sigma=19.0
    # With random variation, it's 100.0 liters (mocked)
    assert daily_demand == 100.0
    
    # Check that the read_csv function was called with the correct arguments
    mock_read_csv.assert_called_once()
    
    # Check that the random.normal function was called with the correct arguments
    # For occupants=4, liters_per_day=95, sigma=19.0
    mock_normal.assert_called_once_with(95.0, 19.0)

@patch('pandas.read_csv')
@patch('numpy.random.normal')
def test_calculate_daily_demand_with_household_type(mock_normal, mock_read_csv, mock_obj_household_type, mock_data, mock_demand_by_household_type_data):
    """Test _calculate_daily_demand method with household type."""
    # Mock the read_csv function to return mock demand data
    mock_read_csv.return_value = mock_demand_by_household_type_data
    
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 130.0
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(UserDefinedDHW, 'get_data_file', return_value='dhw_demand_by_household_type.csv'):
        dhw = UserDefinedDHW()
        daily_demand = dhw._calculate_daily_demand(mock_obj_household_type, mock_data)
    
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
@patch('entise.methods.dhw.user.logger')
def test_calculate_daily_demand_with_no_params(mock_logger, mock_normal, mock_read_csv, mock_obj_no_params, mock_data):
    """Test _calculate_daily_demand method with no specific parameters."""
    # Mock the random.normal function to return a fixed value
    mock_normal.return_value = 110.0
    
    dhw = UserDefinedDHW()
    daily_demand = dhw._calculate_daily_demand(mock_obj_no_params, mock_data)
    
    # Check that the daily demand is calculated correctly
    # For no specific parameters, it should use the default value of 100 liters per day
    # With random variation, it's 110.0 liters (mocked)
    assert daily_demand == 110.0
    
    # Check that the logger.warning was called
    mock_logger.warning.assert_called_once()
    
    # Check that the random.normal function was called with the correct arguments
    # For default values, daily_demand_l=100, sigma=20
    mock_normal.assert_called_once_with(100, 20)

def test_get_default_activity_file():
    """Test get_default_activity_file method."""
    # Test with user-provided activity file that exists
    with patch('os.path.exists', return_value=True):
        dhw = UserDefinedDHW()
        path = dhw.get_default_activity_file()
        assert path == os.path.join('entise', 'data', 'dhw', 'user', 'dhw_activity.csv')
    
    # Test with user-provided activity file that doesn't exist
    with patch('os.path.exists', return_value=False):
        dhw = UserDefinedDHW()
        path = dhw.get_default_activity_file()
        assert path == os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

@patch('pandas.read_csv')
def test_generate(mock_read_csv, mock_obj_dwelling, mock_data, mock_activity_data, mock_demand_by_dwelling_data):
    """Test generate method."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_by_dwelling_data]
    
    # Mock the _calculate_daily_demand method to return a fixed value
    with patch.object(UserDefinedDHW, '_calculate_daily_demand', return_value=100.0):
        dhw = UserDefinedDHW()
        result = dhw.generate(mock_obj_dwelling, mock_data, Types.DHW)
    
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
def test_with_custom_demand_file(mock_read_csv, mock_obj_dwelling, mock_data, mock_activity_data, mock_demand_by_dwelling_data):
    """Test with custom demand file."""
    # Add a custom demand file to the object
    obj_with_custom_file = mock_obj_dwelling.copy()
    obj_with_custom_file[O.DHW_DEMAND_FILE] = 'custom_demand_file.csv'
    
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_by_dwelling_data]
    
    # Mock the get_data_file method to return a fixed path
    with patch.object(UserDefinedDHW, 'get_data_file', return_value='custom_demand_file.csv'):
        # Mock the _calculate_daily_demand method to return a fixed value
        with patch.object(UserDefinedDHW, '_calculate_daily_demand', return_value=100.0):
            dhw = UserDefinedDHW()
            result = dhw.generate(obj_with_custom_file, mock_data, Types.DHW)
    
    # Check that the result has the expected structure
    assert 'summary' in result
    assert 'timeseries' in result

@patch('pandas.read_csv')
def test_with_custom_activity_file(mock_read_csv, mock_obj_dwelling, mock_data, mock_activity_data, mock_demand_by_dwelling_data):
    """Test with custom activity file."""
    # Add a custom activity file to the object
    obj_with_custom_file = mock_obj_dwelling.copy()
    obj_with_custom_file[O.DHW_ACTIVITY_FILE] = 'custom_activity_file.csv'
    
    # Mock the read_csv function to return mock activity data
    mock_read_csv.side_effect = [mock_activity_data, mock_demand_by_dwelling_data]
    
    # Mock the _calculate_daily_demand method to return a fixed value
    with patch.object(UserDefinedDHW, '_calculate_daily_demand', return_value=100.0):
        dhw = UserDefinedDHW()
        result = dhw.generate(obj_with_custom_file, mock_data, Types.DHW)
    
    # Check that the result has the expected structure
    assert 'summary' in result
    assert 'timeseries' in result