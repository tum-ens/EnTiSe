import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

# Create a concrete implementation of BaseProbabilisticDHW for testing
class ConcreteDHW(BaseProbabilisticDHW):
    """Concrete implementation of BaseProbabilisticDHW for testing."""
    name = "ConcreteDHW"
    
    def _calculate_daily_demand(self, obj, data):
        """Return a fixed daily demand for testing."""
        return 100.0  # 100 liters per day
    
    def get_default_activity_file(self):
        """Return a test activity file path."""
        return os.path.join('tests', 'fixtures', 'dhw_activity_test.csv')

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

def test_init():
    """Test initialization of BaseProbabilisticDHW."""
    dhw = ConcreteDHW()
    assert dhw.name == "ConcreteDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.DHW_ACTIVITY_FILE in dhw.optional_keys
    assert O.WEATHER in dhw.required_timeseries

@patch('pandas.read_csv')
def test_generate(mock_read_csv, mock_obj, mock_data, mock_activity_data):
    """Test generate method."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.return_value = mock_activity_data
    
    dhw = ConcreteDHW()
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

def test_get_data_file():
    """Test get_data_file method."""
    # Test with user_file that exists
    with patch('os.path.exists', return_value=True):
        path = BaseProbabilisticDHW.get_data_file('source', 'filename.csv', 'user_file.csv')
        assert path == 'user_file.csv'
    
    # Test with user_file that doesn't exist but exists in user directory
    with patch('os.path.exists', side_effect=[False, True]):
        with patch.object(BaseProbabilisticDHW, 'get_user_data_path', return_value='user/path/user_file.csv'):
            path = BaseProbabilisticDHW.get_data_file('source', 'filename.csv', 'user_file.csv')
            assert path == 'user/path/user_file.csv'
    
    # Test with default filename that exists in user directory
    with patch('os.path.exists', return_value=False):
        with patch.object(BaseProbabilisticDHW, 'get_user_data_path', side_effect=[None, 'user/path/filename.csv']):
            path = BaseProbabilisticDHW.get_data_file('source', 'filename.csv', 'user_file.csv')
            assert path == 'user/path/filename.csv'
    
    # Test with source-specific file
    with patch('os.path.exists', return_value=False):
        with patch.object(BaseProbabilisticDHW, 'get_user_data_path', side_effect=[None, None]):
            path = BaseProbabilisticDHW.get_data_file('source', 'filename.csv', 'user_file.csv')
            assert path == os.path.join('entise', 'data', 'dhw', 'source', 'filename.csv')

def test_get_user_data_path():
    """Test get_user_data_path method."""
    # Test with file that exists
    with patch('os.path.exists', return_value=True):
        path = BaseProbabilisticDHW.get_user_data_path('filename.csv')
        assert path == os.path.join('entise', 'data', 'dhw', 'user', 'filename.csv')
    
    # Test with file that doesn't exist
    with patch('os.path.exists', return_value=False):
        path = BaseProbabilisticDHW.get_user_data_path('filename.csv')
        assert path is None

@patch('pandas.read_csv')
def test_generate_timeseries(mock_read_csv, mock_obj, mock_data, mock_activity_data):
    """Test _generate_timeseries method."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.return_value = mock_activity_data
    
    dhw = ConcreteDHW()
    
    # Call _generate_timeseries directly
    ts_volume, ts_energy = dhw._generate_timeseries(
        mock_data[O.WEATHER],
        mock_activity_data,
        100.0,  # daily_demand
        10.0,   # temp_cold
        60.0,   # temp_hot
        0.1,    # seasonal_variation
        15      # seasonal_peak_day
    )
    
    # Check that the timeseries have the expected length
    assert len(ts_volume) == 24
    assert len(ts_energy) == 24
    
    # Check that the energy is calculated correctly
    # Energy = volume * density * specific_heat * (temp_hot - temp_cold) / 3600
    energy_factor = 1000 * 4186 * (60 - 10) / 3600  # Convert to Wh
    assert np.isclose(ts_energy.sum(), ts_volume.sum() * energy_factor)