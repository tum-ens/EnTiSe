import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.ashrae import ASHRAEWeekendActivityDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

@pytest.fixture
def mock_activity_data():
    """Create mock activity data for testing."""
    return pd.DataFrame({
        'day': [5, 5, 5, 6, 6, 6],  # Weekend days (5=Saturday, 6=Sunday)
        'time': ['08:00:00', '12:00:00', '18:00:00', '08:00:00', '12:00:00', '18:00:00'],
        'event': ['shower', 'sink', 'bath', 'shower', 'sink', 'bath'],
        'probability': [0.12, 0.09, 0.02, 0.12, 0.09, 0.02],
        'duration': [300, 60, 600, 300, 60, 600],
        'flow_rate': [0.133, 0.067, 0.2, 0.133, 0.067, 0.2],
        'sigma_duration': [120, 20, 180, 120, 20, 180],
        'sigma_flow_rate': [0.033, 0.015, 0.05, 0.033, 0.015, 0.05]
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

def test_ashrae_weekend_activity_init():
    """Test initialization of ASHRAEWeekendActivityDHW."""
    dhw = ASHRAEWeekendActivityDHW()
    assert dhw.name == "ASHRAEWeekendActivityDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys

def test_get_default_activity_file():
    """Test get_default_activity_file method."""
    dhw = ASHRAEWeekendActivityDHW()
    path = dhw.get_default_activity_file()
    assert path == os.path.join('entise', 'data', 'dhw', 'ashrae', 'dhw_activity_weekend.csv')

def test_calculate_daily_demand():
    """Test _calculate_daily_demand method."""
    dhw = ASHRAEWeekendActivityDHW()
    with pytest.raises(NotImplementedError):
        dhw._calculate_daily_demand({}, {})

# Create a concrete implementation of ASHRAEWeekendActivityDHW for testing
class ConcreteASHRAEWeekendActivityDHW(ASHRAEWeekendActivityDHW):
    """Concrete implementation of ASHRAEWeekendActivityDHW for testing."""
    
    def _calculate_daily_demand(self, obj, data):
        """Return a fixed daily demand for testing."""
        return 100.0  # 100 liters per day

@patch('pandas.read_csv')
def test_generate_with_concrete_implementation(mock_read_csv, mock_obj, mock_data, mock_activity_data):
    """Test generate method with a concrete implementation."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.return_value = mock_activity_data
    
    dhw = ConcreteASHRAEWeekendActivityDHW()
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
def test_weekend_activity_patterns(mock_read_csv, mock_obj, mock_data, mock_activity_data):
    """Test that weekend activity patterns are used."""
    # Mock the read_csv function to return mock activity data
    mock_read_csv.return_value = mock_activity_data
    
    # Create a weather dataframe with weekend days
    weekend_index = pd.date_range('2025-01-04', periods=48, freq='H')  # Jan 4-5, 2025 is a weekend
    weekend_weather = pd.DataFrame({
        C.DATETIME: weekend_index,
        C.TEMP_OUT: np.full(48, 10.0)
    }, index=weekend_index)
    
    weekend_data = {
        O.WEATHER: weekend_weather
    }
    
    # Create a spy on the _generate_timeseries method to check its inputs
    with patch.object(ConcreteASHRAEWeekendActivityDHW, '_generate_timeseries', wraps=ConcreteASHRAEWeekendActivityDHW._generate_timeseries) as spy:
        dhw = ConcreteASHRAEWeekendActivityDHW()
        result = dhw.generate(mock_obj, weekend_data, Types.DHW)
    
    # Check that the _generate_timeseries method was called with the weekend activity data
    spy.assert_called_once()
    _, activity_data, *_ = spy.call_args[0]
    
    # Check that the activity data contains weekend days
    assert 5 in activity_data['day'].values  # Saturday
    assert 6 in activity_data['day'].values  # Sunday