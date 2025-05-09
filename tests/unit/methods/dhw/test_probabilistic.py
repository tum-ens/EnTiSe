import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from entise.methods.dhw.probabilistic import ProbabilisticDHW
from entise.methods.dhw.jordan_vajen import JordanVajenDwellingSizeDHW
from entise.methods.dhw.hendron_burch import HendronBurchOccupantsDHW
from entise.methods.dhw.iea_annex42 import IEAAnnex42HouseholdTypeDHW
from entise.methods.dhw.vdi4655 import VDI4655ColdWaterTemperatureDHW
from entise.methods.dhw.user import UserDefinedDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

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
def mock_data(mock_weather_data):
    """Create mock data for testing."""
    return {
        O.WEATHER: mock_weather_data
    }

def test_probabilistic_dhw_init():
    """Test initialization of ProbabilisticDHW."""
    dhw = ProbabilisticDHW()
    assert dhw.name == "ProbabilisticDHW"
    assert dhw.types == [Types.DHW]
    assert O.WEATHER in dhw.required_keys
    assert O.DWELLING_SIZE in dhw.optional_keys
    assert O.OCCUPANTS in dhw.optional_keys
    assert O.HOUSEHOLD_TYPE in dhw.optional_keys
    assert O.SOURCE in dhw.optional_keys
    assert O.WEEKEND_ACTIVITY in dhw.optional_keys

@patch.object(JordanVajenDwellingSizeDHW, 'generate')
def test_source_selection_jordan_vajen(mock_generate, mock_data):
    """Test source selection for jordan_vajen."""
    # Create an object with jordan_vajen source and dwelling_size
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'jordan_vajen',
        O.DWELLING_SIZE: 100
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0] == obj
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(JordanVajenDwellingSizeDHW, 'generate')
@patch('entise.methods.dhw.probabilistic.logger')
def test_source_selection_jordan_vajen_no_params(mock_logger, mock_generate, mock_data):
    """Test source selection for jordan_vajen with no specific parameters."""
    # Create an object with jordan_vajen source but no dwelling_size
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'jordan_vajen'
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the logger.warning was called
    mock_logger.warning.assert_called_once()
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0][O.DWELLING_SIZE] == 100  # Default dwelling size
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(HendronBurchOccupantsDHW, 'generate')
def test_source_selection_hendron_burch(mock_generate, mock_data):
    """Test source selection for hendron_burch."""
    # Create an object with hendron_burch source and occupants
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'hendron_burch',
        O.OCCUPANTS: 4
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0] == obj
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(HendronBurchOccupantsDHW, 'generate')
@patch('entise.methods.dhw.probabilistic.logger')
def test_source_selection_hendron_burch_no_params(mock_logger, mock_generate, mock_data):
    """Test source selection for hendron_burch with no specific parameters."""
    # Create an object with hendron_burch source but no occupants
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'hendron_burch'
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the logger.warning was called
    mock_logger.warning.assert_called_once()
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0][O.OCCUPANTS] == 3  # Default occupants
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(IEAAnnex42HouseholdTypeDHW, 'generate')
def test_source_selection_iea_annex42(mock_generate, mock_data):
    """Test source selection for iea_annex42."""
    # Create an object with iea_annex42 source and household_type
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'iea_annex42',
        O.HOUSEHOLD_TYPE: 'family_with_children'
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0] == obj
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(JordanVajenDwellingSizeDHW, 'generate')
def test_source_selection_vdi4655(mock_generate, mock_data):
    """Test source selection for vdi4655."""
    # Create an object with vdi4655 source and dwelling_size
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'vdi4655',
        O.DWELLING_SIZE: 100
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    
    # We need to patch the VDI4655DwellingSizeDHW class that's created dynamically
    with patch('entise.methods.dhw.probabilistic.VDI4655DwellingSizeDHW') as mock_class:
        # Mock the generate method of the instance
        mock_instance = mock_class.return_value
        mock_instance.generate.return_value = mock_result
        
        result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the class was instantiated
    mock_class.assert_called_once()
    
    # Check that the generate method was called with the correct arguments
    mock_instance.generate.assert_called_once()
    args, kwargs = mock_instance.generate.call_args
    assert args[0] == obj
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(UserDefinedDHW, 'generate')
def test_source_selection_user(mock_generate, mock_data):
    """Test source selection for user."""
    # Create an object with user source
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'user',
        O.DWELLING_SIZE: 100
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0] == obj
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(JordanVajenDwellingSizeDHW, 'generate')
def test_source_selection_unknown(mock_generate, mock_data):
    """Test source selection for unknown source."""
    # Create an object with unknown source but with dwelling_size
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'unknown',
        O.DWELLING_SIZE: 100
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0] == obj
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(JordanVajenDwellingSizeDHW, 'generate')
def test_weekend_activity(mock_generate, mock_data):
    """Test weekend activity."""
    # Create an object with jordan_vajen source, dwelling_size, and weekend_activity
    obj = {
        O.WEATHER: 'weather',
        O.SOURCE: 'jordan_vajen',
        O.DWELLING_SIZE: 100,
        O.WEEKEND_ACTIVITY: True
    }
    
    # Mock the generate method to return a fixed result
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_generate.return_value = mock_result
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    result = dhw.generate(obj, mock_data, Types.DHW)
    
    # Check that the result is the mocked result
    assert result == mock_result
    
    # Check that the generate method was called with the correct arguments
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert args[0][O.DHW_ACTIVITY_FILE] == os.path.join('entise', 'data', 'dhw', 'ashrae', 'dhw_activity_weekend.csv')
    assert args[1] == mock_data
    assert args[2] == Types.DHW

@patch.object(JordanVajenDwellingSizeDHW, 'generate')
@patch.object(HendronBurchOccupantsDHW, 'generate')
@patch.object(IEAAnnex42HouseholdTypeDHW, 'generate')
def test_parameter_based_selection(mock_iea_generate, mock_hendron_generate, mock_jordan_generate, mock_data):
    """Test parameter-based selection."""
    # Create objects with different parameters but no source
    obj_dwelling = {
        O.WEATHER: 'weather',
        O.DWELLING_SIZE: 100
    }
    
    obj_occupants = {
        O.WEATHER: 'weather',
        O.OCCUPANTS: 4
    }
    
    obj_household_type = {
        O.WEATHER: 'weather',
        O.HOUSEHOLD_TYPE: 'family_with_children'
    }
    
    # Mock the generate methods to return fixed results
    mock_result = {
        'summary': {f'{C.DEMAND}_{Types.DHW}_volume': 100.0, f'{C.DEMAND}_{Types.DHW}_energy': 5.0},
        'timeseries': pd.DataFrame()
    }
    mock_jordan_generate.return_value = mock_result
    mock_hendron_generate.return_value = mock_result
    mock_iea_generate.return_value = mock_result
    
    # Call the generate method for each object
    dhw = ProbabilisticDHW()
    result_dwelling = dhw.generate(obj_dwelling, mock_data, Types.DHW)
    result_occupants = dhw.generate(obj_occupants, mock_data, Types.DHW)
    result_household_type = dhw.generate(obj_household_type, mock_data, Types.DHW)
    
    # Check that the results are the mocked results
    assert result_dwelling == mock_result
    assert result_occupants == mock_result
    assert result_household_type == mock_result
    
    # Check that the generate methods were called with the correct arguments
    mock_jordan_generate.assert_called_once()
    args, kwargs = mock_jordan_generate.call_args
    assert args[0] == obj_dwelling
    assert args[1] == mock_data
    assert args[2] == Types.DHW
    
    mock_hendron_generate.assert_called_once()
    args, kwargs = mock_hendron_generate.call_args
    assert args[0] == obj_occupants
    assert args[1] == mock_data
    assert args[2] == Types.DHW
    
    mock_iea_generate.assert_called_once()
    args, kwargs = mock_iea_generate.call_args
    assert args[0] == obj_household_type
    assert args[1] == mock_data
    assert args[2] == Types.DHW

def test_missing_parameters():
    """Test missing parameters."""
    # Create an object with no source and no parameters
    obj = {
        O.WEATHER: 'weather'
    }
    
    # Call the generate method
    dhw = ProbabilisticDHW()
    with pytest.raises(ValueError):
        dhw.generate(obj, {}, Types.DHW)