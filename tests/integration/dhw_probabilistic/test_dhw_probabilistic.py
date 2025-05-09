import os
import pandas as pd
import pytest

from entise.core.generator import TimeSeriesGenerator
from entise.constants import Types, Columns as C

DATA_DIR = os.path.dirname(__file__)

@pytest.fixture(scope="module")
def inputs():
    """Load input data for testing."""
    objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))
    
    # Load weather data from the hvac_rc example
    data = {}
    weather_file = os.path.join('..', 'hvac_rc', 'data', 'weather.csv')
    data['weather'] = pd.read_csv(weather_file, parse_dates=['datetime'])
    
    return objects, data

def test_dhw_probabilistic_all_objects(inputs):
    """Test that all objects can be processed without errors."""
    objects_df, shared_data = inputs
    
    # Process all objects
    gen = TimeSeriesGenerator()
    gen.add_objects(objects_df)
    summary, df = gen.generate(shared_data, workers=1)
    
    # Check that all objects have been processed
    assert len(summary) == len(objects_df)
    assert len(df) == len(objects_df)
    
    # Check that all objects have DHW data
    for obj_id in objects_df['id']:
        assert obj_id in df
        assert Types.DHW in df[obj_id]
        
        # Check that the DHW data has the expected columns
        dhw_data = df[obj_id][Types.DHW]
        assert f'load_{Types.DHW}_volume' in dhw_data.columns
        assert f'load_{Types.DHW}_energy' in dhw_data.columns
        
        # Check that the summary has the expected keys
        assert f'demand_{Types.DHW}_volume' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy' in summary.loc[obj_id]

def test_dhw_probabilistic_individual_objects(inputs):
    """Test each object individually to isolate any issues."""
    objects_df, shared_data = inputs
    
    for _, obj_row in objects_df.iterrows():
        obj_id = obj_row['id']
        
        # Process this object
        gen = TimeSeriesGenerator()
        gen.add_objects(obj_row.to_dict())
        summary, df = gen.generate(shared_data, workers=1)
        
        # Check that the object has been processed
        assert obj_id in df
        assert Types.DHW in df[obj_id]
        
        # Check that the DHW data has the expected columns
        dhw_data = df[obj_id][Types.DHW]
        assert f'load_{Types.DHW}_volume' in dhw_data.columns
        assert f'load_{Types.DHW}_energy' in dhw_data.columns
        
        # Check that the summary has the expected keys
        assert f'demand_{Types.DHW}_volume' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy' in summary.loc[obj_id]
        
        # Check that the values are reasonable
        assert summary.loc[obj_id, f'demand_{Types.DHW}_volume'] > 0
        assert summary.loc[obj_id, f'demand_{Types.DHW}_energy'] > 0
        assert dhw_data[f'load_{Types.DHW}_volume'].sum() > 0
        assert dhw_data[f'load_{Types.DHW}_energy'].sum() > 0

def test_dhw_probabilistic_source_comparison(inputs):
    """Test that different sources produce different results."""
    objects_df, shared_data = inputs
    
    # Get objects with different sources
    jordan_vajen_obj = objects_df[objects_df['source'] == 'jordan_vajen'].iloc[0]
    hendron_burch_obj = objects_df[objects_df['source'] == 'hendron_burch'].iloc[0]
    iea_annex42_obj = objects_df[objects_df['source'] == 'iea_annex42'].iloc[0]
    
    # Process these objects
    gen = TimeSeriesGenerator()
    gen.add_objects(jordan_vajen_obj.to_dict())
    gen.add_objects(hendron_burch_obj.to_dict())
    gen.add_objects(iea_annex42_obj.to_dict())
    summary, df = gen.generate(shared_data, workers=1)
    
    # Check that the results are different
    jordan_vajen_demand = summary.loc[jordan_vajen_obj['id'], f'demand_{Types.DHW}_volume']
    hendron_burch_demand = summary.loc[hendron_burch_obj['id'], f'demand_{Types.DHW}_volume']
    iea_annex42_demand = summary.loc[iea_annex42_obj['id'], f'demand_{Types.DHW}_volume']
    
    assert jordan_vajen_demand != hendron_burch_demand
    assert jordan_vajen_demand != iea_annex42_demand
    assert hendron_burch_demand != iea_annex42_demand

def test_dhw_probabilistic_weekend_activity(inputs):
    """Test that weekend activity produces different results for weekdays and weekends."""
    objects_df, shared_data = inputs
    
    # Get object with weekend activity
    weekend_obj = objects_df[objects_df['weekend_activity'] == True].iloc[0]
    
    # Process this object
    gen = TimeSeriesGenerator()
    gen.add_objects(weekend_obj.to_dict())
    summary, df = gen.generate(shared_data, workers=1)
    
    # Get the DHW data
    dhw_data = df[weekend_obj['id']][Types.DHW]
    
    # Add day of week column
    dhw_data['day_of_week'] = dhw_data.index.dayofweek
    
    # Calculate average demand for weekdays and weekends
    weekday_demand = dhw_data[dhw_data['day_of_week'] < 5][f'load_{Types.DHW}_volume'].mean()
    weekend_demand = dhw_data[dhw_data['day_of_week'] >= 5][f'load_{Types.DHW}_volume'].mean()
    
    # Check that the weekend demand is different from the weekday demand
    assert weekday_demand != weekend_demand

def test_dhw_probabilistic_parameter_selection(inputs):
    """Test that objects with no source but with different parameters use different methods."""
    objects_df, shared_data = inputs
    
    # Get objects with no source but with different parameters
    dwelling_size_obj = objects_df[(objects_df['source'].isna()) & (objects_df['dwelling_size'].notna())].iloc[0]
    occupants_obj = objects_df[(objects_df['source'].isna()) & (objects_df['occupants'].notna())].iloc[0]
    household_type_obj = objects_df[(objects_df['source'].isna()) & (objects_df['household_type'].notna())].iloc[0]
    
    # Process these objects
    gen = TimeSeriesGenerator()
    gen.add_objects(dwelling_size_obj.to_dict())
    gen.add_objects(occupants_obj.to_dict())
    gen.add_objects(household_type_obj.to_dict())
    summary, df = gen.generate(shared_data, workers=1)
    
    # Check that the results are different
    dwelling_size_demand = summary.loc[dwelling_size_obj['id'], f'demand_{Types.DHW}_volume']
    occupants_demand = summary.loc[occupants_obj['id'], f'demand_{Types.DHW}_volume']
    household_type_demand = summary.loc[household_type_obj['id'], f'demand_{Types.DHW}_volume']
    
    assert dwelling_size_demand != occupants_demand
    assert dwelling_size_demand != household_type_demand
    assert occupants_demand != household_type_demand

def test_dhw_probabilistic_edge_cases(inputs):
    """Test edge cases like very small or very large values."""
    objects_df, shared_data = inputs
    
    # Create objects with edge case values
    edge_cases = [
        # Very small dwelling size
        {'id': 101, 'dhw': 'ProbabilisticDHW', 'weather': 'weather', 'dwelling_size': 1},
        # Very large dwelling size
        {'id': 102, 'dhw': 'ProbabilisticDHW', 'weather': 'weather', 'dwelling_size': 1000},
        # Very small number of occupants
        {'id': 103, 'dhw': 'ProbabilisticDHW', 'weather': 'weather', 'occupants': 1},
        # Very large number of occupants
        {'id': 104, 'dhw': 'ProbabilisticDHW', 'weather': 'weather', 'occupants': 20},
    ]
    
    # Process these objects
    for edge_case in edge_cases:
        gen = TimeSeriesGenerator()
        gen.add_objects(edge_case)
        summary, df = gen.generate(shared_data, workers=1)
        
        # Check that the object has been processed
        assert edge_case['id'] in df
        assert Types.DHW in df[edge_case['id']]
        
        # Check that the DHW data has the expected columns
        dhw_data = df[edge_case['id']][Types.DHW]
        assert f'load_{Types.DHW}_volume' in dhw_data.columns
        assert f'load_{Types.DHW}_energy' in dhw_data.columns
        
        # Check that the summary has the expected keys
        assert f'demand_{Types.DHW}_volume' in summary.loc[edge_case['id']]
        assert f'demand_{Types.DHW}_energy' in summary.loc[edge_case['id']]
        
        # Check that the values are reasonable (non-negative)
        assert summary.loc[edge_case['id'], f'demand_{Types.DHW}_volume'] >= 0
        assert summary.loc[edge_case['id'], f'demand_{Types.DHW}_energy'] >= 0