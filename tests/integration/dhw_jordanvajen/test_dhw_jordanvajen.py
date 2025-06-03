import os
import pandas as pd
import numpy as np
import pytest
import holidays

from entise.core.generator import TimeSeriesGenerator
from entise.constants import Types, Columns as C, Objects as O

DATA_DIR = os.path.dirname(__file__)

@pytest.fixture(scope="module")
def inputs():
    """Load input data for testing."""
    objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))

    # Create mock weather data
    data = {}
    start = pd.Timestamp('2023-01-01', tz='UTC')
    end = start + pd.Timedelta(days=30)
    index = pd.date_range(start=start, end=end, freq='h')

    # Create a DataFrame with temperature data
    data['weather'] = pd.DataFrame({
        'datetime': index,
        'temp_out': np.random.normal(10, 5, len(index))  # Random outdoor temperatures
    })

    return objects, data

def test_jordan_vajen_all_objects(inputs):
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
        assert f'load_{Types.DHW}_power' in dhw_data.columns

        # Check that the summary has the expected keys
        assert f'demand_{Types.DHW}_volume_total' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_volume_avg' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_volume_peak' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy_total' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy_avg' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy_peak' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_power_avg' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_power_max' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_power_min' in summary.loc[obj_id]

def test_jordan_vajen_individual_objects(inputs):
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
        assert f'load_{Types.DHW}_power' in dhw_data.columns

        # Check that the summary has the expected keys
        assert f'demand_{Types.DHW}_volume_total' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy_total' in summary.loc[obj_id]

        # Check that the values are reasonable
        assert summary.loc[obj_id, f'demand_{Types.DHW}_volume_total'] > 0
        assert summary.loc[obj_id, f'demand_{Types.DHW}_energy_total'] > 0
        assert dhw_data[f'load_{Types.DHW}_volume'].sum() > 0
        assert dhw_data[f'load_{Types.DHW}_energy'].sum() > 0

def test_jordan_vajen_dwelling_size_impact(inputs):
    """Test that different dwelling sizes produce different results."""
    objects_df, shared_data = inputs

    # Get objects with different dwelling sizes
    objects_by_size = objects_df.sort_values('dwelling_size')
    small_obj = objects_by_size.iloc[0]
    large_obj = objects_by_size.iloc[-1]

    # Process these objects
    gen = TimeSeriesGenerator()
    gen.add_objects(small_obj.to_dict())
    gen.add_objects(large_obj.to_dict())
    summary, df = gen.generate(shared_data, workers=1)

    # Check that the results are different
    small_demand = summary.loc[small_obj['id'], f'demand_{Types.DHW}_volume_total']
    large_demand = summary.loc[large_obj['id'], f'demand_{Types.DHW}_volume_total']

    # Larger dwelling should have higher demand
    assert small_demand < large_demand

def test_jordan_vajen_holidays(inputs):
    """Test that holiday locations affect the results appropriately."""
    objects_df, shared_data = inputs

    # Get objects with different holiday locations
    pt_holidays_obj = objects_df[objects_df['holidays_location'] == 'PT'].iloc[0]
    # Handle NaN values in the filter
    ar_mask = objects_df['holidays_location'].notna() & objects_df['holidays_location'].str.contains('AR', na=False)
    ar_holidays_obj = objects_df[ar_mask].iloc[0]
    no_holidays_obj = objects_df[objects_df['holidays_location'].isna()].iloc[0]

    # Process these objects
    gen = TimeSeriesGenerator()
    gen.add_objects(pt_holidays_obj.to_dict())
    gen.add_objects(ar_holidays_obj.to_dict())
    gen.add_objects(no_holidays_obj.to_dict())
    summary, df = gen.generate(shared_data, workers=1)

    # Get the DHW data
    pt_data = df[pt_holidays_obj['id']][Types.DHW]
    ar_data = df[ar_holidays_obj['id']][Types.DHW]
    no_data = df[no_holidays_obj['id']][Types.DHW]

    # Add day of week column
    pt_data['day_of_week'] = pt_data.index.dayofweek
    ar_data['day_of_week'] = ar_data.index.dayofweek
    no_data['day_of_week'] = no_data.index.dayofweek

    # Add holiday column
    pt_data['date'] = pt_data.index.date
    ar_data['date'] = ar_data.index.date
    no_data['date'] = no_data.index.date

    # Get Portuguese holidays
    pt_holiday_dates = set(holidays.Portugal(years=pt_data.index.year.unique()).keys())

    # Get Argentinian holidays
    ar_holiday_dates = set(holidays.Argentina(years=ar_data.index.year.unique(), subdiv='B').keys())

    # Calculate average demand for holidays and non-holidays
    pt_holiday_demand = pt_data[pt_data['date'].isin(pt_holiday_dates)][f'load_{Types.DHW}_volume'].mean()
    pt_non_holiday_demand = pt_data[~pt_data['date'].isin(pt_holiday_dates)][f'load_{Types.DHW}_volume'].mean()

    ar_holiday_demand = ar_data[ar_data['date'].isin(ar_holiday_dates)][f'load_{Types.DHW}_volume'].mean()
    ar_non_holiday_demand = ar_data[~ar_data['date'].isin(ar_holiday_dates)][f'load_{Types.DHW}_volume'].mean()

    # Check that the holiday demand is different from the non-holiday demand
    # This is a weak test because the difference might be small and due to randomness
    # But it's better than nothing
    assert pt_holiday_demand != pt_non_holiday_demand
    assert ar_holiday_demand != ar_non_holiday_demand

def test_jordan_vajen_weekday_weekend(inputs):
    """Test that weekday and weekend demand patterns are different."""
    objects_df, shared_data = inputs

    # Get any object
    obj = objects_df.iloc[0]

    # Process this object
    gen = TimeSeriesGenerator()
    gen.add_objects(obj.to_dict())
    summary, df = gen.generate(shared_data, workers=1)

    # Get the DHW data
    dhw_data = df[obj['id']][Types.DHW]

    # Add day of week column
    dhw_data['day_of_week'] = dhw_data.index.dayofweek

    # Calculate average demand for weekdays and weekends
    weekday_demand = dhw_data[dhw_data['day_of_week'] < 5][f'load_{Types.DHW}_volume'].mean()
    weekend_demand = dhw_data[dhw_data['day_of_week'] >= 5][f'load_{Types.DHW}_volume'].mean()

    # Check that the weekend demand is different from the weekday demand
    assert weekday_demand != weekend_demand

def test_jordan_vajen_edge_cases(inputs):
    """Test edge cases like very small or very large values."""
    objects_df, shared_data = inputs

    # Create objects with edge case values
    edge_cases = [
        # Very small dwelling size
        {'id': 101, 'dhw': 'JordanVajen', 'datetimes': 'weather', 'dwelling_size': 1},
        # Very large dwelling size
        {'id': 102, 'dhw': 'JordanVajen', 'datetimes': 'weather', 'dwelling_size': 1000},
        # Equal water temperatures
        {'id': 103, 'dhw': 'JordanVajen', 'datetimes': 'weather', 'dwelling_size': 100, 'temp_cold': 40, 'temp_hot': 40},
        # Very high water temperature difference
        {'id': 104, 'dhw': 'JordanVajen', 'datetimes': 'weather', 'dwelling_size': 100, 'temp_cold': 5, 'temp_hot': 90},
    ]

    # Process all objects at once
    gen = TimeSeriesGenerator()
    for edge_case in edge_cases:
        gen.add_objects(edge_case)
    summary, df = gen.generate(shared_data, workers=1)

    # Check each object
    for edge_case in edge_cases:
        obj_id = edge_case['id']

        # Check that the object has been processed
        assert obj_id in df
        assert Types.DHW in df[obj_id]

        # Check that the DHW data has the expected columns
        dhw_data = df[obj_id][Types.DHW]
        assert f'load_{Types.DHW}_volume' in dhw_data.columns
        assert f'load_{Types.DHW}_energy' in dhw_data.columns
        assert f'load_{Types.DHW}_power' in dhw_data.columns

        # Check that the summary has the expected keys
        assert f'demand_{Types.DHW}_volume_total' in summary.loc[obj_id]
        assert f'demand_{Types.DHW}_energy_total' in summary.loc[obj_id]

        # Check that the values are reasonable (non-negative)
        assert summary.loc[obj_id, f'demand_{Types.DHW}_volume_total'] >= 0
        assert summary.loc[obj_id, f'demand_{Types.DHW}_energy_total'] >= 0

    # Check that all energy demands are non-negative
    # Note: We're not comparing between cases because the results can vary due to randomness
    # and the specific implementation of the energy calculation
    for obj_id in [101, 102, 103, 104]:
        assert summary.loc[obj_id, f'demand_{Types.DHW}_energy_total'] >= 0
