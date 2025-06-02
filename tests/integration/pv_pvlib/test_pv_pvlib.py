import os
import pandas as pd
import pytest
from entise.core.generator import TimeSeriesGenerator
from entise.constants import Types

DATA_DIR = os.path.dirname(__file__)

@pytest.fixture(scope="module")
def inputs():
    """Load test data from files."""
    objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))
    data = {}
    data_folder = 'data'
    for file in os.listdir(os.path.join(DATA_DIR, data_folder)):
        if file.endswith('.csv'):
            name = file.split('.')[0]
            data[name] = pd.read_csv(os.path.join(os.path.join(DATA_DIR, data_folder, file)), parse_dates=True)

    return objects, data

def test_pv_pvlib_all_objects(inputs):
    """Test the PVLib method with all objects in the test data."""
    objects_df, shared_data = inputs

    # Test each object individually
    for _, obj_row in objects_df.iterrows():
        gen = TimeSeriesGenerator()
        gen.add_objects(obj_row.to_dict())
        summary, df = gen.generate(shared_data, workers=1)

        # Check that the summary contains the expected keys
        obj_id = obj_row['id']
        assert f'generation_{Types.PV}' in summary.loc[obj_id]
        assert f'maximum_generation_{Types.PV}' in summary.loc[obj_id]
        assert f'full_load_hours_{Types.PV}' in summary.loc[obj_id]

        # Check that the time series contains the expected data
        assert obj_id in df
        assert Types.PV in df[obj_id]
        assert f'power_{Types.PV}' in df[obj_id][Types.PV].columns

        # Check that the time series has the expected length
        assert len(df[obj_id][Types.PV]) == len(shared_data['weather'])

        # Check that the generation values are non-negative
        assert (df[obj_id][Types.PV] >= 0).all().all()

def test_pv_pvlib_all_objects_together(inputs):
    """Test the PVLib method with all objects together."""
    objects_df, shared_data = inputs

    # Test all objects together
    gen = TimeSeriesGenerator()
    gen.add_objects(objects_df)
    summary, df = gen.generate(shared_data, workers=1)

    # Check that the summary contains entries for all objects
    for _, obj_row in objects_df.iterrows():
        obj_id = obj_row['id']
        assert obj_id in summary.index

        # Check that the summary contains the expected keys
        assert f'generation_{Types.PV}' in summary.loc[obj_id]
        assert f'maximum_generation_{Types.PV}' in summary.loc[obj_id]
        assert f'full_load_hours_{Types.PV}' in summary.loc[obj_id]

        # Check that the time series contains the expected data
        assert obj_id in df
        assert Types.PV in df[obj_id]
        assert f'power_{Types.PV}' in df[obj_id][Types.PV].columns

        # Check that the time series has the expected length
        assert len(df[obj_id][Types.PV]) == len(shared_data['weather'])

        # Check that the generation values are non-negative
        assert (df[obj_id][Types.PV] >= 0).all().all()

def test_pv_pvlib_orientation_effects(inputs):
    """Test that different orientations produce different generation profiles."""
    objects_df, shared_data = inputs

    # Get objects with different orientations
    south_obj = objects_df[objects_df['azimuth'] == 180].iloc[0]
    east_obj = objects_df[objects_df['azimuth'] == 90].iloc[0]
    west_obj = objects_df[objects_df['azimuth'] == 270].iloc[0]

    # Generate time series for each orientation
    gen = TimeSeriesGenerator()
    gen.add_objects([south_obj.to_dict(), east_obj.to_dict(), west_obj.to_dict()])
    summary, df = gen.generate(shared_data, workers=1)

    # Convert time series indices to datetime for time-of-day analysis
    for obj_id in df:
        df[obj_id][Types.PV].index = pd.to_datetime(df[obj_id][Types.PV].index)

    # Check that east-facing panels generate more in the morning
    morning_hours = [9, 10, 11]
    morning_data = {
        'south': df[south_obj['id']][Types.PV][df[south_obj['id']][Types.PV].index.hour.isin(morning_hours)],
        'east': df[east_obj['id']][Types.PV][df[east_obj['id']][Types.PV].index.hour.isin(morning_hours)],
        'west': df[west_obj['id']][Types.PV][df[west_obj['id']][Types.PV].index.hour.isin(morning_hours)]
    }
    assert morning_data['east'].sum().iloc[0] >= morning_data['west'].sum().iloc[0]

    # Check that west-facing panels generate more in the afternoon
    afternoon_hours = [15, 16, 17]
    afternoon_data = {
        'south': df[south_obj['id']][Types.PV][df[south_obj['id']][Types.PV].index.hour.isin(afternoon_hours)],
        'east': df[east_obj['id']][Types.PV][df[east_obj['id']][Types.PV].index.hour.isin(afternoon_hours)],
        'west': df[west_obj['id']][Types.PV][df[west_obj['id']][Types.PV].index.hour.isin(afternoon_hours)]
    }
    assert afternoon_data['west'].sum().iloc[0] >= afternoon_data['east'].sum().iloc[0]

def test_pv_pvlib_tilt_effects(inputs):
    """Test that different tilts produce different generation profiles."""
    objects_df, shared_data = inputs

    # Get objects with different tilts
    flat_obj = objects_df[objects_df['tilt'] == 0].iloc[0]
    tilted_obj = objects_df[objects_df['tilt'] == 30].iloc[0]
    steep_obj = objects_df[objects_df['tilt'] == 45].iloc[0]
    vertical_obj = objects_df[objects_df['tilt'] == 90].iloc[0]

    # Generate time series for each tilt
    gen = TimeSeriesGenerator()
    gen.add_objects([flat_obj.to_dict(), tilted_obj.to_dict(), steep_obj.to_dict(), vertical_obj.to_dict()])
    summary, df = gen.generate(shared_data, workers=1)

    # Check that tilted panels generate more than flat panels in winter
    # (This is a simplification since we only have one day of data)
    assert summary.loc[tilted_obj['id'], f'generation_{Types.PV}'].item() > summary.loc[flat_obj['id'], f'generation_{Types.PV}'].item()

    # Check that vertical panels generate more than tilted panels in winter
    # Note: The actual relationship depends on the specific weather data and time of year
    assert summary.loc[tilted_obj['id'], f'generation_{Types.PV}'].item() < summary.loc[vertical_obj['id'], f'generation_{Types.PV}'].item()

def test_pv_pvlib_power_scaling(inputs):
    """Test that generation scales linearly with power."""
    objects_df, shared_data = inputs

    # Get objects with different powers
    low_power_obj = objects_df[objects_df['power'] == 2500].iloc[0]
    mid_power_obj = objects_df[objects_df['power'] == 5000].iloc[0]
    high_power_obj = objects_df[objects_df['power'] == 10000].iloc[0]

    # Generate time series for each power
    gen = TimeSeriesGenerator()
    gen.add_objects([low_power_obj.to_dict(), mid_power_obj.to_dict(), high_power_obj.to_dict()])
    summary, df = gen.generate(shared_data, workers=1)

    # Check that generation scales linearly with power
    low_gen = summary.loc[low_power_obj['id'], f'generation_{Types.PV}'].item()
    mid_gen = summary.loc[mid_power_obj['id'], f'generation_{Types.PV}'].item()
    high_gen = summary.loc[high_power_obj['id'], f'generation_{Types.PV}'].item()

    # Allow for small numerical differences
    assert abs(mid_gen / low_gen - 2.0) < 0.1
    assert abs(high_gen / mid_gen - 2.0) < 0.1
    assert abs(high_gen / low_gen - 4.0) < 0.1
