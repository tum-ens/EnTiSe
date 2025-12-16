import os

import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.core.generator import TimeSeriesGenerator

DATA_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def inputs():
    """Load objects and data for integration tests."""
    objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))
    data = {}
    parent_dir = os.path.dirname(DATA_DIR)
    common_data_folder = os.path.join(parent_dir, "common_data")
    for file in os.listdir(common_data_folder):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            data[name] = pd.read_csv(os.path.join(common_data_folder, file), parse_dates=True)

    return objects, data


def test_hvac_7r2c_all_objects(inputs):
    """Test 7R2C generation for all objects in objects.csv."""
    objects_df, shared_data = inputs

    for _, obj_row in objects_df.iterrows():
        gen = TimeSeriesGenerator()
        gen.add_objects(obj_row.to_dict())
        summary, ts = gen.generate(shared_data, workers=1)

        # Basic sanity checks
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(ts, dict)
        assert len(summary) > 0, "Summary should contain results"


def test_hvac_7r2c_output_structure(inputs):
    """Test that 7R2C outputs have the correct structure."""
    objects_df, shared_data = inputs
    obj_row = objects_df.iloc[0]

    gen = TimeSeriesGenerator()
    gen.add_objects(obj_row.to_dict())
    summary, ts = gen.generate(shared_data, workers=1)

    obj_id = obj_row["id"]

    # Check summary structure
    assert obj_id in summary.index, "Object ID should be in summary"

    # Check for expected summary columns
    expected_summary_cols = [
        "heating:demand[Wh]",
        "heating:load_max[W]",
        "cooling:demand[Wh]",
        "cooling:load_max[W]",
    ]
    for col in expected_summary_cols:
        assert col in summary.columns, f"Summary should contain {col}"

    # Check timeseries structure
    assert obj_id in ts, "Object ID should be in timeseries dict"
    assert "hvac" in ts[obj_id], "HVAC timeseries should exist"

    hvac_ts = ts[obj_id]["hvac"]
    expected_ts_cols = [
        C.TEMP_IN,
        "heating:load[W]",
        "cooling:load[W]",
    ]
    for col in expected_ts_cols:
        assert col in hvac_ts.columns, f"Timeseries should contain {col}"


def test_hvac_7r2c_physical_constraints(inputs):
    """Test that 7R2C respects physical constraints."""
    objects_df, shared_data = inputs
    obj_row = objects_df.iloc[0]

    gen = TimeSeriesGenerator()
    gen.add_objects(obj_row.to_dict())
    summary, ts = gen.generate(shared_data, workers=1)

    obj_id = obj_row["id"]
    hvac_ts = ts[obj_id]["hvac"]

    # Check that demands are non-negative
    assert summary.loc[obj_id, "heating:demand[Wh]"] >= 0, "Heating demand must be non-negative"
    assert summary.loc[obj_id, "cooling:demand[Wh]"] >= 0, "Cooling demand must be non-negative"

    # Check that loads are non-negative
    assert (hvac_ts["heating:load[W]"] >= 0).all(), "Heating loads must be non-negative"
    assert (hvac_ts["cooling:load[W]"] >= 0).all(), "Cooling loads must be non-negative"

    # Check that temperatures are reasonable (not extreme)
    temps = hvac_ts[C.TEMP_IN]
    assert temps.min() > -50, "Indoor temperature should be physically reasonable"
    assert temps.max() < 100, "Indoor temperature should be physically reasonable"


def test_hvac_7r2c_multiple_objects(inputs):
    """Test 7R2C with multiple objects simultaneously."""
    objects_df, shared_data = inputs

    gen = TimeSeriesGenerator()
    gen.add_objects(objects_df)
    summary, ts = gen.generate(shared_data, workers=1)

    # Check that all objects were processed
    assert len(summary) == len(objects_df), "All objects should be in summary"
    assert len(ts) == len(objects_df), "All objects should have timeseries"

    # Check that each object has valid results
    for _, obj_row in objects_df.iterrows():
        obj_id = obj_row["id"]
        assert obj_id in summary.index, f"Object {obj_id} should be in summary"
        assert obj_id in ts, f"Object {obj_id} should have timeseries"
