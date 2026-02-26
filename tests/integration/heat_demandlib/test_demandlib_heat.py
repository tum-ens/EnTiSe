"""
Integration tests for EnTiSe demandlib heat method.

This module runs the public examples end to end through the high-level
simulation pipeline to validate that:
- The pipeline executes without errors for the demandlib heat example
- Summary and per-object results are produced
- Timeseries contain the expected column, have no NaNs, and are non-negative
"""

from pathlib import Path

import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from examples.utils import load_input, run_simulation

OUT_COL = f"{Types.HEATING}{SEP}{C.LOAD}[W]"


def _repo_root() -> Path:
    """Return repository root path from this test file's location."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def example_inputs():
    """Load example objects and data for the heat demandlib example from the repo.

    Ensures required inputs exist and that weather contains the C.DATETIME column.
    """
    repo = _repo_root()
    example_dir = repo / "examples" / "heat_demandlib"

    objects, data = load_input(
        base_path=str(example_dir),
        load_common_data=True,
        parse_dates=True,
    )

    assert len(objects) > 0
    assert "weather" in data
    assert C.DATETIME in data["weather"].columns

    return objects, data


def test_heat_demandlib_example_runs_end_to_end(example_inputs):
    """
    Integration test:
    - Runs full Generator pipeline
    - Ensures heating results exist
    - Ensures no NaNs
    - Ensures loads are non-negative
    """
    objects, data = example_inputs

    summary, results = run_simulation(objects, data, workers=1)

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(results, dict)

    for _, row in objects.iterrows():
        obj_id = row["id"]  # keep original type (int or str)

        # Object must exist in results
        assert obj_id in results
        assert Types.HEATING in results[obj_id]

        ts_df = results[obj_id][Types.HEATING]

        assert isinstance(ts_df, pd.DataFrame)
        assert OUT_COL in ts_df.columns
        assert ts_df[OUT_COL].isna().sum() == 0
        assert (ts_df[OUT_COL] >= 0).all()
