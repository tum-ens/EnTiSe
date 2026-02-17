import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.methods.occupancy import PHT


@pytest.fixture
def dummy_inputs():
    np.random.seed(42)
    # One month hourly to stabilize statistics
    dt = pd.date_range(start="2025-01-01", end="2025-01-31 23:00", freq="h", tz="UTC")
    demand = pd.DataFrame({C.POWER: np.random.uniform(50, 5000, size=len(dt)), C.DATETIME: dt}, index=dt)

    # Base object with PHT params
    objs = [
        {
            O.ID: "1",
            Types.OCCUPANCY: "PHT",
            O.LAMBDA: 0.1,
            O.BASELINE_OFFSET: 0.0,
            O.DETECTION_THRESHOLD: 2.0,
            O.NIGHT_SCHEDULE: False,
        },
        {
            O.ID: "2",
            Types.OCCUPANCY: "PHT",
            O.LAMBDA: 0.2,
            O.BASELINE_OFFSET: 0.0,
            O.DETECTION_THRESHOLD: 2.0,
            O.NIGHT_SCHEDULE: False,
        },
        {
            O.ID: "3",
            Types.OCCUPANCY: "PHT",
            O.LAMBDA: 0.1,
            O.BASELINE_OFFSET: 0.0,
            O.DETECTION_THRESHOLD: 2.0,
            O.NIGHT_SCHEDULE: True,
        },
    ]

    df_objs = pd.DataFrame(objs)

    results = {Types.ELECTRICITY: {K.TIMESERIES: demand}}
    data = {}  # not used by PHT.generate; input comes from results

    return df_objs, data, results


def test_pht_outputs(dummy_inputs):
    objs, data, results = dummy_inputs
    obj = objs.loc[objs[O.ID] == "1"].iloc[0]
    method = PHT()
    result = method.generate(obj, data, results, Types.OCCUPANCY)

    assert K.TIMESERIES in result
    ts = result[K.TIMESERIES]
    assert f"{Types.OCCUPANCY}{SEP}{C.OCCUPANCY}" in ts.columns
    assert ts.index.name == C.DATETIME
    assert len(ts) == len(results[Types.ELECTRICITY][K.TIMESERIES])

    assert K.SUMMARY in result
    mean_occ = result[K.SUMMARY].get(f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}")
    assert 0.0 <= mean_occ <= 1.0


def test_lambda_sensitivity(dummy_inputs):
    objs, data, results = dummy_inputs
    obj1 = objs.loc[objs[O.ID] == "1"].iloc[0]
    obj2 = objs.loc[objs[O.ID] == "2"].iloc[0]
    method = PHT()

    r1 = method.generate(obj1, data, results, Types.OCCUPANCY)
    r2 = method.generate(obj2, data, results, Types.OCCUPANCY)

    m1 = r1[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    m2 = r2[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    # Expect a difference when lambda changes
    assert m1 != m2


def test_nightly_schedule_increases_occupancy(dummy_inputs):
    objs, data, results = dummy_inputs
    obj_off = objs.loc[objs[O.ID] == "1"].iloc[0]
    obj_night = objs.loc[objs[O.ID] == "3"].iloc[0]
    method = PHT()

    r_off = method.generate(obj_off, data, results, Types.OCCUPANCY)
    r_night = method.generate(obj_night, data, results, Types.OCCUPANCY)

    m_off = r_off[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    m_night = r_night[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    # Per implementation, nightly schedule zeros occupancy in specified hours
    assert m_night >= m_off


@pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0, 4.0])
def test_detection_threshold_sensitivity(dummy_inputs, threshold):
    objs, data, results = dummy_inputs
    base = objs.loc[objs[O.ID] == "1"].iloc[0].copy()
    base[O.DETECTION_THRESHOLD] = threshold
    method = PHT()
    res = method.generate(base, data, results, Types.OCCUPANCY)
    mean_occ = res[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    assert 0.0 <= mean_occ <= 1.0


def test_baseline_offset_sensitivity(dummy_inputs):
    objs, data, results = dummy_inputs
    base = objs.loc[objs[O.ID] == "1"].iloc[0].copy()
    method = PHT()

    base_low = base.copy()
    base_low[O.BASELINE_OFFSET] = 0.0
    base_high = base.copy()
    base_high[O.BASELINE_OFFSET] = 0.5

    r_low = method.generate(base_low, data, results, Types.OCCUPANCY)
    r_high = method.generate(base_high, data, results, Types.OCCUPANCY)

    m_low = r_low[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    m_high = r_high[K.SUMMARY][f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}"]
    assert m_high <= m_low
