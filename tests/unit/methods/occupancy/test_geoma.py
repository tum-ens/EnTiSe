import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.constants.general import SEP
from entise.constants.general import Keys as K
from entise.methods.occupancy import GeoMA


@pytest.fixture
def dummy_inputs():
    np.random.seed(42)

    datetime = pd.date_range(start="2025-01-01", end="2025-01-31 23:00", freq="h", tz="UTC")

    demand = pd.DataFrame(
        {C.POWER: np.random.uniform(0, 5000, size=len(datetime)), C.DATETIME: datetime}, index=datetime
    )

    objs = [
        {
            O.ID: "1",
            Types.OCCUPANCY: "GeoMA",
            O.DEMAND: O.DEMAND,
            O.LAMBDA: 0.05,
            O.NIGHT_SCHEDULE: False,
        },
        {
            O.ID: "2",
            Types.OCCUPANCY: "GeoMA",
            O.DEMAND: O.DEMAND,
            O.LAMBDA: 0.15,
            O.NIGHT_SCHEDULE: False,
        },
        {
            O.ID: "3",
            Types.OCCUPANCY: "GeoMA",
            O.DEMAND: O.DEMAND,
            O.LAMBDA: 0.05,
            O.NIGHT_SCHEDULE: True,
        },
    ]

    df_objs = pd.DataFrame(objs)

    data = {
        O.DEMAND: demand,
    }

    results = {
        Types.ELECTRICITY: {K.TIMESERIES: demand},
    }

    return df_objs, data, results


def test_geoma_outputs(dummy_inputs):
    objs, data, results = dummy_inputs
    obj = objs.loc[objs[O.ID] == "1"].iloc[0]
    geoma = GeoMA()
    result = geoma.generate(obj, data, results, Types.OCCUPANCY)

    assert K.TIMESERIES in result
    ts = result[K.TIMESERIES]
    assert all(col in ts.columns for col in [f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY}"])
    assert ts.index.name == C.DATETIME
    assert len(ts) == len(data[O.DEMAND])


def test_lambda_difference(dummy_inputs):
    objs, data, results = dummy_inputs
    obj_1 = objs.loc[objs[O.ID] == "1"].iloc[0]
    obj_2 = objs.loc[objs[O.ID] == "2"].iloc[0]
    geoma = GeoMA()
    result_1 = geoma.generate(obj_1, data, results, Types.OCCUPANCY)
    result_2 = geoma.generate(obj_2, data, results, Types.OCCUPANCY)

    assert K.SUMMARY in result_1
    assert K.SUMMARY in result_2

    mean_occ_1 = result_1[K.SUMMARY].get(f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}")
    mean_occ_2 = result_2[K.SUMMARY].get(f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}")
    assert mean_occ_1 != mean_occ_2


def test_higher_occupancy_with_nightly_schedule(dummy_inputs):
    objs, data, results = dummy_inputs
    obj_1 = objs.loc[objs[O.ID] == "1"].iloc[0]
    obj_3 = objs.loc[objs[O.ID] == "3"].iloc[0]
    geoma = GeoMA()
    result_1 = geoma.generate(obj_1, data, results, Types.OCCUPANCY)
    result_3 = geoma.generate(obj_3, data, results, Types.OCCUPANCY)

    assert K.SUMMARY in result_1
    assert K.SUMMARY in result_3

    mean_occ_1 = result_1[K.SUMMARY].get(f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}")
    mean_occ_3 = result_3[K.SUMMARY].get(f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}")
    assert mean_occ_1 < mean_occ_3
