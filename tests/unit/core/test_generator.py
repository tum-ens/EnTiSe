import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method
from entise.core.generator import TimeSeriesGenerator


class DummyHVAC(Method):
    types = [Types.HVAC]
    name = "dummy"
    required_keys = []
    required_timeseries = []

    def generate(self, obj, data, results, ts_type):
        n = 24
        ts = pd.DataFrame(
            {
                f"{C.LOAD}_{Types.HEATING}": np.ones(n),
                f"{C.LOAD}_{Types.COOLING}": np.zeros(n),
                f"{C.TEMP_IN}": np.full(n, 21.5),
            }
        )
        summary = {f"{O.DEMAND}_{Types.HEATING}": float(ts[f"{C.LOAD}_{Types.HEATING}"].sum())}
        return {K.SUMMARY: summary, K.TIMESERIES: ts}


def test_generator_runs_basic_case():
    gen = TimeSeriesGenerator()
    objects = pd.DataFrame([{O.ID: "obj1", Types.HVAC: "dummy"}])
    data = {}

    gen.add_objects(objects)
    summary_df, timeseries = gen.generate(data, workers=1)

    assert "obj1" in summary_df.index
    assert "obj1" in timeseries
    assert Types.HVAC in timeseries["obj1"]
    assert not timeseries["obj1"][Types.HVAC].empty


def test_generator_raises_for_missing_strategy():
    gen = TimeSeriesGenerator()
    objects = pd.DataFrame([{O.ID: "objX", Types.HVAC: "not_registered"}])
    data = {}

    gen.add_objects(objects)

    with pytest.raises(ValueError, match="not_registered"):
        gen.generate(data, workers=1)
