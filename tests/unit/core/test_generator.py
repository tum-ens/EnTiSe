import numpy as np
import pandas as pd

from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method
from entise.core.generator import Generator


class DummyHVAC(Method):
    types = [Types.HVAC]
    name = "dummy"
    required_keys = []
    required_data = []

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
    gen = Generator()
    objects = pd.DataFrame([{O.ID: "obj1", Types.HVAC: "dummy"}])
    data = {}

    gen.add_objects(objects)
    summary_df, timeseries = gen.generate(data, workers=1)

    assert "obj1" in summary_df.index
    assert "obj1" in timeseries
    assert Types.HVAC in timeseries["obj1"]
    assert not timeseries["obj1"][Types.HVAC].empty


class SummaryOrderProbe(Method):
    """Fails for objects with ``fail`` set and reports nothing for objects with ``quiet`` set."""

    types = [Types.HVAC]
    name = "summary_order_probe"
    required_keys = []
    optional_keys = ["fail", "quiet"]

    def generate(self, obj=None, data=None, results=None, ts_type=Types.HVAC, **inputs):
        obj = {**(obj or {}), **inputs}
        if obj.get("fail") is True:
            raise RuntimeError("failed")
        summary = {} if obj.get("quiet") is True else {f"{O.DEMAND}_{Types.HEATING}": 1.0}
        return {K.SUMMARY: summary, K.TIMESERIES: pd.DataFrame({C.TEMP_IN: np.full(4, 20.0)})}


def _summary(objects):
    gen = Generator()
    gen.add_objects([{O.ID: f"b{i}", Types.HVAC: "summary_order_probe", **obj} for i, obj in enumerate(objects)])
    summary, _ = gen.generate({}, workers=1, show_progress=False)
    return summary


def test_summary_rows_follow_the_input_order():
    summary = _summary([{}, {"fail": True}, {}, {}])

    assert list(summary.index) == ["b0", "b1", "b2", "b3"]
    assert summary.loc["b1", "error"] == "failed"


def test_an_object_with_an_empty_summary_keeps_its_row():
    summary = _summary([{}, {"quiet": True}, {}])

    assert list(summary.index) == ["b0", "b1", "b2"]
    assert summary.loc["b1"].isna().all()
