import numpy as np
import pandas as pd

from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method
from entise.core.runner import RowExecutor


class DummyHVAC(Method):
    types = [Types.HVAC]
    name = "runner_dummy"
    required_keys = []

    def generate(self, obj, data, ts_type):
        ts = pd.DataFrame({f"{C.LOAD}_{Types.HEATING}": np.ones(10), f"{C.TEMP_IN}": np.full(10, 22.0)})
        summary = {f"{O.DEMAND}_{Types.HEATING}": float(ts[f"{C.LOAD}_{Types.HEATING}"].sum())}
        return {K.SUMMARY: summary, K.TIMESERIES: ts}


def test_runner_executes_main_method():
    static_inputs = {O.ID: "test_obj", Types.HVAC: "runner_dummy"}
    data = {}
    strategies = {Types.HVAC: "runner_dummy"}

    runner = RowExecutor(static_inputs, data, strategies)
    results = runner.run_main_methods()

    assert Types.HVAC in results
    assert isinstance(results[Types.HVAC][K.SUMMARY], dict)
    assert isinstance(results[Types.HVAC][K.TIMESERIES], pd.DataFrame)
    assert not results[Types.HVAC][K.TIMESERIES].empty


class CallCounter:
    """Helper to track how many times a method is executed."""

    count = 0


class CachingDummy(Method):
    types = [Types.HVAC]
    name = "cache_test"
    required_keys = []

    def generate(self, obj, data, ts_type):
        CallCounter.count += 1
        ts = pd.DataFrame({f"{C.LOAD}_{Types.HEATING}": np.ones(5)})
        summary = {f"{O.DEMAND}_{Types.HEATING}": float(ts[f"{C.LOAD}_{Types.HEATING}"].sum())}
        return {K.SUMMARY: summary, K.TIMESERIES: ts}


def test_runner_uses_cache():
    CallCounter.count = 0
    static_inputs = {O.ID: "obj42", Types.HVAC: "cache_test"}
    data = {}
    strategies = {Types.HVAC: "cache_test"}

    runner = RowExecutor(static_inputs, data, strategies)

    # Call twice — second one should hit the cache
    result_1 = runner.resolve(Types.HVAC)
    result_2 = runner.resolve(Types.HVAC)

    assert result_1 == result_2
    assert CallCounter.count == 1  # Only one call to `generate`
