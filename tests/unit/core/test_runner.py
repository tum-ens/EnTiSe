import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method, method_registry
from entise.core.runner import Runner


class DummyHVAC(Method):
    types = [Types.HVAC]
    name = "runner_dummy"
    required_keys = []

    def generate(self, obj, data, results, ts_type):
        ts = pd.DataFrame({f"{C.LOAD}_{Types.HEATING}": np.ones(10), f"{C.TEMP_IN}": np.full(10, 22.0)})
        summary = {f"{O.DEMAND}_{Types.HEATING}": float(ts[f"{C.LOAD}_{Types.HEATING}"].sum())}
        return {K.SUMMARY: summary, K.TIMESERIES: ts}


def test_runner_executes_main_method():
    static_inputs = {O.ID: "test_obj", Types.HVAC: "runner_dummy"}
    data = {}
    strategies = {Types.HVAC: "runner_dummy"}

    runner = Runner(static_inputs, data, strategies)
    results = runner.run_methods()

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

    def generate(self, obj, data, results, ts_type):
        CallCounter.count += 1
        ts = pd.DataFrame({f"{C.LOAD}_{Types.HEATING}": np.ones(5)})
        summary = {f"{O.DEMAND}_{Types.HEATING}": float(ts[f"{C.LOAD}_{Types.HEATING}"].sum())}
        return {K.SUMMARY: summary, K.TIMESERIES: ts}


def test_runner_uses_cache():
    CallCounter.count = 0
    static_inputs = {O.ID: "obj42", Types.HVAC: "cache_test"}
    data = {}
    strategies = {Types.HVAC: "cache_test"}

    runner = Runner(static_inputs, data, strategies)

    # Call twice — second one should hit the cache
    result_1 = runner.resolve(Types.HVAC)
    result_2 = runner.resolve(Types.HVAC)

    assert result_1 == result_2
    assert CallCounter.count == 1  # Only one call to `generate`


# -- strategies that produce several types in one calculation (#127) --------------------------------------------------


class GroupedDummy(Method):
    """Produces heating and cooling together and records which group each call received."""

    register = False
    name = "runner_group_test"
    types = [Types.HEATING, Types.COOLING]
    required_keys = []
    calls: list = []

    def generate(self, obj, data, results, ts_type):
        GroupedDummy.calls.append(("generate", ts_type))
        return {K.TIMESERIES: pd.DataFrame({ts_type: [1.0]})}

    def generate_group(self, obj, data, results, ts_types):
        GroupedDummy.calls.append(("group", tuple(ts_types)))
        return {kind: {K.TIMESERIES: pd.DataFrame({kind: [1.0]})} for kind in ts_types}


class IncompleteGroup(GroupedDummy):
    name = "runner_incomplete_group_test"

    def generate_group(self, obj, data, results, ts_types):
        return {ts_types[0]: {K.TIMESERIES: pd.DataFrame()}}


@pytest.fixture
def grouped(monkeypatch):
    GroupedDummy.calls = []
    for cls in (GroupedDummy, IncompleteGroup):
        monkeypatch.setitem(method_registry, cls.name, cls)
    return GroupedDummy.calls


@pytest.mark.parametrize("first", [Types.HEATING, Types.COOLING])
def test_a_group_runs_once_for_the_types_that_select_it(grouped, first):
    strategies = {Types.HEATING: "runner_group_test", Types.COOLING: "RUNNER_GROUP_TEST"}
    runner = Runner({}, {}, strategies)

    runner.resolve(first)
    runner.run_methods()

    assert grouped == [("group", (Types.HEATING, Types.COOLING))]
    assert set(runner.results) == {Types.HEATING, Types.COOLING}


def test_a_group_leaves_out_types_that_select_another_strategy(grouped):
    runner = Runner({}, {}, {Types.HEATING: "runner_group_test", Types.COOLING: "cache_test"})

    runner.resolve(Types.HEATING)

    assert grouped == [("group", (Types.HEATING,))]
    assert set(runner.results) == {Types.HEATING}


def test_a_type_the_strategy_does_not_declare_runs_alone(grouped):
    runner = Runner({}, {}, {Types.HEATING: "runner_group_test", Types.HVAC: "runner_group_test"})

    runner.run_methods()

    assert grouped == [("group", (Types.HEATING,)), ("generate", Types.HVAC)]


def test_methods_without_a_group_hook_run_per_type():
    assert Method.generate_group is None
    assert CachingDummy.generate_group is None


def test_an_incomplete_group_leaves_no_partial_results(grouped):
    runner = Runner({}, {}, dict.fromkeys(IncompleteGroup.types, IncompleteGroup.name))

    with pytest.raises(ValueError, match="expected results for"):
        runner.resolve(Types.HEATING)
    assert not runner.results

    # A retry reports the same error, not a dependency cycle left over from the failed call.
    with pytest.raises(ValueError, match="expected results for"):
        runner.resolve(Types.COOLING)


def test_a_dependency_on_a_type_of_the_same_group_reports_a_cycle(grouped, monkeypatch):
    monkeypatch.setattr(GroupedDummy, "required_data", [Types.COOLING])
    runner = Runner({}, {}, dict.fromkeys(GroupedDummy.types, GroupedDummy.name))

    with pytest.raises(ValueError, match="Cyclic strategy dependency.*cooling"):
        runner.resolve(Types.HEATING)
    assert not runner.results
