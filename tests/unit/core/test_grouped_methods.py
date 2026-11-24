"""A method that calculates several of its types in one call (``GroupedMethod``).

The method here is not an RC model. It stands for a demand generator that yields
heat, cold and electricity from shared inputs, so the tests pin what any such method
gets from core: one call per object, one result per selected type, and inputs that
are resolved for all the selected types at once.
"""

import numpy as np
import pandas as pd
import pytest

from entise import Generator
from entise.constants import SEP, Types
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.core.base import Method, method_registry
from entise.core.grouped import GroupedMethod, split_by_type

DEMAND = "annual_demand[Wh]"
SHARE = "electricity_share"
HOURS = 24
KINDS = (Types.HEATING, Types.COOLING, Types.ELECTRICITY)


class SiteDemand(GroupedMethod, Method):
    register = False
    name = "test_site_demand"
    types = list(KINDS)
    required_keys = []
    optional_keys = {DEMAND: 8760.0, SHARE: 0.25}
    shared_columns = ("occupied[1]",)
    output_timeseries = {f"{kind}{SEP}load[W]": f"{kind} load" for kind in KINDS}
    output_summary = {f"{kind}{SEP}demand[Wh]": f"{kind} demand" for kind in KINDS}
    calls: list = []

    def generate(self, obj=None, data=None, results=None, ts_type=Types.HEATING, **inputs):
        obj, data = self._process_kwargs(obj, data, ts_type, **inputs)
        return self.group_result(obj, data, results, ts_type)

    def generate_combined(self, obj, data, results, ts_types):
        SiteDemand.calls.append((ts_types, obj[DEMAND], obj[SHARE]))
        index = pd.date_range("2025-01-01", periods=HOURS, freq="h", tz="UTC")
        hourly = obj[DEMAND] / HOURS
        loads = {
            Types.HEATING: hourly * (1 - obj[SHARE]) * 0.8,
            Types.COOLING: hourly * (1 - obj[SHARE]) * 0.2,
            Types.ELECTRICITY: hourly * obj[SHARE],
        }
        frame = pd.DataFrame({f"{kind}{SEP}load[W]": np.full(HOURS, load) for kind, load in loads.items()}, index)
        frame["occupied[1]"] = 1
        summary = {f"{kind}{SEP}demand[Wh]": float(frame[f"{kind}{SEP}load[W]"].sum()) for kind in KINDS}
        return {K.TIMESERIES: frame, K.SUMMARY: summary}


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    SiteDemand.calls = []
    monkeypatch.setitem(method_registry, SiteDemand.name, SiteDemand)
    return SiteDemand.calls


def run(obj):
    gen = Generator(raise_on_error=True)
    gen.add_objects([obj])
    summary, timeseries = gen.generate({}, workers=1, show_progress=False)
    return summary.loc[obj[O.ID]].dropna(), timeseries[obj[O.ID]]


@pytest.mark.parametrize("order", [(Types.ELECTRICITY, Types.HEATING), (Types.HEATING, Types.ELECTRICITY)])
def test_one_call_fills_the_selected_types(registered, order):
    summary, timeseries = run({O.ID: "site", **dict.fromkeys(order, SiteDemand.name)})

    assert registered == [((Types.HEATING, Types.ELECTRICITY), 8760.0, 0.25)]
    assert set(timeseries) == {Types.HEATING, Types.ELECTRICITY}
    for kind in order:
        assert timeseries[kind].columns.tolist() == [f"{kind}{SEP}load[W]", "occupied[1]"]
    assert sorted(summary.index) == [f"{Types.ELECTRICITY}{SEP}demand[Wh]", f"{Types.HEATING}{SEP}demand[Wh]"]


def test_inputs_are_resolved_for_the_selected_types(registered):
    run(
        {
            O.ID: "site",
            Types.HEATING: SiteDemand.name,
            Types.ELECTRICITY: SiteDemand.name,
            f"{Types.ELECTRICITY}{SEP}{SHARE}": 0.5,
            # Cooling is not selected, so its scoped key is not read.
            f"{Types.COOLING}{SEP}{DEMAND}": 1.0,
        }
    )

    assert registered == [((Types.HEATING, Types.ELECTRICITY), 8760.0, 0.5)]


def test_inputs_given_for_two_selected_types_must_agree():
    scoped = {f"{Types.HEATING}{SEP}{DEMAND}": 1000.0, f"{Types.ELECTRICITY}{SEP}{DEMAND}": 2000.0}
    with pytest.raises(ValueError, match="different values"):
        run({O.ID: "site", Types.HEATING: SiteDemand.name, Types.ELECTRICITY: SiteDemand.name, **scoped})


def test_a_direct_call_returns_the_result_of_its_type(registered):
    summary, timeseries = run({O.ID: "site", Types.ELECTRICITY: SiteDemand.name})
    registered.clear()

    result = SiteDemand().generate({O.ID: "site"}, {}, ts_type=Types.ELECTRICITY)

    assert registered == [((Types.ELECTRICITY,), 8760.0, 0.25)]
    pd.testing.assert_frame_equal(result[K.TIMESERIES], timeseries[Types.ELECTRICITY])
    assert result[K.SUMMARY] == summary.to_dict()


def test_split_by_type_keeps_the_shared_columns_in_every_type():
    frame = pd.DataFrame({"heating:load[W]": [1.0], "cooling:load[W]": [2.0], "occupied[1]": [1]})
    result = {K.TIMESERIES: frame, K.SUMMARY: {"heating:demand[Wh]": 1.0, "cooling:demand[Wh]": 2.0}}

    parts = split_by_type(result, [Types.COOLING], shared_columns=["occupied[1]"])

    assert list(parts) == [Types.COOLING]
    assert parts[Types.COOLING][K.TIMESERIES].columns.tolist() == ["cooling:load[W]", "occupied[1]"]
    assert parts[Types.COOLING][K.SUMMARY] == {"cooling:demand[Wh]": 2.0}
