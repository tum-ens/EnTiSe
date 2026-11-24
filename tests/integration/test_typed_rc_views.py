"""Typed heating and cooling from one solve of each RC model (#127).

Each RC model can fill the ``heating`` and ``cooling`` types of an object. The typed
results come from one solve and equal ``hvac`` with the matching controls. Every test
runs the first object of each model's example, with its weather, windows and gains.
"""

import functools
import logging
import os
import re
from importlib import import_module

import pandas as pd
import pytest

from entise import Generator
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.core.base import Method, method_registry
from entise.core.registry import get_strategy

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
RC_EXAMPLES = {"1R0C": "hvac_1r0c", "1R1C": "hvac_1r1c", "5R1C": "hvac_5r1c", "7R2C": "hvac_7r2c"}
MODELS = list(RC_EXAMPLES)
SOLVERS = {
    "1R0C": ("entise.methods.hvac.R1C0", "calculate_timeseries_1r0c"),
    "1R1C": ("entise.methods.hvac.R1C1", "calculate_timeseries_1r1c"),
    "5R1C": ("entise.methods.hvac.R5C1", "calculate_timeseries_5r1c"),
    "7R2C": ("entise.methods.hvac.R7C2", "calculate_timeseries_7r2c"),
}
TYPED = (Types.HEATING, Types.COOLING)
SELECTIONS = [(Types.HEATING,), (Types.COOLING,), TYPED]
SWITCH = {Types.HEATING: O.ACTIVE_HEATING, Types.COOLING: O.ACTIVE_COOLING}


@functools.cache
def _example(model: str) -> tuple[dict, dict]:
    example = RC_EXAMPLES[model]
    data = {}
    for folder in (os.path.join(EXAMPLES, "common_data"), os.path.join(EXAMPLES, example, "data")):
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".csv"):
                data[os.path.splitext(name)[0]] = pd.read_csv(os.path.join(folder, name))
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    obj = {key: value for key, value in obj.items() if key != Types.HVAC and not pd.isna(value)}
    return obj, data


def building(model: str, **keys) -> tuple[dict, dict]:
    """The example building of a model without any strategy, plus the given keys."""
    obj, data = _example(model)
    return {**obj, **keys}, dict(data)


def run(objects, data, **kwargs):
    gen = Generator(raise_on_error=True)
    gen.add_objects(objects)
    return gen.generate(data, workers=1, show_progress=False, **kwargs)


def run_one(obj, data, **kwargs):
    summary, timeseries = run([obj], data, **kwargs)
    return summary.loc[obj[O.ID]].dropna(), timeseries[obj[O.ID]]


def legacy(model, obj, data, selected, **kwargs):
    """``hvac`` with the controls of the selected types switched on and the others off."""
    controls = {SWITCH[kind]: kind in selected for kind in TYPED}
    return run_one({**obj, Types.HVAC: model, **controls}, data, **kwargs)


def view(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    return frame[[col for col in frame if col == C.TEMP_IN or col.startswith(f"{kind}{SEP}")]]


def own_keys(summary: pd.Series, kinds) -> pd.Series:
    return summary[[key for key in summary.index if key.split(SEP, 1)[0] in kinds]]


@pytest.fixture
def solves(monkeypatch):
    """Count the calls of each model's solver."""
    calls = []
    for model, (module_name, function) in SOLVERS.items():
        module = import_module(module_name)
        solver = getattr(module, function)

        def counted(*args, _model=model, _solver=solver, **kwargs):
            calls.append(_model)
            return _solver(*args, **kwargs)

        monkeypatch.setattr(module, function, counted)
    return calls


@pytest.mark.parametrize("selected", SELECTIONS, ids=lambda s: "+".join(s))
@pytest.mark.parametrize("model", MODELS)
def test_typed_views_equal_hvac_with_the_same_controls(model, selected):
    obj, data = building(model)
    legacy_summary, legacy_ts = legacy(model, obj, data, selected)
    summary, typed = run_one({**obj, **dict.fromkeys(selected, model)}, data)

    assert set(typed) == set(selected)
    for kind in selected:
        pd.testing.assert_frame_equal(typed[kind], view(legacy_ts[Types.HVAC], kind), check_exact=True)
    pd.testing.assert_series_equal(summary, own_keys(legacy_summary, selected), check_exact=True)


@pytest.mark.parametrize("order", [TYPED, TYPED[::-1]], ids=["heating-first", "cooling-first"])
@pytest.mark.parametrize("model", MODELS)
def test_one_solve_serves_both_types_in_either_order(model, order, solves, monkeypatch):
    class Consumer(Method):
        register = False
        name = "typed_load_consumer"
        types = [Types.HP]
        required_keys = []
        required_data = list(order)

        def generate(self, obj=None, data=None, results=None, ts_type=Types.HP, **inputs):
            loads = {kind: results[kind][K.TIMESERIES][f"{kind}{SEP}{C.LOAD}[W]"] for kind in order}
            return {K.TIMESERIES: pd.DataFrame(loads)}

    monkeypatch.setitem(method_registry, Consumer.name, Consumer)
    obj, data = building(model)
    # The consumer is listed first, so it asks for the typed loads before the runner reaches them.
    _, typed = run_one({Types.HP: Consumer.name, **obj, order[0]: model, order[1]: model.lower()}, data)

    assert solves == [model]
    for kind in order:
        pd.testing.assert_series_equal(
            typed[Types.HP][kind], typed[kind][f"{kind}{SEP}{C.LOAD}[W]"], check_names=False, check_exact=True
        )
    pd.testing.assert_series_equal(typed[Types.HEATING][C.TEMP_IN], typed[Types.COOLING][C.TEMP_IN])


@pytest.mark.parametrize("model", MODELS)
def test_another_strategy_keeps_the_other_type(model):
    obj, data = building(model)
    weather = data[O.WEATHER]
    data["metered_cooling"] = pd.DataFrame({C.DATETIME: weather[C.DATETIME], f"{Types.COOLING}{SEP}{C.LOAD}[W]": 1.0})
    legacy_summary, legacy_ts = legacy(model, obj, data, (Types.HEATING,))
    summary, typed = run_one(
        {**obj, Types.HEATING: model, Types.COOLING: "file", f"{Types.COOLING}{SEP}filename": "metered_cooling"},
        data,
    )

    pd.testing.assert_frame_equal(typed[Types.HEATING], view(legacy_ts[Types.HVAC], Types.HEATING))
    assert typed[Types.COOLING].columns.tolist() == [f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    assert typed[Types.COOLING].iloc[:, 0].eq(1.0).all()
    pd.testing.assert_series_equal(summary, own_keys(legacy_summary, [Types.HEATING]))


@pytest.mark.parametrize("model", MODELS)
def test_hvac_and_a_typed_key_on_the_same_model_raise(model):
    obj, data = building(model)
    with pytest.raises(ValueError, match=rf"'{Types.HVAC}'.*'{Types.HEATING}'.*{model}"):
        run_one({**obj, Types.HVAC: model, Types.HEATING: model}, data)


def test_hvac_of_one_model_runs_next_to_a_typed_view_of_another():
    obj, data = building("1R1C")
    _, hvac_only = run_one({**obj, Types.HVAC: "1R0C"}, data)
    _, heating_only = run_one({**obj, Types.HEATING: "1R1C"}, data)
    _, both = run_one({**obj, Types.HVAC: "1R0C", Types.HEATING: "1R1C"}, data)

    pd.testing.assert_frame_equal(both[Types.HVAC], hvac_only[Types.HVAC])
    pd.testing.assert_frame_equal(both[Types.HEATING], heating_only[Types.HEATING])


@pytest.mark.parametrize("model", MODELS)
def test_a_switched_off_control_stays_off_and_warns(model, caplog):
    obj, data = building(model, **{O.ACTIVE_HEATING: False})
    legacy_summary, legacy_ts = legacy(model, obj, data, ())
    with caplog.at_level(logging.WARNING):
        summary, typed = run_one({**obj, Types.HEATING: model}, data)

    pd.testing.assert_frame_equal(typed[Types.HEATING], view(legacy_ts[Types.HVAC], Types.HEATING))
    assert summary[f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] == 0
    assert any(O.ACTIVE_HEATING in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("model", ["5R1C", "7R2C"])
def test_a_heating_schedule_applies_to_the_typed_view(model):
    obj, data = building(model)
    weather = data[O.WEATHER]
    month = pd.to_datetime(weather[C.DATETIME], utc=True).dt.month
    data["season"] = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "on": month.isin([1, 2, 3, 11, 12]).astype(int)})
    obj = {**obj, O.ACTIVE_HEATING: "season|on"}
    legacy_summary, legacy_ts = run_one({**obj, Types.HVAC: model, O.ACTIVE_COOLING: False}, data)
    summary, typed = run_one({**obj, Types.HEATING: model}, data)

    pd.testing.assert_frame_equal(typed[Types.HEATING], view(legacy_ts[Types.HVAC], Types.HEATING))
    assert typed[Types.HEATING].loc[month.eq(7).to_numpy(), f"{Types.HEATING}{SEP}{C.LOAD}[W]"].eq(0).all()
    pd.testing.assert_series_equal(summary, own_keys(legacy_summary, [Types.HEATING]))


@pytest.mark.parametrize("model", MODELS)
def test_scoped_control_inputs_apply_to_their_control(model):
    obj, data = building(model)
    _, legacy_ts = run_one({**obj, Types.HVAC: model, O.POWER_HEATING: 1500, O.POWER_COOLING: 1000}, data)
    scoped = {f"{Types.HEATING}{SEP}{O.POWER_HEATING}": 1500, f"{Types.COOLING}{SEP}{O.POWER_COOLING}": 1000}
    _, typed = run_one({**obj, **scoped, **dict.fromkeys(TYPED, model)}, data)

    for kind in TYPED:
        pd.testing.assert_frame_equal(typed[kind], view(legacy_ts[Types.HVAC], kind))


@pytest.mark.parametrize("model", MODELS)
def test_scoped_building_inputs_must_agree(model):
    obj, data = building(model)
    scoped = {f"{Types.HEATING}{SEP}{O.TEMP_INIT}": 20, f"{Types.COOLING}{SEP}{O.TEMP_INIT}": 21}
    with pytest.raises(ValueError, match="different values"):
        run_one({**obj, **scoped, **dict.fromkeys(TYPED, model)}, data)


@pytest.mark.parametrize("model", MODELS)
def test_an_input_of_the_other_control_cannot_be_scoped_to_a_type(model):
    obj, data = building(model, **{f"{Types.HEATING}{SEP}{O.POWER_COOLING}": 500})
    key = re.escape(f"'{Types.HEATING}{SEP}{O.POWER_COOLING}'")
    with pytest.raises(ValueError, match=rf"{key}.*{Types.COOLING} control"):
        run_one({**obj, Types.HEATING: model}, data)


@pytest.mark.parametrize("utc", [False, True], ids=["local", "utc"])
@pytest.mark.parametrize("model", MODELS)
def test_typed_views_keep_the_hvac_index(model, utc):
    obj, data = building(model)
    _, legacy_ts = run_one({**obj, Types.HVAC: model}, data, utc=utc)
    _, typed = run_one({**obj, **dict.fromkeys(TYPED, model)}, data, utc=utc)

    for kind in TYPED:
        pd.testing.assert_index_equal(typed[kind].index, legacy_ts[Types.HVAC].index)


@pytest.mark.parametrize("kind", TYPED)
@pytest.mark.parametrize("model", MODELS)
def test_a_direct_typed_call_equals_the_generator(model, kind):
    obj, data = building(model)
    summary, typed = run_one({**obj, kind: model}, data)
    result = get_strategy(model)().generate(obj, data, ts_type=kind)

    pd.testing.assert_frame_equal(result[K.TIMESERIES], typed[kind])
    assert result[K.SUMMARY] == summary.to_dict()


@pytest.mark.parametrize("model", MODELS)
def test_a_keyword_of_a_direct_typed_call_overrides_the_scoped_key(model):
    obj, data = building(model, **{f"{Types.HEATING}{SEP}{O.POWER_HEATING}": 500})
    by_keyword = get_strategy(model)().generate(obj, data, ts_type=Types.HEATING, power_heating=1500)
    # The reference drops the scoped key, which would otherwise win over the plain one.
    plain = {key: value for key, value in obj.items() if not key.startswith(f"{Types.HEATING}{SEP}")}
    by_key = get_strategy(model)().generate({**plain, O.POWER_HEATING: 1500}, data, ts_type=Types.HEATING)

    heat = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
    assert by_keyword[K.TIMESERIES].columns.tolist() == [C.TEMP_IN, heat]
    pd.testing.assert_frame_equal(by_keyword[K.TIMESERIES], by_key[K.TIMESERIES])
    assert by_keyword[K.TIMESERIES][heat].max() > 500


@pytest.mark.parametrize("model", MODELS)
def test_scoped_table_references_must_agree(model):
    obj, data = building(model)
    data["other_windows"] = data[O.WINDOWS]
    scoped = {f"{Types.HEATING}{SEP}{O.WINDOWS}": O.WINDOWS, f"{Types.COOLING}{SEP}{O.WINDOWS}": "other_windows"}
    with pytest.raises(ValueError, match="different values"):
        run_one({**obj, **scoped, **dict.fromkeys(TYPED, model)}, data)


@pytest.mark.parametrize("model", MODELS)
def test_every_table_input_is_a_declared_key(model):
    # The typed results resolve the object through the declared keys only.
    method = get_strategy(model)
    tables = {key for key in (*method.required_data, *method.optional_data) if key not in Types}
    assert tables <= set(method.key_defaults())


@pytest.mark.parametrize("model", MODELS)
def test_a_scoped_table_reference_is_read(model):
    obj, data = building(model)
    _, legacy_ts = legacy(model, obj, data, (Types.HEATING,))
    scoped = {key: value for key, value in obj.items() if key != O.WINDOWS}
    scoped[f"{Types.HEATING}{SEP}{O.WINDOWS}"] = obj[O.WINDOWS]
    _, typed = run_one({**scoped, Types.HEATING: model}, data)

    pd.testing.assert_frame_equal(typed[Types.HEATING], view(legacy_ts[Types.HVAC], Types.HEATING))


@pytest.mark.parametrize("model", MODELS)
def test_inputs_scoped_to_hvac_are_not_read_for_typed_results(model):
    obj, data = building(model)
    _, plain = run_one({**obj, Types.HEATING: model}, data)
    _, scoped = run_one({**obj, f"{Types.HVAC}{SEP}{O.POWER_HEATING}": 100, Types.HEATING: model}, data)

    pd.testing.assert_frame_equal(scoped[Types.HEATING], plain[Types.HEATING])
