"""Every declared input of an RC model changes its result (#119).

For the first object of each RC example, one declared key at a time takes another
value. A key that is declared but never read keeps the result unchanged, which is
how ``ventilation_split``, ``H_ve[W K-1]`` and supplied solar gains went unnoticed.
Keys whose effect needs another input are listed with the reason; they are marked
as expected failures, so the test reports when one of them starts to act.
"""

import functools
import os

import pandas as pd
import pytest

from entise import Generator
from entise.constants import Objects as O
from entise.core.registry import get_strategy

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
RC_EXAMPLES = {"1R0C": "hvac_1r0c", "1R1C": "hvac_1r1c", "5R1C": "hvac_5r1c", "7R2C": "hvac_7r2c"}

# Not changed here: the site's time zone moves only the output index, and these name tables or columns.
NOT_VALUES = {O.TIMEZONE, O.WEATHER, O.WINDOWS, O.GAINS_INTERNAL_COL, O.VENTILATION_COL, O.T_EQ, O.T_EQ_COL}

# Values for keys that the example objects leave unset and that have no default.
UNSET = {
    O.TEMP_SUPPLY: 20.0,
    O.H_VE: 500.0,
    O.GAINS_SOLAR: 0.0,
    O.H_TR_OP_SKY: 50.0,
    O.AREA_M: 350.0,
    O.AREA_TOT: 630.0,
}

NEEDS_ANOTHER_INPUT = {
    (model, key): "area and height only convert ventilation given in air changes per hour"
    for model in ("1R0C", "1R1C")
    for key in (O.AREA, O.HEIGHT)
}
NEEDS_ANOTHER_INPUT |= {
    ("5R1C", O.TEMP_SUPPLY): "the supply temperature acts on the share that ventilation_split gives it",
    ("7R2C", O.TEMP_SUPPLY): "the supply temperature acts on the share that ventilation_split gives it",
    ("5R1C", O.VENTILATION_SPLIT): "the split acts only with a supply temperature other than the outdoor air",
    ("7R2C", O.VENTILATION_SPLIT): "the split acts only with a supply temperature other than the outdoor air",
    ("5R1C", O.AREA_M): "the mass area splits the gains only together with the total area",
    ("5R1C", O.AREA_TOT): "the total area splits the gains only together with the mass area",
}


@functools.cache
def _inputs(model: str) -> tuple[dict, dict]:
    example = RC_EXAMPLES[model]
    data = {}
    for folder in (os.path.join(EXAMPLES, "common_data"), os.path.join(EXAMPLES, example, "data")):
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".csv"):
                data[os.path.splitext(name)[0]] = pd.read_csv(os.path.join(folder, name))
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    obj = {key: value for key, value in obj.items() if not (isinstance(value, float) and pd.isna(value))}
    # A location gives the example's windows solar gains.
    return {O.LAT: 49.72, O.LON: 11.06, **obj}, data


def _result(obj: dict, data: dict) -> tuple:
    generator = Generator(raise_on_error=True)
    generator.add_objects(pd.DataFrame([obj]))
    summary, series = generator.generate(data, workers=1)
    temperature = series[obj[O.ID]]["hvac"].iloc[:, 0]
    return (*summary.iloc[0].round().tolist(), round(float(temperature.mean()), 4))


@functools.cache
def _base(model: str) -> tuple:
    return _result(*_inputs(model))


def _other_value(key: str, value):
    """Return a different, valid value for a key, or None if there is no rule for it."""
    if value is None:
        return UNSET.get(key)
    if key.startswith("active_"):
        return not bool(value)
    number = float(value)
    if number == float("inf"):
        return 2000.0
    if "temperature" in key:
        return number + 2.0
    if key in (O.LAT, O.LON):
        return number + 5.0
    if any(part in key for part in ("sigma", "fraction", "split", "humidity")):
        return 0.9 if number < 0.5 else 0.1
    return number * 2.0 + 1.0


def _cases():
    for model in RC_EXAMPLES:
        for key in get_strategy(model).key_defaults():
            if key in NOT_VALUES:
                continue
            reason = NEEDS_ANOTHER_INPUT.get((model, key))
            marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
            yield pytest.param(model, key, id=f"{model}-{key}", marks=marks)


@pytest.mark.parametrize("model, key", list(_cases()))
def test_a_declared_input_changes_the_result(model, key):
    obj, data = _inputs(model)
    current = obj.get(key, get_strategy(model).key_defaults()[key])
    value = _other_value(key, current)
    assert value is not None, f"no value to try for '{key}': add one to UNSET"

    assert _result({**obj, key: value}, data) != _base(model), f"{model} ignores '{key}' ({current!r} -> {value!r})"


def test_5r1c_splits_the_gains_by_mass_and_total_area_together():
    obj, data = _inputs("5R1C")

    assert _result({**obj, O.AREA_M: UNSET[O.AREA_M], O.AREA_TOT: UNSET[O.AREA_TOT]}, data) != _base("5R1C")
