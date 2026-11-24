"""The RC models use the inputs they declare (#119).

Each case changes one declared input of a shipped example object and checks that
the result follows. Before #119, 5R1C dropped ``ventilation_split``, 5R1C and 7R2C
ignored ``H_ve[W K-1]``, and all four models replaced supplied solar gains with
calculated ones.
"""

import os

import pandas as pd
import pytest

from entise import Generator
from entise.constants import Objects as O

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
RC_EXAMPLES = {"1R0C": "hvac_1r0c", "1R1C": "hvac_1r1c", "5R1C": "hvac_5r1c", "7R2C": "hvac_7r2c"}
LOCATION = {O.LAT: 49.72, O.LON: 11.06}
DEMANDS = ["heating:demand[Wh]", "cooling:demand[Wh]"]


def _inputs(example: str) -> tuple[dict, dict]:
    """Return the first object of an example, with a location so its windows get sun, and its tables."""
    data = {}
    for folder in (os.path.join(EXAMPLES, "common_data"), os.path.join(EXAMPLES, example, "data")):
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".csv"):
                data[os.path.splitext(name)[0]] = pd.read_csv(os.path.join(folder, name))
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    return {**obj, **{key: obj.get(key) if pd.notna(obj.get(key)) else value for key, value in LOCATION.items()}}, data


def _demands(obj: dict, data: dict) -> list[float]:
    generator = Generator(raise_on_error=True)
    generator.add_objects(pd.DataFrame([obj]))
    summary, _ = generator.generate(data, workers=1)
    return [float(summary[column].iloc[0]) for column in DEMANDS]


def _without_solar_gains(model: str, obj: dict) -> dict:
    """1R0C has no solar gains flag; without windows it gets no solar gains."""
    if model == "1R0C":
        return {key: value for key, value in obj.items() if key != O.WINDOWS}
    return {**obj, O.ACTIVE_GAINS_SOLAR: False}


def _gains_table(data: dict, column: str, value: float) -> pd.DataFrame:
    return pd.DataFrame({"datetime": data["weather"]["datetime"], column: value})


@pytest.mark.parametrize("model", ["5R1C", "7R2C"])
def test_ventilation_split_moves_ventilation_to_the_supply_air(model):
    obj, data = _inputs(RC_EXAMPLES[model])
    obj = {**obj, O.TEMP_SUPPLY: 20.0}

    assert _demands({**obj, O.VENTILATION_SPLIT: 0.0}, data) != _demands({**obj, O.VENTILATION_SPLIT: 1.0}, data)


@pytest.mark.parametrize("model", ["5R1C", "7R2C"])
def test_ventilation_coefficient_takes_the_place_of_the_ventilation_input(model):
    obj, data = _inputs(RC_EXAMPLES[model])

    with_coefficient = _demands({**obj, O.H_VE: 500.0}, data)

    assert with_coefficient == _demands({**obj, O.VENTILATION: 500.0}, data)
    assert with_coefficient != _demands(obj, data)


@pytest.mark.parametrize("model", list(RC_EXAMPLES))
def test_supplied_solar_gains_take_the_place_of_calculated_ones(model):
    obj, data = _inputs(RC_EXAMPLES[model])
    solar_off = _demands(_without_solar_gains(model, obj), data)
    assert _demands(obj, data) != solar_off, "the example object needs calculated solar gains"

    assert _demands({**obj, O.GAINS_SOLAR: 0.0}, data) == solar_off


@pytest.mark.parametrize("model", list(RC_EXAMPLES))
def test_supplied_solar_gains_can_be_a_table(model):
    obj, data = _inputs(RC_EXAMPLES[model])
    constant = _demands({**obj, O.GAINS_SOLAR: 1000.0}, data)
    assert constant != _demands(obj, data)

    by_id = {**data, "solar": _gains_table(data, str(obj[O.ID]), 1000.0)}
    by_column = {**data, "solar": _gains_table(data, "south", 1000.0)}

    assert _demands({**obj, O.GAINS_SOLAR: "solar"}, by_id) == constant
    assert _demands({**obj, O.GAINS_SOLAR: "solar|south"}, by_column) == constant


def test_active_gains_solar_off_overrides_supplied_gains():
    obj, data = _inputs(RC_EXAMPLES["1R1C"])

    assert _demands({**obj, O.GAINS_SOLAR: 1000.0, O.ACTIVE_GAINS_SOLAR: False}, data) == _demands(
        {**obj, O.ACTIVE_GAINS_SOLAR: False}, data
    )
