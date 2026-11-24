"""An HVAC object that names a windows table needs that table in the data."""

import os

import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.core.generator import Generator

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")


def _inputs(example: str) -> tuple[dict, dict]:
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    folder = os.path.join(EXAMPLES, example, "data")
    data = {os.path.splitext(name)[0]: pd.read_csv(os.path.join(folder, name)) for name in os.listdir(folder)}
    data[O.WEATHER] = pd.read_csv(os.path.join(EXAMPLES, "common_data", "weather.csv")).iloc[:24]
    return obj, data


@pytest.mark.parametrize("example", ["hvac_1r0c", "hvac_1r1c", "hvac_5r1c", "hvac_7r2c"])
def test_missing_windows_table_raises(example):
    obj, data = _inputs(example)
    obj[O.WINDOWS] = "no_such_table"
    generator = Generator(raise_on_error=True)
    generator.add_objects([obj])

    with pytest.raises(ValueError, match="no_such_table"):
        generator.generate(data, workers=1, show_progress=False)


@pytest.mark.parametrize("example", ["hvac_1r0c", "hvac_1r1c", "hvac_5r1c", "hvac_7r2c"])
def test_windows_add_solar_gains(example):
    obj, data = _inputs(example)
    data[O.WEATHER] = pd.read_csv(os.path.join(EXAMPLES, "common_data", "weather.csv")).iloc[4000:4168]
    with_windows = {**obj, O.WINDOWS: "windows"}
    without_windows = {k: v for k, v in obj.items() if k != O.WINDOWS}

    results = []
    for case in (with_windows, without_windows):
        generator = Generator(raise_on_error=True)
        generator.add_objects([case])
        summary, _ = generator.generate(data, workers=1, show_progress=False)
        results.append(summary.iloc[0])

    assert not results[0].equals(results[1])


RC_EXAMPLES = ["hvac_1r0c", "hvac_1r1c", "hvac_5r1c", "hvac_7r2c"]
LOCATION = {O.LAT: 49.72, O.LON: 11.06}


def _run(obj: dict, data: dict):
    generator = Generator(raise_on_error=True)
    generator.add_objects([obj])
    return generator.generate(data, workers=1, show_progress=False)


@pytest.mark.parametrize("example", RC_EXAMPLES)
def test_windows_without_a_location_raise(example):
    """Solar gains need the location; without it, named windows would silently add nothing."""
    obj, data = _inputs(example)
    obj = {k: v for k, v in {**obj, O.WINDOWS: "windows"}.items() if k not in LOCATION}

    with pytest.raises(ValueError, match="latitude"):
        _run(obj, data)


@pytest.mark.parametrize("example", RC_EXAMPLES)
def test_windows_without_rows_for_the_object_raise(example):
    obj, data = _inputs(example)
    obj = {**obj, **LOCATION, O.WINDOWS: "windows", O.ID: "not_in_the_table"}

    with pytest.raises(ValueError, match="not_in_the_table"):
        _run(obj, data)


def test_an_id_of_another_type_than_the_table_ids_raises_and_names_both_types():
    obj, data = _inputs("hvac_1r1c")
    windows = data["windows"]
    data["windows"] = windows[windows[O.ID] == obj[O.ID]].assign(**{O.ID: 1})
    obj = {**obj, **LOCATION, O.WINDOWS: "windows", O.ID: "1"}

    with pytest.raises(ValueError, match=r"'1' \(str\).*1 \(int\)"):
        _run(obj, data)


@pytest.mark.parametrize("example", RC_EXAMPLES)
@pytest.mark.parametrize("opt_out", ["inactive", "supplied"])
def test_objects_without_calculated_solar_gains_need_no_location(example, opt_out):
    """Switched-off or supplied solar gains don't use the windows' location."""
    obj, data = _inputs(example)
    obj = {k: v for k, v in {**obj, O.WINDOWS: "windows"}.items() if k not in LOCATION}
    if opt_out == "inactive":
        obj[O.ACTIVE_GAINS_SOLAR] = False
    else:
        obj[O.GAINS_SOLAR] = 300.0

    summary, _ = _run(obj, data)
    assert len(summary) == 1
