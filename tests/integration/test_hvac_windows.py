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
