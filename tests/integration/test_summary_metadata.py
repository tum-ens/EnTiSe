"""Each method declares the summary keys it reports (``output_summary``), as the method pages list them."""

import os

import pandas as pd
import pytest

from entise.constants import VALID_TYPES
from entise.constants import Objects as O
from entise.core.generator import Generator, _merge_summaries
from entise.core.registry import get_strategy

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
# PyLPG runs an external engine, and DistrictHeatingSim is installed by hand; their own tests cover them.
SKIP = {"electricity_pylpg": "pylpg", "heat_districtheatingsim": "districtheatingsim"}


def _examples() -> list[str]:
    return sorted(name for name in os.listdir(EXAMPLES) if os.path.isfile(os.path.join(EXAMPLES, name, "objects.csv")))


def _tables(example: str) -> dict:
    tables = {}
    for folder in (os.path.join(EXAMPLES, "common_data"), os.path.join(EXAMPLES, example, "data")):
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                stem, ext = os.path.splitext(name)
                if ext == ".csv":
                    tables[stem] = pd.read_csv(os.path.join(folder, name))
    # Two weeks are enough to produce every summary key.
    tables[O.WEATHER] = tables[O.WEATHER].iloc[: 14 * 24]
    return tables


@pytest.mark.parametrize("example", _examples())
def test_summary_keys_are_the_declared_output_summary(example):
    if example in SKIP:
        pytest.skip(f"{example} needs {SKIP[example]}")
    obj = pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv")).iloc[0].to_dict()
    methods = {t: get_strategy(obj[t]) for t in VALID_TYPES if t in obj and pd.notna(obj[t])}

    generator = Generator(raise_on_error=True)
    generator.add_objects([obj])
    summary, _ = generator.generate(_tables(example), workers=1, show_progress=False)

    declared = _merge_summaries(obj[O.ID], {t: dict.fromkeys(m.output_summary) for t, m in methods.items()})
    emitted = set(summary.columns)
    # A method serving several types may report only part of its keys for a secondary type,
    # but every key it reports is declared, and its primary type reports all of them.
    assert emitted <= set(declared), {t: m.name for t, m in methods.items()}
    # The file loader serves every type as an unordered set, so each of them counts as primary.
    if all(not isinstance(m.types, (list, tuple)) or t == m.types[0] for t, m in methods.items()):
        assert emitted == set(declared), {t: m.name for t, m in methods.items()}
