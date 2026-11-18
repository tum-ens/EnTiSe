"""The example objects use keys and tables that the configured methods read."""

import os

import pandas as pd
import pytest

from entise.constants import VALID_TYPES
from entise.constants import Objects as O
from entise.core.registry import get_strategy

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "examples")
TABLE_KEYS = (O.WEATHER, O.WINDOWS, O.DATETIMES)


def _objects(example: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(EXAMPLES, example, "objects.csv"))


def _table_names(example: str) -> set[str]:
    folders = [os.path.join(EXAMPLES, "common_data"), os.path.join(EXAMPLES, example, "data")]
    return {os.path.splitext(name)[0] for folder in folders if os.path.isdir(folder) for name in os.listdir(folder)}


def _examples() -> list[str]:
    return sorted(name for name in os.listdir(EXAMPLES) if os.path.isfile(os.path.join(EXAMPLES, name, "objects.csv")))


@pytest.mark.parametrize("example", _examples())
def test_example_objects_reference_tables_that_exist(example):
    objects = _objects(example)
    tables = _table_names(example)

    for key in TABLE_KEYS:
        if key in objects:
            missing = set(objects[key].dropna()) - tables
            assert not missing, f"{example}: '{key}' names tables that do not exist: {sorted(missing)}"


@pytest.mark.parametrize("example", ["dhw_jordanvajen", "electricity_demandlib"])
def test_example_object_columns_are_keys_of_the_method(example):
    objects = _objects(example)
    keys = {O.ID}
    for ts_type in VALID_TYPES:
        if ts_type in objects:
            keys.add(ts_type)
            for strategy in objects[ts_type].dropna().unique():
                method = get_strategy(strategy)
                keys.update(method.required_keys, method.optional_keys)

    assert set(objects.columns) <= keys, f"{example}: unknown columns {sorted(set(objects.columns) - keys)}"
