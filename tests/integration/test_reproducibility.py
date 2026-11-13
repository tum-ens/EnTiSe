"""Generating the same object again must give bit-identical time series.

The result of an object may not depend on the order of the objects, on the number of
workers, on the batch size, or on the process that happens to calculate it.
"""

import os

import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.core.generator import Generator

HERE = os.path.dirname(__file__)
HOURS = 14 * 24

# How the objects are run: (workers, batch_size, reverse the object order)
REFERENCE = (1, None, False)
VARIANTS = {
    "reversed order": (1, None, True),
    "two workers, one object per batch": (2, 1, False),
    "two workers, even batches, reversed order": (2, None, True),
}


def _weather() -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"), parse_dates=True)
    return weather.iloc[:HOURS].reset_index(drop=True)


def _hvac_1r1c():
    folder = os.path.join(HERE, "hvac_rc")
    objects = pd.read_csv(os.path.join(folder, "objects.csv"))
    objects = objects.loc[objects["hvac"] == "1R1C"]
    data = {"weather": _weather()}
    for name in ("windows", "internal_gains", "ventilation"):
        frame = pd.read_csv(os.path.join(folder, "data", f"{name}.csv"), parse_dates=True)
        data[name] = frame.iloc[:HOURS] if len(frame) > HOURS else frame
    return objects, data


def _hp_ruhnau():
    objects = pd.DataFrame(
        {
            O.ID: [1, 2, 3],
            "hp": "ruhnau",
            O.WEATHER: "weather",
            "hp_source": ["air", "soil", "water"],
            "hp_sink": ["radiator", "floor", "radiator"],
        }
    )
    return objects, {"weather": _weather()}


def _dhw_jordanvajen():
    objects = pd.read_csv(os.path.join(HERE, "dhw_jordanvajen", "objects.csv"))
    objects[O.SEED] = 7
    return objects, {"weather": _weather()}


def _pv_pvlib():
    objects = pd.read_csv(os.path.join(HERE, "pv_pvlib", "objects.csv"))
    return objects, {"weather": _weather()}


CASES = {
    "hvac_1r1c": _hvac_1r1c,
    "hp_ruhnau": _hp_ruhnau,
    "dhw_jordanvajen": _dhw_jordanvajen,
    "pv_pvlib": _pv_pvlib,
}


def _generate(objects: pd.DataFrame, data: dict, workers: int, batch_size, reverse: bool):
    gen = Generator(raise_on_error=True)
    gen.add_objects(objects.iloc[::-1] if reverse else objects)
    return gen.generate(data, workers=workers, batch_size=batch_size, show_progress=False)


@pytest.mark.parametrize("case", CASES)
def test_timeseries_are_bit_identical_however_the_objects_are_run(case):
    objects, data = CASES[case]()
    assert len(objects) > 1

    ref_summary, ref_timeseries = _generate(objects, data, *REFERENCE)
    assert set(ref_timeseries) == set(objects[O.ID])

    for label, variant in VARIANTS.items():
        summary, timeseries = _generate(objects, data, *variant)
        assert set(timeseries) == set(ref_timeseries), label
        for obj_id, by_type in ref_timeseries.items():
            assert set(timeseries[obj_id]) == set(by_type), f"{label}, object {obj_id}"
            for ts_type, expected in by_type.items():
                pd.testing.assert_frame_equal(
                    timeseries[obj_id][ts_type],
                    expected,
                    check_exact=True,
                    obj=f"{case} ({label}), object {obj_id}, {ts_type}",
                )
        pd.testing.assert_frame_equal(summary.loc[ref_summary.index], ref_summary, check_exact=True, obj=label)
